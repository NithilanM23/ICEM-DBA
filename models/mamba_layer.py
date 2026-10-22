import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .icem_dba_modules import (
    MACGPathfinder,
    DirectionalBoundaryAffinity,
    InstanceConditionedExemplarMemory,
    GumbelPromptRouter
)

# =========================================================================
# Pure PyTorch Selective Scan with DIG-SM (C + P_dyn) and DBA-T (dt * guide)
# =========================================================================

class PyTorchSelectiveScan(nn.Module):
    """
    Pure PyTorch implementation of Selective Scan SSM with:
    1. DIG-SM Non-Causal Prompt Injection: C_att = C + P_dyn (Section 3.1 & 3.2 in paper)
    2. DBA-T Continuous Discretization Modulation: Delta_guided = Delta * (1 + tanh(W_align * kappa)) (Section 3.3)
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.x_proj = nn.Linear(self.d_inner, self.d_inner // 16 + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner // 16, self.d_inner, bias=True)
        # Initialize dt_proj bias to negative value for stable initialization (~0.05)
        nn.init.constant_(self.dt_proj.bias, -3.0)

        # S4D real initialization for A
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, u, P_dyn=None, delta_guide=None):
        """
        u: [B, L, D] sequence tokens in MACG traversal order
        P_dyn: [B, L, d_state] optional dynamic prompt injection from GumbelPromptRouter
        delta_guide: [B, L, 1] optional 2D boundary gradient alignment factor from DBA-T
        Returns: [B, L, D]
        """
        B, L, D = u.shape
        xz = self.in_proj(u)  # [B, L, 2 * d_inner]
        x, z = xz.chunk(2, dim=-1)

        # 1D Depthwise causal convolution
        x_conv = x.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = F.silu(x_conv).transpose(1, 2)  # [B, L, d_inner]

        # Project for dt, B, C
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + 2 * d_state]
        dt_rank = self.d_inner // 16
        dt, B_ssm, C_ssm = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).clamp(min=1e-4, max=5.0)  # [B, L, d_inner]

        # --- Component C (DBA-T): Modulate continuous SSM discretization dt ---
        if delta_guide is not None:
            dt = dt * delta_guide  # [B, L, d_inner]

        # Discrete A, B computed in float32 for numerical stability in FP16/AMP
        dt_f = dt.float()
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        deltaA = torch.exp(torch.einsum('bld,dn->bldn', dt_f, A).clamp(max=0.0))  # [B, L, d_inner, d_state]
        deltaB_u = torch.einsum('bld,bln,bld->bldn', dt_f, B_ssm.float(), x_conv.float())  # [B, L, d_inner, d_state]

        # --- Component A (DIG-SM): Prompt Injection C_att = C + P_dyn ---
        if P_dyn is not None:
            C_att = (C_ssm + P_dyn).float()  # [B, L, d_state]
        else:
            C_att = C_ssm.float()

        # Recurrent state propagation maintained strictly in float32 to prevent overflow
        h = torch.zeros(B, self.d_inner, self.d_state, device=u.device, dtype=torch.float32)
        ys = []
        for t in range(L):
            h = deltaA[:, t] * h + deltaB_u[:, t]
            y_t = torch.einsum('bdn,bn->bd', h, C_att[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1).to(u.dtype)  # [B, L, d_inner]
        y = y + x_conv * self.D.to(u.dtype)
        y = y * F.silu(z)
        out = self.out_proj(y)  # [B, L, D]
        return out


# =========================================================================
# Hardware Mamba Dispatcher (Triton / CUDA Kernel & Fallback)
# =========================================================================

class MambaDispatchModule(nn.Module):
    """
    Manages dispatch between hardware-accelerated Triton/CUDA Mamba kernels and
    the pure PyTorch Selective Scan implementation.
    Documents and cleanly handles DBA-T step modulation and DIG-SM prompt injection.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        
        try:
            from mamba_ssm import Mamba
            self.cuda_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.use_cuda_kernel = True
            self.prompt_proj = nn.Linear(d_state, d_model, bias=False)
        except (ImportError, Exception):
            self.cuda_mamba = None
            self.use_cuda_kernel = False

        self.pytorch_mamba = PyTorchSelectiveScan(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, u, P_dyn=None, delta_guide=None):
        """
        If running with precompiled CUDA kernel (mamba_ssm), delta_guide modulates
        pre-SSM throughput, and P_dyn is injected as an unattenuated residual projection.
        When running in pure PyTorch (or custom Triton loop), delta_guide and P_dyn
        are directly injected inside the discrete state transition and C_att readout.
        """
        if self.use_cuda_kernel and u.is_cuda:
            # Hardware-accelerated Triton/CUDA path
            u_guided = u * delta_guide if delta_guide is not None else u
            out = self.cuda_mamba(u_guided)
            if P_dyn is not None:
                out = out + self.prompt_proj(P_dyn)
            return out
        else:
            # Mathematical reference path (PyTorch Selective Scan)
            return self.pytorch_mamba(u, P_dyn=P_dyn, delta_guide=delta_guide)


def build_mamba_block(d_model, d_state=16, d_conv=4, expand=2):
    return MambaDispatchModule(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)


# =========================================================================
# Unified ICEM-DBA Mamba Layer
# =========================================================================

class ICEM_DBAMambaLayer(nn.Module):
    """
    Unified Mamba Layer integrating:
    1. VAMamba MACG Macro-Level Pathfinding (confidence-guided scan from known -> boundary -> hole)
    2. Component A: DIG-SM (Dynamic Instance-Grounded Semantic Modulation via GumbelPromptRouter)
    3. Component B: LC-VER (Linear-Complexity Visible Exemplar Readout fused in 2D coordinate space)
    4. Component C: DBA-T (Directional Boundary Affinity Transition with physical 2D gradient)
    5. Mamba State Space Engine with O(L) complexity
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, inner_rank=32, d_key=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.inner_rank = inner_rank
        self.d_key = d_key

        # Global base prompt dictionary N_global for RSSD / DIG-SM
        self.shared_N = nn.Parameter(torch.randn(inner_rank, d_state) * 0.02)

        # Macro-Level Confidence Pathfinder (VAMamba core)
        self.pathfinder = MACGPathfinder(dim=dim)

        # Component C: Directional Boundary Affinity Transition
        self.dba_t = DirectionalBoundaryAffinity(dim=dim)

        # Components A & B: Instance-Conditioned Exemplar Memory
        self.icem_memory = InstanceConditionedExemplarMemory(
            dim=dim, d_state=d_state, inner_rank=inner_rank, d_key=d_key
        )

        # Component A Bridge: Gumbel-Softmax Router R over N_inst
        self.gumbel_router = GumbelPromptRouter(
            dim=dim, inner_rank=inner_rank, d_state=d_state
        )

        # Mamba Core SSM
        self.mamba = build_mamba_block(
            d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand
        )

    def forward(self, x, pe, mask=None, return_path=False):
        """
        x: [B, C, H, W] feature map in 2D raster space
        pe: [max_len, C] positional encodings
        mask: [B, 1, H, W] binary mask (1 = hole, 0 = known)
        """
        if x.dtype == torch.float16:
            x = x.type(torch.float32)

        B, C, H, W = x.shape
        assert C == self.dim, f"Input channels {C} do not match layer dim {self.dim}"
        n_tokens = H * W

        # Ensure mask is 4D [B, 1, H, W]
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.shape[1] != 1:
                mask = mask[:, :1, :, :]
        else:
            mask = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        # ---------------------------------------------------------------------
        # 1. Macro-Level Pathfinding (MACG / VAMamba Core)
        # ---------------------------------------------------------------------
        # order_tensor: [B, n_tokens] permutation mapping
        # distance_map: [B, 1, H, W] physical distance increasing inward for DBA-T
        order_tensor, distance_map, patch_size = self.pathfinder(x, mask)

        # ---------------------------------------------------------------------
        # 2. Component C: Directional Boundary Affinity Transition (DBA-T)
        # ---------------------------------------------------------------------
        kappa, mod_factor = self.dba_t(x, distance_map)  # [B, 1, H, W]

        # ---------------------------------------------------------------------
        # 3. Components A & B: DIG-SM Codebook & LC-VER Exemplar Readout
        # ---------------------------------------------------------------------
        # N_inst: [B, inner_rank, d_state] dynamic codebook
        # exemplar_out: [B, C, H, W] in 2D raster space (gated by gamma_t)
        N_inst, exemplar_out = self.icem_memory(x, mask, self.shared_N)

        # ---------------------------------------------------------------------
        # 4. Token Sequencing and Permutation
        # ---------------------------------------------------------------------
        x_flat = x.view(B, C, n_tokens)
        x_reordered = torch.gather(x_flat, 2, order_tensor.unsqueeze(1).expand(-1, C, -1))  # [B, C, n_tokens]

        pe_slice = pe[:n_tokens, :].unsqueeze(0).expand(B, -1, -1)  # [B, n_tokens, C]
        pe_reordered = torch.gather(pe_slice, 1, order_tensor.unsqueeze(2).expand(-1, -1, self.dim))

        # Sequence input for Mamba SSM
        x_seq = x_reordered.transpose(1, 2) + pe_reordered  # [B, n_tokens, C]

        # Component A Bridge: Gumbel-Softmax Router generates token-level prompts P_dyn
        P_dyn = self.gumbel_router(x_seq, N_inst)  # [B, n_tokens, d_state]

        # Gather DBA-T 2D boundary gradient modulation into sequence order
        mod_flat = mod_factor.view(B, 1, n_tokens)
        delta_guide = torch.gather(mod_flat, 2, order_tensor.unsqueeze(1)).transpose(1, 2)  # [B, n_tokens, 1]

        # ---------------------------------------------------------------------
        # 5. Mamba SSM Sequence Forward Pass (with C + P_dyn and dt * delta_guide)
        # ---------------------------------------------------------------------
        y_seq = self.mamba(x_seq, P_dyn=P_dyn, delta_guide=delta_guide)  # [B, n_tokens, C]

        # ---------------------------------------------------------------------
        # 6. Inverse Permutation (\pi^-1): Un-shuffle sequence back to 2D Raster Space
        # ---------------------------------------------------------------------
        inverse_order = torch.argsort(order_tensor, dim=1)  # [B, n_tokens]
        y_seq_t = y_seq.transpose(1, 2)  # [B, C, n_tokens]
        y_unshuffled = torch.gather(y_seq_t, 2, inverse_order.unsqueeze(1).expand(-1, C, -1))
        y_mamba = y_unshuffled.view(B, C, H, W)

        # ---------------------------------------------------------------------
        # 7. Coordinate-Aligned Exemplar Integration (Both in 2D Raster Space [B, C, H, W])
        # ---------------------------------------------------------------------
        # Direct unattenuated visible exemplar readout added strictly in physical coordinate space
        y_final = y_mamba + exemplar_out

        if return_path:
            return y_final, order_tensor
        return y_final
