import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# Geodesic Distance Map Calculator
# =========================================================================

def get_distance_map(mask, max_iters=128):
    r"""
    Computes both:
    1. dist_priority: [B, 1, H, W] inverse distance confidence (1.0 at boundary -> 0.0 at center)
       used to enforce MACG traversal: Known -> Boundary Fringe -> Hole Center.
    2. dist_physical: [B, 1, H, W] physical geodesic distance (0.0 at boundary -> 1.0 at center)
       whose 2D spatial gradient \nabla D points inward into the hole (used for DBA-T).
    """
    B, C, H, W = mask.shape
    dist_from_edge = torch.zeros_like(mask, dtype=torch.float32)
    known = 1.0 - mask
    
    for i in range(1, max_iters + 1):
        next_known = F.max_pool2d(known, kernel_size=3, stride=1, padding=1)
        new_pixels = next_known - known
        dist_from_edge += new_pixels * float(i)
        known = next_known
        if known.min() == 1.0:
            break
            
    unreached = 1.0 - known
    dist_from_edge += unreached * float(max_iters + 1)
    
    max_dist = dist_from_edge.view(B, -1).max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    max_dist = torch.clamp(max_dist, min=1.0)
    
    # Priority distance (inversely proportional to distance: 1.0 at outer boundary, 0.0 at center)
    dist_priority = ((max_dist - dist_from_edge) / max_dist) * mask

    # Physical distance increasing inwards (0.0 at outer boundary, 1.0 at deepest center)
    dist_physical = (dist_from_edge / max_dist) * mask
    
    return dist_priority, dist_physical


# =========================================================================
# VAMamba CNNScoreMap (Macro-Level Texture & Content Saliency)
# =========================================================================

class CNNScoreMap(nn.Module):
    """
    Lightweight depthwise-separable convolutional network with Squeeze-and-Excitation
    channel attention and local variance heuristic to compute structural importance scores.
    """
    def __init__(self, in_channels, embed_dim=64):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(
            in_channels, embed_dim, kernel_size=3, padding=2, dilation=2,
            groups=in_channels if in_channels <= embed_dim else 1
        )
        self.act1 = nn.GELU()
        
        # Channel Attention (Squeeze-and-Excitation)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_fc1 = nn.Conv2d(embed_dim, max(1, embed_dim // 4), 1, bias=False)
        self.ca_act = nn.ReLU(inplace=False)
        self.ca_fc2 = nn.Conv2d(max(1, embed_dim // 4), embed_dim, 1, bias=False)
        self.ca_sigmoid = nn.Sigmoid()

        self.conv2 = nn.Conv2d(
            embed_dim, embed_dim, kernel_size=3, padding=2, dilation=2, groups=embed_dim
        )
        self.act2 = nn.GELU()
        
        # Score projection head
        self.score_head = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x):
        # x is [B, C, H, W]
        if x.dim() == 4 and x.shape[1] != self.conv1.in_channels:
            x = x.permute(0, 3, 1, 2).contiguous()
            
        feat = self.act1(self.conv1(x))
        
        # Apply Channel Attention
        ca = self.global_pool(feat)
        ca = self.ca_fc1(ca)
        ca = self.ca_act(ca)
        ca = self.ca_fc2(ca)
        ca = self.ca_sigmoid(ca)
        feat = feat * ca
        
        feat = self.act2(self.conv2(feat))
        scores = self.score_head(feat)  # [B, 1, H, W]
        scores = torch.sigmoid(scores).squeeze(1)  # [B, H, W]
        
        # Blend in local patch variance as a fast content heuristic
        patch_variance = torch.var(x, dim=1)  # [B, H, W]
        content_score = torch.sigmoid(patch_variance * 10.0)
        
        final_scores = 0.7 * scores + 0.3 * content_score
        return final_scores


# =========================================================================
# Component C: Directional Boundary Affinity Transition (DBA-T)
# =========================================================================

class DirectionalBoundaryAffinity(nn.Module):
    r"""
    Computes 2D physical gradient of the distance field \nabla D(p) and feature gradient \nabla F(p),
    evaluates directional cosine alignment \kappa_t, and generates state retention modulation.
    Locks state retention along continuous physical contours (jawline, hair flow, architectural lines).
    """
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # Fixed 2D Sobel Filters for horizontal (dx) and vertical (dy) gradients
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Align projection: maps alignment score \kappa_t to SSM scale factor
        self.w_align = nn.Parameter(torch.ones(1, 1, 1))

    def compute_gradients(self, tensor_map):
        """
        Computes 2D spatial gradients dx, dy via Sobel convolution.
        tensor_map: [B, 1, H, W]
        Returns: grad_x, grad_y: [B, 1, H, W]
        """
        sobel_x = self.sobel_x.to(dtype=tensor_map.dtype, device=tensor_map.device)
        sobel_y = self.sobel_y.to(dtype=tensor_map.dtype, device=tensor_map.device)
        grad_x = F.conv2d(tensor_map, sobel_x, padding=1)
        grad_y = F.conv2d(tensor_map, sobel_y, padding=1)
        return grad_x, grad_y

    def forward(self, feature_map, distance_map):
        """
        feature_map: [B, C, H, W]
        distance_map: [B, 1, H, W]
        Returns:
            kappa: [B, 1, H, W] cosine alignment metric in [-1, 1]
            mod_factor: [B, 1, H, W] continuous contour modulation (1.0 + tanh(W_align * kappa))
        """
        # Collapse feature to 1-channel magnitude representation for edge direction
        feat_mag = torch.mean(torch.abs(feature_map), dim=1, keepdim=True)
        
        # 1. Feature gradients \nabla F(p)
        df_x, df_y = self.compute_gradients(feat_mag)
        df_norm = torch.sqrt(df_x ** 2 + df_y ** 2 + self.eps)
        
        # 2. Distance field gradients \nabla D(p)
        distance_map = distance_map.to(dtype=feat_mag.dtype, device=feat_mag.device)
        dd_x, dd_y = self.compute_gradients(distance_map)
        dd_norm = torch.sqrt(dd_x ** 2 + dd_y ** 2 + self.eps)
        
        # 3. Directional cosine alignment \kappa_t
        dot_product = (df_x * dd_x) + (df_y * dd_y)
        kappa = dot_product / (df_norm * dd_norm + self.eps)
        kappa = torch.clamp(kappa, min=-1.0, max=1.0)
        
        # 4. Continuous SSM Discretization Modulation: 1.0 + tanh(W_align * \kappa)
        mod_factor = 1.0 + torch.tanh(self.w_align.to(feat_mag.dtype) * kappa)
        return kappa, mod_factor


# =========================================================================
# Components A & B: Instance-Conditioned Exemplar Memory (DIG-SM + LC-VER)
# =========================================================================

class InstanceConditionedExemplarMemory(nn.Module):
    """
    Component A: DIG-SM (Dynamic Instance-Grounded Semantic Modulation)
    Component B: LC-VER (Linear-Complexity Visible Exemplar Readout)
    Strictly follows the PyTorch Implementation Blueprint in Section 3.4 of the research report.
    """
    def __init__(self, dim, d_state=16, inner_rank=32, d_key=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.inner_rank = inner_rank
        self.d_key = d_key

        # --- Component A: DIG-SM Anchor & Codebook Projections ---
        self.vis_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.inst_proj = nn.Sequential(
            nn.Linear(dim * 16, dim),
            nn.GELU(),
            nn.Linear(dim, inner_rank * d_state)
        )
        self.inst_beta = nn.Parameter(torch.tensor(0.1))

        # --- Component B: LC-VER Kernelized Linear Key-Value Cache ---
        self.to_q = nn.Linear(dim, d_key, bias=False)
        self.to_k = nn.Linear(dim, d_key, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gamma_gate = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.Sigmoid()
        )

    def forward(self, x, mask, shared_N):
        """
        x: [B, C, H, W]
        mask: [B, 1, H, W] (1 for hole, 0 for known context)
        shared_N: [inner_rank, d_state] global base prompt dictionary
        Returns:
            N_inst: [B, inner_rank, d_state] dynamic instance codebook
            exemplar_out: [B, C, H, W] unattenuated visible exemplar readout
        """
        B, C, H, W = x.shape
        L = H * W
        known_mask = (1.0 - mask).float()  # [B, 1, H, W]

        # ---------------------------------------------------------------------
        # A. DIG-SM: Dynamic Instance Codebook Synthesis
        # ---------------------------------------------------------------------
        # Extract visible features and spatial-semantic anchor pooling
        anchors = self.vis_pool(x * known_mask).flatten(1)  # [B, C * 16]
        delta_N = torch.tanh(self.inst_proj(anchors)).view(B, self.inner_rank, self.d_state)
        N_inst = shared_N.unsqueeze(0) + self.inst_beta * delta_N  # [B, inner_rank, d_state]

        # ---------------------------------------------------------------------
        # B. LC-VER: Kernelized Linear Visible Memory Cache O(L)
        # ---------------------------------------------------------------------
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # [B, L, C]
        k_flat = known_mask.view(B, L, 1)                           # [B, L, 1]
        m_flat = mask.view(B, L, 1)                                 # [B, L, 1]

        # Positive kernel mapping \phi(x) = elu(x) + 1.0 on visible keys
        phi_k = (F.elu(self.to_k(x_flat)) + 1.0) * k_flat           # [B, L, d_key]
        v_vis = self.to_v(x_flat) * k_flat                          # [B, L, C]

        # Global visible memory descriptor matrix S_vis: [B, d_key, C] - COMPUTED ONCE
        S_vis = torch.bmm(phi_k.transpose(1, 2), v_vis)             # [B, d_key, C]
        z_vis = phi_k.sum(dim=1, keepdim=True)                      # [B, 1, d_key]

        # Query visible exemplars for all tokens
        phi_q = F.elu(self.to_q(x_flat)) + 1.0                      # [B, L, d_key]
        
        # O(1) constant-time readout per token
        denom = torch.bmm(phi_q, z_vis.transpose(1, 2)).clamp(min=1e-4) # [B, L, 1]
        exemplar = torch.bmm(phi_q, S_vis) / denom                  # [B, L, C]

        # Dynamic distance/confidence-weighted gating \gamma_t
        gamma = self.gamma_gate(torch.cat([x_flat, m_flat], dim=-1)) * m_flat  # [B, L, C]
        exemplar_out = (gamma * exemplar).permute(0, 2, 1).view(B, C, H, W)   # [B, C, H, W]

        return N_inst, exemplar_out


# =========================================================================
# Macro-Level Confidence Pathfinding (MACG / VAMamba Core)
# =========================================================================

class MACGPathfinder(nn.Module):
    """
    Mask-Aware Confidence Guidance (MACG) Token Sequencing Pathfinder.
    Enforces macro traversal:
        Known Context (C=1) -> Boundary Fringe -> Deep Hole Center.
    Supports multi-scale dynamic patch grouping (4x4 -> 2x2 -> 1x1).
    """
    def __init__(self, dim):
        super().__init__()
        self.cnn_score_map = CNNScoreMap(in_channels=dim, embed_dim=dim)

    def forward(self, x, mask=None):
        """
        x: [B, C, H, W]
        mask: [B, 1, H, W] or None
        Returns:
            order_tensor: [B, H*W] 1D permutation index tensor
            distance_map: [B, 1, H, W] normalized distance map
            patch_size: int (4, 2, or 1)
        """
        B, C, H, W = x.shape
        n_tokens = H * W
        
        orig_score_map = self.cnn_score_map(x)  # [B, H, W]

        # Dynamically set patch size: Attempt 4x4, fallback to 2x2 or 1x1
        patch_size = 1
        for ps in [4, 2]:
            if H % ps == 0 and W % ps == 0:
                patch_size = ps
                break

        if patch_size > 1:
            score_map = F.avg_pool2d(
                orig_score_map.unsqueeze(1), kernel_size=patch_size, stride=patch_size
            ).squeeze(1)
        else:
            score_map = orig_score_map

        distance_map = None
        if mask is not None:
            if patch_size > 1:
                mask_pooled = F.max_pool2d(
                    mask.float(), kernel_size=patch_size, stride=patch_size
                )
            else:
                mask_pooled = mask.float()
                
            dist_priority_pooled, _ = get_distance_map(mask_pooled)  # [B, 1, H_p, W_p]
            dist_squeezed = dist_priority_pooled.squeeze(1)
            mask_squeezed = mask_pooled.squeeze(1)

            # Inward confidence traversal scoring:
            # Known Tokens (10,000) >> Outer Boundary (100) >> Deep Hole Center (0 + texture score)
            known_boost = (1.0 - mask_squeezed) * 10000.0
            spiral_boost = mask_squeezed * dist_squeezed * 100.0
            texture_scores = score_map * 1.0

            final_scores = known_boost + spiral_boost + texture_scores
            
            # Compute full resolution physical distance map for DBA-T gradient computation
            _, distance_map = get_distance_map(mask.float())
        else:
            final_scores = score_map
            distance_map = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)

        B, H_p, W_p = final_scores.shape
        num_patches = H_p * W_p
        flat_scores = final_scores.view(B, num_patches)
        sorted_patch_indices = torch.argsort(flat_scores, dim=1, descending=True)  # [B, num_patches]

        if patch_size == 1:
            order_tensor = sorted_patch_indices  # [B, n_tokens]
        else:
            p_y = sorted_patch_indices // W_p
            p_x = sorted_patch_indices % W_p

            offsets_y = torch.arange(patch_size, device=x.device).view(patch_size, 1).repeat(1, patch_size).flatten()
            offsets_x = torch.arange(patch_size, device=x.device).view(1, patch_size).repeat(patch_size, 1).flatten()

            pixel_y = (p_y.unsqueeze(2) * patch_size) + offsets_y.view(1, 1, -1)
            pixel_x = (p_x.unsqueeze(2) * patch_size) + offsets_x.view(1, 1, -1)

            order_tensor = (pixel_y * W + pixel_x).view(B, n_tokens)  # [B, n_tokens]

        return order_tensor, distance_map, patch_size


# =========================================================================
# Gumbel-Softmax Dynamic Prompt Router (Component A Bridge)
# =========================================================================

class GumbelPromptRouter(nn.Module):
    """
    Gumbel-Softmax Router R (Section 3.1 & 3.2 in Research Paper):
    Bridges the instance-conditioned codebook N_inst [B, r, d_state] with token queries.
    Routes each token feature x_t over the r prompt keys to synthesize token-level
    dynamic non-causal prompts P_dyn [B, L, d_state].
    Injected into Mamba SSM state readout: C_att = C + P_dyn.
    """
    def __init__(self, dim, inner_rank=32, d_state=16, temperature=1.0):
        super().__init__()
        self.dim = dim
        self.inner_rank = inner_rank
        self.d_state = d_state
        self.temperature = temperature

        # Token-to-dictionary router head
        self.router_head = nn.Sequential(
            nn.Linear(dim, inner_rank * 2),
            nn.GELU(),
            nn.Linear(inner_rank * 2, inner_rank)
        )
        self.prompt_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x_seq, N_inst, hard=False):
        """
        x_seq: [B, L, C] sequence tokens in permuted traversal order
        N_inst: [B, r, d_state] dynamic instance codebook from DIG-SM
        Returns:
            P_dyn: [B, L, d_state] dynamic non-causal prompt injection
        """
        B, L, C = x_seq.shape
        logits = self.router_head(x_seq)  # [B, L, inner_rank]

        # Compute Gumbel-Softmax in float32 to prevent half-precision exp overflow
        logits_f32 = logits.float()
        if self.training:
            weights = F.gumbel_softmax(logits_f32, tau=self.temperature, hard=hard, dim=-1).to(x_seq.dtype)
        else:
            weights = F.softmax(logits_f32 / self.temperature, dim=-1).to(x_seq.dtype)

        # Token prompt P_dyn = weights @ N_inst: [B, L, r] @ [B, r, d_state] -> [B, L, d_state]
        P_dyn = torch.bmm(weights, N_inst.to(x_seq.dtype)) * self.prompt_scale.to(x_seq.dtype)
        return P_dyn
