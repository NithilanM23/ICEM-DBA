import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =========================================================================
# Geodesic Distance Map Calculator
# =========================================================================

def get_distance_map(mask, max_iters=None):
    r"""
    Computes both:
    1. dist_priority: [B, 1, H, W] inverse distance confidence (1.0 at boundary -> 0.0 at center)
       used to enforce MACG traversal: Known -> Boundary Fringe -> Hole Center.
    2. dist_physical: [B, 1, H, W] physical geodesic distance (0.0 at boundary -> 1.0 at center)
       whose 2D spatial gradient \nabla D points inward into the hole (used for DBA-T).
    Eliminates GPU-CPU blocking synchronization (known.min() == 1.0) to maintain high CUDA throughput.
    """
    B, C, H, W = mask.shape
    if max_iters is None:
        max_iters = min(32, max(4, max(H, W) // 2))

    dist_from_edge = torch.zeros_like(mask, dtype=torch.float32)
    known = 1.0 - mask
    
    for i in range(1, max_iters + 1):
        next_known = F.max_pool2d(known, kernel_size=3, stride=1, padding=1)
        new_pixels = next_known - known
        dist_from_edge = dist_from_edge + new_pixels * float(i)
        known = next_known
            
    unreached = 1.0 - known
    dist_from_edge = dist_from_edge + unreached * float(max_iters + 1)
    
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
    Enhanced 4-Directional Dyadic Multi-Scale Directional Boundary Affinity (DBA-T):
    Computes 2D physical gradient of the distance field \nabla D(p) and feature gradient \nabla F(p)
    across 4 spatial orientations (Horizontal 0°, Vertical 90°, Diagonal 45°, Diagonal 135°)
    at both fine micro-edge (dilation=1, 3x3) and coarse contour (dilation=2, 5x5 effective field) scales.
    Evaluates multi-scale directional cosine alignments \kappa^{(1)} and \kappa^{(2)},
    fuses them via learned scale-attention, and modulates continuous SSM state retention \Delta t.
    Locks state retention along continuous physical contours (jawline, diagonal cheekbones, lip vermilion borders).
    """
    def __init__(self, dim, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.eps = eps
        
        # 4-Directional Sobel Filters: 0° (dx), 90° (dy), 45° (diag1), 135° (diag2)
        s0 = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32) / 8.0
        s90 = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], dtype=torch.float32) / 8.0
        s45 = torch.tensor([[ 0.,  1.,  2.],
                            [-1.,  0.,  1.],
                            [-2., -1.,  0.]], dtype=torch.float32) / 8.0
        s135 = torch.tensor([[-2., -1.,  0.],
                             [-1.,  0.,  1.],
                             [ 0.,  1.,  2.]], dtype=torch.float32) / 8.0
        
        filters = torch.stack([s0, s90, s45, s135]).unsqueeze(1)  # [4, 1, 3, 3]
        self.register_buffer('directional_sobel', filters)
        
        # Learned scale attention across fine (dilation=1) and coarse (dilation=2) edge scales
        self.scale_attn = nn.Parameter(torch.zeros(2))
        
        # Align projection: maps alignment score \kappa_t to SSM scale factor
        self.w_align = nn.Parameter(torch.ones(1, 1, 1))

    def compute_directional_gradients(self, tensor_map, dilation=1, padding=1):
        """
        Computes 4-directional spatial gradients (0°, 90°, 45°, 135°) via Sobel convolution.
        tensor_map: [B, 1, H, W]
        Returns: grads: [B, 4, H, W]
        """
        filters = self.directional_sobel.to(dtype=tensor_map.dtype, device=tensor_map.device)
        return F.conv2d(tensor_map, filters, padding=padding, dilation=dilation)

    def forward(self, feature_map, distance_map):
        """
        feature_map: [B, C, H, W]
        distance_map: [B, 1, H, W]
        Returns:
            kappa: [B, 1, H, W] cosine alignment metric in [-1, 1]
            mod_factor: [B, 1, H, W] continuous contour modulation (1.0 + tanh(W_align * kappa))
        """
        feat_mag = torch.mean(torch.abs(feature_map), dim=1, keepdim=True)
        distance_map = distance_map.to(dtype=feat_mag.dtype, device=feat_mag.device)
        
        # Scale 1: Fine micro-edges across 4 directions (dilation=1, padding=1)
        df1 = self.compute_directional_gradients(feat_mag, dilation=1, padding=1)     # [B, 4, H, W]
        dd1 = self.compute_directional_gradients(distance_map, dilation=1, padding=1) # [B, 4, H, W]
        dot1 = (df1 * dd1).sum(dim=1, keepdim=True)                                  # [B, 1, H, W]
        norm1 = (torch.sqrt((df1 ** 2).sum(dim=1, keepdim=True) + self.eps) *
                 torch.sqrt((dd1 ** 2).sum(dim=1, keepdim=True) + self.eps))
        kappa1 = torch.clamp(dot1 / (norm1 + self.eps), -1.0, 1.0)

        # Scale 2: Coarse contours across 4 directions (dilation=2, padding=2)
        df2 = self.compute_directional_gradients(feat_mag, dilation=2, padding=2)     # [B, 4, H, W]
        dd2 = self.compute_directional_gradients(distance_map, dilation=2, padding=2) # [B, 4, H, W]
        dot2 = (df2 * dd2).sum(dim=1, keepdim=True)                                  # [B, 1, H, W]
        norm2 = (torch.sqrt((df2 ** 2).sum(dim=1, keepdim=True) + self.eps) *
                 torch.sqrt((dd2 ** 2).sum(dim=1, keepdim=True) + self.eps))
        kappa2 = torch.clamp(dot2 / (norm2 + self.eps), -1.0, 1.0)

        # Learned multi-scale dyadic fusion
        scale_weights = F.softmax(self.scale_attn, dim=0)
        kappa = scale_weights[0] * kappa1 + scale_weights[1] * kappa2
        
        # Continuous SSM Discretization Modulation: 1.0 + tanh(W_align * \kappa)
        mod_factor = 1.0 + torch.tanh(self.w_align.to(feat_mag.dtype) * kappa)
        return kappa, mod_factor


# =========================================================================
# Components A & B: Instance-Conditioned Exemplar Memory (DIG-SM + LC-VER)
# =========================================================================

class InstanceConditionedExemplarMemory(nn.Module):
    """
    Enhanced ICEM:
    Component A: DIG-SM with Spatially-Disentangled Global & Bilateral Quadrant Instance Codebooks.
                 Enforces Left-Right eye symmetry and preserves localized facial anatomy.
    Component B: LC-VER with Tri-Scale Exemplar Retrieval:
                 1. Global Visible Memory (S_glob): Overall face tone, lighting, age traits.
                 2. Bilateral Symmetric Memory (S_sym): Horizontal reflection (Right Eye <-> Left Eye).
                 3. Regional Multi-Cell Memory (S_reg): Localized fine texture details (wrinkles, teeth).
                 Fused via a learned 3-way softmax gate in linear O(L) time.
    """
    def __init__(self, dim, d_state=16, inner_rank=32, d_key=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.inner_rank = inner_rank
        self.d_key = d_key

        # Partition inner_rank (32 slots) into:
        # 16 Global Identity Slots + 16 Regional Quadrant Slots (4 slots x 4 quadrants)
        self.r_glob = inner_rank // 2  # 16
        self.r_quad = (inner_rank - self.r_glob) // 4  # 4 slots per quadrant

        # --- Component A: DIG-SM Spatially-Disentangled Projections ---
        # Global identity projection (conditioned on 1x1 mask-normalized visible pool)
        self.inst_proj_glob = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, self.r_glob * d_state)
        )
        # Regional quadrant projection (conditioned on 2x2 mask-normalized visible pool + bilateral prior)
        self.inst_proj_quad = nn.Sequential(
            nn.Linear(dim * 2, dim),  # Takes [quadrant_feat, mirrored_quadrant_feat]
            nn.GELU(),
            nn.Linear(dim, self.r_quad * d_state)
        )
        self.inst_beta = nn.Parameter(torch.tensor(0.1))

        # --- Component B: LC-VER Tri-Scale Linear Key-Value Projections ---
        self.to_q = nn.Linear(dim, d_key, bias=False)
        self.to_k = nn.Linear(dim, d_key, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Dynamic 3-way softmax fusion gate: blends Global, Bilateral Symmetric, and Regional memories
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 4 + 1, 3),
            nn.Softmax(dim=-1)
        )

        # Dynamic distance/confidence-weighted gating
        self.gamma_gate = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.Sigmoid()
        )

    def _extract_spatially_disentangled_codebook(self, x, known_mask, shared_N):
        """
        Synthesizes spatially-disentangled instance codebook:
        - 16 Global identity slots (invariant tone, age, lighting)
        - 16 Bilateral quadrant slots (Left Eye, Right Eye, Left Lower, Right Lower)
        Uses bilateral horizontal reflection so masked quadrants receive the opposite visible feature!
        """
        B, C, H, W = x.shape

        # 1. Global Anchor (1x1 mask-normalized)
        feat_1x1 = F.adaptive_avg_pool2d(x * known_mask, (1, 1))
        mask_1x1 = F.adaptive_avg_pool2d(known_mask, (1, 1))
        norm_1x1 = (feat_1x1 / (mask_1x1 + 1e-4)) * (mask_1x1 > 1e-4).float()
        delta_N_glob = torch.tanh(self.inst_proj_glob(norm_1x1.flatten(1))).view(B, self.r_glob, self.d_state)

        # 2. Quadrant Anchors (2x2 mask-normalized: [TL, TR, BL, BR])
        feat_2x2 = F.adaptive_avg_pool2d(x * known_mask, (2, 2))  # [B, C, 2, 2]
        mask_2x2 = F.adaptive_avg_pool2d(known_mask, (2, 2))      # [B, 1, 2, 2]
        norm_2x2 = (feat_2x2 / (mask_2x2 + 1e-4)) * (mask_2x2 > 1e-4).float()  # [B, C, 2, 2]

        # Bilateral horizontal reflection of quadrants (TL <-> TR, BL <-> BR)
        norm_2x2_sym = torch.flip(norm_2x2, dims=[-1])  # [B, C, 2, 2]

        # Reshape to [B, 4, C]
        q_feats = norm_2x2.permute(0, 2, 3, 1).contiguous().view(B, 4, C)
        q_feats_sym = norm_2x2_sym.permute(0, 2, 3, 1).contiguous().view(B, 4, C)

        # Pair each quadrant with its bilateral mirror: [B, 4, 2*C]
        q_paired = torch.cat([q_feats, q_feats_sym], dim=-1)
        delta_N_quad = torch.tanh(self.inst_proj_quad(q_paired)).view(B, 4 * self.r_quad, self.d_state)

        # Assemble full instance codebook [B, inner_rank, d_state]
        delta_N = torch.cat([delta_N_glob, delta_N_quad], dim=1)
        N_inst = shared_N.unsqueeze(0) + self.inst_beta * delta_N
        return N_inst

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
        # A. DIG-SM: Spatially-Disentangled Instance Codebook Synthesis
        # ---------------------------------------------------------------------
        N_inst = self._extract_spatially_disentangled_codebook(x, known_mask, shared_N)

        # ---------------------------------------------------------------------
        # B. LC-VER: Tri-Scale Visible Memory Retrieval O(L)
        # ---------------------------------------------------------------------
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # [B, L, C]
        k_flat = known_mask.view(B, L, 1)                           # [B, L, 1]
        m_flat = mask.view(B, L, 1)                                 # [B, L, 1]
        phi_q = F.elu(self.to_q(x_flat)) + 1.0                      # [B, L, d_key]

        # 1. Global Visible Memory (S_glob): Overall facial tone and lighting
        phi_k_glob = (F.elu(self.to_k(x_flat)) + 1.0) * k_flat     # [B, L, d_key]
        v_vis_glob = self.to_v(x_flat) * k_flat                     # [B, L, C]
        S_glob = torch.bmm(phi_k_glob.transpose(1, 2), v_vis_glob) # [B, d_key, C]
        z_glob = phi_k_glob.sum(dim=1, keepdim=True)                # [B, 1, d_key]

        denom_glob = torch.bmm(phi_q, z_glob.transpose(1, 2)).clamp(min=1e-4)
        exemplar_glob = torch.bmm(phi_q, S_glob) / denom_glob         # [B, L, C]

        # 2. Bilateral Symmetric Memory (S_sym): Direct mirror reflection (Right Eye <-> Left Eye)
        x_sym = torch.flip(x, dims=[-1])
        known_sym = torch.flip(known_mask, dims=[-1])
        x_sym_flat = x_sym.permute(0, 2, 3, 1).contiguous().view(B, L, C)
        k_sym_flat = known_sym.view(B, L, 1)

        phi_k_sym = (F.elu(self.to_k(x_sym_flat)) + 1.0) * k_sym_flat
        v_vis_sym = self.to_v(x_sym_flat) * k_sym_flat
        S_sym = torch.bmm(phi_k_sym.transpose(1, 2), v_vis_sym)     # [B, d_key, C]
        z_sym = phi_k_sym.sum(dim=1, keepdim=True)                  # [B, 1, d_key]

        denom_sym = torch.bmm(phi_q, z_sym.transpose(1, 2)).clamp(min=1e-4)
        exemplar_sym = torch.bmm(phi_q, S_sym) / denom_sym           # [B, L, C]

        # 3. Regional Multi-Cell Memory (S_reg on 4x4 spatial cells)
        if H >= 4 and W >= 4 and (H % 4 == 0) and (W % 4 == 0):
            G = 4
            H_c, W_c = H // G, W // G
            L_c = H_c * W_c
            
            # Partition tokens into 16 spatial cells: [B, 16, L_c, ...]
            phi_k_grid = (phi_k_glob.view(B, H, W, self.d_key)
                          .view(B, G, H_c, G, W_c, self.d_key)
                          .permute(0, 1, 3, 2, 4, 5)
                          .contiguous()
                          .view(B, 16, L_c, self.d_key))
            v_vis_grid = (v_vis_glob.view(B, H, W, C)
                          .view(B, G, H_c, G, W_c, C)
                          .permute(0, 1, 3, 2, 4, 5)
                          .contiguous()
                          .view(B, 16, L_c, C))
            
            # Regional key-value memory: [B, 16, d_key, C]
            S_reg = torch.matmul(phi_k_grid.transpose(2, 3), v_vis_grid)
            z_reg = phi_k_grid.sum(dim=2, keepdim=True)  # [B, 16, 1, d_key]

            phi_q_grid = (phi_q.view(B, H, W, self.d_key)
                          .view(B, G, H_c, G, W_c, self.d_key)
                          .permute(0, 1, 3, 2, 4, 5)
                          .contiguous()
                          .view(B, 16, L_c, self.d_key))
            denom_reg = torch.matmul(phi_q_grid, z_reg.transpose(2, 3)).clamp(min=1e-4)
            exemplar_reg_grid = torch.matmul(phi_q_grid, S_reg) / denom_reg

            exemplar_reg = (exemplar_reg_grid.view(B, G, G, H_c, W_c, C)
                            .permute(0, 1, 3, 2, 4, 5)
                            .contiguous()
                            .view(B, L, C))

            # Regional cell confidence fallback (if a cell contains few/no visible tokens)
            cell_vis = z_reg.sum(dim=-1, keepdim=True)  # [B, 16, 1, 1]
            cell_conf = torch.sigmoid((cell_vis - 1.0) * 2.0).expand(-1, -1, L_c, C).contiguous().view(B, L, C)
            exemplar_reg_safe = cell_conf * exemplar_reg + (1.0 - cell_conf) * exemplar_glob
        else:
            exemplar_reg_safe = exemplar_glob

        # 4. Learned 3-Way Softmax Fusion of Global, Bilateral Symmetric, and Regional Readouts
        gate_weights = self.fusion_gate(torch.cat([x_flat, exemplar_glob, exemplar_sym, exemplar_reg_safe, m_flat], dim=-1))
        exemplar_fused = (gate_weights[:, :, 0:1] * exemplar_glob +
                          gate_weights[:, :, 1:2] * exemplar_sym +
                          gate_weights[:, :, 2:3] * exemplar_reg_safe)

        # 5. Dynamic distance/confidence-weighted gating \gamma_t
        gamma = self.gamma_gate(torch.cat([x_flat, m_flat], dim=-1)) * m_flat  # [B, L, C]
        exemplar_out = (gamma * exemplar_fused).permute(0, 2, 1).view(B, C, H, W)

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
            # Multi-scale distance cache on mask tensor avoids recomputing 88 times across 44 blocks per step
            cache = getattr(mask, '_dist_cache', None)
            if cache is None:
                cache = {}
                try:
                    setattr(mask, '_dist_cache', cache)
                except Exception:
                    pass

            cache_key = (H, W, patch_size)
            if cache is not None and cache_key in cache:
                dist_priority_pooled, distance_map, mask_pooled = cache[cache_key]
            else:
                if patch_size > 1:
                    mask_pooled = F.max_pool2d(
                        mask.float(), kernel_size=patch_size, stride=patch_size
                    )
                else:
                    mask_pooled = mask.float()
                    
                dist_priority_pooled, _ = get_distance_map(mask_pooled)  # [B, 1, H_p, W_p]
                _, distance_map = get_distance_map(mask.float())
                if cache is not None:
                    cache[cache_key] = (dist_priority_pooled, distance_map, mask_pooled)

            dist_squeezed = dist_priority_pooled.squeeze(1)
            mask_squeezed = mask_pooled.squeeze(1)
            _, H_p, W_p = dist_squeezed.shape

            # Inward confidence traversal scoring with concentric rings and 2D spatial locality guidance:
            # 1. Known Tokens (10,000) >> Inward Boundary Fringe >> Deep Hole Center
            known_boost = (1.0 - mask_squeezed) * 10000.0

            # 2. Concentric geodesic rings (quantized isocontours inward from outer boundary to center)
            # 16 discrete concentric rings enforce steady inwards wavefront without cross-facial jumps
            ring_idx = torch.floor(dist_squeezed * 15.999).clamp(min=0.0, max=15.0)
            ring_boost = mask_squeezed * (ring_idx * 50.0)

            # 3. Continuous contour wavefront angular positioning relative to mask center of mass
            # Sweeps around the contour perimeter smoothly in an inward circular arc
            y_coords = torch.arange(H_p, device=x.device, dtype=x.dtype).view(H_p, 1).repeat(1, W_p)
            x_coords = torch.arange(W_p, device=x.device, dtype=x.dtype).view(1, W_p).repeat(H_p, 1)

            denom_m = mask_squeezed.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            yc = (mask_squeezed * y_coords.unsqueeze(0)).sum(dim=(-2, -1), keepdim=True) / denom_m
            xc = (mask_squeezed * x_coords.unsqueeze(0)).sum(dim=(-2, -1), keepdim=True) / denom_m

            theta = torch.atan2(y_coords.unsqueeze(0) - yc, x_coords.unsqueeze(0) - xc) + math.pi
            contour_angle_score = (theta / (2.0 * math.pi)) * 2.0

            # 4. Fine content & texture saliency for local feature ordering
            texture_scores = score_map * 0.5

            final_scores = known_boost + ring_boost + contour_angle_score + texture_scores
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
