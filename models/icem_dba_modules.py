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
    Enhanced Dyadic Multi-Scale Directional Boundary Affinity (DBA-T):
    Computes 2D physical gradient of the distance field \nabla D(p) and feature gradient \nabla F(p)
    at both fine micro-edge (dilation=1) and coarse contour (dilation=2) scales.
    Evaluates multi-scale directional cosine alignments \kappa^{(1)} and \kappa^{(2)},
    fuses them via learned scale-attention, and modulates continuous SSM state retention \Delta t.
    Locks state retention along continuous physical contours (jawline, hair flow, architectural lines, lip edges).
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
        
        # Learned scale attention across fine (dilation=1) and coarse (dilation=2) edge scales
        self.scale_attn = nn.Parameter(torch.zeros(2))
        
        # Align projection: maps alignment score \kappa_t to SSM scale factor
        self.w_align = nn.Parameter(torch.ones(1, 1, 1))

    def compute_gradients(self, tensor_map, dilation=1, padding=1):
        """
        Computes 2D spatial gradients dx, dy via Sobel convolution at specified dilation.
        tensor_map: [B, 1, H, W]
        Returns: grad_x, grad_y: [B, 1, H, W]
        """
        sobel_x = self.sobel_x.to(dtype=tensor_map.dtype, device=tensor_map.device)
        sobel_y = self.sobel_y.to(dtype=tensor_map.dtype, device=tensor_map.device)
        grad_x = F.conv2d(tensor_map, sobel_x, padding=padding, dilation=dilation)
        grad_y = F.conv2d(tensor_map, sobel_y, padding=padding, dilation=dilation)
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
        distance_map = distance_map.to(dtype=feat_mag.dtype, device=feat_mag.device)
        
        # Scale 1: Fine micro-edges (dilation=1, padding=1)
        df_x1, df_y1 = self.compute_gradients(feat_mag, dilation=1, padding=1)
        dd_x1, dd_y1 = self.compute_gradients(distance_map, dilation=1, padding=1)
        df_norm1 = torch.sqrt(df_x1 ** 2 + df_y1 ** 2 + self.eps)
        dd_norm1 = torch.sqrt(dd_x1 ** 2 + dd_y1 ** 2 + self.eps)
        kappa1 = torch.clamp(((df_x1 * dd_x1) + (df_y1 * dd_y1)) / (df_norm1 * dd_norm1 + self.eps), -1.0, 1.0)

        # Scale 2: Coarse contours (dilation=2, padding=2)
        df_x2, df_y2 = self.compute_gradients(feat_mag, dilation=2, padding=2)
        dd_x2, dd_y2 = self.compute_gradients(distance_map, dilation=2, padding=2)
        df_norm2 = torch.sqrt(df_x2 ** 2 + df_y2 ** 2 + self.eps)
        dd_norm2 = torch.sqrt(dd_x2 ** 2 + dd_y2 ** 2 + self.eps)
        kappa2 = torch.clamp(((df_x2 * dd_x2) + (df_y2 * dd_y2)) / (df_norm2 * dd_norm2 + self.eps), -1.0, 1.0)

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
    Component A: DIG-SM with Spatially Preserved Spatial Pyramid Anchors
                 (Global 1x1, Quadrants 2x2, Sub-regions 4x4) with mask-normalization.
    Component B: LC-VER with Hierarchical Multi-Scale Exemplar Retrieval
                 (Global Visible Memory + Regional Grid Visible Memory) in linear O(L) time.
    """
    def __init__(self, dim, d_state=16, inner_rank=32, d_key=16):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.inner_rank = inner_rank
        self.d_key = d_key

        # --- Component A: DIG-SM Spatial Pyramid Anchor & Codebook Projections ---
        # 1x1 (1 bin) + 2x2 (4 bins) + 4x4 (16 bins) = 21 spatial anchors
        self.total_pyramid_bins = 1 + 4 + 16  # 21
        self.inst_proj = nn.Sequential(
            nn.Linear(dim * self.total_pyramid_bins, dim),
            nn.GELU(),
            nn.Linear(dim, inner_rank * d_state)
        )
        self.inst_beta = nn.Parameter(torch.tensor(0.1))

        # --- Component B: LC-VER Kernelized Linear Key-Value Projections ---
        self.to_q = nn.Linear(dim, d_key, bias=False)
        self.to_k = nn.Linear(dim, d_key, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Hierarchical adaptive gate fusing global context with fine regional exemplars
        self.hier_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.Sigmoid()
        )

        # Dynamic distance/confidence-weighted gating
        self.gamma_gate = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.Sigmoid()
        )

    def _extract_pyramid_anchors(self, x, known_mask):
        """
        Extracts mask-normalized spatial pyramid anchors:
        Global (1x1) + Bilateral Quadrants (2x2) + Local Sub-regions (4x4).
        Mask-normalization prevents magnitude dilution in heavily masked regions.
        """
        anchors = []
        for grid_size in [(1, 1), (2, 2), (4, 4)]:
            feat_sum = F.adaptive_avg_pool2d(x * known_mask, grid_size)
            mask_sum = F.adaptive_avg_pool2d(known_mask, grid_size)
            # Mask-normalized pool
            normed = feat_sum / (mask_sum + 1e-4)
            valid = (mask_sum > 1e-4).float()
            anchors.append((normed * valid).flatten(1))
        return torch.cat(anchors, dim=1)  # [B, C * 21]

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
        # A. DIG-SM: Spatially Preserved Instance Codebook Synthesis
        # ---------------------------------------------------------------------
        pyramid_anchors = self._extract_pyramid_anchors(x, known_mask)  # [B, C * 21]
        delta_N = torch.tanh(self.inst_proj(pyramid_anchors)).view(B, self.inner_rank, self.d_state)
        N_inst = shared_N.unsqueeze(0) + self.inst_beta * delta_N  # [B, inner_rank, d_state]

        # ---------------------------------------------------------------------
        # B. LC-VER: Hierarchical Multi-Scale Visible Memory Retrieval O(L)
        # ---------------------------------------------------------------------
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # [B, L, C]
        k_flat = known_mask.view(B, L, 1)                           # [B, L, 1]
        m_flat = mask.view(B, L, 1)                                 # [B, L, 1]

        # Positive kernel mapping \phi(x) = elu(x) + 1.0 on visible keys
        phi_k = (F.elu(self.to_k(x_flat)) + 1.0) * k_flat           # [B, L, d_key]
        v_vis = self.to_v(x_flat) * k_flat                          # [B, L, C]
        phi_q = F.elu(self.to_q(x_flat)) + 1.0                      # [B, L, d_key]

        # 1. Global visible memory descriptor matrix S_vis: [B, d_key, C]
        S_vis = torch.bmm(phi_k.transpose(1, 2), v_vis)             # [B, d_key, C]
        z_vis = phi_k.sum(dim=1, keepdim=True)                      # [B, 1, d_key]

        denom_global = torch.bmm(phi_q, z_vis.transpose(1, 2)).clamp(min=1e-4)  # [B, L, 1]
        exemplar_global = torch.bmm(phi_q, S_vis) / denom_global                  # [B, L, C]

        # 2. Regional visible memory grid (4x4 spatial cells)
        if H >= 4 and W >= 4 and (H % 4 == 0) and (W % 4 == 0):
            G = 4
            H_c, W_c = H // G, W // G
            L_c = H_c * W_c
            
            # Partition tokens into 16 spatial cells: [B, 16, L_c, ...]
            phi_k_grid = (phi_k.view(B, H, W, self.d_key)
                          .view(B, G, H_c, G, W_c, self.d_key)
                          .permute(0, 1, 3, 2, 4, 5)
                          .contiguous()
                          .view(B, 16, L_c, self.d_key))
            v_vis_grid = (v_vis.view(B, H, W, C)
                          .view(B, G, H_c, G, W_c, C)
                          .permute(0, 1, 3, 2, 4, 5)
                          .contiguous()
                          .view(B, 16, L_c, C))
            
            # Batched regional key-value memory: [B, 16, d_key, C]
            S_reg = torch.matmul(phi_k_grid.transpose(2, 3), v_vis_grid)
            z_reg = phi_k_grid.sum(dim=2, keepdim=True)  # [B, 16, 1, d_key]

            phi_q_grid = (phi_q.view(B, H, W, self.d_key)
                          .view(B, G, H_c, G, W_c, self.d_key)
                          .permute(0, 1, 3, 2, 4, 5)
                          .contiguous()
                          .view(B, 16, L_c, self.d_key))
            denom_reg = torch.matmul(phi_q_grid, z_reg.transpose(2, 3)).clamp(min=1e-4)  # [B, 16, L_c, 1]
            exemplar_reg_grid = torch.matmul(phi_q_grid, S_reg) / denom_reg              # [B, 16, L_c, C]

            # Reconstruct raster sequence: [B, L, C]
            exemplar_reg = (exemplar_reg_grid.view(B, G, G, H_c, W_c, C)
                            .permute(0, 1, 3, 2, 4, 5)
                            .contiguous()
                            .view(B, L, C))

            # Regional cell confidence fallback (if a cell contains few/no visible tokens)
            cell_vis = z_reg.sum(dim=-1, keepdim=True)  # [B, 16, 1, 1]
            cell_conf = torch.sigmoid((cell_vis - 1.0) * 2.0).expand(-1, -1, L_c, C).contiguous().view(B, L, C)
            exemplar_reg_safe = cell_conf * exemplar_reg + (1.0 - cell_conf) * exemplar_global
        else:
            exemplar_reg_safe = exemplar_global

        # 3. Hierarchical fusion of regional and global exemplar readouts
        hier_gate = self.hier_gate(torch.cat([x_flat, exemplar_global, exemplar_reg_safe], dim=-1))
        exemplar_fused = hier_gate * exemplar_reg_safe + (1.0 - hier_gate) * exemplar_global

        # 4. Dynamic distance/confidence-weighted gating \gamma_t
        gamma = self.gamma_gate(torch.cat([x_flat, m_flat], dim=-1)) * m_flat  # [B, L, C]
        exemplar_out = (gamma * exemplar_fused).permute(0, 2, 1).view(B, C, H, W)   # [B, C, H, W]

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

            # 3. 2D Serpentine continuous locality bias to preserve spatial contiguity within rings
            y_coords = torch.arange(H_p, device=x.device, dtype=x.dtype).view(H_p, 1).repeat(1, W_p)
            x_coords = torch.arange(W_p, device=x.device, dtype=x.dtype).view(1, W_p).repeat(H_p, 1)
            serp_x = torch.where(y_coords.long() % 2 == 0, x_coords, (float(W_p) - 1.0) - x_coords)
            serp_pos = (y_coords * float(W_p) + serp_x) / float(max(1, H_p * W_p))
            locality_scores = serp_pos.unsqueeze(0).expand(B, -1, -1) * 2.0

            # 4. Fine content & texture saliency for local feature ordering
            texture_scores = score_map * 0.5

            final_scores = known_boost + ring_boost + locality_scores + texture_scores
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
