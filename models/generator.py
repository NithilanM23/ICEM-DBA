import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import ICEM_DBATransformerBlock

# =========================================================================
# Positional Encoding Helper
# =========================================================================

def PositionalEncoding(d_model, max_len=50000, device=None):
    """
    Standard 1D Sinusoidal Positional Encoding for visual sequences.
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if device is not None:
        pe = pe.to(device)
    return pe


# =========================================================================
# Patch Embedding & Resizing Modules
# =========================================================================

class OverlapPatchEmbed(nn.Module):
    """
    Initial feature embedding: Projects (masked_img + mask) to base feature dim.
    Supports patch_size 4 (standard Vision Mamba) for O(1) step execution.
    """
    def __init__(self, in_c=4, embed_dim=48, bias=False, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        if patch_size == 4:
            self.proj = nn.Sequential(
                nn.Conv2d(in_c, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=bias),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)
            )
        elif patch_size == 2:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=2, padding=1, bias=bias)
        else:
            self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Downsample(nn.Module):
    """
    Downsampling via PixelUnshuffle (factor of 2) followed by 1x1 Conv projection.
    [B, C, H, W] -> PixelUnshuffle(2) -> [B, C*4, H/2, W/2] -> Conv1x1 -> [B, C*2, H/2, W/2].
    E.g. 48ch -> PixelUnshuffle(2) -> 192ch -> Conv1x1 -> 96ch.
    """
    def __init__(self, n_feat, bias=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(n_feat * 4, n_feat * 2, kernel_size=1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x, mask=None):
        return self.body(x)


class Upsample(nn.Module):
    """
    Upsampling via 1x1 Conv expansion followed by PixelShuffle (factor of 2).
    [B, C, H, W] -> Conv1x1 -> [B, (C//2)*4, H, W] -> PixelShuffle(2) -> [B, C//2, H*2, W*2].
    E.g. 384ch -> Conv1x1 -> 768ch -> PixelShuffle(2) -> 192ch.
    """
    def __init__(self, n_feat, bias=False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, (n_feat // 2) * 4, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.PixelShuffle(2)
        )

    def forward(self, x, mask=None):
        return self.body(x)


# =========================================================================
# Full Proposed ICEM-DBA U-Net Generator
# =========================================================================

class ICEM_DBA_Net(nn.Module):
    """
    Full Proposed ICEM-DBA Inpainting Architecture.
    Hierarchical 4-Level U-Net with Latent Stage, Skip Connections, and Refinement Block,
    powered by ICEM_DBATransformerBlock (MACG + DIG-SM + LC-VER + DBA-T + SSM).
    """
    def __init__(
        self,
        inp_channels=4,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
        d_state=16,
        inner_rank=32,
        d_key=16,
        use_checkpoint=True,
        patch_size=4
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size

        # Patch Embedding
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, bias=bias, patch_size=patch_size)

        # ----------------- ENCODER HIERARCHY -----------------
        # Level 1: [B, dim, H, W]
        self.encoder_level1 = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_blocks[0])
        ])

        # Downsample Level 1 -> Level 2
        self.down1_2 = Downsample(dim)
        dim_l2 = int(dim * 2)

        # Level 2: [B, dim*2, H/2, W/2]
        self.encoder_level2 = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim_l2, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_blocks[1])
        ])

        # Downsample Level 2 -> Level 3
        self.down2_3 = Downsample(dim_l2)
        dim_l3 = int(dim * 4)

        # Level 3: [B, dim*4, H/4, W/4]
        self.encoder_level3 = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim_l3, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_blocks[2])
        ])

        # Downsample Level 3 -> Level 4 (Latent)
        self.down3_4 = Downsample(dim_l3)
        dim_l4 = int(dim * 8)

        # Latent Stage: [B, dim*8, H/8, W/8]
        self.latent = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim_l4, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_blocks[3])
        ])

        # ----------------- DECODER HIERARCHY -----------------
        # Upsample Level 4 -> Level 3
        self.up4_3 = Upsample(dim_l4)
        self.reduce_chan_level3 = nn.Conv2d(dim_l4, dim_l3, kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim_l3, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_blocks[2])
        ])

        # Upsample Level 3 -> Level 2
        self.up3_2 = Upsample(dim_l3)
        self.reduce_chan_level2 = nn.Conv2d(dim_l3, dim_l2, kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim_l2, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_blocks[1])
        ])

        # Upsample Level 2 -> Level 1
        self.up2_1 = Upsample(dim_l2)
        # Skip connection concatenates upsampled (dim) + enc1 (dim) = dim * 2
        self.decoder_level1 = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim_l2, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_blocks[0])
        ])

        # Refinement Stage
        self.refinement = nn.Sequential(*[
            ICEM_DBATransformerBlock(
                dim=dim_l2, ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                LayerNorm_type=LayerNorm_type, d_state=d_state, inner_rank=inner_rank, d_key=d_key
            ) for _ in range(num_refinement_blocks)
        ])

        # Final RGB Output Projection (upsamples back to full resolution if patch_size > 1)
        if patch_size == 4:
            self.output = nn.Sequential(
                nn.Conv2d(dim_l2, dim_l2 * 2, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.PixelShuffle(2),
                nn.LeakyReLU(0.2, inplace=False),
                nn.Conv2d(dim_l2 // 2, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.PixelShuffle(2)
            )
        elif patch_size == 2:
            self.output = nn.Sequential(
                nn.Conv2d(dim_l2, out_channels * 4, kernel_size=3, stride=1, padding=1, bias=bias),
                nn.PixelShuffle(2)
            )
        else:
            self.output = nn.Conv2d(dim_l2, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def _run_stage(self, stage_module, feat, pe, mask):
        """
        Runs a stage of ICEM_DBATransformerBlocks, applying gradient checkpointing
        per-block during training to drop activation memory by >80%.
        """
        cur = feat
        for block in stage_module:
            if self.training and getattr(self, 'use_checkpoint', True):
                cur = torch.utils.checkpoint.checkpoint(
                    block, cur, pe, mask,
                    use_reentrant=False
                )
            else:
                cur = block(cur, pe, mask)
        return {0: cur, 1: pe, 2: mask}

    def forward(self, images, masks, mask1=None, mask2=None, mask3=None, mask4=None,
                pos1=None, pos2=None, pos3=None, pos4=None, pos1_dec=None):
        """
        images: [B, 3, H, W] input image (masked or clean)
        masks: [B, 1, H, W] binary mask (1 = hole, 0 = known)
        """
        device = images.device
        B, _, H, W = images.shape
        ps = getattr(self, 'patch_size', 4)

        H1, W1 = H // ps, W // ps
        H2, W2 = H1 // 2, W1 // 2
        H3, W3 = H1 // 4, W1 // 4
        H4, W4 = H1 // 8, W1 // 8

        # Multi-scale mask interpolation
        if mask1 is None:
            mask1 = F.interpolate(masks.float(), size=[H1, W1], mode='nearest') if ps > 1 else masks.float()
        if mask2 is None:
            mask2 = F.interpolate(masks.float(), size=[H2, W2], mode='nearest')
        if mask3 is None:
            mask3 = F.interpolate(masks.float(), size=[H3, W3], mode='nearest')
        if mask4 is None:
            mask4 = F.interpolate(masks.float(), size=[H4, W4], mode='nearest')

        # Auto-generate positional encodings if not provided
        if pos1 is None:
            pos1 = PositionalEncoding(self.dim, max_len=H1 * W1, device=device)
        if pos2 is None:
            pos2 = PositionalEncoding(self.dim * 2, max_len=H2 * W2, device=device)
        if pos3 is None:
            pos3 = PositionalEncoding(self.dim * 4, max_len=H3 * W3, device=device)
        if pos4 is None:
            pos4 = PositionalEncoding(self.dim * 8, max_len=H4 * W4, device=device)
        if pos1_dec is None:
            pos1_dec = PositionalEncoding(self.dim * 2, max_len=H1 * W1, device=device)

        # 1. Input embedding: masked image concatenated with mask [B, 4, H, W] -> [B, dim, H1, W1]
        masked_img = (images * (1.0 - masks).float()) + masks.float()
        inp = torch.cat([masked_img, masks.float()], dim=1)
        inp_enc1 = self.patch_embed(inp)

        # 2. Encoder Level 1
        out_enc1 = self._run_stage(self.encoder_level1, inp_enc1, pos1, mask1)

        # 3. Encoder Level 2
        inp_enc2 = self.down1_2(out_enc1[0], mask1)
        out_enc2 = self._run_stage(self.encoder_level2, inp_enc2, pos2, mask2)

        # 4. Encoder Level 3
        inp_enc3 = self.down2_3(out_enc2[0], mask2)
        out_enc3 = self._run_stage(self.encoder_level3, inp_enc3, pos3, mask3)

        # 5. Latent Stage
        inp_latent = self.down3_4(out_enc3[0], mask3)
        out_latent = self._run_stage(self.latent, inp_latent, pos4, mask4)

        # 6. Decoder Level 3
        inp_dec3 = self.up4_3(out_latent[0], mask4)
        inp_dec3 = torch.cat([inp_dec3, out_enc3[0]], dim=1)
        inp_dec3 = self.reduce_chan_level3(inp_dec3)
        out_dec3 = self._run_stage(self.decoder_level3, inp_dec3, pos3, mask3)

        # 7. Decoder Level 2
        inp_dec2 = self.up3_2(out_dec3[0], mask3)
        inp_dec2 = torch.cat([inp_dec2, out_enc2[0]], dim=1)
        inp_dec2 = self.reduce_chan_level2(inp_dec2)
        out_dec2 = self._run_stage(self.decoder_level2, inp_dec2, pos2, mask2)

        # 8. Decoder Level 1
        inp_dec1 = self.up2_1(out_dec2[0], mask2)
        inp_dec1 = torch.cat([inp_dec1, out_enc1[0]], dim=1)
        out_dec1 = self._run_stage(self.decoder_level1, inp_dec1, pos1_dec, mask1)

        # 9. Refinement Stage
        refined = self._run_stage(self.refinement, out_dec1[0], pos1_dec, mask1)

        # 10. RGB Output Projection: scaled to [0, 1] [B, 3, H, W]
        out_img = self.output(refined[0])
        out_img = (torch.tanh(out_img) + 1.0) / 2.0
        return out_img
