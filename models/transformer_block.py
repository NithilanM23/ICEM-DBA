import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .mamba_layer import ICEM_DBAMambaLayer

# =========================================================================
# Restormer / SEM-Net LayerNorm Utilities
# =========================================================================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# =========================================================================
# Spatially Enhanced Feed-Forward Network (SEFN)
# =========================================================================

class FeedForward(nn.Module):
    """
    Spatially Enhanced Feed-Forward Network (SEFN) fusing multi-scale
    spatial context via avg pooling and gating.
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.fusion = nn.Conv2d(hidden_features + dim, hidden_features, kernel_size=1, bias=bias)
        self.dwconv_afterfusion = nn.Conv2d(
            hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
            groups=hidden_features, bias=bias
        )
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=False),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True),
            LayerNorm(dim, 'WithBias'),
            nn.ReLU(inplace=False)
        )
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, x, spatial):
        x = self.project_in(x)

        # Spatial multi-scale branch
        y = self.avg_pool(spatial)
        y = self.conv(y)
        y = self.upsample(y)

        # Gated depthwise fusion
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.fusion(torch.cat((x1, y), dim=1))
        x1 = self.dwconv_afterfusion(x1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# =========================================================================
# ICEM-DBA Transformer Block
# =========================================================================

class ICEM_DBATransformerBlock(nn.Module):
    """
    Transformer Block utilizing ICEM_DBAMambaLayer (MACG + DIG-SM + LC-VER + DBA-T + SSM)
    and SEFN (Spatially Enhanced Feed-Forward Network).
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias',
                 d_state=16, inner_rank=32, d_key=16):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = ICEM_DBAMambaLayer(
            dim=dim,
            d_state=d_state,
            inner_rank=inner_rank,
            d_key=d_key
        )
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

    def forward(self, x, pos=None, mask=None):
        """
        Supports both direct tensor signature forward(x, pos, mask)
        and legacy dictionary signature forward({0: x, 1: pos, 2: mask}).
        """
        is_dict = isinstance(x, dict)
        if is_dict:
            x, pos, mask = x[0], x[1], x[2]

        x_spatial = x

        # 1. ICEM-DBA Mamba Layer with Skip Connection
        x = x + self.attn(self.norm1(x), pos, mask)

        # 2. Spatially Enhanced Feed-Forward Network with Skip Connection
        x = x + self.ffn(self.norm2(x), x_spatial)

        if is_dict:
            return {0: x, 1: pos, 2: mask}
        return x
