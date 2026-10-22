import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

class PSNR(nn.Module):
    """PSNR metric calculation module in PyTorch."""
    def __init__(self, max_val=255.0):
        super().__init__()
        base10 = torch.log(torch.tensor(10.0))
        max_val = torch.tensor(max_val).float()
        self.register_buffer('base10', base10)
        self.register_buffer('max_val', 20.0 * torch.log(max_val) / base10)

    def forward(self, a, b):
        mse = torch.mean((a.float() - b.float()) ** 2)
        if mse == 0:
            return torch.tensor(100.0, device=a.device)
        return self.max_val - 10.0 * torch.log(mse) / self.base10

def compute_metrics(gt_tensor, pred_tensor):
    """
    Computes quantitative evaluation metrics between ground truth and inpainted images:
    gt_tensor, pred_tensor: [B, C, H, W] in [0, 1] or [0, 255]
    Returns: psnr, ssim, l1
    """
    if gt_tensor.max() <= 1.0:
        gt_scaled = torch.clamp(gt_tensor, 0.0, 1.0) * 255.0
        pred_scaled = torch.clamp(pred_tensor, 0.0, 1.0) * 255.0
    else:
        gt_scaled = gt_tensor
        pred_scaled = pred_tensor

    gt_np = gt_scaled.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
    pred_np = pred_scaled.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

    psnr_scores = []
    ssim_scores = []

    for b in range(gt_np.shape[0]):
        g = gt_np[b]
        p = pred_np[b]
        p_val = min(100.0, compare_psnr(g, p, data_range=255))
        s_val = compare_ssim(g, p, channel_axis=-1, data_range=255)
        psnr_scores.append(p_val)
        ssim_scores.append(s_val)

    l1_val = torch.nn.functional.l1_loss(pred_tensor, gt_tensor, reduction='mean').item()

    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores)), l1_val

def compute_bsd(gt_tensor, pred_tensor, mask_tensor, kernel_size=3):
    """
    Computes Boundary Seam Discrepancy (BSD) metric (Hypothesis 2 in ICEM-DBA research report).
    Measures L1 discrepancy across a narrow 1-pixel boundary band around the mask perimeter.
    """
    dilated = torch.nn.functional.max_pool2d(mask_tensor.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    eroded = 1.0 - torch.nn.functional.max_pool2d(1.0 - mask_tensor.float(), kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    boundary_band = dilated - eroded

    diff = torch.abs(pred_tensor - gt_tensor) * boundary_band
    normalizer = torch.clamp(torch.sum(boundary_band), min=1.0)
    bsd_metric = (torch.sum(diff) / normalizer).item()
    return bsd_metric
