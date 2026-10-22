import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.icem_dba_modules import (
    get_distance_map,
    CNNScoreMap,
    DirectionalBoundaryAffinity,
    InstanceConditionedExemplarMemory,
    MACGPathfinder,
    GumbelPromptRouter
)
from models.mamba_layer import ICEM_DBAMambaLayer, PyTorchSelectiveScan
from models.transformer_block import ICEM_DBATransformerBlock
from models.generator import ICEM_DBA_Net, PositionalEncoding, Downsample, Upsample
from models.discriminator import Discriminator
from models.loss import AdversarialLoss, TVLoss, BoundarySeamLoss
from models.inpainting_model import ICEM_DBA_InpaintingModel
from utils.config import Config

def print_test_header(title):
    print("\n" + "=" * 65)
    print(f" [TEST] {title}")
    print("=" * 65)

def test_distance_map():
    print_test_header("1. Geodesic Distance Map Verification (Priority & Physical)")
    B, C, H, W = 2, 1, 64, 64
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 16:48, 16:48] = 1.0  # Center hole
    dist_priority, dist_physical = get_distance_map(mask)
    
    assert dist_priority.shape == (B, 1, H, W), f"Priority map shape mismatch: {dist_priority.shape}"
    assert dist_physical.shape == (B, 1, H, W), f"Physical map shape mismatch: {dist_physical.shape}"
    
    # Priority distance: boundary is highest (close to 1), center is lowest (close to 0)
    boundary_priority = dist_priority[:, :, 16, 16].mean()
    center_priority = dist_priority[:, :, 32, 32].mean()
    assert boundary_priority > center_priority, f"Priority inverted! Boundary: {boundary_priority}, Center: {center_priority}"

    # Physical distance: boundary is lowest, center is highest
    boundary_physical = dist_physical[:, :, 16, 16].mean()
    center_physical = dist_physical[:, :, 32, 32].mean()
    assert center_physical > boundary_physical, f"Physical inverted! Boundary: {boundary_physical}, Center: {center_physical}"
    
    # Known region must be zero in both
    assert torch.all(dist_priority[:, :, :16, :] == 0.0)
    assert torch.all(dist_physical[:, :, :16, :] == 0.0)
    print(f"  Passed! Priority: boundary={boundary_priority:.3f} > center={center_priority:.3f}")
    print(f"          Physical: center={center_physical:.3f} > boundary={boundary_physical:.3f}")

def test_cnn_score_map():
    print_test_header("2. CNNScoreMap (VAMamba Texture/Content Saliency)")
    B, C, H, W = 2, 48, 64, 64
    x = torch.randn(B, C, H, W)
    score_map_module = CNNScoreMap(in_channels=C, embed_dim=C)
    scores = score_map_module(x)
    assert scores.shape == (B, H, W), f"Score shape mismatch: {scores.shape}"
    assert scores.min() >= 0.0 and scores.max() <= 1.0, "Scores not in [0, 1]"
    print(f"  Passed! CNNScoreMap output shape: {scores.shape}, range: [{scores.min():.3f}, {scores.max():.3f}]")

def test_dba_t():
    print_test_header("3. Directional Boundary Affinity Transition (DBA-T)")
    B, C, H, W = 2, 48, 64, 64
    feat = torch.randn(B, C, H, W)
    _, dist_physical = get_distance_map(torch.zeros(B, 1, H, W))
    dist_physical[:, :, 16:48, 16:48] = torch.rand(B, 1, 32, 32)
    dba = DirectionalBoundaryAffinity(dim=C)
    kappa, mod_factor = dba(feat, dist_physical)
    assert kappa.shape == (B, 1, H, W), f"Kappa shape mismatch: {kappa.shape}"
    assert mod_factor.shape == (B, 1, H, W), f"Mod factor shape mismatch: {mod_factor.shape}"
    assert kappa.min() >= -1.0 and kappa.max() <= 1.0, f"Kappa not in [-1, 1]: [{kappa.min()}, {kappa.max()}]"
    print(f"  Passed! Cosine alignment kappa in [{kappa.min():.3f}, {kappa.max():.3f}], mod_factor in [{mod_factor.min():.3f}, {mod_factor.max():.3f}]")

def test_icem_memory_and_gumbel_router():
    print_test_header("4. DIG-SM, LC-VER & Gumbel Router (Prompt Bridge C + P_dyn)")
    B, C, H, W = 2, 48, 32, 32
    L = H * W
    d_state, inner_rank, d_key = 16, 32, 16
    x = torch.randn(B, C, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 8:24, 8:24] = 1.0
    shared_N = torch.randn(inner_rank, d_state)

    # 1. DIG-SM & LC-VER
    icem = InstanceConditionedExemplarMemory(dim=C, d_state=d_state, inner_rank=inner_rank, d_key=d_key)
    N_inst, exemplar_out = icem(x, mask, shared_N)
    assert N_inst.shape == (B, inner_rank, d_state), f"N_inst shape mismatch: {N_inst.shape}"
    assert exemplar_out.shape == (B, C, H, W), f"Exemplar readout shape mismatch: {exemplar_out.shape}"
    # Known region must be zero due to mask gating
    assert torch.all(exemplar_out[:, :, :8, :] == 0.0), "Exemplar readout leaked outside hole mask"

    # 2. Gumbel-Softmax Router
    router = GumbelPromptRouter(dim=C, inner_rank=inner_rank, d_state=d_state)
    x_seq = torch.randn(B, L, C)
    P_dyn = router(x_seq, N_inst)
    assert P_dyn.shape == (B, L, d_state), f"P_dyn shape mismatch: {P_dyn.shape}"
    print(f"  Passed! Dynamic N_inst shape: {N_inst.shape}, Gated exemplar readout: {exemplar_out.shape}")
    print(f"          Gumbel Router P_dyn synthesized: {P_dyn.shape} successfully bridged to C + P_dyn.")

def test_macg_order_non_inverted():
    print_test_header("5. MACG Pathfinder Non-Inverted Traversal Order Verification")
    B, C, H, W = 2, 48, 32, 32
    x = torch.randn(B, C, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 8:24, 8:24] = 1.0  # Hole from [8..23, 8..23]
    
    pathfinder = MACGPathfinder(dim=C)
    order, dist_map, patch_size = pathfinder(x, mask)
    assert order.shape == (B, H * W), f"Order tensor shape mismatch: {order.shape}"

    # Pick specific pixel locations:
    # 1. Known pixel: (0, 0) -> 0
    # 2. Hole boundary pixel: (8, 8) -> 8 * 32 + 8 = 264
    # 3. Deep hole center pixel: (16, 16) -> 16 * 32 + 16 = 528
    known_idx = 0
    boundary_idx = 8 * W + 8
    center_idx = 16 * W + 16

    order_b0 = order[0].tolist()
    rank_known = order_b0.index(known_idx)
    rank_boundary = order_b0.index(boundary_idx)
    rank_center = order_b0.index(center_idx)

    print(f"  Traversal step ranks: Known={rank_known}, Boundary={rank_boundary}, Deep Center={rank_center}")
    assert rank_known < rank_boundary, f"Known tokens must be traversed BEFORE boundary! {rank_known} >= {rank_boundary}"
    assert rank_boundary < rank_center, f"Boundary tokens must be traversed BEFORE deep center! {rank_boundary} >= {rank_center}"
    print("  Passed! Traversal order is strictly: Known (1st) -> Boundary (2nd) -> Center (3rd)!")

def test_channel_dimensions_resizing():
    print_test_header("6. Resizing Channel Arithmetic Verification (PixelUnshuffle & PixelShuffle)")
    B, C, H, W = 2, 48, 64, 64
    x = torch.randn(B, C, H, W)
    down = Downsample(n_feat=C)
    x_down = down(x)
    assert x_down.shape == (B, C * 2, H // 2, W // 2), f"Downsample shape error: {x_down.shape}"
    
    up = Upsample(n_feat=C * 2)
    x_up = up(x_down)
    assert x_up.shape == (B, C, H, W), f"Upsample shape error: {x_up.shape}"
    print(f"  Passed! PixelUnshuffle(2) 48ch -> 192ch -> Conv1x1 -> 96ch ({x_down.shape})")
    print(f"          Upsample 96ch -> Conv1x1 -> 192ch -> PixelShuffle(2) -> 48ch ({x_up.shape})")

def test_mamba_layer():
    print_test_header("7. Unified ICEM_DBAMambaLayer Forward Pass")
    B, C, H, W = 2, 48, 32, 32
    x = torch.randn(B, C, H, W)
    pe = PositionalEncoding(C, max_len=H * W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 8:24, 8:24] = 1.0

    layer = ICEM_DBAMambaLayer(dim=C, d_state=16, d_conv=4, expand=2, inner_rank=32, d_key=16)
    out = layer(x, pe, mask)
    assert out.shape == (B, C, H, W), f"Mamba layer output shape mismatch: {out.shape}"
    print(f"  Passed! ICEM-DBA Mamba layer forward output shape: {out.shape}")

def test_full_generator():
    print_test_header("8. Full Proposed ICEM_DBA_Net Generator Forward Pass")
    B, H, W = 2, 64, 64
    images = torch.rand(B, 3, H, W)
    masks = torch.zeros(B, 1, H, W)
    masks[:, :, 16:48, 16:48] = 1.0

    net = ICEM_DBA_Net(
        inp_channels=4,
        out_channels=3,
        dim=32,
        num_blocks=[1, 1, 1, 1],
        num_refinement_blocks=1,
        d_state=16,
        inner_rank=16,
        d_key=16
    )
    out = net(images, masks)
    assert out.shape == (B, 3, H, W), f"Generator output shape mismatch: {out.shape}"
    assert out.min() >= 0.0 and out.max() <= 1.0, f"Generator output out of [0, 1]: [{out.min()}, {out.max()}]"
    print(f"  Passed! Full Generator inpainted RGB output shape: {out.shape}, range: [{out.min():.3f}, {out.max():.3f}]")

def test_discriminator_and_losses():
    print_test_header("9. Discriminator & Loss Objectives")
    B, H, W = 2, 64, 64
    real = torch.rand(B, 3, H, W)
    fake = torch.rand(B, 3, H, W)
    mask = torch.zeros(B, 1, H, W)
    mask[:, :, 16:48, 16:48] = 1.0

    dis = Discriminator(in_channels=3, use_sigmoid=True)
    d_out, feats = dis(fake)
    assert d_out.shape[1] == 1, f"Discriminator out channels mismatch: {d_out.shape}"

    adv_loss = AdversarialLoss(type='lsgan')
    l_adv = adv_loss(d_out, True, False)
    assert l_adv.item() >= 0.0

    tv = TVLoss()
    l_tv = tv(fake)
    assert l_tv.item() >= 0.0

    bsd = BoundarySeamLoss()
    l_bsd = bsd(fake, real, mask)
    assert l_bsd.item() >= 0.0

    print(f"  Passed! Dis out shape: {d_out.shape}, Adv loss: {l_adv.item():.4f}, TV loss: {l_tv.item():.4f}, BSD loss: {l_bsd.item():.4f}")

def test_end_to_end_training_step():
    print_test_header("10. Full InpaintingModel 2-Phase Optimization Step")
    B, H, W = 2, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = torch.rand(B, 3, H, W).to(device)
    masks = torch.zeros(B, 1, H, W).to(device)
    masks[:, :, 16:48, 16:48] = 1.0

    cfg = Config(
        DIM=32,
        NUM_BLOCKS=[1, 1, 1, 1],
        NUM_REFINEMENT_BLOCKS=1,
        D_STATE=16,
        INNER_RANK=16,
        D_KEY=16,
        BATCH_SIZE=2,
        INPUT_SIZE=64,
        LR=1e-4,
        VGG_PRETRAINED=False,
        DEVICE=device
    )

    model = ICEM_DBA_InpaintingModel(cfg)
    outputs_img, gen_loss, dis_loss, logs, g_gan, g_l1, g_cnt, g_stl = model.process(images, masks)
    print(f"  Diagnostics: gen_loss={gen_loss.item():.4f}, dis_loss={dis_loss.item():.4f}")
    print(f"  Detailed logs: {dict(logs)}")
    assert outputs_img.shape == (B, 3, H, W), "Output image shape error"
    assert not torch.isnan(gen_loss) and gen_loss.item() > 0.0, f"Gen loss invalid: {gen_loss.item()}"
    assert not torch.isnan(dis_loss) and dis_loss.item() > 0.0, f"Dis loss invalid: {dis_loss.item()}"

    print(f"  Passed! Forward and backward pass completed cleanly. Final logs: {dict(logs)}")

def run_all_tests():
    print("\n" + "#" * 65)
    print("  LAUNCHING FULL ICEM-DBA PIPELINE VERIFICATION SUITE")
    print("#" * 65)

    test_distance_map()
    test_cnn_score_map()
    test_dba_t()
    test_icem_memory_and_gumbel_router()
    test_macg_order_non_inverted()
    test_channel_dimensions_resizing()
    test_mamba_layer()
    test_full_generator()
    test_discriminator_and_losses()
    test_end_to_end_training_step()

    print("\n" + "#" * 65)
    print("  ALL 10 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("#" * 65 + "\n")

if __name__ == '__main__':
    run_all_tests()
