import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tools import create_dir, stitch_images, imsave, postprocess
from utils.metrics import compute_metrics, compute_bsd
from data.dataset import InpaintingDataset
from models.inpainting_model import ICEM_DBA_InpaintingModel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ICEM-DBA Inpainting Architecture")
    parser.add_argument('--config', type=str, default='./configs/config.yml', help='Path to configuration YAML')
    parser.add_argument('--input', type=str, default=None, help='Path to test images folder or .flist')
    parser.add_argument('--mask', type=str, default=None, help='Path to test masks folder or .flist')
    parser.add_argument('--output', type=str, default='./results/test', help='Output directory for inpainted results')
    parser.add_argument('--weights', type=str, default=None, help='Path to custom checkpoint weights')
    parser.add_argument('--input_size', type=int, default=256, help='Inference resolution')
    parser.add_argument('--num_samples', type=int, default=None, help='Max number of test samples to evaluate')
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(args.config)
    config.INPUT_SIZE = args.input_size
    config.VGG_PRETRAINED = False  # Not needed during pure inference evaluation

    if torch.cuda.is_available():
        config.DEVICE = torch.device('cuda')
    else:
        config.DEVICE = torch.device('cpu')

    out_dir = args.output
    create_dir(out_dir)
    inpainted_dir = os.path.join(out_dir, 'inpainted')
    stitched_dir = os.path.join(out_dir, 'stitched')
    create_dir(inpainted_dir)
    create_dir(stitched_dir)

    # Resolve test image source
    test_img_source = args.input
    if test_img_source is None:
        if os.path.exists('./data/celeba_test'):
            test_img_source = './data/celeba_test'
        else:
            test_img_source = getattr(config, 'TEST_IMAGE_FLIST', None)

    test_mask_source = args.mask if args.mask is not None else getattr(config, 'TEST_MASK_FLIST', None)

    dataset = InpaintingDataset(
        img_source=test_img_source,
        mask_source=test_mask_source,
        input_size=config.INPUT_SIZE,
        augment=False,
        training=False
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_available = len(dataset)
    num_to_eval = min(total_available, args.num_samples) if args.num_samples else total_available
    print(f"[Inference] Loaded test dataset with {total_available} samples. Evaluating {num_to_eval} samples.")

    # Initialize model
    model = ICEM_DBA_InpaintingModel(config)

    # Resolve checkpoint weights
    weights_path = args.weights
    if weights_path is None or not os.path.exists(weights_path):
        candidates = [
            './icem_dba_gen_best.pth',
            './checkpoints/icem_dba_gen_best.pth',
            './checkpoints/icem_dba_gen.pth'
        ]
        for c in candidates:
            if os.path.exists(c):
                weights_path = c
                break

    if weights_path and os.path.exists(weights_path):
        print(f"[Inference] Loading checkpoint weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location=config.DEVICE)
        state_dict = checkpoint.get('generator', checkpoint)
        # Strip any potential DataParallel 'module.' prefixes
        clean_state_dict = {
            (k[7:] if k.startswith('module.') else k): v
            for k, v in state_dict.items()
        }
        load_res = model.generator.load_state_dict(clean_state_dict, strict=False)
        print(f"            Weights loaded successfully: {load_res}")
        if 'psnr' in checkpoint:
            print(f"            Checkpoint Record PSNR: {checkpoint['psnr']:.2f} dB (Iteration {checkpoint.get('iteration', 'N/A')})")
    else:
        model.load()

    model.eval()

    psnr_list, ssim_list, l1_list, bsd_list, mask_pct_list = [], [], [], [], []
    records = []

    print("\n" + "=" * 65)
    print("            EVALUATING TEST DATASET (CELEBA-HQ)")
    print("=" * 65)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            if args.num_samples and idx >= args.num_samples:
                break

            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            # Compute exact mask percentage
            mask_pct = float((masks.sum() / masks.numel()).item() * 100.0)
            mask_pct_list.append(mask_pct)

            # Generator forward
            outputs_img = model(images, masks)
            merged = (outputs_img * masks) + (images * (1.0 - masks))

            psnr, ssim, l1 = compute_metrics(images, merged)
            bsd = compute_bsd(images, merged, masks)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            l1_list.append(l1)
            bsd_list.append(bsd)

            # Save qualitative images with explicit column titles and metrics footer banner
            inputs_masked = images * (1.0 - masks)
            inpainted_file = f"result_{idx+1:04d}.png"
            stitched_file = f"stitched_{idx+1:04d}.png"
            imsave(merged[0], os.path.join(inpainted_dir, inpainted_file))

            metric_banner = f"Sample {idx+1:02d} | Mask Area: {mask_pct:.1f}% | PSNR: {psnr:5.2f} dB | SSIM: {ssim:.4f} | L1: {l1:.4f} | BSD: {bsd:.4f}"
            col_headers = [
                "(a) Ground Truth",
                f"(b) Masked Input ({mask_pct:.1f}%)",
                "(c) ICEM-DBA Output",
                "(d) Final Composite"
            ]

            grid = stitch_images(
                postprocess(images),
                postprocess(inputs_masked),
                postprocess(outputs_img),
                postprocess(merged),
                img_per_row=1,
                column_titles=col_headers,
                metric_text=metric_banner
            )
            grid.save(os.path.join(stitched_dir, stitched_file))

            records.append({
                'sample': idx + 1,
                'mask_pct': round(mask_pct, 1),
                'psnr': round(psnr, 2),
                'ssim': round(ssim, 4),
                'l1': round(l1, 4),
                'bsd': round(bsd, 4),
                'file': stitched_file
            })

            print(f"Sample [{idx+1:3d}/{num_to_eval:3d}] | Mask: {mask_pct:4.1f}% | PSNR: {psnr:5.2f} dB | SSIM: {ssim:.4f} | L1: {l1:.4f} | BSD: {bsd:.4f}")

    # Compute aggregate statistics
    mean_psnr = float(np.mean(psnr_list))
    std_psnr = float(np.std(psnr_list))
    mean_ssim = float(np.mean(ssim_list))
    std_ssim = float(np.std(ssim_list))
    mean_l1 = float(np.mean(l1_list))
    std_l1 = float(np.std(l1_list))
    mean_bsd = float(np.mean(bsd_list))
    std_bsd = float(np.std(bsd_list))
    mean_mask_pct = float(np.mean(mask_pct_list))
    std_mask_pct = float(np.std(mask_pct_list))

    print("\n" + "=" * 65)
    print("            FINAL QUANTITATIVE TEST RESULTS")
    print("=" * 65)
    print(f"  Evaluated Samples : {len(psnr_list)}")
    print(f"  Mean Mask Area    : {mean_mask_pct:.1f} ± {std_mask_pct:.1f}%")
    print(f"  Mean PSNR         : {mean_psnr:.3f} ± {std_psnr:.3f} dB")
    print(f"  Mean SSIM         : {mean_ssim:.4f} ± {std_ssim:.4f}")
    print(f"  Mean L1 (MAE)     : {mean_l1:.4f} ± {std_l1:.4f}")
    print(f"  Mean BSD          : {mean_bsd:.4f} ± {std_bsd:.4f}")
    print(f"  Inpainted images  : {inpainted_dir}")
    print(f"  Stitched previews : {stitched_dir}")
    print("=" * 65 + "\n")

    # Save metrics to CSV with mask percentage
    csv_path = os.path.join(out_dir, 'evaluation_metrics.csv')
    with open(csv_path, 'w') as f:
        f.write("sample,mask_pct,psnr,ssim,l1,bsd,preview_file\n")
        for r in records:
            f.write(f"{r['sample']},{r['mask_pct']},{r['psnr']},{r['ssim']},{r['l1']},{r['bsd']},{r['file']}\n")
    print(f"[Results] Detailed per-image metrics saved to: {csv_path}")

    # Save summary JSON
    import json
    summary_path = os.path.join(out_dir, 'test_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'num_samples': len(psnr_list),
            'mean_mask_pct': mean_mask_pct,
            'std_mask_pct': std_mask_pct,
            'mean_psnr': mean_psnr,
            'std_psnr': std_psnr,
            'mean_ssim': mean_ssim,
            'std_ssim': std_ssim,
            'mean_l1': mean_l1,
            'std_l1': std_l1,
            'mean_bsd': mean_bsd,
            'std_bsd': std_bsd,
            'weights_evaluated': weights_path,
        }, f, indent=4)
    print(f"[Results] Summary metrics saved to: {summary_path}")

if __name__ == '__main__':
    main()
