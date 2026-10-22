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
    return parser.parse_args()

def main():
    args = parse_args()
    config = Config(args.config)
    config.INPUT_SIZE = args.input_size

    if torch.cuda.is_available():
        config.DEVICE = torch.device('cuda')
    else:
        config.DEVICE = torch.device('cpu')

    out_dir = args.output
    create_dir(out_dir)
    create_dir(os.path.join(out_dir, 'inpainted'))
    create_dir(os.path.join(out_dir, 'stitched'))

    test_img_source = args.input if args.input is not None else getattr(config, 'TEST_IMAGE_FLIST', None)
    test_mask_source = args.mask if args.mask is not None else getattr(config, 'TEST_MASK_FLIST', None)

    dataset = InpaintingDataset(
        img_source=test_img_source,
        mask_source=test_mask_source,
        input_size=config.INPUT_SIZE,
        augment=False,
        training=False
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    print(f"[Inference] Loaded test dataset with {len(dataset)} samples.")
    model = ICEM_DBA_InpaintingModel(config)
    if args.weights and os.path.exists(args.weights):
        print(f"[Inference] Loading custom weights from {args.weights}...")
        model.generator.load_state_dict(torch.load(args.weights, map_location=config.DEVICE)['generator'], strict=False)
    else:
        model.load()

    model.eval()

    psnr_list, ssim_list, l1_list, bsd_list = [], [], [], []

    print("\n" + "=" * 50)
    print("Evaluating Test Set...")
    print("=" * 50)

    with torch.no_grad():
        for idx, (images, masks) in enumerate(dataloader):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            # Generator forward
            outputs_img = model(images, masks)
            merged = (outputs_img * masks) + (images * (1.0 - masks))

            psnr, ssim, l1 = compute_metrics(images, merged)
            bsd = compute_bsd(images, merged, masks)

            psnr_list.append(psnr)
            ssim_list.append(ssim)
            l1_list.append(l1)
            bsd_list.append(bsd)

            # Save qualitative images
            inputs_masked = images * (1.0 - masks)
            imsave(merged[0], os.path.join(out_dir, 'inpainted', f"result_{idx:05d}.png"))
            grid = stitch_images(
                postprocess(images),
                postprocess(inputs_masked),
                postprocess(outputs_img),
                postprocess(merged),
                img_per_row=1
            )
            grid.save(os.path.join(out_dir, 'stitched', f"stitched_{idx:05d}.png"))

            print(f"Sample [{idx+1}/{len(dataset)}] - PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}, L1: {l1:.4f}, BSD: {bsd:.4f}")

    print("\n" + "=" * 50)
    print("Final Quantitative Test Results:")
    print("=" * 50)
    print(f"  Mean PSNR : {np.mean(psnr_list):.3f} dB")
    print(f"  Mean SSIM : {np.mean(ssim_list):.4f}")
    print(f"  Mean L1   : {np.mean(l1_list):.4f}")
    print(f"  Mean BSD  : {np.mean(bsd_list):.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
