import os
import time
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tools import create_dir, stitch_images, imsave, Progbar, postprocess
from utils.metrics import PSNR
from data.dataset import InpaintingDataset
from models.inpainting_model import ICEM_DBA_InpaintingModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train ICEM-DBA Inpainting Architecture")
    parser.add_argument('--config', type=str, default='./configs/config.yml', help='Path to configuration YAML')
    parser.add_argument('--batch_size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--lr', type=float, default=None, help='Override generator learning rate')
    parser.add_argument('--max_iters', type=int, default=None, help='Override max iterations')
    parser.add_argument('--input_size', type=int, default=None, help='Override image resolution (e.g. 256)')
    parser.add_argument('--gpu', type=str, default=None, help='CUDA visible devices (e.g. "0" or "0,1")')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    config = Config(args.config)

    # Set CUDA device visibility from argument or config
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    elif hasattr(config, 'GPU') and isinstance(config.GPU, list):
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in config.GPU)
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LR = args.lr
    if args.max_iters is not None:
        config.MAX_ITERS = args.max_iters
    if args.input_size is not None:
        config.INPUT_SIZE = args.input_size

    # Device selection
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpus = getattr(config, 'GPU', [0])
        print(f"[Device] Detected {gpu_count} CUDA Device(s):")
        for i in range(gpu_count):
            print(f"  - GPU [{i}]: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB)")
        if len(gpus) > 1 and gpu_count > 1:
            print(f"[Hardware] Multi-GPU Mode: Active parallelization across GPUs {gpus}")
        config.DEVICE = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = True
    else:
        print("[Device] CUDA not available, using CPU.")
        config.DEVICE = torch.device('cpu')

    # Seed initialization
    seed = getattr(config, 'SEED', 10)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config.print()

    # Directories
    checkpoint_dir = getattr(config, 'PATH', './checkpoints')
    results_dir = os.path.join(getattr(config, 'RESULTS', './results'), 'train_samples')
    create_dir(checkpoint_dir)
    create_dir(results_dir)

    # Dataset & DataLoader
    train_dataset = InpaintingDataset(
        img_source=getattr(config, 'TRAIN_IMAGE_FLIST', None),
        mask_source=getattr(config, 'TRAIN_MASK_FLIST', None),
        input_size=config.INPUT_SIZE,
        augment=True,
        training=True
    )

    batch_size = config.BATCH_SIZE
    num_workers = 4 if os.name != 'nt' else 0  # Multi-threaded dataloading on Linux
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

    # Model initialization
    print("\n[Model] Initializing ICEM-DBA Inpainting Model...")
    model = ICEM_DBA_InpaintingModel(config)
    model.load()

    psnr_metric = PSNR(255.0).to(config.DEVICE)
    max_iteration = int(config.MAX_ITERS)
    epoch = 0

    steps_per_epoch = len(train_loader)
    log_interval = int(getattr(config, 'LOG_INTERVAL', 20))

    print("\n" + "=" * 65)
    print("           ICEM-DBA TRAINING CONFIGURATION SUMMARY")
    print("=" * 65)
    print(f" Dataset Size:      {len(train_dataset)} images ({steps_per_epoch} batches/epoch)")
    print(f" Batch Size:        {batch_size} (distributed across active GPUs)")
    print(f" Resolution:        {config.INPUT_SIZE}x{config.INPUT_SIZE}")
    print(f" Learning Rate:     Gen: {config.LR} | Dis: {config.LR * getattr(config, 'D2G_LR', 0.1)}")
    print(f" Checkpoints:       Every {getattr(config, 'SAVE_INTERVAL', 1000)} iterations -> {checkpoint_dir}")
    print(f" Visual Previews:   Every {getattr(config, 'SAMPLE_INTERVAL', 500)} iterations -> {results_dir}")
    print(f" Log Frequency:     Every {log_interval} iterations")
    print("=" * 65 + "\n")

    step_start_time = time.time()
    merged = None

    keep_training = True
    while keep_training:
        epoch += 1
        print(f"\n>>> Starting Epoch {epoch}...", flush=True)

        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            model.train()
            outputs_img, gen_loss, dis_loss, logs, gen_gan, gen_l1, gen_cnt, gen_stl = model.process(images, masks)
            model.backward(gen_loss, dis_loss)

            iteration = model.iteration

            # Clean periodic logging every LOG_INTERVAL steps
            if step % log_interval == 0 or step == 1 or step == steps_per_epoch:
                with torch.no_grad():
                    merged = (outputs_img * masks) + (images * (1.0 - masks))
                    current_psnr = psnr_metric(postprocess(images), postprocess(merged)).item()

                elapsed = time.time() - step_start_time
                steps_elapsed = log_interval if step > 1 else 1
                speed = elapsed / max(1, steps_elapsed)
                remaining_steps = steps_per_epoch - step
                eta_secs = int(remaining_steps * speed)
                eta_str = f"{eta_secs // 60}m {eta_secs % 60:02d}s"

                percent = (step / steps_per_epoch) * 100.0
                bsd_val = [v for k, v in logs if k == 'bsd']
                bsd_str = f"{bsd_val[0]:.4f}" if bsd_val else "N/A"

                print(f"[Epoch {epoch:2d} | Step {step:4d}/{steps_per_epoch} ({percent:4.1f}%)] "
                      f"Iter: {iteration:6d} | "
                      f"gLoss: {gen_loss.item():.4f} | "
                      f"dLoss: {dis_loss.item():.4f} | "
                      f"l1: {gen_l1.item():.4f} | "
                      f"bsd: {bsd_str} | "
                      f"psnr: {current_psnr:5.2f}dB | "
                      f"{speed:.2f}s/it (ETA: {eta_str})", flush=True)

                step_start_time = time.time()

            # Sample visual preview
            if iteration % getattr(config, 'SAMPLE_INTERVAL', 500) == 0 or iteration == 1:
                with torch.no_grad():
                    if merged is None:
                        merged = (outputs_img * masks) + (images * (1.0 - masks))
                    inputs_masked = images * (1.0 - masks)
                    grid = stitch_images(
                        postprocess(images),
                        postprocess(inputs_masked),
                        postprocess(outputs_img),
                        postprocess(merged),
                        img_per_row=1
                    )
                    sample_path = os.path.join(results_dir, f"sample_iter_{iteration:07d}.png")
                    grid.save(sample_path)
                    print(f"  >>> [Visual Preview] Saved preview image: {sample_path}", flush=True)

            # Save checkpoint
            if iteration % getattr(config, 'SAVE_INTERVAL', 1000) == 0:
                model.save()

            if iteration >= max_iteration:
                keep_training = False
                break

    print("\n[Done] Training completed successfully!", flush=True)

if __name__ == '__main__':
    main()
