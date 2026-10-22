import os
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
    num_workers = 2 if os.name != 'nt' else 0  # 0 on Windows to prevent multiprocessing deadlocks
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )

    # Model initialization
    print("\n[Model] Initializing ICEM-DBA Inpainting Model...")
    model = ICEM_DBA_InpaintingModel(config)
    model.load()

    psnr_metric = PSNR(255.0).to(config.DEVICE)
    max_iteration = int(config.MAX_ITERS)
    epoch = 0

    print("\n" + "=" * 50)
    print("Beginning Training Routine...")
    print("=" * 50)

    keep_training = True
    while keep_training:
        epoch += 1
        print(f"\n--- Epoch: {epoch} ---")
        progbar = Progbar(len(train_loader), width=25, verbose=config.VERBOSE)

        for step, (images, masks) in enumerate(train_loader):
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            model.train()
            outputs_img, gen_loss, dis_loss, logs, gen_gan, gen_l1, gen_cnt, gen_stl = model.process(images, masks)
            model.backward(gen_loss, dis_loss)

            iteration = model.iteration

            # Evaluate training PSNR
            with torch.no_grad():
                merged = (outputs_img * masks) + (images * (1.0 - masks))
                current_psnr = psnr_metric(postprocess(images), postprocess(merged)).item()
                logs.append(('psnr', current_psnr))

            progbar.add(1, values=logs)

            # Sample visualization
            if iteration % getattr(config, 'SAMPLE_INTERVAL', 500) == 0 or iteration == 1:
                with torch.no_grad():
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
                    print(f"\n[Sample] Saved visual preview to {sample_path}")

            # Save checkpoint
            if iteration % getattr(config, 'SAVE_INTERVAL', 1000) == 0:
                model.save()

            if iteration >= max_iteration:
                keep_training = False
                break

    print("\n[Done] Training completed successfully!")

if __name__ == '__main__':
    main()
