import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import math
import argparse
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# Clean up any leftover torch shared memory handles from killed Docker containers
if os.path.exists('/dev/shm'):
    try:
        import glob
        for f in glob.glob('/dev/shm/torch_*'):
            try:
                os.remove(f)
            except Exception:
                pass
    except Exception:
        pass

# Use file_system sharing strategy if multiprocessing is engaged
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass

from utils.config import Config
from utils.tools import create_dir, stitch_images, imsave, Progbar, postprocess
from utils.metrics import PSNR
from data.dataset import InpaintingDataset
from models.inpainting_model import ICEM_DBA_InpaintingModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train ICEM-DBA Inpainting Architecture")
    parser.add_argument('--config', type=str, default='./configs/config.yml', help='Path to configuration YAML')
    parser.add_argument('--batch_size', type=int, default=None, help='Override training batch size')
    parser.add_argument('--grad_accum_steps', type=int, default=None, help='Override gradient accumulation steps')
    parser.add_argument('--num_workers', type=int, default=None, help='Override DataLoader num_workers')
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
    if args.grad_accum_steps is not None:
        config.GRAD_ACCUM_STEPS = args.grad_accum_steps
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
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
        primary_gpu = gpus[0] if isinstance(gpus, list) and len(gpus) > 0 else 0
        config.DEVICE = torch.device(f'cuda:{primary_gpu}')
        torch.cuda.set_device(config.DEVICE)
        if len(gpus) > 1 and gpu_count > 1 and not getattr(config, 'USE_CHECKPOINT', True):
            print(f"[Hardware] Multi-GPU Mode: Active parallelization across GPUs {gpus}")
        else:
            print(f"[Hardware] Single-GPU Mode: Running on primary GPU [{primary_gpu}] ({torch.cuda.get_device_properties(primary_gpu).total_memory / 1e9:.2f} GB)")
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
    num_workers = int(getattr(config, 'NUM_WORKERS', 0))
    pin_mem = bool(getattr(config, 'PIN_MEMORY', False)) and (num_workers > 0)
    try:
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=True
        )
    except Exception as e:
        print(f"[DataLoader] Notice: fallback to in-process loader (num_workers=0): {e}")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
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

    accum_steps = int(getattr(config, 'GRAD_ACCUM_STEPS', 1))
    use_checkpoint = bool(getattr(config, 'USE_CHECKPOINT', True))
    effective_batch = batch_size * accum_steps
    patch_size = int(getattr(config, 'PATCH_SIZE', 4))
    token_dim = config.INPUT_SIZE // patch_size

    print("\n" + "=" * 65)
    print("           ICEM-DBA TRAINING CONFIGURATION SUMMARY")
    print("=" * 65)
    print(f" Dataset Size:      {len(train_dataset)} images ({steps_per_epoch} batches/epoch)")
    print(f" Micro-Batch Size:  {batch_size} (distributed across active GPUs)")
    print(f" Effective Batch:   {effective_batch} ({batch_size} micro-batch x {accum_steps} accum steps)")
    print(f" DataLoader:        Workers: {num_workers} | Sharing: file_system (Docker /dev/shm safe)")
    print(f" Memory Mode:       Checkpointing: {use_checkpoint} | Allocator: expandable_segments")
    print(f" Resolution:        {config.INPUT_SIZE}x{config.INPUT_SIZE} -> {token_dim}x{token_dim} Tokens ({token_dim * token_dim} tokens/layer)")
    print(f" Learning Rate:     Gen: {config.LR} | Dis: {config.LR * getattr(config, 'D2G_LR', 0.1)}")
    print(f" Checkpoints:       Every {getattr(config, 'SAVE_INTERVAL', 1000)} iterations -> {checkpoint_dir}")
    print(f" Visual Previews:   Every {getattr(config, 'SAMPLE_INTERVAL', 500)} iterations -> {results_dir}")
    print(f" Log Frequency:     Every {log_interval} iterations")
    print("=" * 65 + "\n")

    last_log_time = time.time()
    last_log_step = 0
    merged = None
    last_sample_iter = -1
    last_save_iter = -1
    best_psnr = -1.0

    sample_interval = getattr(config, 'SAMPLE_INTERVAL', 500)
    save_interval = getattr(config, 'SAVE_INTERVAL', 1000)
    total_epochs = math.ceil(max_iteration / max(1, steps_per_epoch))

    # Metric CSV logger for plotting loss curves and evaluation history
    csv_log_path = os.path.join(checkpoint_dir, 'training_metrics.csv')
    if not os.path.exists(csv_log_path):
        with open(csv_log_path, 'w') as f:
            f.write('epoch,step,iteration,gLoss,dLoss,l1,bsd,psnr,best_psnr,sec_per_it\n')

    keep_training = True
    while keep_training:
        epoch += 1
        print(f"\n>>> Starting Epoch {epoch}/{total_epochs}...", flush=True)
        if epoch == 1:
            print("    [Pipeline] Fetching initial training batch from DataLoader...", flush=True)
        last_log_time = time.time()
        last_log_step = 0

        for step, (images, masks) in enumerate(train_loader, start=1):
            if step == 1 and epoch == 1:
                print("    [Pipeline] Initial batch loaded! Starting forward & backward optimization...", flush=True)

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

                steps_elapsed = step - last_log_step if last_log_step > 0 else 1
                elapsed = time.time() - last_log_time
                speed = elapsed / max(1, steps_elapsed)
                remaining_steps = steps_per_epoch - step
                eta_secs = int(remaining_steps * speed)
                if eta_secs >= 3600:
                    eta_str = f"{eta_secs // 3600}h {(eta_secs % 3600) // 60:02d}m"
                else:
                    eta_str = f"{eta_secs // 60}m {eta_secs % 60:02d}s"

                # Global total ETA across all epochs
                remaining_total = max(0, max_iteration - iteration)
                total_eta_secs = int(remaining_total * speed)
                if total_eta_secs >= 3600:
                    total_eta_str = f"{total_eta_secs // 3600}h {(total_eta_secs % 3600) // 60:02d}m"
                else:
                    total_eta_str = f"{total_eta_secs // 60}m {total_eta_secs % 60:02d}s"

                if step == 1 and epoch == 1:
                    eta_str = "calculating..."
                    total_eta_str = "calculating..."

                percent = (step / steps_per_epoch) * 100.0
                bsd_val = [v for k, v in logs if k == 'bsd']
                bsd_str = f"{bsd_val[0]:.4f}" if bsd_val else "N/A"

                # Check and save best model snapshot based on PSNR
                if current_psnr > best_psnr and iteration > 50:
                    best_psnr = current_psnr
                    model.save_best(best_psnr)

                best_str = f"{best_psnr:5.2f}dB" if best_psnr > 0 else "N/A"

                print(f"[Epoch {epoch:2d}/{total_epochs:2d} | Step {step:4d}/{steps_per_epoch} ({percent:4.1f}%)] "
                      f"Iter: {iteration:6d}/{max_iteration} | "
                      f"gLoss: {gen_loss.item():.4f} | "
                      f"dLoss: {dis_loss.item():.4f} | "
                      f"l1: {gen_l1.item():.4f} | "
                      f"bsd: {bsd_str} | "
                      f"psnr: {current_psnr:5.2f}dB (Best: {best_str}) | "
                      f"{speed:.3f}s/it | "
                      f"ETA: {eta_str} (Total: {total_eta_str})", flush=True)

                # Append metrics to CSV for persistent curve plotting
                try:
                    with open(csv_log_path, 'a') as f:
                        f.write(f"{epoch},{step},{iteration},{gen_loss.item():.4f},{dis_loss.item():.4f},{gen_l1.item():.4f},{bsd_str},{current_psnr:.2f},{best_psnr:.2f},{speed:.4f}\n")
                except Exception:
                    pass

                last_log_time = time.time()
                last_log_step = step

            # Sample visual preview
            should_sample = (
                (step == 1 and epoch == 1 and last_sample_iter < 0) or
                (iteration > 0 and iteration % sample_interval == 0 and iteration != last_sample_iter)
            )
            if should_sample:
                last_sample_iter = iteration
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
            if iteration > 0 and iteration % save_interval == 0 and iteration != last_save_iter:
                last_save_iter = iteration
                model.save()

            if iteration >= max_iteration:
                keep_training = False
                break

    print("\n[Done] Training completed successfully!", flush=True)

if __name__ == '__main__':
    main()
