"""
Download and extract CelebA-HQ (256x256) directly into ~/datasets/celeba_hq_256
Fast multi-threaded extraction using Hugging Face datasets.
"""
import os
import sys
import argparse
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser(description="Download CelebA-HQ 256x256 dataset")
    parser.add_argument('--output_dir', type=str, default='~/datasets/celeba_hq_256',
                        help='Destination directory for CelebA-HQ images')
    parser.add_argument('--val_size', type=int, default=2000,
                        help='Number of images reserved for validation/testing (default: 2000)')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of parallel disk writing threads')
    args = parser.parse_args()

    out_base = os.path.expanduser(args.output_dir)
    train_dir = os.path.join(out_base, 'train')
    val_dir = os.path.join(out_base, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    print("=" * 65)
    print(" Downloading CelebA-HQ (256x256) via Hugging Face...")
    print(f" Destination Directory: {out_base}")
    print("=" * 65)

    try:
        from datasets import load_dataset
    except ImportError:
        print("[Notice] Installing required 'datasets' package...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "pyarrow"])
        from datasets import load_dataset

    print("\n[1/3] Streaming/Downloading dataset shards from Hugging Face...")
    dataset = load_dataset("korexyz/celeba-hq-256x256", split="train")
    total_imgs = len(dataset)
    print(f"      Total images loaded: {total_imgs}")

    val_split_idx = total_imgs - args.val_size
    print(f"      Splitting into: {val_split_idx} Train images | {args.val_size} Val/Test images")

    print("\n[2/3] Extracting and saving images to disk with parallel I/O...")

    def save_item(item_tuple):
        idx, item = item_tuple
        img = item['image']
        if idx < val_split_idx:
            filename = os.path.join(train_dir, f"{idx:05d}.jpg")
        else:
            filename = os.path.join(val_dir, f"{idx:05d}.jpg")
        img.save(filename, format='JPEG', quality=95)
        return filename

    train_flist_paths = []
    val_flist_paths = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        items = list(enumerate(dataset))
        saved_paths = list(executor.map(save_item, items))

    train_flist_paths = saved_paths[:val_split_idx]
    val_flist_paths = saved_paths[val_split_idx:]

    print("\n[3/3] Generating file lists (.flist)...")
    flist_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(flist_dir, exist_ok=True)

    train_flist_file = os.path.join(flist_dir, 'train_celebahq.flist')
    val_flist_file = os.path.join(flist_dir, 'val_celebahq.flist')

    with open(train_flist_file, 'w') as f:
        for p in train_flist_paths:
            f.write(p + '\n')

    with open(val_flist_file, 'w') as f:
        for p in val_flist_paths:
            f.write(p + '\n')

    print(f"      Saved {len(train_flist_paths)} paths to {train_flist_file}")
    print(f"      Saved {len(val_flist_paths)} paths to {val_flist_file}")

    print("\n" + "=" * 65)
    print(" CelebA-HQ 256x256 Download & Setup Complete!")
    print(f" Training folder:   {train_dir}")
    print(f" Validation folder: {val_dir}")
    print("=" * 65)

if __name__ == '__main__':
    main()
