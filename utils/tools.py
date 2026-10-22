import os
import sys
import time
import random
import torch
import numpy as np
from PIL import Image

def create_dir(path):
    """Creates a directory if it doesn't already exist."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def create_mask(width, height, mask_width, mask_height, x=None, y=None):
    """Generates a random rectangular hole mask."""
    mask = np.zeros((height, width), dtype=np.float32)
    mask_x = x if x is not None else random.randint(0, max(0, width - mask_width))
    mask_y = y if y is not None else random.randint(0, max(0, height - mask_height))
    mask[mask_y:mask_y + mask_height, mask_x:mask_x + mask_width] = 1.0
    return mask

def postprocess(img):
    """
    Converts model tensor [B, C, H, W] in [0, 1] to [B, H, W, C] uint8 in [0, 255].
    """
    img = torch.clamp(img, 0.0, 1.0) * 255.0
    img = img.permute(0, 2, 3, 1).contiguous()
    return img.to(torch.uint8)

def imsave(img_tensor, path):
    """
    Saves a single image tensor or numpy array to file.
    img_tensor: [H, W, C] or [C, H, W] tensor or ndarray
    """
    if isinstance(img_tensor, torch.Tensor):
        arr = img_tensor.detach().cpu().numpy()
    else:
        arr = np.array(img_tensor)
        
    if arr.ndim == 3 and arr.shape[0] in [1, 3]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0:
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr.squeeze(-1)
        im = Image.fromarray(arr, mode='L')
    else:
        im = Image.fromarray(arr, mode='RGB')
    im.save(path)

def stitch_images(inputs, *outputs, img_per_row=1):
    """
    Stitches batch images horizontally and vertically for qualitative logging.
    """
    gap = 4
    all_imgs = [inputs, *outputs]
    n_categories = len(all_imgs)
    b_size = len(inputs)

    # First image dimensions
    first = all_imgs[0][0]
    if isinstance(first, torch.Tensor):
        first_np = first.cpu().numpy()
    else:
        first_np = np.array(first)
    height, width = first_np.shape[:2]

    total_w = width * n_categories * img_per_row + gap * (img_per_row * n_categories - 1)
    n_rows = int(np.ceil(b_size / img_per_row))
    total_h = height * n_rows + gap * (n_rows - 1)

    grid = Image.new('RGB', (total_w, total_h), (255, 255, 255))

    for b in range(b_size):
        row = b // img_per_row
        col = b % img_per_row
        for cat_idx, cat_list in enumerate(all_imgs):
            img_item = cat_list[b]
            if isinstance(img_item, torch.Tensor):
                arr = img_item.cpu().numpy().astype(np.uint8)
            else:
                arr = np.array(img_item).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            elif arr.shape[-1] == 1:
                arr = np.repeat(arr, 3, axis=-1)
            im = Image.fromarray(arr)
            x_pos = (col * n_categories + cat_idx) * (width + gap)
            y_pos = row * (height + gap)
            grid.paste(im, (x_pos, y_pos))

    return grid

class Progbar:
    """Clean terminal progress bar utility."""
    def __init__(self, target, width=25, verbose=1, stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.stateful_metrics = set(stateful_metrics) if stateful_metrics else set()
        self._seen_so_far = 0
        self._values = {}
        self._start = time.time()

    def add(self, n, values=None):
        self._seen_so_far += n
        if values:
            for k, v in values:
                if k not in self._values:
                    self._values[k] = [v * n, n]
                else:
                    self._values[k][0] += v * n
                    self._values[k][1] += n

        if self.verbose == 1:
            prev_total_width = self.width
            bar = 'Prog: ['
            ratio = float(self._seen_so_far) / max(1, self.target)
            percent = int(ratio * 100)
            fill_len = int(self.width * ratio)
            bar += '=' * fill_len + ('>' if fill_len < self.width else '')
            bar += '.' * (self.width - fill_len - 1 if fill_len < self.width else 0)
            bar += f'] {self._seen_so_far}/{self.target} ({percent}%)'

            info = ''
            for k in self._values:
                if k in self.stateful_metrics:
                    info += f' - {k}: {self._values[k][0]:.4f}'
                else:
                    avg = self._values[k][0] / max(1, self._values[k][1])
                    info += f' - {k}: {avg:.4f}'

            sys.stdout.write(f'\r{bar}{info}')
            sys.stdout.flush()
            if self._seen_so_far >= self.target:
                sys.stdout.write('\n')
                sys.stdout.flush()
