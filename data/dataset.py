import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

from ..utils.tools import create_mask

class InpaintingDataset(Dataset):
    """
    Dataset loader for visual completion / inpainting.
    Supports file list (.flist), directories of images, and procedural/external masks.
    """
    def __init__(self, img_source, mask_source=None, input_size=256, mask_type=3, augment=True, training=True):
        super().__init__()
        self.input_size = input_size
        self.mask_type = mask_type
        self.augment = augment
        self.training = training

        self.image_files = self._load_flist(img_source)
        self.mask_files = self._load_flist(mask_source) if mask_source else []

    def _load_flist(self, source):
        if not source:
            return []
        if isinstance(source, list):
            return source
        if os.path.isfile(source):
            if source.endswith('.flist') or source.endswith('.txt'):
                with open(source, 'r') as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                return lines
            else:
                return [source]
        elif os.path.isdir(source):
            exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(source, ext)))
                files.extend(glob.glob(os.path.join(source, '**', ext), recursive=True))
            return sorted(list(set(files)))
        return []

    def __len__(self):
        return max(1, len(self.image_files))

    def _generate_procedural_mask(self, h, w):
        """Generates synthetic irregular brush strokes and random block holes."""
        mask = np.zeros((h, w), dtype=np.float32)
        
        # 1. Random block mask
        if random.random() > 0.5:
            bw = random.randint(w // 4, w // 2)
            bh = random.randint(h // 4, h // 2)
            bx = random.randint(0, w - bw)
            by = random.randint(0, h - bh)
            mask[by:by + bh, bx:bx + bw] = 1.0

        # 2. Random irregular strokes
        num_strokes = random.randint(4, 12)
        from PIL import ImageDraw
        pil_mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        draw = ImageDraw.Draw(pil_mask)

        for _ in range(num_strokes):
            x1, y1 = random.randint(0, w), random.randint(0, h)
            x2, y2 = random.randint(0, w), random.randint(0, h)
            width = random.randint(8, 28)
            draw.line([(x1, y1), (x2, y2)], fill=255, width=width)
            draw.ellipse([x1 - width // 2, y1 - width // 2, x1 + width // 2, y1 + width // 2], fill=255)

        mask = np.array(pil_mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        return mask

    def __getitem__(self, index):
        # Fallback if no images are present: generate a synthetic color pattern
        if len(self.image_files) == 0:
            synthetic_img = torch.rand(3, self.input_size, self.input_size)
            synthetic_mask = torch.from_numpy(self._generate_procedural_mask(self.input_size, self.input_size)).unsqueeze(0)
            return synthetic_img, synthetic_mask

        img_path = self.image_files[index % len(self.image_files)]
        img = Image.open(img_path).convert('RGB')

        # Load or generate mask
        if len(self.mask_files) > 0:
            mask_path = self.mask_files[index % len(self.mask_files)]
            mask = Image.open(mask_path).convert('L')
        else:
            mask = None

        # Resize
        if self.input_size > 0:
            img = TF.resize(img, [self.input_size, self.input_size])
            if mask is not None:
                mask = TF.resize(mask, [self.input_size, self.input_size], interpolation=TF.InterpolationMode.NEAREST)

        # Augmentation
        if self.augment and self.training and random.random() > 0.5:
            img = TF.hflip(img)
            if mask is not None:
                mask = TF.hflip(mask)

        # To tensor
        img_tensor = TF.to_tensor(img)  # [3, H, W] in [0, 1]

        if mask is not None:
            mask_tensor = TF.to_tensor(mask)  # [1, H, W] in [0, 1]
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            h, w = img_tensor.shape[1], img_tensor.shape[2]
            mask_np = self._generate_procedural_mask(h, w)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return img_tensor, mask_tensor
