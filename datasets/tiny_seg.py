# datasets/tiny_seg.py
import os
import glob
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ------------------------------ #
# Utilities for Cityscapes pairs #
# ------------------------------ #
def _enumerate_cityscapes_pairs(root: str, split: str) -> List[Tuple[str, str]]:
    """
    List (image, mask) pairs for a prepared Cityscapes layout:

        root/
          images/<split>/**/<file>.png
          masks/<split>/**/<file>.png     (same structure as images)

    We try several Cityscapes-style mask filename conversions if the
    1:1 path does not exist, e.g.
      *_leftImg8bit.png -> *_labelTrainIds.png
      *_leftImg8bit.png -> *_gtFine_labelTrainIds.png
      *_leftImg8bit.png -> *_gtFine_labelIds.png
    """
    img_dir = os.path.join(root, "images", split)
    msk_dir = os.path.join(root, "masks", split)

    img_paths = glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True)
    pairs: List[Tuple[str, str]] = []

    for ip in img_paths:
        rel = os.path.relpath(ip, img_dir)  # keep nested folder structure
        mp = os.path.join(msk_dir, rel)

        if not os.path.exists(mp):
            base_rel = rel
            # Try common Cityscapes suffixes
            candidates = [
                base_rel.replace("_leftImg8bit.png", "_labelTrainIds.png"),
                base_rel.replace("_leftImg8bit.png",
                                 "_gtFine_labelTrainIds.png"),
                base_rel.replace("_leftImg8bit.png", "_gtFine_labelIds.png"),
            ]
            for r in candidates:
                cand = os.path.join(msk_dir, r)
                if os.path.exists(cand):
                    mp = cand
                    break

        if os.path.exists(mp):
            pairs.append((ip, mp))

    return pairs


# ------------------------------------------------ #
# labelIds -> trainIds mapping LUT (uint8 -> uint8) #
# ------------------------------------------------ #
def _cityscapes_labelids_to_trainids_lut() -> np.ndarray:
    """
    Build a 256-entry LUT mapping Cityscapes labelIds -> trainIds.
    Unused/void classes are mapped to 255 (ignore).

    Mapping (official 19-class subset):
      road=7->0, sidewalk=8->1, building=11->2, wall=12->3, fence=13->4,
      pole=17->5, traffic light=19->6, traffic sign=20->7, vegetation=21->8,
      terrain=22->9, sky=23->10, person=24->11, rider=25->12, car=26->13,
      truck=27->14, bus=28->15, train=31->16, motorcycle=32->17, bicycle=33->18.
    """
    lut = np.full(256, 255, dtype=np.uint8)  # 255 = ignore
    mapping = {
        7: 0, 8: 1, 11: 2, 12: 3, 13: 4,
        17: 5, 19: 6, 20: 7, 21: 8, 22: 9,
        23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
        28: 15, 31: 16, 32: 17, 33: 18,
    }
    for k, v in mapping.items():
        lut[k] = v
    return lut


_LUT_LABELIDS_TO_TRAINIDS = _cityscapes_labelids_to_trainids_lut()


class TinySegDataset(Dataset):
    """
    Minimal segmentation dataset:
      - Reads PNG images/masks
      - Resizes to (img_h, img_w), then pads on right/bottom if needed
      - Masks returned as integer class IDs (long), with 255 as ignore
      - If masks are still Cityscapes 'labelIds', auto-remap to 'trainIds'
    """

    def __init__(self, root: str, split: str, img_h: int, img_w: int, num_classes: int):
        super().__init__()
        self.root = root
        self.split = split
        self.img_h = img_h
        self.img_w = img_w
        self.num_classes = num_classes

        # Find pairs (recursive)
        self.pairs = _enumerate_cityscapes_pairs(root, split)

        if len(self.pairs) == 0:
            print(f"[WARN] No pairs found at {root}/images/{split} (recursive). "
                  f"Check directory structure and file names.")
        else:
            print(f"[INFO] Found {len(self.pairs)} {split} pairs under {root}")

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _pad_to_size(img: Image.Image, h: int, w: int, fill=0) -> Image.Image:
        iw, ih = img.size
        pad_w = max(0, w - iw)
        pad_h = max(0, h - ih)
        if pad_w or pad_h:
            # pad(right, bottom)
            img = TF.pad(img, [0, 0, pad_w, pad_h], fill=fill)
        return img

    def __getitem__(self, idx: int):
        img_path, msk_path = self.pairs[idx]

        # --- RGB image ---
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, [self.img_h, self.img_w],
                        interpolation=Image.BILINEAR)
        img = self._pad_to_size(img, self.img_h, self.img_w, fill=0)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[
                           0.229, 0.224, 0.225])

        # --- Mask (may be labelIds or trainIds) ---
        msk = Image.open(msk_path)
        msk = TF.resize(msk, [self.img_h, self.img_w],
                        interpolation=Image.NEAREST)
        msk = self._pad_to_size(msk, self.img_h, self.img_w, fill=255)

        # Convert to numpy (uint8) safely
        msk_np = np.array(msk, dtype=np.uint8)

        # If any value > 18, it's almost certainly labelIds -> remap to trainIds
        if msk_np.max() > 18:
            msk_np = _LUT_LABELIDS_TO_TRAINIDS[msk_np]

        # Convert to tensor LONG shape [H, W]
        msk_t = torch.from_numpy(msk_np.astype(np.int64))

        return img, msk_t


def make_loader(root: str, img_size: int, batch_size: int, split: str,
                shuffle: bool, num_classes: int) -> DataLoader:
    ds = TinySegDataset(root=root, split=split, img_h=img_size, img_w=img_size,
                        num_classes=num_classes)

    # Good defaults: half your CPU cores, persistent workers, prefetch
    nw = max(2, (os.cpu_count() or 4) // 2)
    use_cuda = torch.cuda.is_available()
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=use_cuda,
        persistent_workers=True,
        prefetch_factor=2,
    )
