#!/usr/bin/env python3
import argparse, sys, random
from pathlib import Path
import cv2, numpy as np

def ensure_dirs(dst):
    (dst/'images_train').mkdir(parents=True, exist_ok=True)
    (dst/'masks_train').mkdir(parents=True, exist_ok=True)
    (dst/'images_val').mkdir(parents=True, exist_ok=True)
    (dst/'masks_val').mkdir(parents=True, exist_ok=True)

def list_pairs(src_root: Path):
    rgb = src_root/'RGB'; gt = src_root/'GT_index'
    images = sorted(list(rgb.rglob('*.png')) + list(rgb.rglob('*.jpg')))
    return [(img, gt/(img.stem + '.png')) for img in images if (gt/(img.stem + '.png')).exists()]

def to_binary_mask(gt_path: Path, ids):
    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if gt is None: raise FileNotFoundError(gt_path)
    return (np.isin(gt, ids).astype(np.uint8) * 255)

def process(pairs, dst_root: Path, count, size, ids, split):
    out_img = dst_root/(f'images_{split}')
    out_msk = dst_root/(f'masks_{split}')
    k = 0
    for img_path, gt_path in pairs:
        if k >= count: break
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None: continue
        mask = to_binary_mask(gt_path, ids)
        if size > 0:
            img  = cv2.resize(img,  (size, size), cv2.INTER_AREA)
            mask = cv2.resize(mask, (size, size), cv2.INTER_NEAREST)
        name = f'{img_path.stem}.png'
        cv2.imwrite(str(out_img/name), img)
        cv2.imwrite(str(out_msk/name), mask)
        k += 1
    return k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', default='data/sample')
    ap.add_argument('--train_n', type=int, default=200)
    ap.add_argument('--val_n', type=int, default=50)
    ap.add_argument('--img_size', type=int, default=160)
    ap.add_argument('--traversable_ids', default='1,2')
    a = ap.parse_args()

    ids = [int(x) for x in a.traversable_ids.split(',') if x.strip()]
    src = Path(a.src); dst = Path(a.dst); ensure_dirs(dst)
    pairs = list_pairs(src)
    if not pairs:
        print('No pairs found. Check --src.', file=sys.stderr); sys.exit(1)
    random.seed(0); random.shuffle(pairs)
    kt = process(pairs,      dst, a.train_n, a.img_size, ids, 'train')
    kv = process(pairs[kt:], dst, a.val_n,   a.img_size, ids, 'val')
    print(f'Sample created: {kt} train, {kv} val @ {dst}')

if __name__ == '__main__': main()
