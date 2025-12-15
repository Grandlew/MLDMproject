# scripts/prepare_cityscapes.py
import argparse
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="Cityscapes root containing leftImg8bit/ and gtFine/")
    p.add_argument("--out_root", default="data/cityscapes_512x1024",
                   help="Output root folder")
    p.add_argument("--img_h", type=int, default=512)
    p.add_argument("--img_w", type=int, default=1024)
    p.add_argument(
        "--mask_suffix",
        default="_gtFine_labelTrainIds.png",
        help="Label filename suffix to match. Common options: "
             "'_gtFine_labelTrainIds.png' (19 classes) or '_gtFine_labelIds.png' (34 classes).",
    )
    return p.parse_args()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def find_image_files(left_root: Path):
    # leftImg8bit/{train,val}/{city}/*_leftImg8bit.png
    splits = ["train", "val"]
    files = {s: [] for s in splits}
    for split in splits:
        split_dir = left_root / split
        if not split_dir.exists():
            continue
        for img in split_dir.rglob("*_leftImg8bit.png"):
            files[split].append(img)
    return files


def mask_for_image(img_path: Path, gt_root: Path, mask_suffix: str) -> Path:
    # Example image:
    #   .../leftImg8bit/train/berlin/berlin_000000_000019_leftImg8bit.png
    # We want:
    #   .../gtFine/train/berlin/berlin_000000_000019_gtFine_labelTrainIds.png
    split = img_path.parts[-3]     # train/val
    city = img_path.parts[-2]     # e.g., berlin
    stem = img_path.name.replace("_leftImg8bit.png", "")
    return gt_root / split / city / f"{stem}{mask_suffix}"


def resize_like_cityscapes(im: Image.Image, out_w: int, out_h: int, is_mask: bool) -> Image.Image:
    # Cityscapes images are 1024x2048 (H x W). We standardize to (img_h, img_w).
    # Use bilinear for RGB, nearest for mask to preserve IDs.
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    return im.resize((out_w, out_h), resample=resample)


def main():
    args = parse_args()

    root = Path(args.root)
    left_root = root / "leftImg8bit"
    gt_root = root / "gtFine"

    if not left_root.exists() or not gt_root.exists():
        raise FileNotFoundError(
            f"Could not find expected folders:\n  {left_root}\n  {gt_root}\n"
            "Make sure you extracted Cityscapes to --root so these exist."
        )

    out_root = Path(args.out_root)
    out_img_tr = out_root / "images" / "train"
    out_img_vl = out_root / "images" / "val"
    out_msk_tr = out_root / "masks" / "train"
    out_msk_vl = out_root / "masks" / "val"
    for d in [out_img_tr, out_img_vl, out_msk_tr, out_msk_vl]:
        ensure_dir(d)

    files = find_image_files(left_root)
    total_pairs = 0
    missing_masks = 0

    for split, imgs in files.items():
        if split not in ("train", "val"):
            continue
        if not imgs:
            print(f"[{split}] found 0 images")
            continue

        out_img_dir = out_img_tr if split == "train" else out_img_vl
        out_msk_dir = out_msk_tr if split == "train" else out_msk_vl

        for img_path in tqdm(imgs, desc=f"Processing {split}", ncols=80):
            mask_path = mask_for_image(img_path, gt_root, args.mask_suffix)
            if not mask_path.exists():
                # Try alternate suffix automatically (helpful if user picked the other one)
                alt_suffix = "_gtFine_labelIds.png" if args.mask_suffix.endswith(
                    "TrainIds.png") else "_gtFine_labelTrainIds.png"
                alt_mask = mask_for_image(img_path, gt_root, alt_suffix)
                if alt_mask.exists():
                    mask_path = alt_mask
                else:
                    missing_masks += 1
                    continue

            # Load
            img = Image.open(img_path).convert("RGB")
            msk = Image.open(mask_path)  # keep mode 'L' (ids)

            # Resize to requested size
            img_resized = resize_like_cityscapes(
                img, args.img_w, args.img_h, is_mask=False)
            msk_resized = resize_like_cityscapes(
                msk, args.img_w, args.img_h, is_mask=True)

            # Build output filename: keep original stem, but standardize suffixes
            stem = img_path.name.replace("_leftImg8bit.png", "")
            out_img = out_img_dir / f"{stem}.png"
            out_msk = out_msk_dir / f"{stem}.png"

            img_resized.save(out_img)
            # masks must be uint8 ids
            if msk_resized.mode != "L":
                msk_resized = msk_resized.convert("L")
            msk_resized.save(out_msk)

            total_pairs += 1

    print(f"Done. Output in: {out_root}")
    print(f"Total image/mask pairs written: {total_pairs}")
    if missing_masks:
        print(f"Warning: {missing_masks} images had no matching masks with suffix '{args.mask_suffix}' "
              f"(or its alternate).")


if __name__ == "__main__":
    main()
