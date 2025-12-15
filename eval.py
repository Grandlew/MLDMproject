# eval.py
import os
import argparse
import yaml
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

from datasets.tiny_seg import make_loader
from models.vit import DeiTTiny
from models.heads import SegHead
# overall mIoU (uses ignore_index=255 internally)
from utils.metrics import miou


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to saved .ckpt")
    p.add_argument("--batch_size", type=int, default=None,
                   help="Override batch size for eval")
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed precision (CUDA only)")
    p.add_argument("--per_class", action="store_true",
                   help="Print per-class IoU breakdown")
    return p.parse_args()


def build_model(cfg, img_size, num_classes, device):
    mcfg = cfg["model"]
    model = DeiTTiny(
        img_size=img_size,
        patch=mcfg["patch"],
        embed_dim=mcfg["embed_dim"],
        depth=mcfg["depth"],
        heads=mcfg["heads"],
        mlp_ratio=mcfg["mlp_ratio"],
        drop_path=mcfg["stochastic_depth"],
    ).to(device)
    out_ch = 1 if num_classes <= 2 else num_classes
    head = SegHead(in_ch=mcfg["embed_dim"], out_ch=out_ch).to(device)
    return model, head


@torch.no_grad()
def per_class_iou(pred_classes: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255):
    """
    Per-class IoU for multi-class semantic segmentation.
    pred_classes: [N,H,W] long
    target:       [N,H,W] long (255 = ignore)
    Returns: (ious: [C] tensor with NaN for empty classes)
    """
    if pred_classes.dim() != 3 or target.dim() != 3:
        raise ValueError("per_class_iou expects [N,H,W] integer tensors")

    # Flatten and mask ignore
    pred = pred_classes.view(-1)
    tgt = target.view(-1)
    valid = (tgt != ignore_index)
    pred = pred[valid]
    tgt = tgt[valid]

    ious = pred.new_full((num_classes,), float("nan"), dtype=torch.float32)
    for c in range(num_classes):
        pred_c = (pred == c)
        tgt_c = (tgt == c)
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union > 0:
            ious[c] = (inter + 1e-7) / (union + 1e-7)
    return ious


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    # ---- Load checkpoint ----
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "cfg" not in ckpt:
        raise ValueError(
            "Checkpoint missing 'cfg'. Re-train or provide a config.")
    cfg = ckpt["cfg"]

    # ---- Dataset / loader ----
    img_size = int(cfg["dataset"]["img_size"])
    num_classes = int(cfg["dataset"].get("num_classes", 2))
    bs_eval = int(args.batch_size or cfg["train"].get("batch_size", 8))

    val_loader = make_loader(
        cfg["dataset"]["root"], img_size, bs_eval, "val", False, num_classes)

    # ---- Model ----
    model, head = build_model(cfg, img_size, num_classes, device)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval()
    head.eval()

    # ---- Eval loop ----
    mious = []
    per_class_accum = None
    per_class_count = 0

    with torch.no_grad(), autocast("cuda", enabled=(args.amp and torch.cuda.is_available())):
        for x, y in val_loader:
            x = x.to(device)
            # y: [B,H,W] long (trainIds w/ 255 ignore from dataset)
            y = y.to(device)

            # forward
            _, grid, _ = model(x)
            logits = head(grid)  # [B,C,h,w] or [B,1,h,w]
            logits = F.interpolate(
                logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

            if num_classes <= 2:
                # Binary: cast to classes {0,1} via threshold on sigmoid
                probs = torch.sigmoid(logits)            # [B,1,H,W]
                pred = (probs > 0.5).long().squeeze(1)  # [B,H,W]
                # treat as 2 classes (foreground/background)
                mious.append(miou(pred, y.long(), 2))
                if args.per_class:
                    pci = per_class_iou(pred, y.long(), 2)
                    per_class_accum = (pci if per_class_accum is None
                                       else torch.nan_to_num(per_class_accum, nan=0.0) + torch.nan_to_num(pci, nan=0.0))
                    per_class_count += 1
            else:
                # Multi-class: argmax over channels
                pred = torch.argmax(logits, dim=1)       # [B,H,W]
                mious.append(miou(pred, y.long(), num_classes))
                if args.per_class:
                    pci = per_class_iou(pred, y.long(), num_classes)
                    per_class_accum = (pci if per_class_accum is None
                                       else torch.nan_to_num(per_class_accum, nan=0.0) + torch.nan_to_num(pci, nan=0.0))
                    per_class_count += 1

    overall_miou = float(torch.stack(mious).mean().item()
                         ) if len(mious) else 0.0
    print(f"Overall mIoU: {overall_miou:.4f}")

    if args.per_class and per_class_accum is not None and per_class_count > 0:
        per_class_mean = per_class_accum / per_class_count
        # Print nicely
        print("Per-class IoU:")
        for c, v in enumerate(per_class_mean.tolist()):
            if v != v:  # NaN check
                print(f"  class {c:02d}:  (no pixels) ")
            else:
                print(f"  class {c:02d}:  {v:.4f}")


if __name__ == "__main__":
    main()
