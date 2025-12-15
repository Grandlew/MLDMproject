# scripts/vis_preds.py
from models.heads import SegHead
from models.vit import DeiTTiny
from datasets.tiny_seg import make_loader
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.amp import autocast

# allow running as a module from project root or direct
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Cityscapes trainId color palette (19 classes)
CITY_COLORS = np.array([
    [128,  64, 128], [244,  35, 232], [70,  70,  70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170,  30], [220, 220,   0],
    [107, 142,  35], [152, 251, 152], [70, 130, 180], [220,  20,  60],
    [255,   0,   0], [0,   0, 142], [0,   0,  70], [0,  60, 100],
    [0,  80, 100], [0,   0, 230], [119,  11,  32]
], dtype=np.uint8)


def colorize(train_ids: np.ndarray) -> Image.Image:
    h, w = train_ids.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    m = (train_ids >= 0) & (train_ids < 19)
    rgb[m] = CITY_COLORS[train_ids[m]]
    return Image.fromarray(rgb)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to best/last .ckpt")
    ap.add_argument("--out", default="reports/city_val_vis.jpg",
                    help="Output image path")
    ap.add_argument("--n", type=int, default=12,
                    help="Number of samples to visualize")
    ap.add_argument("--amp", action="store_true",
                    help="Use AMP if CUDA is available")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    img_size = int(cfg["dataset"]["img_size"])
    num_classes = int(cfg["dataset"].get("num_classes", 19))

    model, head = build_model(cfg, img_size, num_classes, device)
    model.load_state_dict(ckpt["model"])
    head.load_state_dict(ckpt["head"])
    model.eval()
    head.eval()

    val_loader = make_loader(
        cfg["dataset"]["root"], img_size, 4, "val", False, num_classes)

    tiles = []
    taken = 0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with torch.no_grad(), autocast("cuda", enabled=(args.amp and torch.cuda.is_available())):
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)  # y: [B,H,W] long
            _, grid, _ = model(x)
            # [B,C,h,w] or [B,1,h,w]
            logits = head(grid)
            logits = F.interpolate(
                logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            pred = torch.argmax(logits, dim=1) if num_classes > 2 else (
                torch.sigmoid(logits) > 0.5).long().squeeze(1)

            for i in range(x.size(0)):
                if taken >= args.n:
                    break
                img = (x[i].detach().cpu().clamp(0, 1).permute(
                    1, 2, 0).numpy() * 255).astype(np.uint8)
                gt = y[i].detach().cpu().numpy().astype(np.int32)
                pr = pred[i].detach().cpu().numpy().astype(np.int32)

                w = img.shape[1]
                panel = Image.new("RGB", (w * 3, img.shape[0]))
                panel.paste(Image.fromarray(img), (0, 0))
                panel.paste(colorize(gt),        (w, 0))
                panel.paste(colorize(pr),        (w * 2, 0))
                tiles.append(panel)
                taken += 1
            if taken >= args.n:
                break

    # stack tiles vertically
    if not tiles:
        raise SystemExit("No samples visualized (empty loader?)")
    H = sum(t.height for t in tiles)
    W = tiles[0].width
    out = Image.new("RGB", (W, H))
    y0 = 0
    for t in tiles:
        out.paste(t, (0, y0))
        y0 += t.height

    out.save(args.out)
    print("Saved", args.out)


if __name__ == "__main__":
    main()
