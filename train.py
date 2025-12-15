# train.py
import os
import yaml
import time
import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.amp import autocast, GradScaler

from datasets.tiny_seg import make_loader
from models.vit import DeiTTiny
from models.heads import SegHead
from utils.metrics import dice_coeff, miou
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs in config")
    p.add_argument("--amp", action="store_true",
                   help="Enable mixed precision (CUDA only)")
    p.add_argument("--save_dir", default=None,
                   help="Override logging.out_dir in config")
    return p.parse_args()


def to_int(v, default=None):
    if v is None:
        return default
    return int(v)


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
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # --------- config values (with safe coercions) ----------
    img_size = to_int(cfg["dataset"]["img_size"])
    num_classes = to_int(cfg["dataset"].get("num_classes", 2))
    epochs = to_int(args.epochs, cfg["train"]["epochs"])
    bs = to_int(cfg["train"]["batch_size"])

    lr_raw = cfg["train"]["lr"]
    wd_raw = cfg["train"]["weight_decay"]
    lr = float(lr_raw) if not isinstance(lr_raw, (float, int)) else lr_raw
    wd = float(wd_raw) if not isinstance(wd_raw, (float, int)) else wd_raw

    out_dir = args.save_dir or cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision("high")

    # --------- data ----------
    train_loader = make_loader(
        cfg["dataset"]["root"], img_size, bs, "train", True,  num_classes)
    val_loader = make_loader(
        cfg["dataset"]["root"], img_size, bs, "val",   False, num_classes)

    # --------- model / opt / loss ----------
    model, head = build_model(cfg, img_size, num_classes, device)
    params = list(model.parameters()) + list(head.parameters())
    opt = optim.AdamW(params, lr=lr, weight_decay=wd)

    scaler = GradScaler("cuda", enabled=(
        args.amp and torch.cuda.is_available()))

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=255)

    best_score = -1.0  # mIoU for multi-class, Dice for binary

    # --------- training loop ----------
    for ep in range(epochs):
        model.train()
        head.train()
        t0 = time.time()
        it = 0

        for x, y in train_loader:
            it += 1
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=(args.amp and torch.cuda.is_available())):
                # forward
                # ViT backbone features (B, H', W', C?) depending on your head
                _, grid, _ = model(x)
                # (B, C, h, w) where C = 1 or num_classes
                logits = head(grid)

                # ensure same spatial size as target
                logits = F.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

                # loss
                if num_classes <= 2:
                    # [B,1,H,W] vs [B,1,H,W]
                    loss = bce(logits, y.unsqueeze(1).float())
                else:
                    # [B,C,H,W] vs [B,H,W]
                    loss = ce(logits, y.long())

            scaler.scale(loss).backward()
            # optional but recommended: clip grads (unscale first for AMP)
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(opt)
            scaler.update()

            if it % 10 == 0:
                print(
                    f"  [ep {ep+1}] iter {it}/{len(train_loader)} loss={loss.detach().item():.4f}")

        # --------- validation ----------
        model.eval()
        head.eval()
        dices, mious = [], []
        with torch.no_grad(), autocast("cuda", enabled=(args.amp and torch.cuda.is_available())):
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                _, grid, _ = model(x)
                logits = head(grid)
                logits = F.interpolate(
                    logits, size=y.shape[-2:], mode="bilinear", align_corners=False)

                if num_classes <= 2:
                    probs = torch.sigmoid(logits)                 # [B,1,H,W]
                    # expects binary masks
                    dices.append(dice_coeff(probs, y))
                else:
                    pred = torch.argmax(logits, dim=1)            # [B,H,W]
                    # ignores 255 internally
                    mious.append(miou(pred, y.long(), num_classes))

        # aggregate metrics
        if num_classes <= 2:
            val_dice = float(torch.stack(dices).mean().item()
                             ) if len(dices) else 0.0
            val_miou = 0.0
            score = val_dice
            print(
                f"Epoch {ep+1}/{epochs} - time {time.time()-t0:.1f}s - val Dice={val_dice:.4f}")
        else:
            val_miou = float(torch.stack(mious).mean().item()
                             ) if len(mious) else 0.0
            val_dice = 0.0
            score = val_miou
            print(
                f"Epoch {ep+1}/{epochs} - time {time.time()-t0:.1f}s - val mIoU={val_miou:.4f}")

        # save best
        if score > best_score:
            best_score = score
            best_path = os.path.join(out_dir, "deit_tiny_best.ckpt")
            torch.save(
                {"model": model.state_dict(), "head": head.state_dict(),
                 "cfg": cfg, "score": best_score},
                best_path,
            )
            print("  -> Saved BEST:", best_path)

        # save last each epoch
        last_path = os.path.join(out_dir, "deit_tiny_last.ckpt")
        torch.save(
            {"model": model.state_dict(), "head": head.state_dict(),
             "cfg": cfg, "score": score},
            last_path,
        )

    print("Done.")


if __name__ == "__main__":
    main()
