import os, yaml, torch
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from datasets.tiny_seg import make_loader
from models.vit import DeiTTiny
from models.heads import SegHead
from utils.metrics import dice_coeff, miou
from utils.augment import apply_augment
import torch.nn.functional as F
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_dir", default="runs")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))

    # --- coerce types to avoid "float" vs "str" errors ---
    img_size = int(cfg["dataset"]["img_size"])
    num_classes = int(cfg["dataset"].get("num_classes", 2))
    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])
    bs = int(cfg["train"]["batch_size"])

    lr_raw = cfg["train"]["lr"]
    wd_raw = cfg["train"]["weight_decay"]
    lr = float(lr_raw) if not isinstance(lr_raw, (float, int)) else lr_raw
    wd = float(wd_raw) if not isinstance(wd_raw, (float, int)) else wd_raw

    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(cfg["dataset"]["root"], img_size, bs, "train", True, num_classes)
    val_loader   = make_loader(cfg["dataset"]["root"], img_size, bs, "val",   False, num_classes)

    model = DeiTTiny(
        img_size=img_size,
        patch=cfg["model"]["patch"],
        embed_dim=cfg["model"]["embed_dim"],
        depth=cfg["model"]["depth"],
        heads=cfg["model"]["heads"],
        mlp_ratio=cfg["model"]["mlp_ratio"],
        drop_path=cfg["model"]["stochastic_depth"],
    ).to(device)
    head = SegHead(in_ch=cfg["model"]["embed_dim"], out_ch=1 if num_classes <= 2 else num_classes).to(device)

    opt = optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=args.amp)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss(ignore_index=255)

    best = 0.0
    for ep in range(epochs):
        # ---- train ----
        model.train()
        head.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            x, y = apply_augment(x, y, cfg.get("train", {}).get("augment", {}))
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp):
                _, grid, _ = model(x)
                logits = head(grid)
                # Ensure logits match target spatial size
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = bce(logits, y) if num_classes <= 2 else ce(logits, y.squeeze(1).long())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # ---- validate ----
        model.eval()
        head.eval()
        dices, mious = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                with autocast(enabled=args.amp):
                    _, grid, _ = model(x)
                    m = head(grid)
                    m = F.interpolate(m, size=y.shape[-2:], mode="bilinear", align_corners=False).sigmoid()
                dices.append(dice_coeff(m, y))
                mious.append(miou(m, y))

        d = sum(dices) / len(dices)
        m = sum(mious) / len(mious)
        print(f"Epoch {ep+1}/{epochs} - val Dice={d:.4f} mIoU={m:.4f}")

        # save if best so far
        if m >= best:
            best = m
            torch.save(
                {"model": model.state_dict(), "head": head.state_dict(), "cfg": cfg},
                os.path.join(out_dir, "deit_tiny.ckpt"),
            )
            print("Saved best:", os.path.join(out_dir, "deit_tiny.ckpt"))

    # ---- after the training loop, always save the final state too ----
    torch.save(
        {"model": model.state_dict(), "head": head.state_dict(), "cfg": cfg},
        os.path.join(out_dir, "deit_tiny_last.ckpt"),
    )
    print("Saved last:", os.path.join(out_dir, "deit_tiny_last.ckpt"))


if __name__ == "__main__":
    main()
