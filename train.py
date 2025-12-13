import os, yaml, torch, csv
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
import torch.nn.functional as F
from datasets.tiny_seg import make_loader
from models.vit import DeiTTiny
from models.heads import SegHead
from utils.metrics import dice_coeff, miou
from utils.augment import apply_augment
import argparse

def build_model(cfg, num_classes, device):
    backbone = cfg["model"].get("backbone", "deit_tiny")
    if backbone == "deit_tiny":
        m = DeiTTiny(
            img_size=int(cfg["dataset"]["img_size"]),
            patch=cfg["model"]["patch"],
            embed_dim=cfg["model"]["embed_dim"],
            depth=cfg["model"]["depth"],
            heads=cfg["model"]["heads"],
            mlp_ratio=cfg["model"]["mlp_ratio"],
            drop_path=cfg["model"]["stochastic_depth"],
        ).to(device)
        h = SegHead(in_ch=cfg["model"]["embed_dim"], out_ch=1 if num_classes<=2 else num_classes).to(device)
        def forward_logits(x):
            _, grid, _ = m(x)
            return h(grid)
        params = list(m.parameters()) + list(h.parameters())
    elif backbone == "tinycnn":
        from models.baseline_cnn import TinySegCNN
        m = TinySegCNN(out_ch=1 if num_classes<=2 else num_classes).to(device)
        h = None
        def forward_logits(x):
            return m(x)
        params = list(m.parameters())
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return m, h, forward_logits, params

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--save_dir", default="runs")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config))

    img_size = int(cfg["dataset"]["img_size"])
    num_classes = int(cfg["dataset"].get("num_classes", 2))
    epochs = int(args.epochs if args.epochs is not None else cfg["train"]["epochs"])
    bs = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"]); wd = float(cfg["train"]["weight_decay"])

    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = make_loader(cfg["dataset"]["root"], img_size, bs, "train", True,  num_classes)
    val_loader   = make_loader(cfg["dataset"]["root"], img_size, bs, "val",   False, num_classes)

    model, head, forward_logits, params = build_model(cfg, num_classes, device)

    opt = optim.AdamW(params, lr=lr, weight_decay=wd)
    scaler = GradScaler(enabled=args.amp)
    bce = nn.BCEWithLogitsLoss(); ce = nn.CrossEntropyLoss(ignore_index=255)

    best = 0.0
    metrics_csv = os.path.join(out_dir, "metrics.csv")
    if not os.path.exists(metrics_csv):
        with open(metrics_csv, "w", newline="") as f: csv.writer(f).writerow(["epoch","dice","miou"])

    for ep in range(epochs):
        model.train(); 
        if head: head.train()
        for it, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            x, y = apply_augment(x, y, cfg.get("train", {}).get("augment", {}))
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=args.amp and torch.cuda.is_available()):
                logits = forward_logits(x)
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                loss = bce(logits, y) if num_classes<=2 else ce(logits, y.squeeze(1).long())
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if it % 10 == 0:
                print(f"  [ep {ep+1}] iter {it}/{len(train_loader)} loss={float(loss):.4f}")

        model.eval(); 
        if head: head.eval()
        dices, mious = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with autocast(enabled=args.amp and torch.cuda.is_available()):
                    logits = forward_logits(x)
                    logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
                    m = logits.sigmoid()
                dices.append(dice_coeff(m, y)); mious.append(miou(m, y))
        d = sum(dices)/len(dices); m = sum(mious)/len(mious)
        print(f"Epoch {ep+1}/{epochs} - val Dice={d:.4f} mIoU={m:.4f}")

        with open(metrics_csv, "a", newline="") as f: csv.writer(f).writerow([ep+1, float(d), float(m)])

        if m >= best:
            best = m
            torch.save({"model": model.state_dict(), "head": None if head is None else head.state_dict(), "cfg": cfg},
                       os.path.join(out_dir, "deit_tiny.ckpt"))
            print("Saved best:", os.path.join(out_dir, "deit_tiny.ckpt"))

    torch.save({"model": model.state_dict(), "head": None if head is None else head.state_dict(), "cfg": cfg},
               os.path.join(out_dir, "deit_tiny_last.ckpt"))
    print("Saved last:", os.path.join(out_dir, "deit_tiny_last.ckpt"))

if __name__ == "__main__":
    main()
