import argparse
import torch
import torch.nn.functional as F

from models.vit import DeiTTiny
from models.heads import SegHead
from datasets.tiny_seg import make_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--amp", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("cfg", {})

    # --- Safe defaults if anything is missing in cfg ---
    ds_cfg = cfg.get("dataset", {})
    mdl_cfg = cfg.get("model", {})
    trn_cfg = cfg.get("train", {})

    img_size    = int(ds_cfg.get("img_size", 160))
    num_classes = int(ds_cfg.get("num_classes", 2))
    root        = ds_cfg.get("root", "data/sample")
    batch_size  = int(trn_cfg.get("batch_size", 64))

    model = DeiTTiny(
        img_size=img_size,
        patch=mdl_cfg.get("patch", 16),
        embed_dim=mdl_cfg.get("embed_dim", 192),
        depth=mdl_cfg.get("depth", 12),
        heads=mdl_cfg.get("heads", 3),
        mlp_ratio=mdl_cfg.get("mlp_ratio", 4.0),
        drop_path=mdl_cfg.get("stochastic_depth", 0.1),
    ).to(device)

    head = SegHead(
        in_ch=mdl_cfg.get("embed_dim", 192),
        out_ch=1 if num_classes <= 2 else num_classes,
    ).to(device)

    # Load weights (lenient)
    model.load_state_dict(ckpt["model"], strict=False)
    head.load_state_dict(ckpt["head"], strict=False)
    model.eval()
    head.eval()

    # DataLoader (num_workers=0 for Windows)
    val_loader = make_loader(root, img_size, batch_size, "val", False, num_classes)

    # Single batch sanity pass
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            _, grid, _ = model(x)
            logits = head(grid)
            # match target spatial size
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            mask = logits.sigmoid()

            print("Eval OK. Batch mask mean:", float(mask.mean()))
            break


if __name__ == "__main__":
    main()
