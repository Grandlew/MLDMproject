# scripts/export_onnx.py
# ------------------------------------------------------------
# Robust ONNX exporter for DeiT-Tiny + segmentation head
# - Auto-builds model from ckpt["cfg"] if present (has safe defaults)
# - Resizes ViT positional embeddings in the checkpoint in-place
#   for the requested img_size (supports keys: 'pos', 'pos_embed', etc.)
# - Exports fixed-size or dynamic H/W models (opset 17)
# ------------------------------------------------------------

import argparse
import io
import sys
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

# Project imports (must exist in your repo)
from models.vit import DeiTTiny
from models.heads import SegHead


# --------------------------- utils ---------------------------

def _to_int(x, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _is_square(n: int) -> bool:
    r = int(n ** 0.5)
    return r * r == n


def resize_any_positional_embedding(sd: Dict[str, torch.Tensor],
                                    img_size: int,
                                    patch: int = 16) -> None:
    """
    Find a ViT positional embedding tensor in the state_dict (common keys: 'pos',
    'pos_embed', 'positional_embedding', etc.), then resize it to a grid that
    corresponds to img_size // patch. Works for [1, 1+Gh*Gw, C] layouts.
    """
    target_g = img_size // patch

    # First: common key names in several ViT implementations.
    candidates = [
        "pos_embed", "pos", "positional_embedding", "position_embedding", "pos_embed.weight"
    ]

    key: Optional[str] = next((k for k in candidates if k in sd), None)

    # Fallback: search for any [1, L, C] where L-1 is a perfect square (class token + grid)
    if key is None:
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[0] == 1:
                L = v.shape[1]
                if L > 1 and _is_square(L - 1):
                    key = k
                    break

    if key is None:
        print(
            "[export_onnx] WARN: No positional embedding tensor found; skipping resize.")
        return

    pe = sd[key]  # expected [1, 1+Gh*Gw, C]
    if not (isinstance(pe, torch.Tensor) and pe.ndim == 3 and pe.shape[0] == 1 and pe.shape[1] > 1):
        print(
            f"[export_onnx] WARN: Unexpected PE shape at '{key}': {tuple(pe.shape)}; skipping.")
        return

    L, C = pe.shape[1], pe.shape[2]
    gh_old_sq = L - 1
    if not _is_square(gh_old_sq):
        print(
            f"[export_onnx] WARN: L-1 is not a square for '{key}' (L={L}); skipping.")
        return

    gh_old = int(gh_old_sq ** 0.5)
    gh_new = int(target_g)

    if gh_new == gh_old:
        print(
            f"[export_onnx] INFO: PE already matches grid {gh_new}x{gh_new}; no resize.")
        return

    cls = pe[:, :1]         # [1,1,C]
    grid = pe[:, 1:]        # [1,Gh*Gh,C] -> [1,C,Gh,Gh]
    grid = grid.reshape(1, gh_old, gh_old, C).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(gh_new, gh_new),
                         mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, gh_new * gh_new, C)
    sd[key] = torch.cat([cls, grid], dim=1)
    print(
        f"[export_onnx] INFO: Resized PE '{key}' from {gh_old}x{gh_old} -> {gh_new}x{gh_new}.")


class SegWrapper(nn.Module):
    """
    Simple wrapper that:
      - runs the DeiT-Tiny backbone
      - applies the segmentation head
      - upsamples logits to input spatial size
    Output: logits [B, out_ch, H, W]
    """

    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, grid, _ = self.backbone(x)              # grid token map
        # [B, out_ch, gh, gw] (head decides shape)
        logits = self.head(grid)
        logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


# ----------------------- main exporter -----------------------

def build_models_from_cfg(cfg: Dict[str, Any], img_size: int, num_classes: int) -> (nn.Module, nn.Module):
    # Safe defaults (DeiT-Tiny canonical)
    patch = _to_int(cfg.get("model", {}).get("patch", 16), 16)
    embed_dim = _to_int(cfg.get("model", {}).get("embed_dim", 192), 192)
    depth = _to_int(cfg.get("model", {}).get("depth", 12), 12)
    heads = _to_int(cfg.get("model", {}).get("heads", 3), 3)
    mlp_ratio = float(cfg.get("model", {}).get("mlp_ratio", 4.0))
    drop_path = float(cfg.get("model", {}).get("stochastic_depth", 0.0))

    backbone = DeiTTiny(
        img_size=img_size,
        patch=patch,
        embed_dim=embed_dim,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
        drop_path=drop_path,
    )

    out_ch = 1 if num_classes <= 2 else int(num_classes)
    head = SegHead(in_ch=embed_dim, out_ch=out_ch)

    return backbone, head


def main():
    p = argparse.ArgumentParser("Export DeiT-Tiny + SegHead to ONNX")
    p.add_argument("--ckpt", required=True,
                   help="Path to training checkpoint (.ckpt/.pth)")
    p.add_argument("--out", required=True, help="Path to output .onnx")
    p.add_argument("--img_size", type=int, default=224,
                   help="Export size (H=W), e.g., 160 or 224")
    p.add_argument("--dynamic", action="store_true",
                   help="Make H/W dynamic in the exported graph")
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset (>=17 recommended)")
    args = p.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Try to obtain config and classes from checkpoint if available
    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    dataset_cfg = cfg.get("dataset", {}) if isinstance(cfg, dict) else {}
    num_classes = _to_int(dataset_cfg.get("num_classes", 2), 2)

    # Build fresh model instances for export
    backbone, head = build_models_from_cfg(
        cfg, img_size=int(args.img_size), num_classes=num_classes)

    # Gather state_dicts to load
    sd_model: Optional[Dict[str, torch.Tensor]] = None
    sd_head: Optional[Dict[str, torch.Tensor]] = None
    if isinstance(ckpt, dict):
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            sd_model = ckpt["model"]
        if "head" in ckpt and isinstance(ckpt["head"], dict):
            sd_head = ckpt["head"]
        # Some checkpoints store everything under a single dict (e.g., pure state_dict)
        if sd_model is None and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            # Try common prefixes
            sd_model = {k.replace("backbone.", "").replace("model.", ""): v
                        for k, v in ckpt["state_dict"].items() if k.startswith(("backbone.", "model.", ""))}
    else:
        # Entire file is a raw state_dict
        sd_model = ckpt

    if sd_model is None:
        raise RuntimeError(
            "Could not locate model weights in checkpoint (expected keys like 'model' or 'state_dict').")

    # Resize positional embeddings (in-place on sd_model) for requested img_size
    patch_for_pe = _to_int(cfg.get("model", {}).get("patch", 16), 16)
    resize_any_positional_embedding(
        sd_model, img_size=int(args.img_size), patch=patch_for_pe)

    # Load state dicts (strict=False is intentional to avoid harmless mismatches)
    missing, unexpected = backbone.load_state_dict(sd_model, strict=False)
    if missing or unexpected:
        print(
            f"[export_onnx] INFO: backbone missing={len(missing)}, unexpected={len(unexpected)}")

    if sd_head is not None:
        hm, hu = head.load_state_dict(sd_head, strict=False)
        if hm or hu:
            print(
                f"[export_onnx] INFO: head missing={len(hm)}, unexpected={len(hu)}")

    # Wrap
    wrapper = SegWrapper(backbone, head).to(device).eval()

    # Dummy input (channels-last/first as PyTorch expects)
    H = W = int(args.img_size)
    dummy = torch.randn(1, 3, H, W, device=device)

    # Dynamic axes (optional)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "images": {2: "height", 3: "width"},
            "logits": {2: "height", 3: "width"},
        }

    print(
        f"[export_onnx] Exporting to: {args.out} (img_size={args.img_size}, dynamic={args.dynamic}, opset={args.opset})")

    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )

    # Quick integrity check: load with onnx to verify the file is well-formed (optional)
    try:
        import onnx  # type: ignore
        onnx.load(args.out)
        print("[export_onnx] ONNX save verified OK.")
    except Exception as e:
        print(f"[export_onnx] WARN: onnx validation skipped/failed: {e}")

    print("[export_onnx] DONE.")


if __name__ == "__main__":
    # Windows-friendly entry (no multiprocessing needed)
    main()
