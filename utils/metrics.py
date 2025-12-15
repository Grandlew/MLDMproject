# utils/metrics.py
import torch


@torch.no_grad()
def dice_coeff(pred_probs: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    Binary Dice on probs in [0,1].
    pred_probs: [N,1,H,W] or [N,H,W] float
    target:     [N,1,H,W] or [N,H,W] float or long in {0,1}
    """
    if pred_probs.dim() == 4 and pred_probs.size(1) == 1:
        pred_bin = (pred_probs > thr).float()
        tgt = target.float()
        if tgt.dim() == 3:      # [N,H,W] -> [N,1,H,W]
            tgt = tgt.unsqueeze(1)
        elif tgt.dim() == 4 and tgt.size(1) == 1:
            pass
        else:
            raise ValueError(
                "dice_coeff expects binary masks of shape [N,1,H,W] or [N,H,W].")

        inter = (pred_bin * tgt).sum(dim=(1, 2, 3))
        denom = pred_bin.sum(dim=(1, 2, 3)) + tgt.sum(dim=(1, 2, 3))
        dice = (2.0 * inter + 1e-7) / (denom + 1e-7)
        return dice.mean()
    elif pred_probs.dim() == 3:  # [N,H,W]
        pred_bin = (pred_probs > thr).float()
        tgt = target.float()
        inter = (pred_bin * tgt).sum(dim=(1, 2))
        denom = pred_bin.sum(dim=(1, 2)) + tgt.sum(dim=(1, 2))
        dice = (2.0 * inter + 1e-7) / (denom + 1e-7)
        return dice.mean()
    else:
        raise ValueError("dice_coeff: unsupported shape.")


@torch.no_grad()
def miou(pred_classes: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> torch.Tensor:
    """
    Mean IoU for multi-class semantic segmentation.
    pred_classes: [N,H,W] long (class ids 0..C-1)
    target:       [N,H,W] long (class ids 0..C-1, possibly 255=ignore)
    """
    if pred_classes.dim() != 3 or target.dim() != 3:
        raise ValueError("miou expects [N,H,W] integer tensors")

    ious = []
    # Flatten per-sample then aggregate
    pred_flat = pred_classes.view(-1)
    tgt_flat = target.view(-1)

    # Mask out ignore
    valid = tgt_flat != ignore_index
    pred_flat = pred_flat[valid]
    tgt_flat = tgt_flat[valid]

    for c in range(num_classes):
        pred_c = pred_flat == c
        tgt_c = tgt_flat == c
        inter = (pred_c & tgt_c).sum().float()
        union = (pred_c | tgt_c).sum().float()
        if union > 0:
            ious.append((inter + 1e-7) / (union + 1e-7))

    if len(ious) == 0:
        return torch.tensor(0.0, device=pred_classes.device)
    return torch.stack(ious).mean()
