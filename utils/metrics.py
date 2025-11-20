def dice_coeff(pred, target, eps=1e-6):
    pred_bin=(pred>0.5).float()
    inter=(pred_bin*target).sum(dim=(1,2,3))
    union=pred_bin.sum(dim=(1,2,3))+target.sum(dim=(1,2,3))
    return float(((2*inter+eps)/(union+eps)).mean().item())

def miou(pred, target, eps=1e-6):
    pred_bin=(pred>0.5).float()
    inter=(pred_bin*target).sum(dim=(1,2,3))
    union=(pred_bin+target - pred_bin*target).sum(dim=(1,2,3))
    return float(((inter+eps)/(union+eps)).mean().item())
