import os, cv2, torch
import torch.nn.functional as F
from models.vit import DeiTTiny
from models.heads import SegHead
from datasets.tiny_seg import make_loader

def make_grid(img, pred, gt):
    img = (img[0].permute(1,2,0).cpu().numpy()*255).astype("uint8")
    pr  = (pred[0,0].cpu().numpy()*255).astype("uint8")
    gt  = (gt[0,0].cpu().numpy()*255).astype("uint8")
    prC = cv2.applyColorMap(pr, cv2.COLORMAP_JET)
    gtC = cv2.applyColorMap(gt, cv2.COLORMAP_OCEAN)
    return cv2.hconcat([img, prC, gtC])

def main(ckpt_path, out="reports/vis.jpg"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["cfg"]; ds = cfg["dataset"]; mdl = cfg["model"]
    img_size = int(ds["img_size"]); ncls = int(ds["num_classes"])

    val = make_loader(ds["root"], img_size, 1, "val", False, ncls)
    model = DeiTTiny(img_size=img_size, patch=mdl["patch"], embed_dim=mdl["embed_dim"],
                     depth=mdl["depth"], heads=mdl["heads"], mlp_ratio=mdl["mlp_ratio"],
                     drop_path=mdl["stochastic_depth"]).to(device)
    head  = SegHead(in_ch=mdl["embed_dim"], out_ch=1 if ncls<=2 else ncls).to(device)
    model.load_state_dict(ck["model"], strict=False)
    head.load_state_dict(ck["head"], strict=False)
    model.eval(); head.eval()

    with torch.no_grad():
        for x,y in val:
            x,y = x.to(device), y.to(device)
            _, g, _ = model(x); logits = head(g)
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            m = logits.sigmoid()
            vis = make_grid(x, m, y)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            import cv2
            cv2.imwrite(out, vis); break

if __name__ == "__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="reports/vis.jpg")
    a=p.parse_args()
    main(a.ckpt, a.out)
