import torch, torch.nn as nn, torch.nn.functional as F, argparse
from models.vit import DeiTTiny
from models.heads import SegHead

p=argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--out", default="runs/deit_tiny.onnx")
args=p.parse_args()

ck=torch.load(args.ckpt, map_location="cpu"); cfg=ck["cfg"]
ds, mdl = cfg["dataset"], cfg["model"]
img_size = int(ds["img_size"]); ncls=int(ds["num_classes"])

vit = DeiTTiny(img_size=img_size, patch=mdl["patch"], embed_dim=mdl["embed_dim"],
               depth=mdl["depth"], heads=mdl["heads"], mlp_ratio=mdl["mlp_ratio"],
               drop_path=mdl["stochastic_depth"])
head=SegHead(in_ch=mdl["embed_dim"], out_ch=1 if ncls<=2 else ncls)
vit.load_state_dict(ck["model"], strict=False); head.load_state_dict(ck["head"], strict=False)
vit.eval(); head.eval()

class Wrapped(nn.Module):
    def __init__(self, vit, head, out_hw):
        super().__init__(); self.vit=vit; self.head=head; self.out_hw=out_hw
    def forward(self, x):
        _, g, _ = self.vit(x); logits = self.head(g)
        return F.interpolate(logits, size=self.out_hw, mode="bilinear", align_corners=False)

wrapped = Wrapped(vit, head, (img_size, img_size))
dummy = torch.randn(1,3,img_size,img_size)
torch.onnx.export(wrapped, dummy, args.out, input_names=["images"], output_names=["logits"],
                  opset_version=17, dynamic_axes={"images":{0:"batch"}, "logits":{0:"batch"}})
print("Saved ONNX to", args.out)
