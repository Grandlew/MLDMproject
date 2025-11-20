import torch.nn as nn
class SegHead(nn.Module):
    def __init__(self,in_ch=192,mid=128,out_ch=1):
        super().__init__()
        self.dec=nn.Sequential(
            nn.Conv2d(in_ch,mid,3,padding=1),nn.GELU(),
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(mid,64,3,padding=1),nn.GELU(),
            nn.Upsample(scale_factor=2,mode="bilinear",align_corners=False),
            nn.Conv2d(64,out_ch,1))
    def forward(self,feat): return self.dec(feat)
