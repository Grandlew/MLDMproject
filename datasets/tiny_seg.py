from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import cv2, numpy as np, torch

class TinySeg(Dataset):
    def __init__(self, root, split="train", img_size=160, num_classes=2):
        self.root = Path(root)
        self.imgs = sorted((self.root/f"images_{split}").glob("*.*"))
        self.masks = [(self.root/f"masks_{split}"/p.name) for p in self.imgs]
        self.size = img_size
        self.num_classes = num_classes
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        x = cv2.cvtColor(cv2.imread(str(self.imgs[i])), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(self.masks[i]), cv2.IMREAD_GRAYSCALE)
        x = cv2.resize(x, (self.size, self.size), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = (x/255.0).astype(np.float32)
        x = torch.from_numpy(x).permute(2,0,1)
        mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
        std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        x = (x - mean)/std
        if self.num_classes <= 2:
            m = torch.from_numpy((m>127).astype(np.float32))[None]
        else:
            m = torch.from_numpy(m.astype(np.int64))
        return x, m

def make_loader(root, img_size, bs, split, shuffle=True, num_classes=2):
    ds = TinySeg(root, split, img_size, num_classes)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)

