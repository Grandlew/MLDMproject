# scripts/make_toy_sample.py
import os, numpy as np, cv2

def gen_split(root, split, n, H=160, W=160):
    os.makedirs(f"{root}/images_{split}", exist_ok=True)
    os.makedirs(f"{root}/masks_{split}",  exist_ok=True)
    rng = np.random.default_rng(0 if split=="train" else 1)
    for i in range(n):
        # noisy RGB image
        img = (rng.random((H,W,3))*255).astype(np.uint8)
        # one bright circular region = traversable (mask=255)
        mask = np.zeros((H,W), np.uint8)
        cx, cy = int(rng.integers(30,130)), int(rng.integers(30,130))
        r = int(rng.integers(20,50))
        cv2.circle(mask, (cx,cy), r, 255, -1)
        # make that region a bit brighter to give a learnable signal
        img[mask==255] = np.clip(img[mask==255] + 60, 0, 255)

        name = f"{i:04d}.png"
        cv2.imwrite(f"{root}/images_{split}/{name}", img)
        cv2.imwrite(f"{root}/masks_{split}/{name}",  mask)

def main():
    root = "data/sample"
    gen_split(root, "train", 200)
    gen_split(root, "val",   50)
    print("Toy sample ready at", root)

if __name__ == "__main__":
    main()
