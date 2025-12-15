import cv2, glob, os
imgs = sorted(glob.glob(r"data\sample\images_val\*.png")) or \
       sorted(glob.glob(r"data\sample\images_val\*.jpg"))
assert imgs, "No images found in data\\sample\\images_val — generate or copy a few first."

first = cv2.imread(imgs[0]); h,w = first.shape[:2]
os.makedirs("samples", exist_ok=True)
out = cv2.VideoWriter(r"samples\urban.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (w,h))
for p in imgs:
    im = cv2.imread(p)
    if im is None: continue
    out.write(cv2.resize(im, (w,h)))
out.release()
print("wrote samples\\urban.mp4")
