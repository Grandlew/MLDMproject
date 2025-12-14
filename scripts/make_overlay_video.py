import cv2, glob, os
ov = sorted(glob.glob(r"logs\urban_run\overlays\*.jpg"))
assert ov, "No overlays found in logs\\urban_run\\overlays"
first = cv2.imread(ov[0]); h,w = first.shape[:2]
os.makedirs("reports", exist_ok=True)
out = cv2.VideoWriter(r"reports\urban_overlay.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (w,h))
for p in ov: out.write(cv2.imread(p))
out.release(); print("wrote reports\\urban_overlay.mp4")
