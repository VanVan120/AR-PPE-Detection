import os, time
os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2, numpy as np
cv2.setNumThreads(2)
from ultralytics.trackers.utils.gmc import GMC
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
for seq in ("MOT17-04-FRCNN", "MOT17-05-FRCNN"):
    g = GMC(method="sparseOptFlow", downscale=2)
    imgs = [cv2.imread(os.path.join(ROOT, "data", "mot17", "MOT17", "train", seq, "img1", f"{f:06d}.jpg")) for f in range(1, 121)]
    t = []
    for im in imgs:
        t0 = time.perf_counter(); g.apply(im); t.append(time.perf_counter() - t0)
    print(seq, imgs[0].shape, "GMC sparseOptFlow ms/frame: mean %.1f median %.1f" % (1000 * np.mean(t[1:]), 1000 * np.median(t[1:])))
