"""Is best_grouped.pt's low MOT17 person recall a scale effect? Same sampled frames at imgsz 640
(as cached/deployed) and 1280, conf > 0.35, recall on visible (>=0.5) pedestrians by height."""
import os
os.environ["YOLO_AUTOINSTALL"] = "false"; os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np, cv2, torch
torch.set_num_threads(2)
from scipy.optimize import linear_sum_assignment
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
from ultralytics import YOLO
W = os.path.join(ROOT, "paper_materials", "retrain", "local_out", "best_grouped.pt")
model = YOLO(W)
hb = [0, 100, 150, 200, 300, 5000]
stats = {sz: {k: [0, 0] for k in range(len(hb) - 1)} for sz in (640, 1280)}
for s in SEQS:
    gt = np.loadtxt(os.path.join(MOT, s, "gt", "gt.txt"), delimiter=",", ndmin=2)
    n = int(gt[:, 0].max())
    for f in range(1, n + 1, 50):
        img = cv2.imread(os.path.join(MOT, s, "img1", f"{f:06d}.jpg"))
        g = gt[(gt[:, 0] == f) & (gt[:, 7] == 1) & (gt[:, 6] == 1) & (gt[:, 8] >= 0.5)]
        if len(g) == 0: continue
        gb = np.c_[g[:, 2], g[:, 3], g[:, 2] + g[:, 4], g[:, 3] + g[:, 5]]
        for sz in (640, 1280):
            r = model.predict(img, imgsz=sz, conf=0.35, classes=[3], device="cpu", verbose=False)[0]
            d = r.boxes.xyxy.cpu().numpy()
            m = np.zeros(len(g), bool)
            if len(d):
                M = iou(gb, d); M[M < 0.5] = 0
                a, b = linear_sum_assignment(-M); ok = M[a, b] > 0; m[a[ok]] = True
            for k in range(len(g)):
                hi = min(np.searchsorted(hb, g[k, 5], side="right") - 1, len(hb) - 2)
                stats[sz][hi][0] += m[k]; stats[sz][hi][1] += 1
    print(s, "done", flush=True)
for sz in (640, 1280):
    tot = [sum(v[0] for v in stats[sz].values()), sum(v[1] for v in stats[sz].values())]
    print(f"imgsz {sz}: overall visible recall {100*tot[0]/tot[1]:.1f}% (n={tot[1]}); by height: " +
          ", ".join(f"[{hb[k]},{hb[k+1]}) {100*v[0]/max(v[1],1):.1f}% (n={v[1]})" for k, v in stats[sz].items()))
