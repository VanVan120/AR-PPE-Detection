"""Independent prefix re-run of boxmot 26.0.0 StrongSort() / DeepOcSort() (defaults) on the cached detections,
compared with the stored Part D files. cwd = a scratch folder holding a COPY of the default ReID weights.
  python boxmot_prefix.py <det> <seq> <n_frames>"""
import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import cv2
import numpy as np
import torch

torch.set_num_threads(2)
cv2.setNumThreads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD")
TR = os.path.join(CACHE, "trackers", "mot_challenge", "MOT17-train")
cwd = os.path.join(HERE, "boxmot_cwd")
os.makedirs(os.path.join(cwd, "models"), exist_ok=True)
w = os.path.join(cwd, "models", "osnet_x0_25_msmt17.pt")
if not os.path.isfile(w):
    shutil.copyfile(os.path.join(CACHE, "boxmot_weights", "models", "osnet_x0_25_msmt17.pt"), w)
os.chdir(cwd)
import boxmot  # noqa: E402

det, seq, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
print("boxmot", boxmot.__version__, "torch", torch.__version__)
z = np.load(os.path.join(CACHE, "detections", f"{det}__{seq}.npz"))
D = {int(f): z["dets"][z["off"][i]:z["off"][i + 1]] for i, f in enumerate(z["frames"])}
for name, cls in (("strongsort", boxmot.StrongSort), ("deepocsort", boxmot.DeepOcSort)):
    t = cls()
    mine = []
    for f in range(1, N + 1):
        img = cv2.imread(os.path.join(MOT, seq, "img1", f"{f:06d}.jpg"))
        out = np.asarray(t.update(D[f].astype(np.float32), img))
        for r in out:
            x1, y1, x2, y2 = [float(v) for v in r[:4]]
            mine.append(f"{f},{int(r[4]) + 1},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
    stored = [l.rstrip("\n") for l in open(os.path.join(TR, f"{det}__{name}", "data", f"{seq}.txt"), encoding="utf-8")
              if int(l.split(",")[0]) <= N]
    nd = sum(1 for a, b in zip(mine, stored) if a != b) + abs(len(mine) - len(stored))
    ids_eq = [l.split(",")[:2] for l in mine] == [l.split(",")[:2] for l in stored]
    maxd = 0.0
    if len(mine) == len(stored):
        for a, b in zip(mine, stored):
            maxd = max(maxd, max(abs(float(x) - float(y)) for x, y in zip(a.split(",")[2:6], b.split(",")[2:6])))
    print(f"  {name}: config {t.config.__class__.__name__} defaults; frames 1..{N}: lines mine {len(mine)} stored {len(stored)}, "
          f"differing lines {nd}, (frame,id) equal {ids_eq}, max |d coord| {maxd:.2f}")
