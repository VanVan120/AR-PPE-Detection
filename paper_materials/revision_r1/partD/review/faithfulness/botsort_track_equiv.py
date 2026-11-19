"""Does the Part D BoT-SORT replay equal what model.track(tracker=botsort with_reid True) does?

For a prefix of a sequence:
  1. run the real model.track (ultralytics 8.4.75, persist=True, conf 0.10, person class, imgsz 640), capturing
     exactly what it hands the BOTSORT tracker (dets, feats, img) and what the tracker returns;
  2. replay the captured inputs through the Part D replay recipe (BOTSORT(args=cfg), Boxes(dets), img,
     NUMPY feats) and require identical outputs;
  3. compare the captured dets/feats with the Part D detection cache;
  4. compare model.track's tracks with the stored Part D tracker file for those frames.
  python botsort_track_equiv.py <det> <seq> <n_frames>
Writes only into this scratch folder.
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2
import numpy as np
import torch

torch.set_num_threads(1)
cv2.setNumThreads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD")
TR = os.path.join(CACHE, "trackers", "mot_challenge", "MOT17-train")
W = {"grouped": (os.path.join(ROOT, "paper_materials", "retrain", "local_out", "best_grouped.pt"), 3),
     "coco": (os.path.join(ROOT, "yolov8s.pt"), 0)}

import ultralytics  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.bot_sort import BOTSORT  # noqa: E402
from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402
from ultralytics.utils import YAML, IterableSimpleNamespace  # noqa: E402
from ultralytics.utils.checks import check_yaml  # noqa: E402

det, seq, N = sys.argv[1], sys.argv[2], int(sys.argv[3])
wpath, pcls = W[det]
print("ultralytics", ultralytics.__version__, "torch", torch.__version__)

src_yaml = check_yaml("botsort.yaml")
text = open(src_yaml, encoding="utf-8").read()
assert text.count("with_reid: False") == 1
yaml_path = os.path.join(HERE, "botsort_withreid.yaml")
with open(yaml_path, "w", encoding="utf-8") as fh:
    fh.write(text.replace("with_reid: False", "with_reid: True"))

captured = []
_orig = BYTETracker.update


def _wrapped(self, results, img=None, feats=None, **kw):
    rec = {"dets": np.asarray(results.data).copy(), "img": None if img is None else img.copy(),
           "feats_type": type(feats).__name__, "kw": sorted(kw)}
    if feats is None:
        rec["feats"] = None
    elif hasattr(feats, "detach"):
        rec["feats"] = feats.detach().float().cpu().numpy().copy()
    elif isinstance(feats, list) and len(feats) == 0:
        rec["feats"] = np.zeros((0, 128), np.float32)
    else:
        rec["feats"] = np.asarray(feats, np.float32).copy()
    out = _orig(self, results, img, feats, **kw)
    rec["out"] = np.asarray(out).copy()
    rec["max_frames_lost"] = self.max_frames_lost
    rec["encoder"] = getattr(self.encoder, "__name__", type(self.encoder).__name__)
    captured.append(rec)
    return out


BOTSORT.update = _wrapped
paths = [os.path.join(MOT, seq, "img1", f"{f:06d}.jpg") for f in range(1, N + 1)]
model = YOLO(wpath)
for _ in model.track(source=paths, tracker=yaml_path, persist=True, conf=0.10, classes=[pcls], imgsz=640,
                     device="cpu", half=False, verbose=False, stream=True, project=os.path.join(HERE, "ul_runs")):
    pass
BOTSORT.update = BYTETracker.update
assert len(captured) == N, len(captured)
print(f"model.track called the tracker {len(captured)} times; feats types {sorted({c['feats_type'] for c in captured})}; "
      f"extra kwargs {sorted({tuple(c['kw']) for c in captured})}; max_frames_lost {captured[0]['max_frames_lost']}; "
      f"encoder {captured[0]['encoder']}")

# 2. replay recipe of tools/revision/mot17_track.py (BotSortReID) on the captured inputs
cfg = IterableSimpleNamespace(**YAML.load(check_yaml("botsort.yaml")))
cfg.with_reid = True
trk = BOTSORT(args=cfg)
n_diff = 0
for i, c in enumerate(captured):
    out = trk.update(Boxes(c["dets"], c["img"].shape[:2]), c["img"], c["feats"])
    if out.shape != c["out"].shape or not np.array_equal(out, c["out"]):
        n_diff += 1
        if n_diff <= 3:
            print(f"  replay != model.track at frame {i + 1}: {out[:3]} vs {c['out'][:3]}")
print(f"[replay vs model.track, same inputs] frames with any difference: {n_diff} / {N}")

# 3. captured inputs vs the Part D cache
z = np.load(os.path.join(CACHE, "detections", f"{det}__{seq}.npz"))
fr, off, cd, cf = z["frames"], z["off"], z["dets"], z["feats"]
cnt_bad, mb, mf = 0, 0.0, 0.0
for i, c in enumerate(captured):
    a, b = cd[off[i]:off[i + 1]], cf[off[i]:off[i + 1]]
    if len(a) != len(c["dets"]):
        cnt_bad += 1
        continue
    if len(a):
        mb = max(mb, float(np.abs(a - c["dets"]).max()))
        mf = max(mf, float(np.abs(b - c["feats"]).max()))
print(f"[model.track inputs vs cache] count mismatches {cnt_bad}, max |d det| {mb:.6f}, max |d feat| {mf:.6f}")

# 4. model.track tracks vs stored Part D file (first N frames)
mine = []
for i, c in enumerate(captured):
    for r in c["out"]:
        x1, y1, x2, y2 = [float(v) for v in r[:4]]
        mine.append(f"{i + 1},{int(r[4])},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
stored = [l.rstrip("\n") for l in open(os.path.join(TR, f"{det}__botsort_reid", "data", f"{seq}.txt"), encoding="utf-8")
          if int(l.split(",")[0]) <= N]
nd = sum(1 for a, b in zip(mine, stored) if a != b) + abs(len(mine) - len(stored))
ids_m = [(l.split(",")[0], l.split(",")[1]) for l in mine]
ids_s = [(l.split(",")[0], l.split(",")[1]) for l in stored]
print(f"[model.track vs stored file, frames 1..{N}] lines mine {len(mine)} stored {len(stored)}, differing lines {nd}, "
      f"(frame,id) sequences equal {ids_m == ids_s}")
