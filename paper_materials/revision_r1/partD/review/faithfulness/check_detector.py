"""(1) .venv: deployed phase2 Detector.detect() (all classes, conf 0.35) person boxes == cache > 0.35 ?
   (2) .venv-gpu: hooked person-only predict at conf 0.10 == cache (boxes + 128-D feats) ?
Usage: python check_detector.py deployed|hooked
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["YOLO_AUTOINSTALL"] = "false"
import cv2
import numpy as np
import torch

torch.set_num_threads(2)
cv2.setNumThreads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "detections")
W = {"grouped": (os.path.join(ROOT, "paper_materials", "retrain", "local_out", "best_grouped.pt"), 3),
     "coco": (os.path.join(ROOT, "yolov8s.pt"), 0)}
PICK = {"MOT17-02-FRCNN": [1, 300, 600], "MOT17-05-FRCNN": [1, 2, 400, 837], "MOT17-13-FRCNN": [1, 375, 750],
        "MOT17-04-FRCNN": [500], "MOT17-10-FRCNN": [100], "MOT17-11-FRCNN": [700], "MOT17-09-FRCNN": [200]}


def cached(det, seq):
    z = np.load(os.path.join(CACHE, f"{det}__{seq}.npz"))
    fr, off, d, ft = z["frames"], z["off"], z["dets"], z["feats"]
    return {int(f): (d[off[i]:off[i + 1]], ft[off[i]:off[i + 1]]) for i, f in enumerate(fr)}


mode = sys.argv[1]
import ultralytics
print("ultralytics", ultralytics.__version__, "torch", torch.__version__)
for det, (wp, pcls) in W.items():
    if mode == "deployed":
        sys.path.insert(0, os.path.join(ROOT, "phase2"))
        from src.detector import Detector
        D = Detector(wp, "cpu", 0.35, 640)
        D.warmup()
    else:
        from ultralytics import YOLO
        model = YOLO(wp)
        kw = dict(imgsz=640, conf=0.10, classes=[pcls], device="cpu", verbose=False, half=False)
        model.predict(np.zeros((640, 640, 3), np.uint8), **kw)
        p = model.predictor
        head = p.model.model.model[-1]

        def _pre_hook(module, inp):
            p._feats = list(inp[0])

        p._hook = head.register_forward_pre_hook(_pre_hook)
    worst_box = worst_conf = worst_feat = 0.0
    n_bad_count = n_frames = 0
    for seq, frames in PICK.items():
        C = cached(det, seq)
        for f in frames:
            path = os.path.join(MOT, seq, "img1", f"{f:06d}.jpg")
            cd, cf = C[f]
            if mode == "deployed":
                d = D.detect(cv2.imread(path))
                m = d.class_id == pcls
                got = np.concatenate([d.xyxy[m], d.confidence[m][:, None]], 1)
                other = int((~m).sum())
                want = cd[cd[:, 4] > 0.35][:, :5]
                gf = None
            else:
                r = model.predict(path, **kw)[0]
                got = r.boxes.data.float().cpu().numpy()[:, :5]
                gf = r.feats.float().cpu().numpy() if len(got) else np.zeros((0, 128), np.float32)
                want = cd[:, :5]
                other = 0
            n_frames += 1
            if len(got) != len(want):
                n_bad_count += 1
                print(f"  COUNT MISMATCH {det} {seq} f{f}: got {len(got)} want {len(want)}")
                continue
            go, wo = np.argsort(-got[:, 4], kind="stable"), np.argsort(-want[:, 4], kind="stable")
            g, w = got[go], want[wo]
            if len(g):
                worst_box = max(worst_box, float(np.abs(g[:, :4] - w[:, :4]).max()))
                worst_conf = max(worst_conf, float(np.abs(g[:, 4] - w[:, 4]).max()))
                if gf is not None:
                    worst_feat = max(worst_feat, float(np.abs(gf[go] - cf[wo]).max()))
            print(f"  {det} {seq} f{f}: {len(got)} person boxes (non-person boxes from deployed: {other})")
    print(f"[{mode}] {det}: frames {n_frames}, count mismatches {n_bad_count}, max |dbox| {worst_box:.5f} px, "
          f"max |dconf| {worst_conf:.6f}, max |dfeat| {worst_feat:.6f}")
