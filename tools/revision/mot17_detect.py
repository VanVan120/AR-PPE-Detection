"""Revision R1, Part D: cache person detections (and native ReID features) on MOT17, once.

One detection pass per (detector, sequence), at imgsz 640, conf 0.10, every frame, person class
only. Every tracker in Part D replays these cached boxes, so they all see the same detections.

  grouped : best_grouped.pt (our retrained detector), class 3 'Person'
  coco    : yolov8s.pt, the COCO-pretrained weights the training started from, class 0 'person'

All other predict settings are ultralytics defaults (NMS IoU 0.7, single-label NMS, max_det 300),
which is also how phase2's detector calls the model. Restricting to the person class inside NMS
is equivalent to keeping the person boxes of an all-class run (the class filter is applied after
the best-class choice and before class-wise NMS).

The same pass stores, for every box, the 128-D "native" appearance feature that ultralytics'
BoT-SORT uses with `with_reid: True, model: auto`: the input of the Detect layer captured by the
same forward pre-hook ultralytics registers in trackers/track.py, reduced per box by
DetectionPredictor.get_obj_feats. BoT-SORT can then be replayed on the cache with exactly the
features model.track would have given it.

    .venv-gpu\\Scripts\\python.exe tools\\revision\\mot17_detect.py --threads 4
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import datetime
import hashlib
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MOT = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
CACHE = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "detections")
OUT = os.path.join(ROOT, "paper_materials", "revision_r1", "partD")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
DETECTORS = {
    "grouped": (os.path.join(ROOT, "paper_materials", "retrain", "local_out", "best_grouped.pt"),
                "679eae32764839df179a5ff8b2a9a2e924e013df860fef0af54daed2eb8c8c94", 3),
    "coco": (os.path.join(ROOT, "yolov8s.pt"),
             "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a", 0),
}
IMGSZ, CONF = 640, 0.10


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def run(det_name, seq, device, threads):
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    wpath, wsha, person = DETECTORS[det_name]
    got = sha256(wpath)
    if got != wsha:
        raise SystemExit(f"[FAIL] {wpath}: sha256 {got} != {wsha}")
    model = YOLO(wpath)
    kw = dict(imgsz=IMGSZ, conf=CONF, classes=[person], device=device, verbose=False, half=False)
    model.predict(np.zeros((640, 640, 3), np.uint8), **kw)          # builds model.predictor
    p = model.predictor
    head = p.model.model.model[-1]
    if type(head).__name__ != "Detect" or getattr(head, "end2end", False):
        raise SystemExit("[FAIL] unexpected head; the native-feature path would not be used")

    def _pre_hook(module, inp):          # identical to ultralytics/trackers/track.py:70-71
        p._feats = list(inp[0])

    p._hook = head.register_forward_pre_hook(_pre_hook)
    img_dir = os.path.join(MOT, seq, "img1")
    files = sorted(f for f in os.listdir(img_dir) if f.endswith(".jpg"))
    frames, offs, dets_all, feats_all, speed = [], [0], [], [], []
    t0 = time.time()
    for f in files:
        r = model.predict(os.path.join(img_dir, f), **kw)[0]
        d = r.boxes.data.float().cpu().numpy().astype(np.float32)          # x1 y1 x2 y2 conf cls
        if len(d):
            ft = r.feats.float().cpu().numpy().astype(np.float32)
            if ft.shape != (len(d), 128):
                raise SystemExit(f"[FAIL] {seq} {f}: feats {ft.shape} for {len(d)} boxes")
        else:
            ft = np.zeros((0, 128), np.float32)
        frames.append(int(os.path.splitext(f)[0]))
        dets_all.append(d)
        feats_all.append(ft)
        offs.append(offs[-1] + len(d))
        speed.append([r.speed.get("preprocess", 0.0), r.speed.get("inference", 0.0),
                      r.speed.get("postprocess", 0.0)])
    secs = time.time() - t0
    p._hook.remove()
    p._feats = None
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{det_name}__{seq}.npz")
    np.savez_compressed(path, frames=np.array(frames, np.int32), off=np.array(offs, np.int64),
                        dets=np.concatenate(dets_all) if dets_all else np.zeros((0, 6), np.float32),
                        feats=np.concatenate(feats_all) if feats_all else np.zeros((0, 128), np.float32),
                        speed_ms=np.array(speed, np.float32))
    # MOT-style detection file for reference (frame, -1, x, y, w, h, conf, -1, -1, -1)
    with open(os.path.join(CACHE, f"{det_name}__{seq}.txt"), "w", encoding="utf-8") as fh:
        for fr, d in zip(frames, dets_all):
            for x1, y1, x2, y2, c, _ in d:
                fh.write(f"{fr},-1,{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{c:.5f},-1,-1,-1\n")
    sp = np.array(speed)
    return {"detector": det_name, "weights": wpath, "weights_sha256": got, "person_class": person,
            "sequence": seq, "frames": len(files), "boxes": int(offs[-1]),
            "seconds": round(secs, 1), "ms_per_frame_total": round(1000 * secs / max(len(files), 1), 1),
            "ms_per_frame_stages_mean": {"preprocess": float(sp[:, 0].mean()),
                                         "inference": float(sp[:, 1].mean()),
                                         "postprocess": float(sp[:, 2].mean())},
            "cache": path, "settings": {"imgsz": IMGSZ, "conf": CONF, "iou": 0.7,
                                        "classes": [person], "device": device, "half": False,
                                        "torch_threads": threads},
            "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                         "python": sys.version.split()[0]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--detectors", nargs="*", default=list(DETECTORS))
    ap.add_argument("--seqs", nargs="*", default=SEQS)
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    rec_path = os.path.join(OUT, "detection_cache.json")
    recs = json.load(open(rec_path, encoding="utf-8")) if os.path.isfile(rec_path) else {}
    for d in a.detectors:
        for s in a.seqs:
            key = f"{d}__{s}"
            if key in recs and os.path.isfile(recs[key]["cache"]):
                print(f"[skip] {key} cached", flush=True)
                continue
            r = run(d, s, a.device, a.threads)
            r["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
            r["command"] = " ".join([sys.executable] + sys.argv)
            recs[key] = r
            with open(rec_path, "w", encoding="utf-8") as fh:
                json.dump(recs, fh, indent=2)
            print(f"[{r['finished']}] {key}: {r['frames']} frames, {r['boxes']} boxes, "
                  f"{r['ms_per_frame_total']} ms/frame", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
