"""Prove the exported models are still ACCURATE — the gate that makes speed meaningful.

A faster model that quietly lost accuracy is a regression, not an optimisation, and
for a *safety* detector ("is this worker wearing a hard hat?") an unnoticed recall
drop is the worst possible failure. So every latency claim in Phase 4 is paired
with a measured accuracy claim, on the same held-out test split that produced the
Phase 1 benchmark.

Two complementary checks, because they fail differently:

  * **Output parity** (`--mode outputs`, fast). Run the baseline `.pt` and the
    exported model over the same images and compare detections directly: are the
    boxes matched at high IoU, and how far do the confidences move? This catches
    *breakage* (wrong preprocessing, lost metadata, layout mismatch) in seconds,
    and is the check to run routinely. An fp32 ONNX export should be numerically
    near-identical — measured max confidence deviation here is ~1e-6.

  * **Metric parity** (`--mode metrics`, authoritative). Run ultralytics'
    validator on both models with *identical* arguments and compare per-class
    precision / recall / mAP50 / mAP50-95. This is the number that belongs in a
    report. Identical args matter: a different `conf`/`iou`/`imgsz` between the
    two runs produces a fake "quantization loss" that is really a
    configuration difference.

Because a full 4,190-image validation on a CPU backend takes a while, `--limit`
builds a reproducible random subset (images + their labels + a generated
data.yaml) and validates on that; `--limit 0` uses the full split for the
authoritative figure.

    python -m phase4_deploy.edge.parity --mode outputs
    python -m phase4_deploy.edge.parity --mode metrics --limit 400
    python -m phase4_deploy.edge.parity --mode metrics --limit 0 --imgsz 640,320
"""
from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .common import (enable_utf8_stdout, file_mb, freeze_environment,
                     preserve_cuda_env, quiet_onnxruntime)

DEFAULT_WEIGHTS = "phase2/models/best.pt"
DEFAULT_ARTIFACTS = "phase4_deploy/artifacts"
DEFAULT_DATA = "data/ppe_download/data.yaml"


# --- output-level parity ------------------------------------------------------
def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of xyxy boxes -> (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    ax1, ay1, ax2, ay2 = (a[:, k][:, None] for k in range(4))
    bx1, by1, bx2, by2 = (b[:, k][None, :] for k in range(4))
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = np.clip((ax2 - ax1) * (ay2 - ay1), 0, None)
    area_b = np.clip((bx2 - bx1) * (by2 - by1), 0, None)
    union = np.maximum(area_a + area_b - inter, 1e-9)
    return inter / union


def match_detections(boxes_a, cls_a, conf_a, boxes_b, cls_b, conf_b,
                     iou_thresh: float = 0.9) -> dict:
    """Greedily match detections of the same class between two models.

    Returns counts plus the largest confidence deviation across matched pairs.
    A correct fp32 export should match everything with a deviation near zero."""
    boxes_a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    boxes_b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)
    cls_a = np.asarray(cls_a).reshape(-1)
    cls_b = np.asarray(cls_b).reshape(-1)
    conf_a = np.asarray(conf_a, dtype=float).reshape(-1)
    conf_b = np.asarray(conf_b, dtype=float).reshape(-1)

    ious = iou_matrix(boxes_a, boxes_b)
    same = (cls_a[:, None] == cls_b[None, :]).astype(float) if len(cls_a) and len(cls_b) \
        else np.zeros_like(ious)
    scored = ious * same
    used_b = set()
    matched = 0
    max_dconf = 0.0
    for i in range(len(boxes_a)):
        if scored.shape[1] == 0:
            break
        order = np.argsort(-scored[i])
        for j in order:
            if scored[i, j] < iou_thresh:
                break
            if j in used_b:
                continue
            used_b.add(int(j))
            matched += 1
            max_dconf = max(max_dconf, abs(float(conf_a[i] - conf_b[j])))
            break
    return {"n_a": int(len(boxes_a)), "n_b": int(len(boxes_b)), "matched": int(matched),
            "unmatched_a": int(len(boxes_a) - matched),
            "unmatched_b": int(len(boxes_b) - matched),
            "max_conf_delta": float(max_dconf)}


def _predict_all(model_path: str, images: Sequence[str], imgsz: int, conf: float,
                 device) -> List[tuple]:
    from ultralytics import YOLO
    out = []
    with preserve_cuda_env():
        model = YOLO(model_path, task="detect")
        for p in images:
            r = model.predict(p, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            b = r.boxes
            out.append((b.xyxy.cpu().numpy(), b.cls.cpu().numpy(), b.conf.cpu().numpy()))
    return out


def output_parity(baseline: str, other: str, images: Sequence[str], imgsz: int = 640,
                  conf: float = 0.35, iou_thresh: float = 0.9, device="cpu") -> dict:
    """Detection-level agreement between two models over the same images."""
    A = _predict_all(baseline, images, imgsz, conf, device)
    B = _predict_all(other, images, imgsz, conf, device)
    tot = {"n_a": 0, "n_b": 0, "matched": 0, "unmatched_a": 0, "unmatched_b": 0}
    max_d = 0.0
    for (ba, ca, fa), (bb, cb, fb) in zip(A, B):
        m = match_detections(ba, ca, fa, bb, cb, fb, iou_thresh)
        for k in tot:
            tot[k] += m[k]
        max_d = max(max_d, m["max_conf_delta"])
    tot["max_conf_delta"] = max_d
    tot["agreement_pct"] = 100.0 * tot["matched"] / max(tot["n_a"], 1)
    tot["images"] = len(images)
    return tot


# --- metric-level parity ------------------------------------------------------
def make_subset(data_yaml: str, split: str, limit: int, outdir: str,
                seed: int = 0) -> Tuple[str, int]:
    """Build a reproducible N-image subset dataset (images + labels + data.yaml).

    Returns (subset_yaml_path, n_images). ultralytics finds labels by swapping
    `/images/` for `/labels/`, so the subset mirrors that layout."""
    import yaml

    with open(data_yaml, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    root = os.path.dirname(os.path.abspath(data_yaml))
    rel = cfg.get(split) or cfg.get("val")
    split_dir = os.path.normpath(os.path.join(root, str(rel)))
    if not os.path.isdir(split_dir):                      # yaml paths are often ../<split>/images
        alt = os.path.normpath(os.path.join(root, str(rel).lstrip("./")))
        split_dir = alt if os.path.isdir(alt) else split_dir
    imgs = sorted(
        os.path.join(split_dir, f) for f in os.listdir(split_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")))
    rng = random.Random(seed)
    rng.shuffle(imgs)
    picked = imgs[:limit] if limit > 0 else imgs

    img_out = os.path.join(outdir, "images")
    lbl_out = os.path.join(outdir, "labels")
    shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)
    lbl_dir = os.path.join(os.path.dirname(split_dir), "labels")
    n_labels = 0
    for p in picked:
        shutil.copy(p, os.path.join(img_out, os.path.basename(p)))
        lb = os.path.join(lbl_dir, os.path.splitext(os.path.basename(p))[0] + ".txt")
        if os.path.isfile(lb):
            shutil.copy(lb, os.path.join(lbl_out, os.path.basename(lb)))
            n_labels += 1
    if not picked:
        raise RuntimeError(f"no images found under {split_dir} - check the dataset layout")
    if n_labels == 0:
        # Without labels ultralytics treats every image as background and reports a
        # meaningless ~0 mAP, which would look like catastrophic quantization loss.
        raise RuntimeError(
            f"found {len(picked)} images but NO labels in {lbl_dir}. The subset would "
            "score ~0 mAP for every model, which is a dataset-layout problem, not an "
            "export problem. Expected a 'labels/' dir beside 'images/'.")
    sub_yaml = os.path.join(outdir, "data.yaml")
    with open(sub_yaml, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"path": os.path.abspath(outdir), "train": "images",
                        "val": "images", "test": "images",
                        "names": cfg.get("names"), "nc": cfg.get("nc")}, fh)
    return sub_yaml, len(picked)


def val_metrics(model_path: str, data_yaml: str, imgsz: int, device, conf: float = 0.001,
                iou: float = 0.6, split: str = "val", batch: int = 1) -> dict:
    """Run ultralytics validation and extract per-class + overall metrics.

    `conf` defaults to 0.001 (the standard mAP protocol — a high conf threshold
    truncates the precision-recall curve and understates mAP)."""
    from ultralytics import YOLO

    with preserve_cuda_env():
        model = YOLO(model_path, task="detect")
        res = model.val(data=data_yaml, imgsz=imgsz, device=device, conf=conf, iou=iou,
                        split=split, batch=batch, plots=False, verbose=False)
    box = res.box
    names = {int(k): str(v) for k, v in (res.names or {}).items()}
    idx = [int(i) for i in getattr(box, "ap_class_index", [])]
    per_class: Dict[str, dict] = {}
    for pos, cid in enumerate(idx):
        try:
            per_class[names.get(cid, str(cid))] = {
                "precision": float(box.p[pos]), "recall": float(box.r[pos]),
                "mAP50": float(box.ap50[pos]), "mAP50-95": float(box.ap[pos]),
            }
        except Exception:
            pass
    return {"mAP50": float(box.map50), "mAP50-95": float(box.map),
            "precision": float(box.mp), "recall": float(box.mr),
            "per_class": per_class}


def _fmt_delta(v: float) -> str:
    return f"{v:+.4f}" if abs(v) >= 1e-4 else "  0.0000"


def main(argv=None) -> int:
    import argparse
    import glob
    import json

    enable_utf8_stdout()
    freeze_environment(quiet_yolo=True)
    quiet_onnxruntime()

    ap = argparse.ArgumentParser(
        description="Verify exported models keep the detector's accuracy")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--artifacts", default=DEFAULT_ARTIFACTS)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--mode", default="outputs", choices=["outputs", "metrics", "both"])
    ap.add_argument("--imgsz", default="640", help="comma-separated for metrics mode")
    ap.add_argument("--images", type=int, default=25, help="images for output parity")
    ap.add_argument("--limit", type=int, default=400,
                    help="metrics: subset size (0 = full split, authoritative)")
    ap.add_argument("--conf", type=float, default=0.35, help="output-parity conf threshold")
    ap.add_argument("--split", default="test")
    ap.add_argument("--json", default="")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.weights):
        print(f"[FAIL] weights not found: {args.weights}")
        return 1
    stem = os.path.splitext(os.path.basename(args.weights))[0]
    payload: Dict[str, object] = {"weights": args.weights, "mode": args.mode}
    sizes = [int(s) for s in str(args.imgsz).split(",") if str(s).strip()]

    print("=" * 78)
    print(" Phase 4 — accuracy parity (exported vs PyTorch baseline)")
    print("=" * 78)

    # ---- output-level parity -------------------------------------------------
    if args.mode in ("outputs", "both"):
        pool = sorted(glob.glob("data/ppe_download/test/images/*.jpg")) or \
            sorted(glob.glob("data/dataset/test/images/*.jpg"))
        if not pool:
            print("[warn] no test images found — skipping output parity")
        else:
            imgs = pool[:max(1, args.images)]
            print(f"\n--- output parity on {len(imgs)} images "
                  f"(conf={args.conf}, match IoU>=0.9) ---")
            print(f"{'model':<22} {'dets(pt)':>9} {'dets':>6} {'matched':>8} "
                  f"{'agree%':>7} {'max dconf':>10}")
            rows = {}
            for imgsz in sizes:
                for tag, label in (("fp32", "onnx fp32"), ("fp16", "onnx fp16"),
                                   ("int8", "onnx int8")):
                    p = os.path.join(args.artifacts, f"{stem}_{tag}_{imgsz}.onnx")
                    if not os.path.isfile(p):
                        continue
                    r = output_parity(args.weights, p, imgs, imgsz, args.conf, device="cpu")
                    rows[f"{label}@{imgsz}"] = r
                    print(f"{label + '@' + str(imgsz):<22} {r['n_a']:>9} {r['n_b']:>6} "
                          f"{r['matched']:>8} {r['agreement_pct']:>6.1f}% "
                          f"{r['max_conf_delta']:>10.6f}")
            payload["output_parity"] = rows

    # ---- metric-level parity -------------------------------------------------
    if args.mode in ("metrics", "both"):
        if not os.path.isfile(args.data):
            print(f"\n[warn] dataset yaml not found: {args.data} — skipping metric parity")
        else:
            work = os.path.join(args.artifacts, ".subset")
            if args.limit > 0:
                data_yaml, n = make_subset(args.data, args.split, args.limit, work)
                split_used = "val"
                print(f"\n--- metric parity on a {n}-image random subset of '{args.split}' "
                      f"(--limit 0 for the full split) ---")
            else:
                data_yaml, split_used = args.data, args.split
                print(f"\n--- metric parity on the FULL '{args.split}' split (authoritative) ---")

            metrics: Dict[str, dict] = {}
            for imgsz in sizes:
                targets = [("pytorch", args.weights)]
                for tag, label in (("fp32", "onnx fp32"), ("fp16", "onnx fp16"),
                                   ("int8", "onnx int8")):
                    p = os.path.join(args.artifacts, f"{stem}_{tag}_{imgsz}.onnx")
                    if os.path.isfile(p):
                        targets.append((label, p))
                print(f"\n  imgsz {imgsz}")
                print(f"  {'model':<16} {'mAP50':>8} {'mAP50-95':>9} {'precision':>10} "
                      f"{'recall':>8}   {'dmAP50':>9}")
                base = None
                for label, path in targets:
                    dev = "cpu" if path.endswith(".onnx") else 0
                    try:
                        m = val_metrics(path, data_yaml, imgsz, dev, split=split_used)
                    except Exception as e:
                        print(f"  {label:<16} FAILED: {type(e).__name__}: {str(e)[:90]}")
                        continue
                    metrics[f"{label}@{imgsz}"] = m
                    if base is None:
                        base = m
                        d = "  baseline"
                    else:
                        d = _fmt_delta(m["mAP50"] - base["mAP50"])
                    print(f"  {label:<16} {m['mAP50']:>8.4f} {m['mAP50-95']:>9.4f} "
                          f"{m['precision']:>10.4f} {m['recall']:>8.4f}   {d:>9}")
            payload["metric_parity"] = metrics

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
