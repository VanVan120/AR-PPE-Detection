"""Revision R1, Part B: detector results at the deployed threshold, and bootstrap intervals.

Two stages, so the expensive one runs once:

  predict   one ultralytics validation pass per (weights, split) at conf 0.001, imgsz 640,
            NMS IoU 0.6 (the task's setting) -- and, from the SAME forward pass, NMS IoU 0.7,
            ultralytics' default and the setting behind Table 3. Every image's predictions,
            ground truth and ultralytics' own true-positive matrix are cached, so nothing
            below needs inference again.
  analyze   from the caches: precision / recall / F1 at IoU 0.5 at (a) conf 0.35, (b) the
            F1-optimal confidence chosen on valid and applied to test, (c) the F1-optimal
            confidence on test itself; and 1,000-resample bootstrap intervals for mAP@50 and
            mAP@50-95, resampling source groups (cluster) and, for comparison, images.

    .venv-gpu\\Scripts\\python.exe tools\\revision\\threshold_eval.py all --threads 4

Why the validator and not model.predict: the cache must reproduce the published evaluation
(tools/eval_grouped.py runs model.val with rect batches, batch 16, multi-label NMS). Capturing
inside the validator means the mAP computed from the cache equals the mAP ultralytics prints,
which `analyze` checks before reporting anything.
"""
from __future__ import annotations

import os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import csv
import datetime
import hashlib
import json
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(HERE))          # tools/ (rescore helpers)

DEFAULT_OUT = os.path.join(ROOT, "paper_materials", "revision_r1", "partB")
CACHE_DIR = os.path.join(ROOT, "data", "revision_r1_cache", "partB")   # gitignored (data/)
SPLIT_CSV = os.path.join(ROOT, "tools", "splits", "ppe_grouped_split.csv")
RETRAIN = os.path.join(ROOT, "paper_materials", "retrain")

WEIGHTS = {   # name -> (local path, sha256 of the asset released as v1.0.1-paper)
    "grouped_s2": (os.path.join(RETRAIN, "local_out", "best_grouped.pt"),                # best_grouped.pt
                   "679eae32764839df179a5ff8b2a9a2e924e013df860fef0af54daed2eb8c8c94"),
    "grouped_s1": (os.path.join(RETRAIN, "local_runs", "grouped_s1", "weights", "best.pt"),  # best_grouped_stage1.pt
                   "1b32d84f8103c5512da4c51d46b32febf763563a9d1b6d6121bbe196fe197656"),
    "refined": (os.path.join(ROOT, "best_refined.pt"),
                "3d4d6c5cac2538202fd77c14faa8f91044c8466fcbd07953eaf1c7aec1d18621"),
}
DATASETS = {
    "grouped": os.path.join(ROOT, "data", "ppe_grouped"),
    "published": os.path.join(ROOT, "data", "ppe_download"),
}
JOBS = [   # (weights, dataset, split)
    ("grouped_s2", "grouped", "valid"), ("grouped_s2", "grouped", "test"),
    ("grouped_s1", "grouped", "valid"), ("grouped_s1", "grouped", "test"),
    ("refined", "published", "test"),
]
CONF, IOU, EXTRA_IOU, IMGSZ, BATCH = 0.001, 0.6, 0.7, 640, 16
DEPLOYED_CONF = 0.35
N_BOOT, BOOT_SEED = 1000, 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def job_name(w, d, s):
    return f"{w}__{d}_{s}"


# ======================================================================= predict

def make_validator_class(extra_iou):
    """A DetectionValidator that keeps every image's predictions, ground truth and TP matrix."""
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import nms

    class CapturingValidator(DetectionValidator):
        last = None

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.captured = []
            self._extra = None
            CapturingValidator.last = self

        def postprocess(self, preds):
            # non_max_suppression converts the boxes to xyxy IN PLACE, so the second pass needs
            # its own copy of the raw output, taken before the first pass touches it.
            raw = (preds[0] if isinstance(preds, (list, tuple)) else preds).clone()
            out = super().postprocess(preds)          # NMS at self.args.iou (0.6)
            # The same raw output, NMS at the second IoU -- identical call, one argument changed.
            ex = nms.non_max_suppression(
                raw, self.args.conf, extra_iou,
                nc=0 if self.args.task == "detect" else self.nc, multi_label=True,
                agnostic=self.args.single_cls or self.args.agnostic_nms,
                max_det=self.args.max_det, end2end=self.end2end,
                rotated=self.args.task == "obb")
            self._extra = [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5]} for x in ex]
            return out

        def update_metrics(self, preds, batch):
            super().update_metrics(preds, batch)
            for si, pred in enumerate(preds):
                pb = self._prepare_batch(si, batch)
                rec = {"im_name": os.path.basename(pb["im_file"]),
                       "ori_shape": [int(v) for v in pb["ori_shape"]],
                       "imgsz": [int(v) for v in pb["imgsz"]],
                       "gt_box": pb["bboxes"].cpu().numpy().astype(np.float32).reshape(-1, 4),
                       "gt_cls": pb["cls"].cpu().numpy().astype(np.int16)}
                for tag, p in (("a", pred), ("b", self._extra[si])):
                    rec[f"{tag}_box"] = p["bboxes"].cpu().numpy().astype(np.float32).reshape(-1, 4)
                    rec[f"{tag}_conf"] = p["conf"].cpu().numpy().astype(np.float32)
                    rec[f"{tag}_cls"] = p["cls"].cpu().numpy().astype(np.int16)
                    rec[f"{tag}_tp"] = self._process_batch(p, pb)["tp"].astype(bool)
                self.captured.append(rec)

    return CapturingValidator


def write_data_yaml(dataset, split, work):
    """Mirror tools/eval_grouped.py: a data.yaml whose split entry is a sorted list of images."""
    import yaml
    import rescore as rs

    root = DATASETS[dataset]
    base = yaml.safe_load(open(os.path.join(root, "data.yaml"), encoding="utf-8"))
    base["path"] = root
    imgs = rs.list_images(os.path.join(root, split, "images"))
    os.makedirs(work, exist_ok=True)
    lst = os.path.join(work, f"{dataset}_{split}.txt")
    with open(lst, "w", encoding="utf-8") as fh:
        fh.write("\n".join(imgs) + "\n")
    spec = dict(base)
    key = "val" if split == "valid" else "test"
    spec[key] = os.path.abspath(lst)
    yp = os.path.join(work, f"data_{dataset}_{split}.yaml")
    with open(yp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    return yp, key, len(imgs), base


def pack(records, prefix):
    """Concatenate per-image arrays with offsets."""
    n = [len(r[f"{prefix}_conf"]) for r in records]
    off = np.concatenate([[0], np.cumsum(n)]).astype(np.int64)
    cat = lambda k, shape, dt: (np.concatenate([r[k] for r in records]).astype(dt)
                                if records else np.zeros(shape, dt))
    return {f"{prefix}_off": off,
            f"{prefix}_box": cat(f"{prefix}_box", (0, 4), np.float32),
            f"{prefix}_conf": cat(f"{prefix}_conf", (0,), np.float32),
            f"{prefix}_cls": cat(f"{prefix}_cls", (0,), np.int16),
            f"{prefix}_tp": cat(f"{prefix}_tp", (0, 10), bool)}


def predict(job, out, threads, limit=0):
    import torch
    import ultralytics
    from ultralytics import YOLO

    torch.set_num_threads(threads)
    w, d, s = job
    name = job_name(*job)
    wpath, wsha = WEIGHTS[w]
    got = sha256(wpath)
    if got != wsha:
        raise SystemExit(f"[FAIL] {wpath} sha256 {got} != expected {wsha}")
    work = os.path.join(CACHE_DIR, "work")
    yp, key, n_list, base = write_data_yaml(d, s, work)
    if limit:
        lst = [l.strip() for l in open(os.path.join(work, f"{d}_{s}.txt"), encoding="utf-8") if l.strip()]
        with open(os.path.join(work, f"{d}_{s}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lst[:limit]) + "\n")
        n_list = min(limit, len(lst))
    V = make_validator_class(EXTRA_IOU)
    model = YOLO(wpath)
    t0 = time.time()
    res = model.val(validator=V, data=yp, split=key, imgsz=IMGSZ, batch=BATCH, conf=CONF, iou=IOU,
                    device="cpu", workers=0, plots=False, half=False, verbose=False,
                    project=os.path.join(CACHE_DIR, "val_runs"), name=name, exist_ok=True)
    secs = time.time() - t0
    v = V.last
    recs = v.captured
    if len(recs) != n_list:
        raise SystemExit(f"[FAIL] captured {len(recs)} images, list has {n_list}")
    names = res.names if hasattr(res, "names") else model.names
    printed = {"mp": float(res.box.mp), "mr": float(res.box.mr), "map50": float(res.box.map50),
               "map": float(res.box.map),
               "per_class": {names[int(c)]: {"p": float(res.box.p[j]), "r": float(res.box.r[j]),
                                             "ap50": float(res.box.ap50[j]), "ap": float(res.box.ap[j])}
                             for j, c in enumerate(res.box.ap_class_index)}}
    arrays = {"im_name": np.array([r["im_name"] for r in recs]),
              "ori_shape": np.array([r["ori_shape"] for r in recs], dtype=np.int32),
              "imgsz": np.array([r["imgsz"] for r in recs], dtype=np.int32)}
    goff = np.concatenate([[0], np.cumsum([len(r["gt_cls"]) for r in recs])]).astype(np.int64)
    arrays.update(gt_off=goff,
                  gt_box=np.concatenate([r["gt_box"] for r in recs]).astype(np.float32),
                  gt_cls=np.concatenate([r["gt_cls"] for r in recs]).astype(np.int16))
    arrays.update(pack(recs, "a"))
    arrays.update(pack(recs, "b"))
    os.makedirs(CACHE_DIR, exist_ok=True)
    cpath = os.path.join(CACHE_DIR, f"{name}{'_limit%d' % limit if limit else ''}.npz")
    np.savez_compressed(cpath, **arrays)
    meta = {"job": name, "weights": wpath, "weights_sha256": got, "dataset_dir": DATASETS[d],
            "split": s, "images": len(recs), "limit": limit or None,
            "conf": CONF, "nms_iou": {"a": IOU, "b": EXTRA_IOU}, "imgsz": IMGSZ, "batch": BATCH,
            "rect": True, "multi_label_nms": True, "max_det": int(v.args.max_det),
            "device": "cpu", "torch_threads": threads, "workers": 0,
            "names": {int(k): v_ for k, v_ in dict(names).items()},
            "validator_printed_iou_a": printed, "seconds": round(secs, 1),
            "versions": versions(), "cache": cpath, "cache_sha256": sha256(cpath),
            "command": " ".join([sys.executable] + sys.argv),
            "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.join(out, "predict"), exist_ok=True)
    with open(os.path.join(out, "predict", f"{name}{'_limit%d' % limit if limit else ''}.json"),
              "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[predict] {name}: {len(recs)} images in {secs / 60:.1f} min -> {cpath}", flush=True)
    return meta


def versions():
    import torch
    import ultralytics
    import cv2
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "ultralytics": ultralytics.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "platform": platform.platform(),
            "cpu": platform.processor()}


# ======================================================================= analysis

class Cache:
    """One job's cached predictions for one NMS variant ('a' = IoU 0.6, 'b' = IoU 0.7)."""

    def __init__(self, path, variant):
        z = np.load(path, allow_pickle=False)
        self.im_name = z["im_name"]
        self.n = len(self.im_name)
        self.gt_off, self.gt_box, self.gt_cls = z["gt_off"], z["gt_box"], z["gt_cls"]
        p = variant
        self.off, self.box = z[f"{p}_off"], z[f"{p}_box"]
        self.conf, self.cls, self.tp = z[f"{p}_conf"], z[f"{p}_cls"].astype(np.int64), z[f"{p}_tp"]
        self.img_of_pred = np.repeat(np.arange(self.n), np.diff(self.off))
        self.img_of_gt = np.repeat(np.arange(self.n), np.diff(self.gt_off))

    def ultralytics_metrics(self):
        from ultralytics.utils.metrics import ap_per_class
        r = ap_per_class(self.tp, self.conf, self.cls.astype(float), self.gt_cls.astype(float))
        tp, fp, p, r_, f1, ap, ucls, p_curve, r_curve, f1_curve, x, _ = r
        return {"classes": ucls, "p": p, "r": r_, "f1": f1, "ap50": ap[:, 0], "ap": ap.mean(1),
                "mp": float(p.mean()), "mr": float(r_.mean()), "map50": float(ap[:, 0].mean()),
                "map": float(ap.mean()), "p_curve": p_curve, "r_curve": r_curve,
                "f1_curve": f1_curve, "x": x}


def f1_optimal_conf(m):
    """ultralytics' rule: argmax of the box-smoothed mean-over-classes F1 curve."""
    from ultralytics.utils.metrics import smooth
    i = int(smooth(m["f1_curve"].mean(0), 0.1).argmax())
    return float(m["x"][i]), i


def match_at(gt_box, gt_cls, box, cls, thr=0.5):
    """ultralytics' match_predictions (greedy by IoU, one-to-one, class-aware) at one IoU."""
    if len(gt_cls) == 0 or len(cls) == 0:
        return np.zeros(len(cls), bool)
    a, b = gt_box, box
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    iou = inter / (area_a[:, None] + area_b[None, :] - inter + 1e-7)
    iou = iou * (gt_cls[:, None] == cls[None, :])
    correct = np.zeros(len(cls), bool)
    matches = np.array(np.nonzero(iou >= thr)).T
    if matches.shape[0]:
        if matches.shape[0] > 1:
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int)] = True
    return correct


def counts_at_threshold(c, t, classes):
    """Re-match every image keeping only predictions with conf >= t (what a run at conf t sees)."""
    tp = {k: 0 for k in classes}
    npred = {k: 0 for k in classes}
    ngt = {k: int((c.gt_cls == k).sum()) for k in classes}
    for i in range(c.n):
        a, b = c.off[i], c.off[i + 1]
        keep = c.conf[a:b] > t           # strict, as ultralytics' NMS and the deployed detector
        box, cls = c.box[a:b][keep], c.cls[a:b][keep]
        g0, g1 = c.gt_off[i], c.gt_off[i + 1]
        corr = match_at(c.gt_box[g0:g1], c.gt_cls[g0:g1].astype(np.int64), box, cls)
        for k in classes:
            sel = cls == k
            npred[k] += int(sel.sum())
            tp[k] += int(corr[sel].sum())
    return tp, npred, ngt


def prf(tp, npred, ngt):
    p = tp / npred if npred else 0.0
    r = tp / ngt if ngt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def threshold_table(c, t, names):
    classes = sorted(int(k) for k in np.unique(c.gt_cls))
    tp, npred, ngt = counts_at_threshold(c, t, classes)
    rows = {}
    for k in classes:
        p, r, f = prf(tp[k], npred[k], ngt[k])
        rows[names[k]] = {"tp": tp[k], "fp": npred[k] - tp[k], "fn": ngt[k] - tp[k],
                          "precision": p, "recall": r, "f1": f}
    mp = float(np.mean([v["precision"] for v in rows.values()]))
    mr = float(np.mean([v["recall"] for v in rows.values()]))
    T, P, G = sum(tp.values()), sum(npred.values()), sum(ngt.values())
    pp, pr, pf = prf(T, P, G)
    return {"conf": t, "per_class": rows,
            "mean": {"precision": mp, "recall": mr,
                     "f1_of_means": 2 * mp * mr / (mp + mr) if (mp + mr) else 0.0,
                     "f1_macro": float(np.mean([v["f1"] for v in rows.values()]))},
            "pooled": {"precision": pp, "recall": pr, "f1": pf, "tp": T, "fp": P - T, "fn": G - T}}


def curve_at(m, t, names):
    """ultralytics-style P/R read off the interpolated curves at confidence t (cross-check)."""
    j = int(np.argmin(np.abs(m["x"] - t)))
    out = {}
    for ci, k in enumerate(m["classes"]):
        p, r = float(m["p_curve"][ci, j]), float(m["r_curve"][ci, j])
        out[names[int(k)]] = {"precision": p, "recall": r,
                              "f1": 2 * p * r / (p + r) if (p + r) else 0.0}
    return out


class WeightedAP:
    """mAP@50 and mAP@50-95 for a resample given as per-image weights (multiplicities).

    Predictions are ordered once by ultralytics' own permutation (np.argsort(-conf)); a resample
    weights each prediction by its image's multiplicity, which is ultralytics' computation on the
    resampled set with every copy of a prediction kept together. With all weights 1 this is
    exactly ap_per_class (checked in `analyze`).
    """

    def __init__(self, c):
        from ultralytics.utils.metrics import compute_ap
        self.compute_ap = compute_ap
        order = np.argsort(-c.conf)
        self.cls_s, self.tp_s = c.cls[order], c.tp[order].astype(np.float64)
        self.img_s = c.img_of_pred[order]
        self.classes = np.unique(c.gt_cls).astype(np.int64)
        self.per = []
        for k in self.classes:
            sel = self.cls_s == k
            ngt_img = np.bincount(c.img_of_gt[c.gt_cls == k], minlength=c.n).astype(np.float64)
            self.per.append((self.img_s[sel], self.tp_s[sel], ngt_img))

    def __call__(self, w):
        aps = []
        for img, tp, ngt_img in self.per:
            n_l = float(ngt_img @ w)
            if n_l == 0:
                continue                      # class absent from this resample: not averaged
            if len(img) == 0:
                aps.append(np.zeros(10))
                continue
            W = w[img]
            keep = W > 0
            if not keep.any():
                aps.append(np.zeros(10))
                continue
            Wk, tpk = W[keep][:, None], tp[keep]
            tpc = np.cumsum(Wk * tpk, 0)
            fpc = np.cumsum(Wk * (1 - tpk), 0)
            recall = tpc / (n_l + 1e-16)
            precision = tpc / (tpc + fpc)
            aps.append(np.array([self.compute_ap(recall[:, j], precision[:, j])[0]
                                 for j in range(10)]))
        a = np.array(aps)
        return float(a[:, 0].mean()), float(a.mean())


def load_groups():
    """file name -> source group, from the split definition."""
    g = {}
    with open(SPLIT_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            g[os.path.basename(r["file"])] = r["group"]
    return g


def draws(n_units, rng):
    return rng.integers(0, n_units, size=(N_BOOT, n_units))


def bootstrap(c, wap, groups_of_img, out_csv, label, draws_cluster, draws_image, units):
    """Percentile intervals; the draws are shared by every checkpoint scored on this test set."""
    rows, res = [], {}
    for kind, D in (("cluster", draws_cluster), ("image", draws_image)):
        s50, s5095 = np.empty(N_BOOT), np.empty(N_BOOT)
        for b in range(N_BOOT):
            if kind == "cluster":
                gw = np.bincount(D[b], minlength=len(units)).astype(np.float64)
                w = gw[groups_of_img]
            else:
                w = np.bincount(D[b], minlength=c.n).astype(np.float64)
            s50[b], s5095[b] = wap(w)
            rows.append({"checkpoint": label, "kind": kind, "resample": b,
                         "map50": s50[b], "map50_95": s5095[b]})
        res[kind] = {"map50": summary(s50), "map50_95": summary(s5095),
                     "samples": {"map50": s50, "map50_95": s5095}}
    with open(out_csv, "a", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=["checkpoint", "kind", "resample", "map50", "map50_95"])
        if fh.tell() == 0:
            wr.writeheader()
        wr.writerows(rows)
    return res


def summary(s):
    lo, hi = np.percentile(s, [2.5, 97.5])
    return {"mean": float(s.mean()), "sd": float(s.std(ddof=1)), "ci95": [float(lo), float(hi)]}


def analyze(out):
    names = None
    report = {"settings": {"conf": CONF, "nms_iou_task": IOU, "nms_iou_table3": EXTRA_IOU,
                           "imgsz": IMGSZ, "batch": BATCH, "deployed_conf": DEPLOYED_CONF,
                           "match_iou": 0.5, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                           "percentile_interval": [2.5, 97.5]},
              "jobs": {}, "checks": {}, "thresholds": {}, "bootstrap": {}}
    caches = {}
    for job in JOBS:
        name = job_name(*job)
        meta = json.load(open(os.path.join(out, "predict", f"{name}.json"), encoding="utf-8"))
        names = {int(k): v for k, v in meta["names"].items()}
        report["jobs"][name] = {k: meta[k] for k in ("weights", "weights_sha256", "dataset_dir",
                                                     "split", "images", "seconds", "cache_sha256")}
        for var in ("a", "b"):
            caches[(name, var)] = Cache(meta["cache"], var)
        # Check 1: the cache reproduces what the validator itself printed (IoU 0.6 variant).
        m = caches[(name, "a")].ultralytics_metrics()
        pr = meta["validator_printed_iou_a"]
        diffs = {k: abs(m[k] - pr[k]) for k in ("mp", "mr", "map50", "map")}
        report["checks"][f"{name}: cache vs validator (NMS 0.6)"] = {
            "max_abs_diff": max(diffs.values()), "diffs": diffs}

    # Check 2: the IoU-0.7 cache against the published Table 3 numbers (GPU run).
    pub = {"grouped_s2": os.path.join(ROOT, "results", "grouped_retrain", "eval_stage2", "eval_stage2.json"),
           "grouped_s1": os.path.join(ROOT, "results", "grouped_retrain", "eval_stage1", "eval_stage1.json")}
    for w, p in pub.items():
        pj = json.load(open(p, encoding="utf-8"))
        full = [r for r in pj["runs"] if r["tag"] == "full_640"][0]
        m = caches[(job_name(w, "grouped", "test"), "b")].ultralytics_metrics()
        report["checks"][f"{w}: cache NMS 0.7 (CPU) vs published full_640 (GPU)"] = {
            "published": {k: full["overall"][k] for k in ("precision", "recall", "map50", "map50_95")},
            "cache": {"precision": m["mp"], "recall": m["mr"], "map50": m["map50"], "map50_95": m["map"]}}

    # Check 3: weighted AP with unit weights == ap_per_class.
    for (name, var), c in caches.items():
        m = c.ultralytics_metrics()
        wap = WeightedAP(c)
        a50, a5095 = wap(np.ones(c.n))
        report["checks"][f"{name} ({var}): WeightedAP(1) vs ap_per_class"] = {
            "map50_diff": abs(a50 - m["map50"]), "map50_95_diff": abs(a5095 - m["map"])}

    # Thresholds, for the two grouped checkpoints, both NMS variants.
    for w in ("grouped_s2", "grouped_s1"):
        for var, iou in (("a", IOU), ("b", EXTRA_IOU)):
            cv = caches[(job_name(w, "grouped", "valid"), var)]
            ct = caches[(job_name(w, "grouped", "test"), var)]
            mv, mt = cv.ultralytics_metrics(), ct.ultralytics_metrics()
            t_valid, _ = f1_optimal_conf(mv)
            t_test, it = f1_optimal_conf(mt)
            key = f"{w} | NMS IoU {iou}"
            report["thresholds"][key] = {
                "a_deployed_0.35": threshold_table(ct, DEPLOYED_CONF, names),
                "b_valid_optimal": {"chosen_on_valid": t_valid,
                                    **threshold_table(ct, t_valid, names)},
                "c_test_optimal_ultralytics": {
                    "conf": t_test,
                    "per_class": {names[int(k)]: {"precision": float(mt["p"][j]),
                                                  "recall": float(mt["r"][j]),
                                                  "f1": float(mt["f1"][j])}
                                  for j, k in enumerate(mt["classes"])},
                    "mean": {"precision": mt["mp"], "recall": mt["mr"],
                             "f1_of_means": 2 * mt["mp"] * mt["mr"] / (mt["mp"] + mt["mr"]),
                             "f1_macro": float(np.mean(mt["f1"]))}},
                "c_test_optimal_rematched": threshold_table(ct, t_test, names),
                "curve_at_0.35_crosscheck": curve_at(mt, DEPLOYED_CONF, names),
                "valid_metrics": {"map50": mv["map50"], "map50_95": mv["map"]},
                "test_metrics": {"map50": mt["map50"], "map50_95": mt["map"]},
            }

    # Bootstrap: grouped test (both grouped checkpoints, shared draws), published test (refined).
    groups = load_groups()
    boot_csv = os.path.join(out, "bootstrap_samples.csv")
    if os.path.exists(boot_csv):
        os.remove(boot_csv)
    for var, iou in (("a", IOU), ("b", EXTRA_IOU)):
        for test_set, ckpts in (("grouped_test", ("grouped_s2", "grouped_s1")),
                                ("published_test", ("refined",))):
            ds = "grouped" if test_set == "grouped_test" else "published"
            c0 = caches[(job_name(ckpts[0], ds, "test"), var)]
            gids = [groups[n] for n in c0.im_name]
            units, gidx = np.unique(gids, return_inverse=True)
            rng_c, rng_i = np.random.default_rng(BOOT_SEED), np.random.default_rng(BOOT_SEED)
            Dc, Di = draws(len(units), rng_c), draws(c0.n, rng_i)
            res = {"images": c0.n, "groups": int(len(units))}
            samples = {}
            for w in ckpts:
                c = caches[(job_name(w, ds, "test"), var)]
                assert list(c.im_name) == list(c0.im_name), "image order differs between checkpoints"
                m = c.ultralytics_metrics()
                b = bootstrap(c, WeightedAP(c), gidx, boot_csv, f"{w} | NMS {iou}", Dc, Di, units)
                samples[w] = b
                res[w] = {"point": {"map50": m["map50"], "map50_95": m["map"]},
                          **{k: {kk: vv for kk, vv in b[k].items() if kk != "samples"}
                             for k in ("cluster", "image")}}
            if len(ckpts) == 2:      # paired difference stage 2 - stage 1 on identical resamples
                res["paired_difference_s2_minus_s1"] = {
                    k: {m_: summary(samples[ckpts[0]][k]["samples"][m_] - samples[ckpts[1]][k]["samples"][m_])
                        for m_ in ("map50", "map50_95")} for k in ("cluster", "image")}
            report["bootstrap"][f"{test_set} | NMS IoU {iou}"] = res

    with open(os.path.join(out, "partB_results.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    write_tables(report, out)
    print(f"[analyze] wrote {out}", flush=True)
    return report


def write_tables(report, out):
    """CSV per threshold table, one row per class plus the means."""
    with open(os.path.join(out, "thresholds_per_class.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["checkpoint_nms", "threshold_rule", "conf", "class", "precision", "recall", "f1",
                     "tp", "fp", "fn"])
        for key, t in report["thresholds"].items():
            for rule in ("a_deployed_0.35", "b_valid_optimal", "c_test_optimal_ultralytics",
                         "c_test_optimal_rematched"):
                tab = t[rule]
                for cls, v in tab["per_class"].items():
                    wr.writerow([key, rule, round(tab["conf"], 6), cls, v["precision"], v["recall"], v["f1"],
                                 v.get("tp", ""), v.get("fp", ""), v.get("fn", "")])
                mn = tab["mean"]
                wr.writerow([key, rule, round(tab["conf"], 6), "mean (macro)", mn["precision"],
                             mn["recall"], mn["f1_macro"], "", "", ""])


def deployed(out, threads, weights="grouped_s2", split="test"):
    """Supplementary: P/R/F1 at IoU 0.5 exactly as phase2's detector calls the model.

    phase2/src/detector.py runs model.predict(frame, conf=0.35, imgsz=640) on every frame:
    single-label NMS at IoU 0.7, per-image letterbox, boxes in original pixels. The cache above
    is validation-style (multi-label NMS, rect batches), so its conf-0.35 figures can differ;
    this pass measures the deployed path directly, against the YOLO label files.
    """
    import torch
    import rescore as rs
    from ultralytics import YOLO
    torch.set_num_threads(threads)
    wpath, wsha = WEIGHTS[weights]
    if sha256(wpath) != wsha:
        raise SystemExit("[FAIL] weights hash")
    root = DATASETS["grouped"]
    imgs = rs.list_images(os.path.join(root, split, "images"))
    model = YOLO(wpath)
    names = model.names
    classes = sorted(names)
    tp = {k: 0 for k in classes}
    npred = {k: 0 for k in classes}
    ngt = {k: 0 for k in classes}
    t0 = time.time()
    for p in imgs:
        r = model.predict(p, conf=DEPLOYED_CONF, imgsz=IMGSZ, device="cpu", verbose=False)[0]
        h, w = r.orig_shape
        lab = rs.label_for(p)
        gt = (np.loadtxt(lab, ndmin=2) if lab and os.path.isfile(lab) and os.path.getsize(lab) else np.zeros((0, 5)))
        gcls = gt[:, 0].astype(np.int64)
        gxy = np.stack([(gt[:, 1] - gt[:, 3] / 2) * w, (gt[:, 2] - gt[:, 4] / 2) * h,
                        (gt[:, 1] + gt[:, 3] / 2) * w, (gt[:, 2] + gt[:, 4] / 2) * h], 1) if len(gt) else np.zeros((0, 4))
        box = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(np.int64)
        corr = match_at(gxy, gcls, box, cls)
        for k in classes:
            npred[k] += int((cls == k).sum())
            tp[k] += int(corr[cls == k].sum())
            ngt[k] += int((gcls == k).sum())
    rows = {}
    for k in classes:
        pp, rr, ff = prf(tp[k], npred[k], ngt[k])
        rows[names[k]] = {"tp": tp[k], "fp": npred[k] - tp[k], "fn": ngt[k] - tp[k],
                          "precision": pp, "recall": rr, "f1": ff}
    mp = float(np.mean([v["precision"] for v in rows.values()]))
    mr = float(np.mean([v["recall"] for v in rows.values()]))
    res = {"weights": wpath, "split": split, "images": len(imgs), "conf": DEPLOYED_CONF,
           "nms": "ultralytics predict defaults: single-label, IoU 0.7, max_det 300",
           "per_class": rows, "mean": {"precision": mp, "recall": mr,
                                       "f1_macro": float(np.mean([v["f1"] for v in rows.values()]))},
           "seconds": round(time.time() - t0, 1), "versions": versions(),
           "command": " ".join([sys.executable] + sys.argv)}
    with open(os.path.join(out, f"deployed_path_{weights}_{split}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"[deployed] {weights} {split}: mean P {mp:.4f} R {mr:.4f} in {res['seconds'] / 60:.1f} min", flush=True)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["predict", "analyze", "all", "deployed"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--job", type=int, default=None, help="predict only job N (0-based)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N images only")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.stage in ("predict", "all"):
        jobs = JOBS if a.job is None else [JOBS[a.job]]
        for j in jobs:
            done = os.path.join(a.out, "predict", f"{job_name(*j)}.json")
            if os.path.exists(done) and not a.limit:
                print(f"[predict] {job_name(*j)} cached, skipping", flush=True)
                continue
            predict(j, a.out, a.threads, a.limit)
    if a.stage in ("analyze", "all") and not a.limit:
        analyze(a.out)
    if a.stage == "deployed":
        deployed(a.out, a.threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
