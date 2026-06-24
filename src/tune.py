"""Per-class confidence-threshold tuning + precision/recall curves for YOLO-World.

Why this exists
---------------
The headline metrics score the detector at a single global confidence threshold
(``confidence_threshold``, default 0.25). But a zero-shot open-vocabulary detector
has a very different precision/recall trade-off per class. On this dataset, for
example, ``Hardhat`` scores P=100% / R=3.6% at 0.25 while its AP@50 is ~70% — i.e.
most of its recall sits *just below* the operating threshold and is thrown away.

This module quantifies that: for each scored class it builds a precision/recall
curve by IoU-matching predictions to ground truth, then picks the F1-optimal
confidence threshold. Crucially the threshold is chosen on the **validation**
split and only *reported* on the **test** split, so the test numbers stay an
honest held-out measurement (no operating-point leakage).

Nothing here trains a model — it only re-chooses the cutoff on existing zero-shot
detections. It therefore cannot help the classes whose AP@50 is ~0 (NO-Hardhat /
NO-Safety Vest): no threshold recovers recall the detector never produced. That
limitation is exactly what motivates fine-tuning (Track A).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import supervision as sv

from .config import Config
from .detector_yolo import Detection


# ---------------------------------------------------------------------------
# Per-class precision/recall curve
# ---------------------------------------------------------------------------
@dataclass
class ClassCurve:
    """Precision/recall sweep for one class, plus its F1-optimal operating point."""
    name: str
    npos: int                                  # number of ground-truth boxes
    conf: list[float] = field(default_factory=list)        # threshold at each step
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)
    f1: list[float] = field(default_factory=list)
    best_threshold: Optional[float] = None
    best_precision: Optional[float] = None
    best_recall: Optional[float] = None
    best_f1: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "npos": self.npos,
            "best_threshold": _r(self.best_threshold),
            "best_precision": _r(self.best_precision),
            "best_recall": _r(self.best_recall),
            "best_f1": _r(self.best_f1),
            # store a downsampled curve so the JSON stays small but plottable
            "curve": _sample_curve(self.conf, self.precision, self.recall, self.f1),
        }


def _scored_pairs_for_class(
    name: str,
    samples_preds: list[list[Detection]],
    samples_gt: list[sv.Detections],
    dataset_classes: list[str],
    iou_thr: float,
) -> tuple[list[tuple[float, int]], int]:
    """Return [(confidence, is_true_positive)] across all images for one class,
    plus the total number of ground-truth boxes (npos) for that class.

    Matching is greedy per image: each prediction (highest confidence first) claims
    the highest-IoU unclaimed GT box of the same class with IoU >= ``iou_thr``.
    """
    pairs: list[tuple[float, int]] = []
    npos = 0
    for dets, gt in zip(samples_preds, samples_gt):
        gt_boxes = _gt_boxes_for_class(gt, name, dataset_classes)
        npos += len(gt_boxes)

        pred = [(d.confidence, d.bbox) for d in dets if d.class_name == name]
        if not pred:
            continue
        pred.sort(key=lambda x: -x[0])
        pboxes = np.array([b for _, b in pred], dtype=float)

        if len(gt_boxes):
            iou = sv.box_iou_batch(gt_boxes, pboxes)   # [n_gt, n_pred]
            claimed = np.zeros(len(gt_boxes), dtype=bool)
        for j, (conf, _) in enumerate(pred):
            tp = 0
            if len(gt_boxes):
                col = np.where(claimed, -1.0, iou[:, j])
                k = int(np.argmax(col))
                if col[k] >= iou_thr:
                    claimed[k] = True
                    tp = 1
            pairs.append((float(conf), tp))
    return pairs, npos


def _gt_boxes_for_class(gt: Optional[sv.Detections], name: str, dataset_classes: list[str]) -> np.ndarray:
    if gt is None or len(gt) == 0:
        return np.empty((0, 4), dtype=float)
    boxes = []
    for box, cid in zip(gt.xyxy, gt.class_id):
        idx = int(cid)
        if 0 <= idx < len(dataset_classes) and dataset_classes[idx] == name:
            boxes.append(box)
    return np.array(boxes, dtype=float) if boxes else np.empty((0, 4), dtype=float)


def class_curve(
    name: str,
    samples_preds: list[list[Detection]],
    samples_gt: list[sv.Detections],
    dataset_classes: list[str],
    iou_thr: float,
) -> ClassCurve:
    pairs, npos = _scored_pairs_for_class(name, samples_preds, samples_gt, dataset_classes, iou_thr)
    curve = ClassCurve(name=name, npos=npos)
    if not pairs or npos == 0:
        return curve

    pairs.sort(key=lambda x: -x[0])
    tp_cum = fp_cum = 0
    for conf, tp in pairs:
        tp_cum += tp
        fp_cum += 1 - tp
        prec = tp_cum / (tp_cum + fp_cum)
        rec = tp_cum / npos
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        curve.conf.append(conf)
        curve.precision.append(prec)
        curve.recall.append(rec)
        curve.f1.append(f1)

    best = int(np.argmax(curve.f1))
    curve.best_threshold = float(curve.conf[best])
    curve.best_precision = float(curve.precision[best])
    curve.best_recall = float(curve.recall[best])
    curve.best_f1 = float(curve.f1[best])
    return curve


# ---------------------------------------------------------------------------
# Operating-point evaluation at a chosen threshold (for before/after on test)
# ---------------------------------------------------------------------------
def operating_point(
    name: str,
    samples_preds: list[list[Detection]],
    samples_gt: list[sv.Detections],
    dataset_classes: list[str],
    iou_thr: float,
    threshold: float,
) -> dict:
    """P/R/F1/TP/FP/FN for one class at a fixed confidence ``threshold``."""
    pairs, npos = _scored_pairs_for_class(name, samples_preds, samples_gt, dataset_classes, iou_thr)
    kept = [(c, tp) for c, tp in pairs if c >= threshold]
    tp = sum(tp for _, tp in kept)
    fp = sum(1 - tp_ for _, tp_ in kept)
    fn = npos - tp
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / npos if npos else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall else (0.0 if (precision is not None and recall is not None) else None))
    return {
        "threshold": round(float(threshold), 4),
        "precision": _r(precision), "recall": _r(recall), "f1": _r(f1),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "support": int(npos),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _r(x: Optional[float], n: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), n)


def _sample_curve(conf, prec, rec, f1, max_points: int = 200) -> list[dict]:
    if not conf:
        return []
    n = len(conf)
    idxs = range(n) if n <= max_points else (int(i * (n - 1) / (max_points - 1)) for i in range(max_points))
    seen = []
    out = []
    for i in idxs:
        if i in seen:
            continue
        seen.append(i)
        out.append({"conf": _r(conf[i]), "p": _r(prec[i]), "r": _r(rec[i]), "f1": _r(f1[i])})
    return out
