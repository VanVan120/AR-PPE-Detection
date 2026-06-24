"""Quantitative evaluation against the dataset's ground-truth labels — the headline.

YOLO-World (detection metrics)
  Predictions are IoU-matched to ground-truth boxes via `supervision` at IoU 0.5.
  Evaluation is restricted to the *scored* PPE classes (the dataset classes that
  prompts map to). Reports per-class precision/recall/F1 (operating point at
  `confidence_threshold`, derived from supervision's confusion matrix), per-class
  AP@50 and overall mAP@50 (full curve, from supervision's MeanAveragePrecision),
  plus the confusion matrix.

VLM (image-level metrics)
  The VLM emits text, not boxes, so it is scored at the image level on the core
  violation classes (NO-Hardhat / NO-Safety Vest): did the image's GT contain the
  class, and did the VLM flag it? Reports per-class precision/recall/F1.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import supervision as sv

from .config import Config
from .detector_yolo import Detection
from .detector_vlm import VlmResult
from .loader import Sample


# ===========================================================================
# YOLO-World detection metrics
# ===========================================================================
def evaluate_yolo(
    samples: list[Sample],
    preds_per_sample: list[list[Detection]],
    cfg: Config,
) -> dict:
    scored = cfg.scored_classes
    if not scored:
        return {"available": False, "reason": "no prompts map to a valid dataset class"}

    class_to_idx = {name: i for i, name in enumerate(scored)}
    pred_list: list[sv.Detections] = []      # full curve (down to floor) -> mAP
    op_pred_list: list[sv.Detections] = []    # filtered at the operating point -> P/R/F1
    gt_list: list[sv.Detections] = []
    support = {name: 0 for name in scored}

    for sample, dets in zip(samples, preds_per_sample):
        pred_list.append(_preds_to_sv(dets, class_to_idx))
        # Operating point: keep each detection only if it meets its per-class
        # threshold (per_class_thresholds override, else global). With no overrides
        # this is just the global threshold, matching the previous behaviour.
        op_dets = [d for d in dets if d.confidence >= cfg.threshold_for(d.class_name)]
        op_pred_list.append(_preds_to_sv(op_dets, class_to_idx))
        gt = _gt_to_sv(sample.gt, cfg.dataset_classes, class_to_idx)
        gt_list.append(gt)
        if len(gt):
            for cid in gt.class_id:
                support[scored[int(cid)]] += 1

    from supervision.metrics import MeanAveragePrecision, MetricTarget

    map_res = (
        MeanAveragePrecision(metric_target=MetricTarget.BOXES)
        .update(pred_list, gt_list)
        .compute()
    )
    # Predictions are already filtered to the operating point, so the confusion
    # matrix keeps everything it is given (conf_threshold=0).
    cm = sv.ConfusionMatrix.from_detections(
        predictions=op_pred_list,
        targets=gt_list,
        classes=scored,
        conf_threshold=0.0,
        iou_threshold=cfg.iou_threshold,
    )

    per_class: dict[str, dict] = {}
    p_vals, r_vals, f_vals = [], [], []
    for i, name in enumerate(scored):
        tp, fp, fn = _cm_counts(cm.matrix, i)
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall else (0.0 if (precision is not None and recall is not None) else None)
        )
        ap = _ap50(map_res, i)
        per_class[name] = {
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "ap50": _round(ap),
            "tp": tp, "fp": fp, "fn": fn,
            "support": support[name],
        }
        # Macro averages over the classes that actually occur in GT (support > 0),
        # using the zero_division=0 convention (no predictions -> precision/F1 count
        # as 0) so all three macros average over the *same* class set. Per-class
        # values above keep `None` to flag "the model predicted nothing for this class".
        if support[name] > 0:
            p_vals.append(precision if precision is not None else 0.0)
            r_vals.append(recall if recall is not None else 0.0)
            f_vals.append(f1 if f1 is not None else 0.0)

    # supervision returns map50 == -1.0 as a sentinel when no scored class has any
    # ground truth across the whole set; surface that as None (rendered as "—"),
    # not a misleading -100%.
    map50 = float(map_res.map50)
    overall = {
        "precision_macro": _round(_mean(p_vals)),
        "recall_macro": _round(_mean(r_vals)),
        "f1_macro": _round(_mean(f_vals)),
        "mAP50": _round(map50 if map50 >= 0 else None),
        "support_total": int(sum(support.values())),
    }

    return {
        "available": True,
        "iou_threshold": cfg.iou_threshold,
        "confidence_threshold": cfg.confidence_threshold,
        "per_class_thresholds": {name: cfg.threshold_for(name) for name in scored},
        "tuned_thresholds_active": cfg.uses_per_class_thresholds,
        "num_images": len(samples),
        "scored_classes": scored,
        "per_class": per_class,
        "overall": overall,
        "confusion_matrix": {
            "labels": scored + ["background"],
            "matrix": cm.matrix.astype(int).tolist(),
            "note": "rows = ground truth, cols = predicted; last index = background (FP row / FN col)",
        },
    }


def _preds_to_sv(detections: list[Detection], class_to_idx: dict[str, int]) -> sv.Detections:
    boxes, confs, cids = [], [], []
    for d in detections:
        if d.class_name in class_to_idx:
            boxes.append(d.bbox)
            confs.append(d.confidence)
            cids.append(class_to_idx[d.class_name])
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=np.array(boxes, dtype=float),
        confidence=np.array(confs, dtype=float),
        class_id=np.array(cids, dtype=int),
    )


def _gt_to_sv(gt: Optional[sv.Detections], dataset_classes: list[str], class_to_idx: dict[str, int]) -> sv.Detections:
    if gt is None or len(gt) == 0:
        return sv.Detections.empty()
    boxes, cids = [], []
    for box, cid in zip(gt.xyxy, gt.class_id):
        idx = int(cid)
        name = dataset_classes[idx] if 0 <= idx < len(dataset_classes) else None
        if name in class_to_idx:
            boxes.append(box)
            cids.append(class_to_idx[name])
    if not boxes:
        return sv.Detections.empty()
    return sv.Detections(xyxy=np.array(boxes, dtype=float), class_id=np.array(cids, dtype=int))


def _cm_counts(matrix: np.ndarray, i: int) -> tuple[int, int, int]:
    """TP/FP/FN for class i from supervision's confusion matrix.

    Convention (verified): rows = ground truth, cols = predicted, last index =
    background. So TP = M[i,i]; FP = (col i sum) - TP; FN = (row i sum) - TP.
    """
    tp = matrix[i, i]
    fp = matrix[:, i].sum() - tp
    fn = matrix[i, :].sum() - tp
    return int(round(tp)), int(round(fp)), int(round(fn))


def _ap50(map_res, class_idx: int) -> Optional[float]:
    matched = list(np.asarray(map_res.matched_classes).tolist())
    if class_idx in matched:
        row = matched.index(class_idx)
        return float(map_res.ap_per_class[row][0])  # column 0 == IoU 0.5
    return None


# ===========================================================================
# VLM image-level metrics
# ===========================================================================
def evaluate_vlm(
    samples: list[Sample],
    vlm_per_sample: list[Optional[VlmResult]],
    cfg: Config,
) -> dict:
    classes = [c for c in cfg.vlm_eval_classes if c in cfg.dataset_classes]
    if not classes:
        return {"available": False, "reason": "no valid vlm_eval_classes in dataset"}

    neg_pattern = _negative_pattern(cfg.vlm_negative_keywords)
    counts = {c: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for c in classes}

    for sample, vres in zip(samples, vlm_per_sample):
        gt_present = _gt_image_classes(sample.gt, cfg.dataset_classes)
        for cls in classes:
            truth = cls in gt_present
            pred = _vlm_flags_class(vres, cls, cfg, neg_pattern)
            cell = counts[cls]
            if truth and pred:
                cell["tp"] += 1
            elif pred and not truth:
                cell["fp"] += 1
            elif truth and not pred:
                cell["fn"] += 1
            else:
                cell["tn"] += 1

    per_class: dict[str, dict] = {}
    p_vals, r_vals, f_vals = [], [], []
    for cls in classes:
        c = counts[cls]
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall else (0.0 if (precision is not None and recall is not None) else None)
        )
        per_class[cls] = {
            "precision": _round(precision),
            "recall": _round(recall),
            "f1": _round(f1),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support_images": tp + fn,
        }
        # Macro over classes present in GT (support_images > 0), zero_division=0.
        if (tp + fn) > 0:
            p_vals.append(precision if precision is not None else 0.0)
            r_vals.append(recall if recall is not None else 0.0)
            f_vals.append(f1 if f1 is not None else 0.0)

    return {
        "available": True,
        "level": "image",
        "num_images": len(samples),
        "eval_classes": classes,
        "per_class": per_class,
        "overall": {
            "precision_macro": _round(_mean(p_vals)),
            "recall_macro": _round(_mean(r_vals)),
            "f1_macro": _round(_mean(f_vals)),
        },
    }


def _gt_image_classes(gt: Optional[sv.Detections], dataset_classes: list[str]) -> set[str]:
    present: set[str] = set()
    if gt is None or len(gt) == 0:
        return present
    for cid in gt.class_id:
        idx = int(cid)
        if 0 <= idx < len(dataset_classes):
            present.add(dataset_classes[idx])
    return present


_NEVER_MATCH = re.compile(r"(?!x)x")  # matches nothing


def _negative_pattern(neg_words: list[str]) -> re.Pattern:
    norm = sorted({_normalize(w) for w in neg_words if w.strip()}, key=len, reverse=True)
    if not norm:
        # No negative words configured -> nothing can be "negated". Returning an
        # empty-alternation regex would instead match at every position and flag
        # every PPE keyword as a violation, so use a never-match pattern.
        return _NEVER_MATCH
    alts = "|".join(re.escape(w) for w in norm)
    return re.compile(rf"(?<![a-z]){alts}(?![a-z])")


def _vlm_flags_class(vres: Optional[VlmResult], cls: str, cfg: Config, neg_pattern: re.Pattern) -> bool:
    if vres is None or not vres.observations:
        return False
    keywords = [_normalize(k) for k in cfg.vlm_violation_keywords.get(cls, [])]
    if not keywords:
        return False
    for obs in vres.observations:
        text = _normalize(f"{obs.type} {obs.description}")
        if _keyword_negated(text, keywords, neg_pattern):
            return True
    return False


# Require a negative word *near* the PPE keyword (within this many characters),
# so "wearing a vest" + an unrelated "no guardrail" elsewhere doesn't count as a
# missing-vest flag. ~40 chars ≈ a short clause.
_NEG_WINDOW = 40


def _keyword_negated(text: str, keywords: list[str], neg_pattern: re.Pattern) -> bool:
    for kw in keywords:
        start = 0
        while True:
            idx = text.find(kw, start)
            if idx == -1:
                break
            lo = max(0, idx - _NEG_WINDOW)
            hi = min(len(text), idx + len(kw) + _NEG_WINDOW)
            if neg_pattern.search(text[lo:hi]):
                return True
            start = idx + len(kw)
    return False


def _normalize(text: str) -> str:
    t = str(text).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", t).strip()


# ===========================================================================
# small helpers
# ===========================================================================
def _round(x: Optional[float], ndigits: int = 4) -> Optional[float]:
    if x is None:
        return None
    return round(float(x), ndigits)


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None
