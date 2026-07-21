"""Assembly101 temporal-action-segmentation metrics: MoF, segmental Edit, F1@{10,25,50}.

Faithful port of the reference implementation (assembly-101/
assembly101-temporal-action-segmentation `utils.py`, itself adapted from the
MS-TCN `eval.py`). Three metrics with **three different aggregation schemes** — a
naive uniform average does NOT reproduce the reference:

  * MoF / frame accuracy  — GLOBAL per-frame micro-average (sum correct / sum frames)
  * Segmental Edit        — PER-VIDEO macro-average (mean of per-video edit scores)
  * F1@{10,25,50}         — GLOBAL per-segment micro-average (accumulate tp/fp/fn,
                            compute precision/recall/F1 once at the end)

CRITICAL Assembly101 detail: `bg_class = []` (empty) — NO background/SIL class is
excluded, unlike GTEA / 50Salads / Breakfast. A non-empty background list gives
wrong Edit/F1 numbers here.

Per-frame labels may be ints (action ids) or strings; comparisons are by equality,
so be consistent within a run.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def get_labels_start_end_time(frame_wise_labels: Sequence, bg_class=()):
    """Collapse a per-frame label sequence into ordered segments (label, start, end).

    `end` is exclusive. Segments whose label is in `bg_class` are dropped. With the
    Assembly101 default `bg_class=()` nothing is dropped.
    """
    labels, starts, ends = [], [], []
    if len(frame_wise_labels) == 0:
        return labels, starts, ends
    last_label = frame_wise_labels[0]
    if last_label not in bg_class:
        labels.append(last_label)
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(len(frame_wise_labels))
    return labels, starts, ends


def levenstein(p, y, norm: bool = False) -> float:
    """Edit distance between two segment-label sequences (ins/del/sub cost 1)."""
    m_row, n_col = len(p), len(y)
    D = np.zeros((m_row + 1, n_col + 1), dtype=float)
    D[:, 0] = np.arange(m_row + 1)
    D[0, :] = np.arange(n_col + 1)
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        return (1.0 - D[-1, -1] / max(m_row, n_col, 1)) * 100.0
    return float(D[-1, -1])


def edit_score(recognized, ground_truth, norm: bool = True, bg_class=()) -> float:
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap: float, bg_class=()):
    """Return (tp, fp, fn) segment counts at one IoU threshold. Each GT segment can
    be matched at most once (greedy over predicted segments, argmax GT)."""
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0.0
    fp = 0.0
    hits = np.zeros(len(y_label))
    y_start = np.array(y_start, dtype=float)
    y_end = np.array(y_end, dtype=float)

    for j in range(len(p_label)):
        if len(y_label) == 0:
            fp += 1
            continue
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        label_match = np.array([p_label[j] == y_label[x] for x in range(len(y_label))], dtype=float)
        IoU = (intersection / union) * label_match
        idx = int(np.array(IoU).argmax())
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - float(np.sum(hits))
    return tp, fp, fn


class TASMetrics:
    """Accumulate per-video predictions, then report the dataset-level numbers with
    the reference's exact aggregation schemes. Feed one (pred, gt) per video."""

    def __init__(self, overlaps=(0.1, 0.25, 0.5), bg_class=()):
        self.overlaps = tuple(overlaps)
        self.bg_class = tuple(bg_class)
        self.n_correct = 0
        self.n_frames = 0
        self.edit_sum = 0.0
        self.n_videos = 0
        self.tp = {o: 0.0 for o in self.overlaps}
        self.fp = {o: 0.0 for o in self.overlaps}
        self.fn = {o: 0.0 for o in self.overlaps}

    def add(self, pred_frames: Sequence, gt_frames: Sequence) -> None:
        if len(pred_frames) != len(gt_frames):
            raise ValueError(f"pred/gt length mismatch: {len(pred_frames)} vs {len(gt_frames)}")
        self.n_frames += len(gt_frames)
        self.n_correct += int(sum(1 for p, g in zip(pred_frames, gt_frames) if p == g))
        self.edit_sum += edit_score(pred_frames, gt_frames, norm=True, bg_class=self.bg_class)
        self.n_videos += 1
        for o in self.overlaps:
            tp, fp, fn = f_score(pred_frames, gt_frames, o, self.bg_class)
            self.tp[o] += tp
            self.fp[o] += fp
            self.fn[o] += fn

    def summary(self) -> dict:
        mof = 100.0 * self.n_correct / max(self.n_frames, 1)
        edit = self.edit_sum / max(self.n_videos, 1)
        f1 = {}
        for o in self.overlaps:
            tp, fp, fn = self.tp[o], self.fp[o], self.fn[o]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1[o] = float(np.nan_to_num(2.0 * precision * recall / (precision + recall))) * 100.0 \
                if (precision + recall) > 0 else 0.0
        return {"MoF": mof, "Edit": edit,
                "F1": {o: f1[o] for o in self.overlaps},
                "n_videos": self.n_videos, "n_frames": self.n_frames}
