"""Visualize a trained TAS model's step predictions vs ground truth as a timeline.

Runs the model on one (or more) Assembly101 annotation(s) and renders, over the
sequence's frames:
  * a **Ground truth** step ribbon,
  * a **Predicted** step ribbon (same colour per step, so a vertical colour-match
    means the model got that stretch right),
  * a green/red **Match** strip (green where predicted == GT frame),
and the per-sequence MoF / Edit / F1 in the title, with a legend of the steps.

    python -m phase3_activity.tas.visualize --ckpt phase3_activity/models/mstcn_best.pt
    python -m phase3_activity.tas.visualize --ckpt ... --fold val --index 0 --count 3

The `render_timeline` function is dependency-light (numpy + matplotlib) so it can be
unit-tested on synthetic labels without the model or dataset.
"""
from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")            # headless: render straight to a file
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .metrics import TASMetrics, get_labels_start_end_time
from .postprocess import upsample_predictions


def _palette(class_ids: Sequence[int]) -> Dict[int, tuple]:
    """A distinct, stable colour per class id (qualitative maps, hsv fallback)."""
    ids = sorted(set(int(c) for c in class_ids))
    k = len(ids)
    if k <= 20:
        cols = plt.cm.tab20(np.linspace(0, 1, 20))[:k]
    else:
        big = np.vstack([plt.cm.tab20(np.linspace(0, 1, 20)),
                         plt.cm.tab20b(np.linspace(0, 1, 20)),
                         plt.cm.tab20c(np.linspace(0, 1, 20))])
        cols = big[:k] if k <= len(big) else plt.cm.hsv(np.linspace(0, 1, k, endpoint=False))
    return {cid: tuple(cols[i]) for i, cid in enumerate(ids)}


def _draw_ribbon(ax, y: float, height: float, frame_labels: Sequence[int], colors: dict) -> None:
    labels, starts, ends = get_labels_start_end_time(list(frame_labels), bg_class=())
    for lab, s, e in zip(labels, starts, ends):
        ax.broken_barh([(s, e - s)], (y, height), facecolors=colors[int(lab)], edgecolors="none")


def _draw_match(ax, y: float, height: float, gt: np.ndarray, pred: np.ndarray) -> None:
    T = len(gt)
    match = gt == pred
    i = 0
    while i < T:
        j = i
        while j < T and match[j] == match[i]:
            j += 1
        ax.broken_barh([(i, j - i)], (y, height),
                       facecolors=("#2e7d32" if match[i] else "#c62828"), edgecolors="none")
        i = j


def render_timeline(gt: Sequence[int], pred: Sequence[int], name: str,
                    id_to_action: Dict[int, str], out_path: str, fps: int = 30) -> dict:
    """Render GT vs predicted step ribbons + match strip to `out_path`; return the
    per-sequence metrics summary. `len(pred)` must equal `len(gt)`."""
    gt = np.asarray(list(gt))
    pred = np.asarray(list(pred))
    if len(gt) != len(pred):
        raise ValueError(f"gt/pred length mismatch: {len(gt)} vs {len(pred)}")
    T = len(gt)
    present = sorted(set(gt.tolist()) | set(pred.tolist()))
    colors = _palette(present)

    metrics = TASMetrics(overlaps=(0.1, 0.25, 0.5), bg_class=())
    metrics.add(pred.tolist(), gt.tolist())
    s = metrics.summary()
    f1 = s["F1"]

    fig, ax = plt.subplots(figsize=(14, 3.4))
    _draw_ribbon(ax, 2.1, 0.8, gt, colors)       # Ground truth (top)
    _draw_ribbon(ax, 1.1, 0.8, pred, colors)     # Predicted (middle)
    _draw_match(ax, 0.1, 0.8, gt, pred)          # Match strip (bottom)

    ax.set_yticks([2.5, 1.5, 0.5])
    ax.set_yticklabels(["Ground truth", "Predicted", "Match"])
    ax.set_ylim(0, 3)
    ax.set_xlim(0, T)
    ax.set_xlabel(f"frame   (0–{T},  ≈{T / fps:.0f}s @ {fps} fps)")
    short = name[:-4] if name.endswith(".txt") else name
    if len(short) > 62:
        short = short[:59] + "..."
    ax.set_title(f"{short}\nMoF {s['MoF']:.1f}    Edit {s['Edit']:.1f}    "
                 f"F1@10/25/50 {f1[0.1]:.1f} / {f1[0.25]:.1f} / {f1[0.5]:.1f}",
                 fontsize=10)

    handles = [Patch(facecolor=colors[c], label=id_to_action.get(c, f"cls-{c}")) for c in present]
    ncol = 1 if len(handles) <= 14 else 2
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1.0), loc="upper left",
              fontsize=7, ncol=ncol, title="steps", title_fontsize=8, frameon=False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return s


def predict_frames(model, ds, i: int, device: str):
    """Run the model on sample i -> (gt_frames, pred_frames, name), frame-resolution."""
    import torch
    with torch.no_grad():
        x, _y, name, orig_len = ds[i]
        logits = model(x.unsqueeze(0).to(device))[-1, 0]        # (C, T')
        pooled_pred = logits.argmax(0).cpu().numpy()
    pred_frames = upsample_predictions(pooled_pred, orig_len, ds.chunk_size)
    gt_frames = ds.gt_frames_for(name)
    return gt_frames, pred_frames, name


def main(argv=None) -> int:
    import argparse

    import torch

    from .dataset import (LmdbFeatureStore, build_samples, keep_present,
                          load_actions_csv, video_ids_for_fold)
    from .model import load_model
    from .torch_dataset import TASDataset

    ap = argparse.ArgumentParser(description="Visualize TAS step predictions vs ground truth")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="phase3_activity/data")
    ap.add_argument("--features-root", default="")
    ap.add_argument("--view", default="C10095_rgb")
    ap.add_argument("--fold", default="val", choices=["train", "val"])
    ap.add_argument("--index", type=int, default=0, help="first sample (by position) to render")
    ap.add_argument("--count", type=int, default=1, help="how many consecutive samples to render")
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--out-dir", default="phase3_activity/outputs")
    args = ap.parse_args(argv)

    ann = os.path.join(args.data_root, "coarse-annotations")
    actions_dict, id_to_action = load_actions_csv(os.path.join(ann, "actions.csv"))
    features_root = args.features_root or args.data_root
    store = LmdbFeatureStore(features_root)
    names = video_ids_for_fold(os.path.join(ann, "coarse_splits"), args.fold)
    samples = keep_present(store, args.view,
                           build_samples(os.path.join(ann, "coarse_labels"), names, actions_dict))
    ds = TASDataset(samples, args.view, store, args.chunk_size, args.max_frames)
    if len(ds) == 0:
        print("no samples for this view/fold"); return 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    n = max(0, min(args.count, len(ds) - args.index))
    for k in range(n):
        i = args.index + k
        gt, pred, name = predict_frames(model, ds, i, device)
        out = os.path.join(args.out_dir, f"timeline_{args.fold}_{i:03d}.png")
        s = render_timeline(gt, pred, name, id_to_action, out)
        print(f"[{i:3d}] MoF {s['MoF']:5.1f}  Edit {s['Edit']:5.1f}  "
              f"F1@50 {s['F1'][0.5]:5.1f}  {name[:46]:46} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
