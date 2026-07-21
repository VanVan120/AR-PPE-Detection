"""Evaluate a trained TAS model on a fold and print MoF / Edit / F1@{10,25,50}.

Predicts at chunk resolution, upsamples to per-frame, and scores with the reference
metrics against the full-resolution GT.
"""
from __future__ import annotations

import numpy as np
import torch

from .metrics import TASMetrics
from .postprocess import upsample_predictions


@torch.no_grad()
def evaluate_model(model, ds, device: str = "cpu", chunk_size=None,
                   overlaps=(0.1, 0.25, 0.5), dump_dir: str = "") -> dict:
    # Derive the pooling stride from the dataset so the prediction->frame upsample can
    # never drift out of sync with how the features were pooled (a silent-accuracy bug).
    if chunk_size is None:
        chunk_size = getattr(ds, "chunk_size", 20)
    model.eval()
    metrics = TASMetrics(overlaps=overlaps, bg_class=())
    for i in range(len(ds)):
        x, _y, vid, orig_len = ds[i]
        x = x.unsqueeze(0).to(device)                      # (1, D, T')
        outputs = model(x)                                 # (S, 1, C, T')
        pooled_pred = outputs[-1, 0].argmax(0).cpu().numpy()   # (T',)
        pred_frames = upsample_predictions(pooled_pred, orig_len, chunk_size)
        gt_frames = np.asarray(ds.gt_frames_for(vid))
        metrics.add(pred_frames.tolist(), gt_frames.tolist())
        if dump_dir:
            import os
            os.makedirs(dump_dir, exist_ok=True)
            np.savetxt(os.path.join(dump_dir, f"{vid.replace('/', '_')}.txt"),
                       pred_frames, fmt="%d")
    return metrics.summary()


def format_summary(s: dict) -> str:
    f1 = s["F1"]
    keys = sorted(f1)
    f1s = " / ".join(f"{f1[k]:.1f}" for k in keys)
    return (f"MoF {s['MoF']:.1f}   Edit {s['Edit']:.1f}   "
            f"F1@{'/'.join(str(int(k*100)) for k in keys)} {f1s}   "
            f"(n_videos={s['n_videos']})")


def main(argv=None) -> int:
    import argparse
    import os
    import pickle

    from .dataset import (FEATURE_DIM, NUM_CLASSES, LmdbFeatureStore, load_actions_csv,
                          load_gt_frames, video_ids_for_fold)
    from .model import MSTCN
    from .torch_dataset import TASDataset

    ap = argparse.ArgumentParser(description="Evaluate an MS-TCN TAS model on Assembly101")
    ap.add_argument("--data-root", default="phase3_activity/data")
    ap.add_argument("--view", default="C10095_rgb")
    ap.add_argument("--fold", default="val", choices=["train", "val"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--num-stages", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=10)
    ap.add_argument("--num-f-maps", type=int, default=64)
    ap.add_argument("--dump-dir", default="")
    args = ap.parse_args(argv)

    ann = os.path.join(args.data_root, "coarse-annotations")
    actions_dict, _ = load_actions_csv(os.path.join(ann, "actions.csv"))
    with open(os.path.join(args.data_root, "statistic_input.pkl"), "rb") as fh:
        statistic = pickle.load(fh)
    ids = video_ids_for_fold(os.path.join(ann, "coarse_splits"), args.fold)
    gt = load_gt_frames(os.path.join(ann, "coarse_labels"), ids, actions_dict)
    store = LmdbFeatureStore(os.path.join(args.data_root, "TSM_features"))
    ds = TASDataset(ids, args.view, store, statistic, gt, args.chunk_size, args.max_frames)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSTCN(args.num_stages, args.num_layers, args.num_f_maps,
                  dim=FEATURE_DIM, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    print(f"Evaluating on {len(ds)} videos ({args.fold}, view {args.view})...")
    print(format_summary(evaluate_model(model, ds, device, args.chunk_size, dump_dir=args.dump_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
