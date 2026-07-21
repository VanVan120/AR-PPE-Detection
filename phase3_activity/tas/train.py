"""Train the MS-TCN TAS baseline on Assembly101 TSM features.

Batch size 1 over variable-length videos (the standard, memory-safe TAS setup), the
MS-TCN objective (per-stage CE + temporal smoothing), Adam + StepLR, early stopping
on validation MoF, best-checkpoint saving. `train_model` is dependency-injected (any
torch datasets) so the whole loop is smoke-testable on a synthetic fixture; `main`
wires the real Assembly101 paths.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

from .evaluate import evaluate_model, format_summary
from .model import MSTCN, MSTCNLoss, save_checkpoint


def train_model(model, train_ds, val_ds, device: str = "cpu", epochs: int = 200,
                lr: float = 1e-4, weight_decay: float = 1e-4, patience: int = 20,
                lr_step: int = 200, lr_gamma: float = 0.5, ckpt_path: str = "",
                seed: int = 0, log_every: int = 10, verbose: bool = True) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = model.to(device)
    criterion = MSTCNLoss(num_classes=model.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_gamma)

    best_score = -1.0
    best_summary: dict = {}
    since_improved = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_ds))
        epoch_loss = 0.0
        for idx in order:
            x, y, _vid, _orig = train_ds[int(idx)]
            x = x.unsqueeze(0).to(device)               # (1, D, T')
            y = y.unsqueeze(0).to(device)               # (1, T')
            optimizer.zero_grad()
            outputs = model(x)                          # (S, 1, C, T')
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        scheduler.step()

        val_summary = evaluate_model(model, val_ds, device)   # chunk_size derived from val_ds
        score = val_summary["MoF"]
        improved = score > best_score
        if improved:
            best_score = score
            best_summary = val_summary
            since_improved = 0
            if ckpt_path:
                save_checkpoint(model, ckpt_path)
        else:
            since_improved += 1

        if verbose and (epoch % log_every == 0 or epoch == 1 or improved):
            print(f"epoch {epoch:3d}  loss {epoch_loss / max(len(train_ds),1):.4f}  "
                  f"val {format_summary(val_summary)}{'  *' if improved else ''}")
        if since_improved >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (no val MoF gain in {patience})")
            break

    return best_summary


def main(argv=None) -> int:
    import argparse

    from .dataset import (FEATURE_DIM, NUM_CLASSES, LmdbFeatureStore, build_samples,
                          keep_present, load_actions_csv, video_ids_for_fold)
    from .torch_dataset import TASDataset

    ap = argparse.ArgumentParser(description="Train an MS-TCN TAS baseline on Assembly101")
    ap.add_argument("--data-root", default="phase3_activity/data")
    ap.add_argument("--features-root", default="", help="dir holding the per-view LMDBs "
                    "(default: data-root, since the download puts the view there)")
    ap.add_argument("--view", default="C10095_rgb")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--chunk-size", type=int, default=20)
    ap.add_argument("--max-frames", type=int, default=1200)
    ap.add_argument("--num-stages", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=10)
    ap.add_argument("--num-f-maps", type=int, default=64)
    ap.add_argument("--limit-train", type=int, default=0, help="cap #train samples (0=all)")
    ap.add_argument("--limit-val", type=int, default=0, help="cap #val samples (0=all)")
    ap.add_argument("--ckpt", default="phase3_activity/models/mstcn_best.pt")
    args = ap.parse_args(argv)

    ann = os.path.join(args.data_root, "coarse-annotations")
    actions_dict, _ = load_actions_csv(os.path.join(ann, "actions.csv"))
    splits = os.path.join(ann, "coarse_splits")
    labels = os.path.join(ann, "coarse_labels")
    features_root = args.features_root or args.data_root
    store = LmdbFeatureStore(features_root)

    train_samples = keep_present(store, args.view,
                                 build_samples(labels, video_ids_for_fold(splits, "train"), actions_dict))
    val_samples = keep_present(store, args.view,
                               build_samples(labels, video_ids_for_fold(splits, "val"), actions_dict))
    if args.limit_train:
        train_samples = train_samples[:args.limit_train]
    if args.limit_val:
        val_samples = val_samples[:args.limit_val]
    train_ds = TASDataset(train_samples, args.view, store, args.chunk_size, args.max_frames)
    val_ds = TASDataset(val_samples, args.view, store, args.chunk_size, args.max_frames)
    print(f"train samples: {len(train_ds)}   val samples: {len(val_ds)}  (view {args.view})")

    from .model import MSTCN
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MSTCN(args.num_stages, args.num_layers, args.num_f_maps,
                  dim=FEATURE_DIM, num_classes=NUM_CLASSES)
    best = train_model(model, train_ds, val_ds, device, epochs=args.epochs, lr=args.lr,
                       weight_decay=args.weight_decay, patience=args.patience,
                       ckpt_path=args.ckpt)
    print("\nbest val:", format_summary(best))
    print(f"checkpoint -> {args.ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
