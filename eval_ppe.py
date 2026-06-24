#!/usr/bin/env python3
"""Evaluate a trained detector on a dataset's held-out test split.

Uses ultralytics' native validation (the authoritative source for per-class
precision / recall / F1 / mAP@50 / mAP@50-95), so the numbers are exactly the
standard detection metrics — no custom scoring to second-guess. Reports per class
and flags which metrics clear the 90% bar.

    python eval_ppe.py --model models/ppe_s.pt --dataset-dir data/ppe_download
    python eval_ppe.py --model models/ppe_s.pt --dataset-dir data/ppe_download --tta
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.config import load_config, _read_dataset_classes
import train  # reuse _resolved_data_yaml


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate a detector on a dataset's test split")
    p.add_argument("--model", required=True, help="path to a trained .pt model")
    p.add_argument("--dataset-dir", required=True, help="dataset dir (with data.yaml + test/)")
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--tta", action="store_true", help="test-time augmentation")
    p.add_argument("--out", default=None, help="optional JSON output path")
    return p.parse_args(argv)


def _arr(x):
    try:
        return list(x)
    except Exception:
        return []


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)
    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        return 1

    cfg = load_config()
    cfg.dataset_dir = args.dataset_dir
    cfg.dataset_classes = _read_dataset_classes(cfg.data_yaml_path)
    data_yaml = train._resolved_data_yaml(cfg)

    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"Evaluating '{args.model}' on '{args.split}' split of {args.dataset_dir} "
          f"(imgsz {args.imgsz}{', TTA' if args.tta else ''})...")
    res = model.val(data=data_yaml, split=args.split, imgsz=args.imgsz,
                    augment=args.tta, device=cfg.device, verbose=False)

    box = res.box
    names = res.names if hasattr(res, "names") else model.names
    idx = _arr(getattr(box, "ap_class_index", []))
    p = _arr(getattr(box, "p", []))
    r = _arr(getattr(box, "r", []))
    f1 = _arr(getattr(box, "f1", []))
    ap50 = _arr(getattr(box, "ap50", []))
    ap = _arr(getattr(box, "ap", []))   # AP@50-95 per class

    per_class = {}
    print("\n" + "=" * 78)
    print(f" Per-class metrics on '{args.split}' ({args.dataset_dir})")
    print("=" * 78)
    print(f"{'class':<14}{'precision':>11}{'recall':>10}{'F1':>9}{'mAP@50':>10}{'mAP50-95':>11}{'  90%?':>8}")
    for j, ci in enumerate(idx):
        name = names.get(int(ci), str(ci)) if isinstance(names, dict) else names[int(ci)]
        row = {
            "precision": float(p[j]) if j < len(p) else None,
            "recall": float(r[j]) if j < len(r) else None,
            "f1": float(f1[j]) if j < len(f1) else None,
            "map50": float(ap50[j]) if j < len(ap50) else None,
            "map50_95": float(ap[j]) if j < len(ap) else None,
        }
        per_class[name] = row
        hits = all((row[k] is not None and row[k] >= 0.90) for k in ("precision", "recall", "f1", "map50"))
        mark = " YES" if hits else ""
        def pc(x): return "  —  " if x is None else f"{x*100:6.1f}%"
        print(f"{name:<14}{pc(row['precision']):>11}{pc(row['recall']):>10}{pc(row['f1']):>9}"
              f"{pc(row['map50']):>10}{pc(row['map50_95']):>11}{mark:>8}")

    overall = {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }
    of1 = (2 * overall["precision"] * overall["recall"] /
           (overall["precision"] + overall["recall"])) if (overall["precision"] + overall["recall"]) else 0.0
    overall["f1"] = of1
    print("-" * 78)
    print(f"{'ALL (mean)':<14}{overall['precision']*100:10.1f}%{overall['recall']*100:9.1f}%"
          f"{of1*100:8.1f}%{overall['map50']*100:9.1f}%{overall['map50_95']*100:10.1f}%")

    n90 = sum(1 for v in per_class.values()
              if all((v[k] is not None and v[k] >= 0.90) for k in ("precision", "recall", "f1", "map50")))
    print(f"\nClasses clearing 90% on P/R/F1/mAP@50: {n90}/{len(per_class)}")
    print(f"Overall mAP@50 = {overall['map50']*100:.1f}%   mean F1 = {of1*100:.1f}%"
          f"{'   (TTA)' if args.tta else ''}")

    out = {"model": args.model, "split": args.split, "tta": args.tta,
           "per_class": per_class, "overall": overall}
    out_path = args.out or os.path.join(cfg.output_dir, "ppe_eval.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
