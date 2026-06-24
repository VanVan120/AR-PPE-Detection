#!/usr/bin/env python3
"""Track A — fine-tune a closed-vocabulary detector on the dataset's own labels.

The validation prototype showed zero-shot perception is not good enough for
construction PPE compliance: YOLO-World detects Person well but collapses on the
PPE classes and scores ~0 on the safety-critical violation classes (NO-Hardhat /
NO-Safety Vest), because open-vocabulary prompts can't ground the *absence* of an
object. This script trains a standard YOLO detector on the labelled train+valid
splits to fix exactly that, then the same evaluation harness (compare_models.py)
scores it head-to-head against the zero-shot baseline on the held-out test split.

    python train.py                       # uses config.yaml knobs
    python train.py --epochs 100 --batch 8

On completion the best checkpoint is copied to `finetuned_model` (config), so
`detector_backend: finetuned` and compare_models.py pick it up automatically.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import yaml

from src.config import load_config, validate_config, Config


def _resolved_data_yaml(cfg: Config) -> str:
    """Write a training data.yaml with ABSOLUTE, verified split paths.

    The Roboflow export's data.yaml uses relative paths ('../train/images') that
    don't resolve from the dataset dir, so ultralytics can't find the splits.
    Rewrite it with an absolute `path` + correct split sub-paths and the class
    names read from the original.
    """
    base = os.path.abspath(cfg.dataset_dir)
    splits = {"train": "train/images", "val": "valid/images", "test": "test/images"}
    for key, sub in splits.items():
        if not os.path.isdir(os.path.join(base, sub)):
            raise FileNotFoundError(f"expected split dir not found: {os.path.join(base, sub)}")
    spec = {
        "path": base,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "nc": len(cfg.dataset_classes),
        "names": cfg.dataset_classes,
    }
    out = os.path.join(os.path.abspath(cfg.output_dir), "train", "data_resolved.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune a detector on the dataset (Track A)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--base-model", default=None, help="override finetune_base_model")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--name", default="finetune", help="ultralytics run name")
    p.add_argument("--out", default=None, help="destination for best.pt (default: config finetuned_model)")
    p.add_argument("--dataset-dir", default=None,
                   help="train on a different dataset dir (its data.yaml defines the classes); "
                        "default: config dataset_dir")
    return p.parse_args(argv)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)
    cfg: Config = load_config(args.config)

    # Optionally train on a different dataset: point at its dir and read ITS
    # classes from its own data.yaml (so e.g. a larger PPE dataset with different
    # class names trains correctly without touching config.yaml).
    if args.dataset_dir:
        from src.config import _read_dataset_classes
        cfg.dataset_dir = args.dataset_dir
        if not os.path.isfile(cfg.data_yaml_path):
            print(f"Cannot train — no data.yaml in {args.dataset_dir}", file=sys.stderr)
            return 1
        cfg.dataset_classes = _read_dataset_classes(cfg.data_yaml_path)
        if not cfg.dataset_classes:
            print(f"Cannot train — no class names in {cfg.data_yaml_path}", file=sys.stderr)
            return 1
    else:
        # The detector trains on the dataset's own data.yaml; make sure it is there.
        errors = [i for i in validate_config(cfg) if i.level == "error"
                  and ("data.yaml" in i.message or "dataset_dir" in i.message)]
        if errors:
            print("Cannot train — dataset problem:", file=sys.stderr)
            for i in errors:
                print(f"  {i}", file=sys.stderr)
            return 1

    base_model = args.base_model or cfg.finetune_base_model
    epochs = args.epochs or cfg.finetune_epochs
    imgsz = args.imgsz or cfg.finetune_imgsz
    batch = args.batch or cfg.finetune_batch
    patience = args.patience if args.patience is not None else cfg.finetune_patience
    device = cfg.device

    data_yaml = _resolved_data_yaml(cfg)
    project = os.path.abspath(os.path.join(cfg.output_dir, "train"))

    print("=" * 64)
    print(" Fine-tuning a detector on the dataset (Track A)")
    print("=" * 64)
    print(f"  base model : {base_model}")
    print(f"  data.yaml  : {data_yaml}")
    print(f"  classes    : {len(cfg.dataset_classes)}  (scored: {cfg.scored_classes})")
    print(f"  epochs     : {epochs}   imgsz: {imgsz}   batch: {batch}   patience: {patience}")
    print(f"  device     : {device}")
    print("-" * 64)

    try:
        from ultralytics import YOLO
    except Exception as e:
        print(f"[FAIL] ultralytics not available: {e}", file=sys.stderr)
        return 1

    try:
        model = YOLO(base_model)
        # Ultralytics writes runs under {project}/{name}; best.pt lands in weights/.
        model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            patience=patience,
            device=device,
            project=project,
            name=args.name,
            exist_ok=True,
            verbose=True,
            plots=True,
        )
    except Exception as e:
        msg = str(e)
        hint = ""
        if "out of memory" in msg.lower() or "cuda" in msg.lower():
            hint = (f"\n  Hint: the GPU may be out of memory — lower finetune_batch "
                    f"(currently {batch}) or finetune_imgsz (currently {imgsz}) in config.yaml, "
                    "or set device: cpu.")
        print(f"[FAIL] training failed: {e}{hint}", file=sys.stderr)
        return 1

    best = os.path.join(project, args.name, "weights", "best.pt")
    if not os.path.isfile(best):
        print(f"[FAIL] training finished but best.pt not found at {best}", file=sys.stderr)
        return 1

    dest = os.path.abspath(args.out or cfg.finetuned_model)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copy2(best, dest)
    print("-" * 64)
    print(f"[ ok ] best checkpoint -> {dest}")
    print("Next:")
    print("  python compare_models.py        # zero-shot vs fine-tuned on test")
    print("  (or set detector_backend: finetuned in config.yaml to run the full prototype on it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
