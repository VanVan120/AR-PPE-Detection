"""Recover the `args.yaml` and `results.csv` that ultralytics embeds in a checkpoint.

`train.py` and the Kaggle notebooks wrote their run directories to machines that are gone
(`outputs/train/` never existed here; the final detector trained on Kaggle). But an
ultralytics checkpoint carries its own training record: `train_args` is the same mapping
that gets written to `args.yaml`, and `train_results` is the same table that gets written
to `results.csv`. So the run is recoverable from the weights alone, without Kaggle and
without re-training.

    python tools/ckpt_train_record.py best_refined.pt --out paper_materials/final_detector_train

Writes args_from_checkpoint.yaml, results_from_checkpoint.csv and summary.json. Prints the
fields a paper needs: base model, epochs requested, epochs actually recorded, imgsz, batch,
optimizer, patience, seed, best epoch, and the ultralytics version that trained it.

Note `torch.load(..., weights_only=False)` is required: an ultralytics checkpoint is a
pickle holding real objects. Only run this on a checkpoint you trust — here, the project's
own weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys


def fitness_weights(version):
    """The [mAP50, mAP50-95] weights `Metric.fitness` uses, for a given ultralytics version.

    This changed between versions, so a checkpoint has to be scored with the weights its OWN
    ultralytics used or the "best epoch" can come out wrong:

      8.4.x  : [0.0, 0.0, 0.0, 1.0] -> mAP@50-95 alone   (utils/metrics.py:120 in 8.4.75)
      older  : [0.0, 0.0, 0.1, 0.9] -> 0.1*mAP50 + 0.9*mAP50-95

    Unknown or missing version falls back to the 8.4 behaviour, since that is what this
    project pins; the fallback is reported by the caller rather than applied silently.
    """
    try:
        major, minor = (int(x) for x in str(version).split(".")[:2])
    except (TypeError, ValueError):
        return (0.0, 1.0), True                      # unknown -> assume 8.4, flag it
    if (major, minor) >= (8, 4):
        return (0.0, 1.0), False
    return (0.1, 0.9), False


def fitness(row, weights=(0.0, 1.0)):
    """Ultralytics' fitness — the value `best.pt` maximises — under `weights`."""
    m50, m5095 = row.get("metrics/mAP50(B)"), row.get("metrics/mAP50-95(B)")
    if m5095 is None:
        return None
    try:
        return weights[0] * float(m50 or 0.0) + weights[1] * float(m5095)
    except (TypeError, ValueError):
        return None


def as_rows(train_results):
    """`train_results` is a dict of column -> list. Turn it into a list of row dicts."""
    if not isinstance(train_results, dict) or not train_results:
        return [], []
    cols = list(train_results.keys())
    n = min(len(v) for v in train_results.values())
    return cols, [{c: train_results[c][i] for c in cols} for i in range(n)]


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Extract the embedded training record from a .pt")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True, help="directory to write the recovered files into")
    a = ap.parse_args(argv)

    if not os.path.isfile(a.checkpoint):
        print(f"checkpoint not found: {a.checkpoint}", file=sys.stderr)
        return 1

    import torch
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        print(f"{a.checkpoint} is not an ultralytics checkpoint dict", file=sys.stderr)
        return 1

    args = ckpt.get("train_args") or {}
    cols, rows = as_rows(ckpt.get("train_results") or {})
    os.makedirs(a.out, exist_ok=True)

    # args.yaml — ultralytics writes a plain mapping, so reproduce that shape
    args_path = os.path.join(a.out, "args_from_checkpoint.yaml")
    try:
        import yaml
        with open(args_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump({k: args[k] for k in sorted(args)}, fh, sort_keys=False,
                           allow_unicode=True, default_flow_style=False)
    except Exception:
        with open(args_path, "w", encoding="utf-8") as fh:
            for k in sorted(args):
                fh.write(f"{k}: {args[k]}\n")

    res_path = os.path.join(a.out, "results_from_checkpoint.csv")
    if rows:
        with open(res_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    # Score with the weights this checkpoint's OWN ultralytics used, not the one installed here.
    fit_w, guessed_version = fitness_weights(ckpt.get("version"))
    print(f"fitness weights for ultralytics {ckpt.get('version')!r}: "
          f"{fit_w[0]}*mAP50 + {fit_w[1]}*mAP50-95"
          + ("  [version unknown - assumed 8.4]" if guessed_version else ""))
    fits = [(fitness(r, fit_w), i) for i, r in enumerate(rows)]
    fits = [(f, i) for f, i in fits if f is not None]
    best_i = max(fits)[1] if fits else None

    def ep(row):
        for k in ("epoch", "                  epoch"):
            if k in row:
                return row[k]
        return None

    summary = {
        "checkpoint": a.checkpoint,
        "checkpoint_bytes": os.path.getsize(a.checkpoint),
        "ultralytics_version": ckpt.get("version"),
        "trained_date": str(ckpt.get("date")) if ckpt.get("date") is not None else None,
        "epoch_field": ckpt.get("epoch"),
        "best_fitness": (float(ckpt["best_fitness"])
                         if ckpt.get("best_fitness") is not None else None),
        "has_train_args": bool(args),
        "has_train_results": bool(rows),
        "requested": {k: args.get(k) for k in
                      ("model", "data", "epochs", "imgsz", "batch", "optimizer",
                       "patience", "seed", "device", "project", "name", "pretrained",
                       "lr0", "workers")},
        "results_columns": cols,
        "epochs_recorded": len(rows),
        "best_epoch_index_0based": best_i,
        "best_epoch_label": ep(rows[best_i]) if best_i is not None else None,
        "best_epoch_fitness": max(fits)[0] if fits else None,
        "best_epoch_row": rows[best_i] if best_i is not None else None,
        "last_epoch_row": rows[-1] if rows else None,
        "train_metrics": ckpt.get("train_metrics"),
    }
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"),
              indent=2, default=str)

    r = summary["requested"]
    print("=" * 74)
    print(f" {a.checkpoint}")
    print("=" * 74)
    print(f"  ultralytics      : {summary['ultralytics_version']}")
    print(f"  trained (date)   : {summary['trained_date']}")
    print(f"  base model       : {r['model']}")
    print(f"  data             : {r['data']}")
    print(f"  epochs requested : {r['epochs']}   imgsz {r['imgsz']}   batch {r['batch']}")
    print(f"  optimizer        : {r['optimizer']}   patience {r['patience']}   seed {r['seed']}")
    print(f"  epochs recorded  : {summary['epochs_recorded']}"
          f"{'  (no results table embedded)' if not rows else ''}")
    if summary["epochs_recorded"] and r["epochs"] is not None:
        early = summary["epochs_recorded"] < int(r["epochs"])
        print(f"  early stopping   : {'YES' if early else 'no'} "
              f"({summary['epochs_recorded']} run of {r['epochs']} requested)")
    print(f"  epoch field      : {summary['epoch_field']}   "
          f"best_fitness {summary['best_fitness']}"
          f"{'   (checkpoint was stripped)' if summary['epoch_field'] in (-1, None) else ''}")
    if best_i is not None:
        print(f"  best epoch       : {summary['best_epoch_label']} "
              f"(row {best_i + 1} of {len(rows)}), fitness {summary['best_epoch_fitness']:.5f}")
    print(f"\nWrote {args_path}")
    if rows:
        print(f"Wrote {res_path}")
    print(f"Wrote {os.path.join(a.out, 'summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
