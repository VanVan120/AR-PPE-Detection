#!/usr/bin/env python3
"""Track B — squeeze zero-shot YOLO-World with per-class threshold tuning.

Picks the F1-optimal confidence threshold *per class* on the validation split and
reports the resulting before/after on the held-out test split. No training: this
only re-chooses the operating point on existing zero-shot detections, so it cannot
help classes the detector never detects (AP@50 ~ 0). Those need Track A.

    python tune.py                 # tune on 'valid', report on 'test'
    python tune.py --val-split valid --test-split test
    python tune.py --refresh       # ignore cached detections and re-run inference

Outputs (under outputs/tuning/):
    thresholds.json   per-class F1-optimal thresholds (paste into config.yaml)
    tuning.json       full curves + baseline-vs-tuned comparison on test
    pr_<class>.png     precision / recall / F1 vs confidence (test curve)
    tuning.html       human-readable report
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.config import load_config, validate_config, Config
from src.detector_yolo import Detection
from src.loader import load_split, Sample
from src import tune


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-class threshold tuning for YOLO-World")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--val-split", default="valid", help="split to tune thresholds on")
    p.add_argument("--test-split", default="test", help="split to report before/after on")
    p.add_argument("--backend", default="yolo_world", choices=["yolo_world", "finetuned"],
                   help="which detector to tune (default: zero-shot yolo_world)")
    p.add_argument("--limit", type=int, default=0, help="cap images per split (0 = all)")
    p.add_argument("--refresh", action="store_true", help="ignore cached detections")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Detection caching — inference is the slow part; cache it so iterating on the
# analysis/report is instant.
# ---------------------------------------------------------------------------
def run_split(detector, samples: list[Sample], cache_path: str, refresh: bool) -> list[list[Detection]]:
    names = [s.name for s in samples]
    if not refresh and os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path, "r", encoding="utf-8"))
            if cached.get("names") == names:
                print(f"  using cached detections: {cache_path}")
                return [[Detection(**d) for d in img] for img in cached["dets"]]
        except Exception:
            pass

    all_dets: list[list[Detection]] = []
    for i, s in enumerate(samples):
        try:
            img = s.read_image()
            dets = detector.detect(img)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {s.name}: failed ({e})")
            dets = []
        all_dets.append(dets)
        if (i + 1) % 20 == 0 or i + 1 == len(samples):
            print(f"  [{i+1}/{len(samples)}] detected")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump({"names": names, "dets": [[d.to_dict() for d in img] for img in all_dets]},
              open(cache_path, "w", encoding="utf-8"))
    return all_dets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_curve(curve: tune.ClassCurve, global_thr: float, tuned_thr, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=110)
    if curve.conf:
        ax.plot(curve.conf, curve.precision, label="precision", color="#2b6cb0", lw=1.6)
        ax.plot(curve.conf, curve.recall, label="recall", color="#c05621", lw=1.6)
        ax.plot(curve.conf, curve.f1, label="F1", color="#2f855a", lw=1.8)
    ax.axvline(global_thr, color="#a0aec0", ls="--", lw=1.1, label=f"global {global_thr:g}")
    if tuned_thr is not None:
        ax.axvline(tuned_thr, color="#000000", ls=":", lw=1.3, label=f"tuned {tuned_thr:.2f}")
    ax.set_xlabel("confidence threshold")
    ax.set_ylabel("score")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 1)
    ax.set_title(f"{curve.name}  (npos={curve.npos})", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def _delta(after, before):
    if after is None or before is None:
        return ""
    d = (after - before) * 100
    sign = "+" if d >= 0 else ""
    color = "#1a7f37" if d > 0.05 else ("#b42318" if d < -0.05 else "#666")
    return f"<span style='color:{color}'>{sign}{d:.1f}</span>"


def write_html(path: str, classes: list[str], rows: dict, macro: dict,
               png_names: dict, meta: dict) -> None:
    def tr(name):
        b, a = rows[name]["baseline"], rows[name]["tuned"]
        return (
            f"<tr><td><b>{name}</b></td>"
            f"<td>{b['threshold']:g}</td>"
            f"<td>{_pct(b['precision'])}</td><td>{_pct(b['recall'])}</td><td>{_pct(b['f1'])}</td>"
            f"<td class=sep>{a['threshold']:g}</td>"
            f"<td>{_pct(a['precision'])}</td><td>{_pct(a['recall'])}</td>"
            f"<td><b>{_pct(a['f1'])}</b> {_delta(a['f1'], b['f1'])}</td>"
            f"<td>{b['support']}</td></tr>"
        )
    body_rows = "\n".join(tr(c) for c in classes)
    mb, ma = macro["baseline"], macro["tuned"]
    macro_row = (
        f"<tr class=macro><td><b>macro</b></td><td>{mb['threshold']}</td>"
        f"<td>{_pct(mb['precision'])}</td><td>{_pct(mb['recall'])}</td><td>{_pct(mb['f1'])}</td>"
        f"<td class=sep>tuned</td><td>{_pct(ma['precision'])}</td><td>{_pct(ma['recall'])}</td>"
        f"<td><b>{_pct(ma['f1'])}</b> {_delta(ma['f1'], mb['f1'])}</td><td></td></tr>"
    )
    plots = "\n".join(
        f"<figure><img src='{png_names[c]}' alt='{c} PR curve'><figcaption>{c}</figcaption></figure>"
        for c in classes if c in png_names
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Threshold tuning — Track B</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#1a202c;background:#fafbfc}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
 .hint{{background:#eef4ff;border:1px solid #d6e2ff;border-radius:8px;padding:10px 12px;color:#3a4a6b}}
 table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;font-size:13px}}
 th,td{{padding:6px 9px;text-align:right;border-bottom:1px solid #edf0f4}}
 th:first-child,td:first-child{{text-align:left}}
 thead th{{background:#eef1f5}} td.sep,th.sep{{border-left:2px solid #cbd5e0}}
 tr.macro td{{background:#f0f3f7;font-weight:600;border-top:2px solid #cbd5e0}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-top:12px}}
 figure{{margin:0;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px}}
 figure img{{width:100%;height:auto}} figcaption{{text-align:center;font-size:12px;color:#555;margin-top:4px}}
</style></head><body>
<h1>Per-class threshold tuning — Track B (zero-shot, no training)</h1>
<p class="hint">Thresholds chosen to maximise F1 on the <b>{meta['val_split']}</b> split
({meta['n_val']} images), reported here on the held-out <b>{meta['test_split']}</b> split
({meta['n_test']} images). Model: {meta['yolo_model']} · IoU {meta['iou']}.
The baseline is the uniform global threshold ({meta['global']:g}).</p>
<h2>Operating point: baseline vs tuned (on {meta['test_split']})</h2>
<table>
<thead><tr>
 <th rowspan=2>class</th><th colspan=4>baseline @ {meta['global']:g}</th>
 <th class=sep colspan=4>tuned (per-class)</th><th rowspan=2>GT</th></tr>
 <tr><th>thr</th><th>P</th><th>R</th><th>F1</th>
 <th class=sep>thr</th><th>P</th><th>R</th><th>F1 Δpts</th></tr></thead>
<tbody>
{body_rows}
{macro_row}
</tbody></table>
<h2>Precision / recall / F1 vs confidence (held-out test curve)</h2>
<div class="grid">
{plots}
</div>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)
    cfg: Config = load_config(args.config)

    errors = [i for i in validate_config(cfg) if i.level == "error"]
    if errors:
        print("Cannot tune — config/dataset problems:", file=sys.stderr)
        for i in errors:
            print(f"  {i}", file=sys.stderr)
        return 1
    scored = cfg.scored_classes
    if not scored:
        print("No scored classes — nothing to tune.", file=sys.stderr)
        return 1

    backend = args.backend
    # Backend-tagged output dir + caches so tuning the fine-tuned model never
    # clobbers the zero-shot artifacts (outputs/tuning/dets_*.json is reused by
    # compare_models.py for the zero-shot baseline).
    out_dir = os.path.join(cfg.output_dir, "tuning") if backend == "yolo_world" \
        else os.path.join(cfg.output_dir, "tuning", backend)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading splits: tune on '{args.val_split}', report on '{args.test_split}'")
    val = load_split(cfg, args.val_split)
    test = load_split(cfg, args.test_split)
    if args.limit > 0:
        val, test = val[:args.limit], test[:args.limit]
    print(f"  {len(val)} {args.val_split} images, {len(test)} {args.test_split} images")

    if backend == "finetuned":
        if not os.path.isfile(cfg.finetuned_model):
            print(f"Fine-tuned model not found: {cfg.finetuned_model} — run train.py first.", file=sys.stderr)
            return 1
        from src.detector_finetuned import FinetunedDetector
        model_label = cfg.finetuned_model
        print(f"Loading fine-tuned detector '{model_label}' on {cfg.device}...")
        detector = FinetunedDetector(cfg)
    else:
        from src.detector_yolo import YoloWorldDetector
        model_label = cfg.yolo_model
        print(f"Loading YOLO-World '{model_label}' on {cfg.device}...")
        detector = YoloWorldDetector(cfg)

    print(f"Running detector on '{args.val_split}' (conf floor {cfg.yolo_conf_floor})...")
    val_preds = run_split(detector, val, os.path.join(out_dir, f"dets_{args.val_split}.json"), args.refresh)
    print(f"Running detector on '{args.test_split}'...")
    test_preds = run_split(detector, test, os.path.join(out_dir, f"dets_{args.test_split}.json"), args.refresh)

    val_gt = [s.gt for s in val]
    test_gt = [s.gt for s in test]
    iou = cfg.iou_threshold
    global_thr = cfg.confidence_threshold

    thresholds: dict[str, float] = {}
    rows: dict[str, dict] = {}
    test_curves: dict[str, tune.ClassCurve] = {}
    png_names: dict[str, str] = {}

    # macro accumulators (zero_division=0 over classes with GT support on test)
    macro = {"baseline": {"p": [], "r": [], "f1": []}, "tuned": {"p": [], "r": [], "f1": []}}

    for name in scored:
        vcurve = tune.class_curve(name, val_preds, val_gt, cfg.dataset_classes, iou)
        # Only trust a tuned threshold if tuning found a *positive-F1* operating
        # point on valid. For classes the detector can't detect (best F1 == 0,
        # e.g. NO-Hardhat), the F1-argmax is an arbitrary junk confidence — fall
        # back to the global threshold so we don't suppress their display boxes.
        if vcurve.best_threshold is not None and (vcurve.best_f1 or 0) > 0:
            tuned_thr = vcurve.best_threshold
        else:
            tuned_thr = global_thr
        thresholds[name] = round(float(tuned_thr), 4)

        baseline = tune.operating_point(name, test_preds, test_gt, cfg.dataset_classes, iou, global_thr)
        tuned = tune.operating_point(name, test_preds, test_gt, cfg.dataset_classes, iou, tuned_thr)
        rows[name] = {"baseline": baseline, "tuned": tuned, "val_best_f1": vcurve.best_f1}

        tcurve = tune.class_curve(name, test_preds, test_gt, cfg.dataset_classes, iou)
        test_curves[name] = tcurve
        png = f"pr_{_safe(name)}.png"
        plot_curve(tcurve, global_thr, tuned_thr, os.path.join(out_dir, png))
        png_names[name] = png

        if baseline["support"] > 0:
            for key, op in (("baseline", baseline), ("tuned", tuned)):
                macro[key]["p"].append(op["precision"] or 0.0)
                macro[key]["r"].append(op["recall"] or 0.0)
                macro[key]["f1"].append(op["f1"] or 0.0)

        print(f"  {name:<16} tuned thr={tuned_thr:.3f}  "
              f"F1 {_pct(baseline['f1'])} -> {_pct(tuned['f1'])}")

    def _avg(xs):
        return sum(xs) / len(xs) if xs else None
    macro_summary = {
        "baseline": {"threshold": f"{global_thr:g}", "precision": _avg(macro["baseline"]["p"]),
                     "recall": _avg(macro["baseline"]["r"]), "f1": _avg(macro["baseline"]["f1"])},
        "tuned": {"threshold": "per-class", "precision": _avg(macro["tuned"]["p"]),
                  "recall": _avg(macro["tuned"]["r"]), "f1": _avg(macro["tuned"]["f1"])},
    }

    meta = {
        "val_split": args.val_split, "test_split": args.test_split,
        "n_val": len(val), "n_test": len(test), "yolo_model": model_label, "backend": backend,
        "iou": iou, "global": global_thr,
    }

    # --- write artifacts ----------------------------------------------------
    json.dump(thresholds, open(os.path.join(out_dir, "thresholds.json"), "w", encoding="utf-8"), indent=2)
    json.dump({
        "meta": meta,
        "thresholds": thresholds,
        "comparison": rows,
        "macro": macro_summary,
        "val_curves": {n: tune.class_curve(n, val_preds, val_gt, cfg.dataset_classes, iou).to_dict()
                       for n in scored},
        "test_curves": {n: c.to_dict() for n, c in test_curves.items()},
    }, open(os.path.join(out_dir, "tuning.json"), "w", encoding="utf-8"), indent=2)
    write_html(os.path.join(out_dir, "tuning.html"), scored, rows, macro_summary, png_names, meta)

    cfg_key = "finetuned_per_class_thresholds" if backend == "finetuned" else "per_class_thresholds"
    print("\n" + "=" * 64)
    print(f" Threshold tuning ({backend}) — macro F1 on held-out test")
    print("=" * 64)
    print(f"  baseline (global {global_thr:g}):  {_pct(macro_summary['baseline']['f1'])}")
    print(f"  tuned (per-class):       {_pct(macro_summary['tuned']['f1'])}")
    print(f"\n  -> paste thresholds.json into config.yaml under '{cfg_key}:' to apply.")
    print("Wrote:")
    for f in ("thresholds.json", "tuning.json", "tuning.html"):
        print(f"  {os.path.join(out_dir, f)}")
    print(f"  {out_dir}/pr_*.png")
    return 0


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


if __name__ == "__main__":
    raise SystemExit(main())
