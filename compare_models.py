#!/usr/bin/env python3
"""Track A — zero-shot YOLO-World vs fine-tuned detector, head to head.

Scores both detectors on the held-out test split through the *same* evaluation
harness (src.evaluator.evaluate_yolo) so the numbers are directly comparable, and
writes a side-by-side report. mAP@50 is the fair headline number because it is
threshold-independent; per-class P/R/F1 are shown at each model's operating point
(zero-shot at its Track-B tuned thresholds, fine-tuned at the global threshold).

    python compare_models.py            # needs models/finetuned.pt (run train.py first)

Outputs (under outputs/comparison/):
    comparison.json     full metrics for both models
    comparison.html     side-by-side metrics table + example image pairs
    zs_*.jpg / ft_*.jpg annotated examples from each model
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

from src.config import load_config, validate_config, Config
from src.detector_yolo import Detection, annotate, save_image
from src.loader import load_split, select_review_sample, Sample
from src import evaluator


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Compare zero-shot vs fine-tuned detector")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--examples", type=int, default=8, help="annotated example pairs in the report")
    p.add_argument("--refresh", action="store_true", help="ignore cached detections")
    return p.parse_args(argv)


def _run_detector(detector, samples: list[Sample], cache_path: str, refresh: bool) -> list[list[Detection]]:
    names = [s.name for s in samples]
    if not refresh and os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path, "r", encoding="utf-8"))
            if cached.get("names") == names:
                print(f"  using cached: {cache_path}")
                return [[Detection(**d) for d in img] for img in cached["dets"]]
        except Exception:
            pass
    if detector is None:
        # Caller passed None expecting a valid cache hit, but the cache is
        # missing/stale/mismatched. Fail loudly instead of silently producing
        # all-empty detections (which would make the comparison meaningless).
        raise RuntimeError(
            f"no usable cached detections at {cache_path} and no detector to compute them — "
            "re-run with --refresh or ensure the cache matches the current split/limit")
    out = []
    for i, s in enumerate(samples):
        try:
            out.append(detector.detect(s.read_image()))
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] {s.name}: failed ({e})")
            out.append([])
        if (i + 1) % 20 == 0 or i + 1 == len(samples):
            print(f"  [{i+1}/{len(samples)}] detected")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump({"names": names, "dets": [[d.to_dict() for d in img] for img in out]},
              open(cache_path, "w", encoding="utf-8"))
    return out


def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


def _delta(a, b):
    if a is None or b is None:
        return ""
    d = (a - b) * 100
    color = "#1a7f37" if d > 0.05 else ("#b42318" if d < -0.05 else "#666")
    sign = "+" if d >= 0 else ""
    return f"<span style='color:{color}'>{sign}{d:.1f}</span>"


def write_html(path: str, scored, zs, ft, examples, meta):
    def row(name):
        z = zs["per_class"].get(name, {})
        f = ft["per_class"].get(name, {})
        return (
            f"<tr><td><b>{name}</b></td>"
            f"<td>{_pct(z.get('precision'))}</td><td>{_pct(z.get('recall'))}</td>"
            f"<td>{_pct(z.get('f1'))}</td><td>{_pct(z.get('ap50'))}</td>"
            f"<td class=sep>{_pct(f.get('precision'))}</td><td>{_pct(f.get('recall'))}</td>"
            f"<td><b>{_pct(f.get('f1'))}</b></td><td>{_pct(f.get('ap50'))} {_delta(f.get('ap50'), z.get('ap50'))}</td>"
            f"<td>{z.get('support', '')}</td></tr>"
        )
    rows = "\n".join(row(n) for n in scored)
    zo, fo = zs["overall"], ft["overall"]
    macro = (
        f"<tr class=macro><td><b>macro / mAP</b></td>"
        f"<td>{_pct(zo['precision_macro'])}</td><td>{_pct(zo['recall_macro'])}</td>"
        f"<td>{_pct(zo['f1_macro'])}</td><td>{_pct(zo['mAP50'])}</td>"
        f"<td class=sep>{_pct(fo['precision_macro'])}</td><td>{_pct(fo['recall_macro'])}</td>"
        f"<td><b>{_pct(fo['f1_macro'])}</b></td>"
        f"<td><b>{_pct(fo['mAP50'])}</b> {_delta(fo['mAP50'], zo['mAP50'])}</td><td></td></tr>"
    )
    cards = "\n".join(
        f"<div class='pair'><figure><img src='{e['zs']}'><figcaption>zero-shot</figcaption></figure>"
        f"<figure><img src='{e['ft']}'><figcaption>fine-tuned</figcaption></figure>"
        f"<div class='cap'>{e['name']}</div></div>"
        for e in examples
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Zero-shot vs fine-tuned — Track A</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#1a202c;background:#fafbfc}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:26px}}
 .hint{{background:#eef4ff;border:1px solid #d6e2ff;border-radius:8px;padding:10px 12px;color:#3a4a6b}}
 table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;font-size:13px}}
 th,td{{padding:6px 9px;text-align:right;border-bottom:1px solid #edf0f4}}
 th:first-child,td:first-child{{text-align:left}}
 thead th{{background:#eef1f5}} td.sep,th.sep{{border-left:2px solid #cbd5e0}}
 tr.macro td{{background:#f0f3f7;font-weight:700;border-top:2px solid #cbd5e0}}
 .pair{{display:inline-block;width:48%;vertical-align:top;margin:0 1% 14px;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:8px}}
 .pair figure{{display:inline-block;width:49%;margin:0}} .pair img{{width:100%;border-radius:4px}}
 figcaption{{text-align:center;font-size:11px;color:#666}} .pair .cap{{font-size:11px;color:#555;margin-top:4px;word-break:break-all}}
</style></head><body>
<h1>Zero-shot YOLO-World vs fine-tuned detector — Track A</h1>
<p class="hint">Both scored on the held-out <b>{meta['split']}</b> split ({meta['n']} images)
through the same evaluation harness. Zero-shot: {meta['zs_model']} (per-class tuned thresholds).
Fine-tuned: {meta['ft_model']}, trained on train+valid (global threshold {meta['global']:g}).
<b>mAP@50 is the fair, threshold-independent headline.</b></p>
<h2>Per-class metrics on {meta['split']}</h2>
<table>
<thead><tr>
 <th rowspan=2>class</th><th colspan=4>zero-shot YOLO-World</th>
 <th class=sep colspan=4>fine-tuned</th><th rowspan=2>GT</th></tr>
 <tr><th>P</th><th>R</th><th>F1</th><th>AP50</th>
 <th class=sep>P</th><th>R</th><th>F1</th><th>AP50 Δ</th></tr></thead>
<tbody>
{rows}
{macro}
</tbody></table>
<h2>Example detections (same image, both models)</h2>
{cards}
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)
    cfg: Config = load_config(args.config)

    if not os.path.isfile(cfg.finetuned_model):
        print(f"Fine-tuned model not found: {cfg.finetuned_model}\n"
              f"Run `python train.py` first.", file=sys.stderr)
        return 1
    errors = [i for i in validate_config(cfg) if i.level == "error"]
    if errors:
        for i in errors:
            print(f"  {i}", file=sys.stderr)
        return 1

    out_dir = os.path.join(cfg.output_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    samples = load_split(cfg, args.split)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Comparing on '{args.split}' ({len(samples)} images)")

    # --- zero-shot: reuse Track-B cached test detections when available --------
    from src.detector_finetuned import FinetunedDetector
    from src.detector_yolo import YoloWorldDetector

    # Reuse the Track-B cached zero-shot detections only if it genuinely matches
    # the current samples (same names/order); otherwise compute with a real
    # detector. Passing None to _run_detector is safe ONLY on a validated hit.
    zs_cache = os.path.join(cfg.output_dir, "tuning", f"dets_{args.split}.json")
    names = [s.name for s in samples]
    cache_ok = False
    if os.path.exists(zs_cache) and not args.refresh:
        try:
            cache_ok = json.load(open(zs_cache, "r", encoding="utf-8")).get("names") == names
        except Exception:
            cache_ok = False
    if cache_ok:
        print("Zero-shot detections: reusing Track-B cache")
        zs_preds = _run_detector(None, samples, zs_cache, refresh=False)
    else:
        print(f"Loading YOLO-World '{cfg.yolo_model}'...")
        zs_preds = _run_detector(YoloWorldDetector(cfg), samples,
                                 os.path.join(out_dir, f"dets_{args.split}_zs.json"), args.refresh)

    print(f"Loading fine-tuned '{cfg.finetuned_model}'...")
    ft_preds = _run_detector(FinetunedDetector(cfg), samples,
                             os.path.join(out_dir, f"dets_{args.split}_ft.json"), args.refresh)

    # --- evaluate both through the same harness -------------------------------
    # Zero-shot keeps its tuned per-class thresholds; fine-tuned uses the global
    # threshold (its own operating point was not separately tuned).
    cfg_ft = copy.copy(cfg)
    cfg_ft.per_class_thresholds = {}
    zs = evaluator.evaluate_yolo(samples, zs_preds, cfg)
    ft = evaluator.evaluate_yolo(samples, ft_preds, cfg_ft)

    scored = cfg.scored_classes

    # --- annotated example pairs ----------------------------------------------
    examples = []
    pick = select_review_sample(samples, args.examples)
    idx_of = {s.name: i for i, s in enumerate(samples)}
    for s in pick:
        i = idx_of[s.name]
        try:
            img = s.read_image()
        except Exception:
            continue
        stem = "".join(c if c.isalnum() else "_" for c in os.path.splitext(s.name)[0])[:48]
        zs_name, ft_name = f"zs_{i:03d}_{stem}.jpg", f"ft_{i:03d}_{stem}.jpg"
        save_image(annotate(img, zs_preds[i], cfg.confidence_threshold, cfg.per_class_thresholds),
                   os.path.join(out_dir, zs_name))
        save_image(annotate(img, ft_preds[i], cfg.confidence_threshold),
                   os.path.join(out_dir, ft_name))
        examples.append({"name": s.name, "zs": zs_name, "ft": ft_name})

    meta = {
        "split": args.split, "n": len(samples),
        "zs_model": cfg.yolo_model, "ft_model": cfg.finetuned_model,
        "global": cfg.confidence_threshold,
    }
    json.dump({"meta": meta, "zero_shot": zs, "finetuned": ft},
              open(os.path.join(out_dir, "comparison.json"), "w", encoding="utf-8"), indent=2)
    write_html(os.path.join(out_dir, "comparison.html"), scored, zs, ft, examples, meta)

    # --- console headline ------------------------------------------------------
    print("\n" + "=" * 64)
    print(f" Zero-shot vs fine-tuned on '{args.split}' (held-out)")
    print("=" * 64)
    print(f"{'':<16}{'zero-shot':>22}{'fine-tuned':>22}")
    print(f"{'mAP@50':<16}{_pct(zs['overall']['mAP50']):>22}{_pct(ft['overall']['mAP50']):>22}")
    print(f"{'macro F1':<16}{_pct(zs['overall']['f1_macro']):>22}{_pct(ft['overall']['f1_macro']):>22}")
    for n in scored:
        z, f = zs['per_class'].get(n, {}), ft['per_class'].get(n, {})
        print(f"  {n:<14} F1 {_pct(z.get('f1')):>8} -> {_pct(f.get('f1')):>8}   "
              f"AP50 {_pct(z.get('ap50')):>8} -> {_pct(f.get('ap50')):>8}")
    print("\nWrote:")
    print(f"  {os.path.join(out_dir, 'comparison.html')}")
    print(f"  {os.path.join(out_dir, 'comparison.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
