#!/usr/bin/env python3
"""Track C — evaluate the detector+VLM fusion and produce richer reports.

Two things:
  1. Quantitative — at the IMAGE level (does this image contain a NO-Hardhat /
     NO-Safety Vest violation?), compare four decision rules on the held-out test
     split through the same scoring style as the rest of the harness:
        detector-only · vlm-only · fusion-OR (either) · fusion-AND (both)
     This shows whether combining helps recall (OR) or precision (AND) vs either
     pipeline alone.
  2. Qualitative — the fused, reconciled safety report per image (detector
     authoritative, VLM confirming / raising review flags / adding context hazards).

Reuses the fine-tuned detections (outputs/comparison/dets_test_ft.json) and the
cached VLM responses (outputs/vlm/), so it needs neither the GPU nor Ollama if
those exist; otherwise it computes what's missing.

    python fusion_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.config import load_config, validate_config, Config
from src.detector_yolo import Detection, annotate, save_image
from src.detector_vlm import VlmResult, load_vlm_output
from src.loader import load_split, select_review_sample, Sample
from src import fusion
from src.evaluator import _gt_image_classes, _vlm_flags_class, _negative_pattern


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate detector+VLM fusion (Track C)")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# detections / vlm loading (reuse caches when present)
# ---------------------------------------------------------------------------
def _load_or_run_finetuned(cfg: Config, samples: list[Sample], split: str, refresh: bool) -> list[list[Detection]]:
    cache = os.path.join(cfg.output_dir, "comparison", f"dets_{split}_ft.json")
    names = [s.name for s in samples]
    if not refresh and os.path.exists(cache):
        try:
            c = json.load(open(cache, "r", encoding="utf-8"))
            if c.get("names") == names:
                print(f"  using cached fine-tuned detections: {cache}")
                return [[Detection(**d) for d in img] for img in c["dets"]]
        except Exception:
            pass
    print("  running fine-tuned detector...")
    from src.detector_finetuned import FinetunedDetector
    det = FinetunedDetector(cfg)
    out = []
    for i, s in enumerate(samples):
        try:
            out.append(det.detect(s.read_image()))
        except Exception:
            out.append([])
        if (i + 1) % 20 == 0 or i + 1 == len(samples):
            print(f"    [{i+1}/{len(samples)}]")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({"names": names, "dets": [[d.to_dict() for d in img] for img in out]},
              open(cache, "w", encoding="utf-8"))
    return out


def _load_cached_vlm(cfg: Config, samples: list[Sample]) -> list[VlmResult | None]:
    """Reuse the canonical run's saved VLM responses, matched by the same
    `{index:04d}_{stem}.json` naming (identical sorted order -> identical index)."""
    out = []
    missing = 0
    for i, s in enumerate(samples):
        stem = os.path.splitext(s.name)[0]
        path = os.path.join(cfg.vlm_out_dir, f"{i:04d}_{stem}.json")
        if os.path.exists(path):
            try:
                out.append(load_vlm_output(path))
                continue
            except Exception:
                pass
        out.append(None)
        missing += 1
    if missing:
        print(f"  [warn] {missing}/{len(samples)} VLM responses missing from {cfg.vlm_out_dir} "
              "(run `python run.py` first to populate); those images score VLM as 'no flag'.")
    return out


# ---------------------------------------------------------------------------
# image-level scoring of the four decision rules
# ---------------------------------------------------------------------------
def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else None
    r = tp / (tp + fn) if (tp + fn) else None
    f = (2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None))
    return {"precision": _r(p), "recall": _r(r), "f1": _r(f), "tp": tp, "fp": fp, "fn": fn}


def _r(x):
    return None if x is None else round(float(x), 4)


def evaluate_fusion(samples, ft_preds, vlm_results, cfg: Config) -> dict:
    classes = [c for c in cfg.vlm_eval_classes if c in cfg.dataset_classes]
    neg = _negative_pattern(cfg.vlm_negative_keywords)
    methods = ["detector", "vlm", "or", "and"]
    counts = {m: {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes} for m in methods}

    for s, dets, vres in zip(samples, ft_preds, vlm_results):
        gt_present = _gt_image_classes(s.gt, cfg.dataset_classes)
        for c in classes:
            truth = c in gt_present
            d = fusion.detector_flags_class(dets, c, cfg)[0]
            v = _vlm_flags_class(vres, c, cfg, neg) if vres is not None else False
            preds = {"detector": d, "vlm": v, "or": d or v, "and": d and v}
            for m, pred in preds.items():
                cell = counts[m][c]
                if truth and pred:
                    cell["tp"] += 1
                elif pred and not truth:
                    cell["fp"] += 1
                elif truth and not pred:
                    cell["fn"] += 1

    result = {"classes": classes, "methods": {}}
    for m in methods:
        per_class = {c: _prf(**counts[m][c]) for c in classes}
        p = [per_class[c]["precision"] or 0.0 for c in classes if (counts[m][c]["tp"] + counts[m][c]["fn"]) > 0]
        r = [per_class[c]["recall"] or 0.0 for c in classes if (counts[m][c]["tp"] + counts[m][c]["fn"]) > 0]
        f = [per_class[c]["f1"] or 0.0 for c in classes if (counts[m][c]["tp"] + counts[m][c]["fn"]) > 0]
        result["methods"][m] = {
            "per_class": per_class,
            "macro": {"precision": _avg(p), "recall": _avg(r), "f1": _avg(f)},
        }
    return result


def _avg(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _pct(x):
    return "—" if x is None else f"{x * 100:.1f}%"


METHOD_LABEL = {"detector": "detector-only", "vlm": "VLM-only",
                "or": "fusion OR (either)", "and": "fusion AND (both)"}


def write_html(path, evalres, examples, meta):
    classes = evalres["classes"]
    # method comparison table: rows = method, cols = per-class F1 + macro P/R/F1
    head_cls = "".join(f"<th class=sep>{c} P</th><th>{c} R</th><th>{c} F1</th>" for c in classes)
    rows = ""
    for m in ["detector", "vlm", "or", "and"]:
        md = evalres["methods"][m]
        cells = ""
        for c in classes:
            pc = md["per_class"][c]
            cells += f"<td class=sep>{_pct(pc['precision'])}</td><td>{_pct(pc['recall'])}</td><td><b>{_pct(pc['f1'])}</b></td>"
        mac = md["macro"]
        rows += (f"<tr><td>{METHOD_LABEL[m]}</td>{cells}"
                 f"<td class=sep>{_pct(mac['precision'])}</td><td>{_pct(mac['recall'])}</td>"
                 f"<td><b>{_pct(mac['f1'])}</b></td></tr>")
    cards = ""
    for e in examples:
        viol = "".join(
            f"<span class='v {v['status']}'>{v['class']}: {v['status']}</span>"
            for v in e["violations"] if v["status"] != "clear"
        ) or "<span class='v clear'>no violations</span>"
        haz = (" · hazards: " + ", ".join(h["type"].replace("_", " ") for h in e["context_hazards"])) \
            if e["context_hazards"] else ""
        cards += (f"<div class='card'><div class='imgwrap'><img src='{e['img']}' loading='lazy'></div>"
                  f"<div class='cap'>{e['name']}</div>"
                  f"<p class='report'>{e['report']}</p>"
                  f"<div class='tags'>{viol}{haz}</div></div>")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Detector + VLM fusion — Track C</title>
<style>
 body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:24px;color:#1a202c;background:#fafbfc}}
 h1{{font-size:20px}} h2{{font-size:16px;margin-top:26px}}
 .hint{{background:#eef4ff;border:1px solid #d6e2ff;border-radius:8px;padding:10px 12px;color:#3a4a6b}}
 table{{border-collapse:collapse;width:100%;background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;font-size:13px}}
 th,td{{padding:6px 9px;text-align:right;border-bottom:1px solid #edf0f4}}
 th:first-child,td:first-child{{text-align:left}}
 thead th{{background:#eef1f5}} td.sep,th.sep{{border-left:2px solid #cbd5e0}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px;margin-top:12px}}
 .card{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:10px}}
 .card img{{width:100%;border-radius:5px}} .cap{{font-size:11px;color:#666;word-break:break-all}}
 .report{{font-size:13px;margin:6px 0}} .tags .v{{display:inline-block;font-size:11px;padding:2px 7px;border-radius:10px;margin:2px 3px 0 0}}
 .v.confirmed{{background:#fde2e1;color:#9b1c1c}} .v.detector{{background:#fdebc8;color:#92400e}}
 .v.review{{background:#e0e7ff;color:#3730a3}} .v.clear{{background:#e8f5e9;color:#256029}}
</style></head><body>
<h1>Detector + VLM fusion — Track C</h1>
<p class="hint">Fine-tuned detector (authoritative on scored classes) + VLM (confirmation,
review flags, context hazards). Image-level violation detection on the held-out
<b>{meta['split']}</b> split ({meta['n']} images): four decision rules compared.
<b>For a safety tool, recall (don't miss a violation) usually outweighs precision</b>,
which favours fusion-OR; AND maximises precision (fewest false alarms).</p>
<h2>Image-level violation detection — which decision rule wins?</h2>
<table>
<thead><tr><th rowspan=2>decision rule</th>{head_cls}<th class=sep>macro P</th><th>macro R</th><th>macro F1</th></tr><tr></tr></thead>
<tbody>{rows}</tbody></table>
<p style="font-size:12px;color:#666">detector-only and VLM-only are the single-pipeline baselines; OR / AND are the fusions.</p>
<h2>Fused safety reports (examples)</h2>
<div class="grid">{cards}</div>
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
    cfg = load_config(args.config)
    # The fusion detector IS the fine-tuned model, so gate off the zero-shot-tuned
    # per-class thresholds (they must not apply to the fine-tuned backend).
    cfg.detector_backend = "finetuned"
    if not os.path.isfile(cfg.finetuned_model):
        print(f"Fine-tuned model not found: {cfg.finetuned_model} — run train.py first.", file=sys.stderr)
        return 1
    errors = [i for i in validate_config(cfg) if i.level == "error"]
    if errors:
        for i in errors:
            print(f"  {i}", file=sys.stderr)
        return 1

    out_dir = os.path.join(cfg.output_dir, "fusion")
    os.makedirs(out_dir, exist_ok=True)

    samples = load_split(cfg, args.split)
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"Fusion eval on '{args.split}' ({len(samples)} images)")

    ft_preds = _load_or_run_finetuned(cfg, samples, args.split, args.refresh)
    vlm_results = _load_cached_vlm(cfg, samples)

    # quantitative
    evalres = evaluate_fusion(samples, ft_preds, vlm_results, cfg)

    # qualitative fused reports for a review sample
    pick = {s.name for s in select_review_sample(samples, args.examples)}
    examples = []
    for i, (s, dets, vres) in enumerate(zip(samples, ft_preds, vlm_results)):
        fr = fusion.fuse(dets, vres, cfg)
        if s.name in pick and len(examples) < args.examples:
            try:
                img = s.read_image()
                stem = "".join(c if c.isalnum() else "_" for c in os.path.splitext(s.name)[0])[:48]
                iname = f"fz_{i:03d}_{stem}.jpg"
                save_image(annotate(img, dets, cfg.confidence_threshold), os.path.join(out_dir, iname))
                examples.append({"name": s.name, "img": iname, "report": fr.report,
                                 "violations": [v.to_dict() for v in fr.violations],
                                 "context_hazards": [o.to_dict() for o in fr.context_hazards]})
            except Exception:
                pass

    meta = {"split": args.split, "n": len(samples), "ft_model": cfg.finetuned_model}
    json.dump({"meta": meta, "image_level": evalres},
              open(os.path.join(out_dir, "fusion.json"), "w", encoding="utf-8"), indent=2)
    write_html(os.path.join(out_dir, "fusion.html"), evalres, examples, meta)

    # console headline
    print("\n" + "=" * 70)
    print(f" Image-level violation detection on '{args.split}' (macro over {evalres['classes']})")
    print("=" * 70)
    print(f"{'rule':<22}{'precision':>12}{'recall':>12}{'F1':>12}")
    for m in ["detector", "vlm", "or", "and"]:
        mac = evalres["methods"][m]["macro"]
        print(f"{METHOD_LABEL[m]:<22}{_pct(mac['precision']):>12}{_pct(mac['recall']):>12}{_pct(mac['f1']):>12}")
    print("\nWrote:")
    print(f"  {os.path.join(out_dir, 'fusion.html')}")
    print(f"  {os.path.join(out_dir, 'fusion.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
