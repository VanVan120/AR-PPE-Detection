#!/usr/bin/env python3
"""Track B (secondary) — does prompt phrasing recover the violation classes?

Per-class threshold tuning lifted the *compliant* PPE classes (Hardhat, Safety
Vest) a lot, but left NO-Hardhat / NO-Safety Vest at ~0 because the open-vocab
detector produced essentially no true positives for them at any confidence. That
could be a phrasing problem: open-vocabulary detectors often ground a positive
visual concept ("a bare head") far better than a negation ("a person WITHOUT a
hard hat"). This script tests that hypothesis directly.

For each candidate phrasing of a violation class it re-runs YOLO-World on the
validation split and reports the best achievable F1 (and recall at that point)
for the target class. If no phrasing lifts the class above the baseline, that is
strong evidence the limitation is fundamental to zero-shot detection here — which
is the case for fine-tuning (Track A).

    python prompts_experiment.py                # all variants, on 'valid'
    python prompts_experiment.py --split valid --limit 60

Reuses the same detector wrapper; each variant is one inference pass over the
split (cached per variant so re-runs are instant).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from src.config import load_config, Config
from src.detector_yolo import Detection
from src.loader import load_split, Sample
from src import tune


# A "variant" swaps the single prompt that maps to one target dataset class while
# keeping a fixed context of the other prompts, so the comparison is controlled.
# target_class -> list of (variant_name, prompt_text)
VARIANTS: dict[str, list[tuple[str, str]]] = {
    "NO-Hardhat": [
        ("baseline: 'person without a hard hat'", "person without a hard hat"),
        ("'a bare human head'", "a bare human head"),
        ("'construction worker with no helmet'", "construction worker with no helmet"),
        ("'person with an uncovered head'", "person with an uncovered head"),
        ("'head without a hard hat'", "head without a hard hat"),
    ],
    "NO-Safety Vest": [
        ("baseline: 'person without a safety vest'", "person without a safety vest"),
        ("'worker in plain clothes, no vest'", "worker in plain clothes without a hi-vis vest"),
        ("'person not wearing a reflective vest'", "person not wearing a reflective vest"),
        ("'worker in a t-shirt'", "worker wearing a plain t-shirt"),
    ],
    "Safety Vest": [
        ("baseline: 'high-visibility safety vest'", "person wearing a high-visibility safety vest"),
        ("'bright orange or yellow safety vest'", "person wearing a bright orange or yellow safety vest"),
        ("'reflective hi-vis vest'", "worker wearing a reflective hi-vis vest"),
    ],
}

# Fixed context prompts (so YOLO has the usual competing classes to suppress
# false positives). The target class's prompt is injected per variant.
CONTEXT_PROMPTS = [
    "person wearing a hard hat",
    "person",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prompt-variant experiment for YOLO-World violation classes")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--split", default="valid")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args(argv)


def _variant_cfg(base: Config, target_class: str, prompt_text: str) -> Config:
    import copy
    c = copy.copy(base)
    c.safety_prompts = [prompt_text] + [p for p in CONTEXT_PROMPTS if p != prompt_text]
    c.prompt_to_class = {prompt_text: target_class}
    for ctx in CONTEXT_PROMPTS:
        if ctx == prompt_text:
            continue
        # map context prompts to their natural class if known, else leave unscored
        c.prompt_to_class[ctx] = {"person wearing a hard hat": "Hardhat",
                                  "person": "Person"}.get(ctx)
    return c


def _run(detector, samples: list[Sample], cache_path: str, refresh: bool) -> list[list[Detection]]:
    names = [s.name for s in samples]
    if not refresh and os.path.exists(cache_path):
        try:
            cached = json.load(open(cache_path, "r", encoding="utf-8"))
            if cached.get("names") == names:
                return [[Detection(**d) for d in img] for img in cached["dets"]]
        except Exception:
            pass
    out = []
    for s in samples:
        try:
            out.append(detector.detect(s.read_image()))
        except Exception:
            out.append([])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    json.dump({"names": names, "dets": [[d.to_dict() for d in img] for img in out]},
              open(cache_path, "w", encoding="utf-8"))
    return out


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args = parse_args(argv)
    cfg = load_config(args.config)
    samples = load_split(cfg, args.split)
    if args.limit > 0:
        samples = samples[:args.limit]
    gts = [s.gt for s in samples]
    iou = cfg.iou_threshold
    print(f"Prompt-variant experiment on '{args.split}' ({len(samples)} images), IoU {iou}")

    from src.detector_yolo import YoloWorldDetector
    out_dir = os.path.join(cfg.output_dir, "tuning", "prompt_variants")
    os.makedirs(out_dir, exist_ok=True)

    results: dict[str, list[dict]] = {}
    for target_class, variants in VARIANTS.items():
        print(f"\n=== target class: {target_class} ===")
        print(f"{'variant':<46}{'bestF1':>8}{'prec':>7}{'rec':>7}{'@conf':>7}{'npos':>6}")
        rows = []
        for vname, prompt_text in variants:
            vcfg = _variant_cfg(cfg, target_class, prompt_text)
            detector = YoloWorldDetector(vcfg)
            safe = "".join(ch if ch.isalnum() else "_" for ch in prompt_text)[:40]
            preds = _run(detector, samples, os.path.join(out_dir, f"{target_class.replace(' ','_')}__{safe}.json"), args.refresh)
            curve = tune.class_curve(target_class, preds, gts, cfg.dataset_classes, iou)
            bf1 = curve.best_f1 or 0.0
            bp = curve.best_precision or 0.0
            br = curve.best_recall or 0.0
            bt = curve.best_threshold if curve.best_threshold is not None else float("nan")
            print(f"{vname:<46}{bf1*100:>7.1f}%{bp*100:>6.0f}%{br*100:>6.0f}%{bt:>7.2f}{curve.npos:>6}")
            rows.append({"variant": vname, "prompt": prompt_text, "best_f1": round(bf1, 4),
                         "best_precision": round(bp, 4), "best_recall": round(br, 4),
                         "best_threshold": None if bt != bt else round(float(bt), 4), "npos": curve.npos})
        results[target_class] = rows

    json.dump(results, open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8"), indent=2)

    print("\n" + "=" * 64)
    print(" Verdict")
    print("=" * 64)
    for target_class, rows in results.items():
        base = rows[0]["best_f1"]
        best = max(rows, key=lambda r: r["best_f1"])
        if best["best_f1"] > base + 0.03:
            print(f"  {target_class}: prompt '{best['prompt']}' improves best-F1 "
                  f"{base*100:.1f}% -> {best['best_f1']*100:.1f}%  (adopt)")
        else:
            print(f"  {target_class}: no phrasing beats baseline ({base*100:.1f}% best-F1) — "
                  f"limitation is fundamental, needs Track A")
    print(f"\nWrote {os.path.join(out_dir, 'results.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
