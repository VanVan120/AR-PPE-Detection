#!/usr/bin/env python3
"""Construction Safety Inspection — approach-validation prototype.

Single entry point. One command runs both perception pipelines over the dataset's
test split, writes annotated images + per-image safety reports, a side-by-side
results.html, and metrics.json with quantitative metrics vs the ground-truth labels.

    python run.py                 # full run on the test split (local Ollama VLM)
    python run.py --check         # environment / dataset readiness check
    python run.py --api           # use the Anthropic vision API instead of Ollama
    python run.py --limit 8       # quick run on the first 8 images
    python run.py --images path/  # run on an ad-hoc image folder (no evaluation)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from src.config import load_config, validate_config, resolve_device, Config
from src import compare, render
from src.detector_yolo import annotate, save_image
from src.detector_vlm import build_vlm, save_vlm_output, load_vlm_output
from src.reporter import build_report
from src.loader import load_eval_split, load_adhoc_folder, select_review_sample, Sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construction safety inspection prototype")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--check", action="store_true", help="run readiness checks and exit")
    p.add_argument("--api", action="store_true", help="use the Anthropic vision API instead of Ollama")
    p.add_argument("--limit", type=int, default=0, help="process only the first N images (0 = all)")
    p.add_argument("--images", default=None, help="ad-hoc image folder (skips evaluation)")
    p.add_argument("--reuse-vlm", action="store_true",
                   help="reuse cached VLM responses in outputs/vlm/ instead of re-running the VLM "
                        "(use when only detection changed, e.g. tuned thresholds)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def run_check(cfg: Config, args: argparse.Namespace) -> int:
    print("=" * 64)
    print(" Readiness check")
    print("=" * 64)
    ok = True

    # Python
    v = sys.version_info
    py_ok = (v.major, v.minor) >= (3, 11)
    print(f"[{'ok' if py_ok else 'warn'}] Python {v.major}.{v.minor}.{v.micro} "
          f"({'>= 3.11' if py_ok else 'older than recommended 3.11'})")

    # Packages
    needed = [("ultralytics", "ultralytics"), ("supervision", "supervision"),
              ("cv2", "opencv-python"), ("numpy", "numpy"), ("yaml", "PyYAML"),
              ("requests", "requests"), ("torch", "torch")]
    if args.api:
        needed.append(("anthropic", "anthropic"))
    for mod, pip_name in needed:
        try:
            __import__(mod)
            print(f"[ ok ] import {mod}")
        except Exception as e:
            ok = False
            print(f"[FAIL] import {mod} — install '{pip_name}' ({e})")

    # Device
    print(f"[ ok ] compute device: {resolve_device(cfg.device_pref)}")

    # Config + dataset
    print("-" * 64)
    for issue in validate_config(cfg):
        print(issue)
        if issue.level == "error":
            ok = False

    # VLM backend health
    print("-" * 64)
    if cfg.runs_pipeline("vlm"):
        try:
            vlm = build_vlm(cfg, args.api)
            good, msg = vlm.health_check()
            print(f"[{'ok' if good else 'warn'}] {msg}")
        except Exception as e:
            print(f"[warn] VLM backend unavailable: {e}")
    else:
        print("[ ok ] VLM pipeline disabled in config")

    print("=" * 64)
    if ok:
        print("READY — run `python run.py` to produce metrics + results.")
        return 0
    print("NOT READY — resolve the [FAIL] items above. See README.md for setup.")
    return 1


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------
def _rel(path: str, base: str) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def run(cfg: Config, args: argparse.Namespace) -> int:
    adhoc = args.images is not None

    # --- load samples --------------------------------------------------------
    if adhoc:
        samples = load_adhoc_folder(args.images)
        if not samples:
            print(f"No images found in {args.images}", file=sys.stderr)
            return 1
        print(f"Loaded {len(samples)} ad-hoc images from {args.images} (no evaluation).")
    else:
        issues = validate_config(cfg)
        errors = [i for i in issues if i.level == "error"]
        if errors:
            print("Cannot run — dataset/config problems:", file=sys.stderr)
            for i in errors:
                print(f"  {i}", file=sys.stderr)
            print("\nPlace the unzipped Roboflow export under "
                  f"'{cfg.dataset_dir}' (data.yaml + test/images + test/labels), "
                  "or run `python run.py --check`.", file=sys.stderr)
            return 1
        samples = load_eval_split(cfg)
        print(f"Loaded {len(samples)} images from the '{cfg.eval_split}' split.")

    if args.limit and args.limit > 0:
        samples = samples[:args.limit]
        print(f"Limiting to first {len(samples)} images.")

    # --- build pipelines -----------------------------------------------------
    run_yolo = cfg.runs_pipeline("yolo_world")
    run_vlm = cfg.runs_pipeline("vlm")
    yolo = None
    vlm = None

    if run_yolo:
        try:
            from src.detector_finetuned import build_detector
            backend = (cfg.detector_backend or "yolo_world").lower()
            model_name = cfg.finetuned_model if backend == "finetuned" else cfg.yolo_model
            print(f"Loading detector backend '{backend}' ('{model_name}') on {cfg.device} "
                  "(weights auto-download on first run)...")
            yolo = build_detector(cfg)
        except Exception as e:
            print(f"[warn] detector unavailable, skipping it: {e}")
            run_yolo = False

    reuse_vlm = args.reuse_vlm
    if run_vlm and reuse_vlm:
        # Reuse previously saved responses; no backend needed.
        print(f"VLM: reusing cached responses from {cfg.vlm_out_dir}/ (--reuse-vlm)")
        vlm = None
    elif run_vlm:
        try:
            vlm = build_vlm(cfg, args.api)
            good, msg = vlm.health_check()
            print(f"VLM: {msg}")
            if not good:
                print("[warn] VLM not ready — skipping the VLM pipeline.")
                run_vlm, vlm = False, None
        except Exception as e:
            print(f"[warn] VLM backend error, skipping it: {e}")
            run_vlm, vlm = False, None

    if not run_yolo and not run_vlm:
        print("No pipelines available to run.", file=sys.stderr)
        return 1

    # --- output dirs ---------------------------------------------------------
    for d in (cfg.output_dir, cfg.yolo_out_dir, cfg.vlm_out_dir, cfg.images_out_dir):
        os.makedirs(d, exist_ok=True)

    # --- per-image processing ------------------------------------------------
    records: list[dict] = []
    yolo_all: list[list] = []
    vlm_all: list = []
    review_names = {s.name for s in select_review_sample(samples, cfg.sample_for_review)}

    # Per-class thresholds are tuned for the zero-shot backend; gate them off for
    # any other backend so display/annotation match the (gated) evaluator + record.
    active_class_thresholds = cfg.per_class_thresholds if cfg.uses_per_class_thresholds else None
    reused_missing = 0

    t0 = time.time()
    for i, sample in enumerate(samples):
        # Unique output stem so two files sharing a stem can't overwrite each other.
        prefix = f"{i:04d}_{os.path.splitext(sample.name)[0]}"
        try:
            image = sample.read_image()

            display_dets = None
            if run_yolo:
                dets_all = yolo.detect(image)
                display_dets = [d for d in dets_all if d.confidence >= cfg.threshold_for(d.class_name)]
                out_img = os.path.join(cfg.yolo_out_dir, f"{prefix}.jpg")
                save_image(
                    annotate(image, dets_all, cfg.confidence_threshold, active_class_thresholds),
                    out_img,
                )
            else:
                dets_all = []
                out_img = os.path.join(cfg.images_out_dir, f"{prefix}.jpg")
                save_image(image, out_img)
            display_rel = _rel(out_img, cfg.output_dir)

            vres = None
            if run_vlm:
                vlm_path = os.path.join(cfg.vlm_out_dir, f"{prefix}.json")
                if reuse_vlm:
                    if os.path.exists(vlm_path):
                        vres = load_vlm_output(vlm_path)
                    else:
                        vres = None
                        reused_missing += 1
                else:
                    vres = vlm.observe(image)
                    save_vlm_output(vlm_path, vres)

            report = build_report(dets_all if run_yolo else None, vres, cfg)
            # Append exactly once per iteration so yolo_all/vlm_all stay aligned
            # with `samples` for the evaluator.
            yolo_all.append(dets_all)
            vlm_all.append(vres)
            records.append(compare.assemble_record(sample, display_rel, display_dets, vres, report))

            nd = len(display_dets) if display_dets is not None else 0
            no = len(vres.observations) if vres else 0
            print(f"[{i+1}/{len(samples)}] {sample.name}: {nd} det, {no} obs — {report}")
        except Exception as e:
            # One bad image must not abort the whole batch; keep arrays aligned.
            yolo_all.append([])
            vlm_all.append(None)
            print(f"[{i+1}/{len(samples)}] {sample.name}: failed ({e}) — skipped")

    elapsed = time.time() - t0
    print(f"Processed {len(samples)} images in {elapsed:.1f}s.")

    if run_vlm and reuse_vlm and reused_missing:
        print(f"[warn] --reuse-vlm: {reused_missing}/{len(samples)} images had no cached VLM "
              f"response in {cfg.vlm_out_dir}/ — they were scored as 'no VLM flag'. "
              "Run `python run.py` (without --reuse-vlm) first to populate the cache.")

    if run_vlm:
        attempted = [v for v in vlm_all if v is not None]
        errored = [v for v in attempted if v.error]
        if attempted and len(errored) == len(attempted):
            print("[warn] every VLM call errored — check the VLM backend "
                  "(model tag / host). VLM metrics will be empty.")

    # --- evaluation ----------------------------------------------------------
    # Never let an evaluation error throw away an entire run's per-image work —
    # write the gallery/json regardless.
    try:
        metrics = _evaluate(samples, yolo_all, vlm_all, cfg, run_yolo, run_vlm, adhoc)
    except Exception as e:
        print(f"[warn] evaluation failed, writing outputs without metrics: {e}")
        reason = f"evaluation error: {e}"
        metrics = {"yolo_world": {"available": False, "reason": reason},
                   "vlm": {"available": False, "reason": reason}}

    # --- write outputs -------------------------------------------------------
    backend = (cfg.detector_backend or "yolo_world").lower()
    meta = {
        "dataset_dir": cfg.dataset_dir,
        "split": "ad-hoc" if adhoc else cfg.eval_split,
        "num_total": len(samples),
        "num_shown": sum(1 for r in records if r["image"] in review_names),
        "detector_backend": backend if run_yolo else None,
        "yolo_model": (cfg.finetuned_model if backend == "finetuned" else cfg.yolo_model) if run_yolo else None,
        "vlm_backend": _vlm_label(cfg, args.api, run_vlm),
        "device": cfg.device,
        "confidence_threshold": cfg.confidence_threshold,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_json = os.path.join(cfg.output_dir, "results.json")
    metrics_json = os.path.join(cfg.output_dir, "metrics.json")
    results_html = os.path.join(cfg.output_dir, "results.html")

    render.write_json(results_json, {
        "meta": meta,
        "summary": compare.summarize(records),
        "images": records,
    })
    render.write_json(metrics_json, {"meta": meta, **metrics})
    review_records = [r for r in records if r["image"] in review_names]
    render.render_results_html(results_html, review_records, metrics, meta)

    _print_headline(metrics, backend)
    print("\nWrote:")
    print(f"  {results_html}")
    print(f"  {results_json}")
    print(f"  {metrics_json}")
    if run_yolo:
        print(f"  {cfg.yolo_out_dir}/  (annotated images)")
    if run_vlm:
        print(f"  {cfg.vlm_out_dir}/  (raw + parsed VLM responses)")
    return 0


def _evaluate(samples, yolo_all, vlm_all, cfg, run_yolo, run_vlm, adhoc) -> dict:
    if adhoc:
        reason = "ad-hoc images have no ground-truth labels"
        return {"yolo_world": {"available": False, "reason": reason},
                "vlm": {"available": False, "reason": reason}}
    from src import evaluator
    out = {}
    if run_yolo:
        print("Evaluating YOLO-World against ground truth...")
        out["yolo_world"] = evaluator.evaluate_yolo(samples, yolo_all, cfg)
    else:
        out["yolo_world"] = {"available": False, "reason": "yolo_world pipeline not run"}
    if run_vlm:
        print("Evaluating VLM (image-level) against ground truth...")
        out["vlm"] = evaluator.evaluate_vlm(samples, vlm_all, cfg)
    else:
        out["vlm"] = {"available": False, "reason": "vlm pipeline not run"}
    return out


def _vlm_label(cfg: Config, use_api: bool, run_vlm: bool) -> str:
    if not run_vlm:
        return "skipped"
    if use_api:
        return f"anthropic ({cfg.api_model})"
    return f"ollama ({cfg.ollama_model})"


def _detector_label(backend: str) -> str:
    return "Fine-tuned detector" if (backend or "").lower() == "finetuned" else "YOLO-World"


def _print_headline(metrics: dict, backend: str = "yolo_world") -> None:
    label = _detector_label(backend)
    print("\n" + "=" * 64)
    print(" Headline metrics (vs test-split ground truth)")
    print("=" * 64)
    y = metrics.get("yolo_world", {})
    if y.get("available"):
        ov = y["overall"]
        tuned = y.get("tuned_thresholds_active")
        thr = y.get("per_class_thresholds", {})
        print(f"{label}  mAP@50 = {_p(ov['mAP50'])}   "
              f"macro P/R/F1 = {_p(ov['precision_macro'])}/{_p(ov['recall_macro'])}/{_p(ov['f1_macro'])}"
              f"{'   [per-class tuned thresholds]' if tuned else ''}")
        for name, c in y["per_class"].items():
            t = thr.get(name)
            tstr = f" @{t:g}" if t is not None else ""
            print(f"   {name:<16} P={_p(c['precision'])} R={_p(c['recall'])} "
                  f"F1={_p(c['f1'])} AP50={_p(c['ap50'])}{tstr} (support {c['support']})")
    else:
        print(f"{label} metrics: {y.get('reason', 'unavailable')}")
    v = metrics.get("vlm", {})
    if v.get("available"):
        print("VLM (image-level):")
        for name, c in v["per_class"].items():
            print(f"   {name:<16} P={_p(c['precision'])} R={_p(c['recall'])} "
                  f"F1={_p(c['f1'])} (support {c['support_images']})")
    else:
        print(f"VLM metrics: {v.get('reason', 'unavailable')}")


def _p(x) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def main(argv=None) -> int:
    # Keep Unicode (em-dash, ≥) safe on Windows consoles that default to cp1252.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.check:
        return run_check(cfg, args)
    return run(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
