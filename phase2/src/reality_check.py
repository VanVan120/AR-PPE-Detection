"""Reality-check — does the clean-benchmark accuracy survive worn-camera video?

Runs the trained detector over a (ideally self-recorded, first-person) clip and
logs detection behaviour: how often each class is detected, at what confidence,
how many people per frame. It then compares that behaviour to the Phase 1 clean
test-split benchmark (benchmark.json) and writes reality_check.json with an honest
verdict on the domain gap.

Two modes:
  * Behavioural (default): no ground truth needed. Reports detection rate and
    confidence on the violation classes vs the clean benchmark. This flags a gap
    (e.g. a class that benchmarked at 94% recall but is barely detected on the
    clip) without claiming a precise recall number.
  * Measured recall (optional): pass a small hand-label JSON to estimate real
    image-level recall on the violation classes and compare it head-to-head.

Hand-label JSON format (frame indices are 0-based, into the processed stream):
    {"labels": {"30": ["No-Helmet", "Person"], "60": ["Person"]}}
"""
from __future__ import annotations

import json
import os
import sys
from typing import Optional

from .config import Config
from .detector import Detector
from .source import FrameSource, SourceError


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_reality_check(cfg: Config, clip_path: str, every: int = 1,
                      max_frames: int = 0, labels_path: Optional[str] = None,
                      out_path: Optional[str] = None) -> dict:
    if not os.path.isfile(clip_path):
        raise SourceError(
            f"reality-check clip not found: {clip_path}\n"
            "Record a short first-person walkthrough (phone at chest/head height, "
            "walking past people with/without hard hats and vests) and save it under "
            "data/clips/, then pass it with --reality-check <path>.")

    det = Detector(cfg.weights_path, cfg.device, cfg.confidence_threshold, cfg.imgsz)
    det.warmup()
    names = det.names
    person_name = cfg.person_class
    violation_names = list(cfg.violation_rules.keys())

    per_class = {n: {"detections": 0, "conf_sum": 0.0, "frames_present": 0,
                     "max_in_frame": 0} for n in names.values()}
    detected_by_frame: dict[int, set] = {}
    dets_per_frame: list[int] = []
    persons_per_frame: list[int] = []
    frames_with_any = 0
    processed = 0

    print(f"Reality-check: running detector on {clip_path} (device {cfg.device})...")
    with FrameSource(clip_path, target_fps=cfg.target_fps) as src:
        raw_idx = -1
        for frame in src.frames():
            raw_idx += 1
            if every > 1 and (raw_idx % every) != 0:
                continue
            d = det.detect(frame)
            class_ids = list(d.class_id) if d.class_id is not None else []
            confs = list(d.confidence) if d.confidence is not None else []
            frame_names = [names.get(int(c), str(int(c))) for c in class_ids]

            present_counts: dict[str, int] = {}
            for nm, cf in zip(frame_names, confs):
                pc = per_class.setdefault(nm, {"detections": 0, "conf_sum": 0.0,
                                               "frames_present": 0, "max_in_frame": 0})
                pc["detections"] += 1
                pc["conf_sum"] += float(cf)
                present_counts[nm] = present_counts.get(nm, 0) + 1
            for nm, cnt in present_counts.items():
                per_class[nm]["frames_present"] += 1
                per_class[nm]["max_in_frame"] = max(per_class[nm]["max_in_frame"], cnt)

            detected_by_frame[processed] = set(present_counts.keys())
            dets_per_frame.append(len(class_ids))
            persons_per_frame.append(present_counts.get(person_name, 0))
            if class_ids:
                frames_with_any += 1

            processed += 1
            if processed % 50 == 0:
                print(f"  ...{processed} frames")
            if max_frames and processed >= max_frames:
                break

    if processed == 0:
        raise SourceError(f"no frames read from {clip_path} — is it a valid video?")

    # --- aggregate clip stats ----------------------------------------------
    clip_stats = {"frames_processed": processed,
                  "frames_with_detection": frames_with_any,
                  "detection_rate": round(frames_with_any / processed, 4),
                  "mean_detections_per_frame": round(sum(dets_per_frame) / processed, 3),
                  "mean_persons_per_frame": round(sum(persons_per_frame) / processed, 3),
                  "per_class": {}}
    for nm, pc in per_class.items():
        if pc["detections"] == 0 and nm not in violation_names and nm != person_name:
            continue
        clip_stats["per_class"][nm] = {
            "detections": pc["detections"],
            "mean_confidence": round(pc["conf_sum"] / pc["detections"], 4) if pc["detections"] else 0.0,
            "frames_present": pc["frames_present"],
            "present_rate": round(pc["frames_present"] / processed, 4),
            "max_in_frame": pc["max_in_frame"],
        }

    # --- benchmark ----------------------------------------------------------
    benchmark = {}
    if os.path.isfile(cfg.benchmark_path):
        try:
            benchmark = _load_json(cfg.benchmark_path)
        except Exception as e:
            print(f"[warn] could not read benchmark {cfg.benchmark_path}: {e}")

    # --- optional measured recall ------------------------------------------
    measured = None
    if labels_path:
        measured = _measure_recall(labels_path, detected_by_frame, violation_names)
        skipped = measured.get("skipped_indices", [])
        if skipped:
            print(f"[warn] {len(skipped)} label index(es) are outside the {processed} processed "
                  f"frames and were ignored: {skipped}. Label indices are 0-based into the "
                  f"PROCESSED stream (raw_frame // --every; --every={every} here).", file=sys.stderr)

    # --- comparison + verdict ----------------------------------------------
    comparison = _compare(clip_stats, benchmark, violation_names, measured)

    report = {
        "meta": {
            "clip": clip_path,
            "model": os.path.basename(cfg.weights_path),
            "device": cfg.device,
            "confidence_threshold": cfg.confidence_threshold,
            "frame_sampling": every,
            "note": "Behavioural comparison unless 'measured_frame_presence_recall' is present. "
                    "Detection rate/confidence are not a substitute for labelled recall. The "
                    "measured metric is FRAME-PRESENCE recall (class detected anywhere in a "
                    "labelled frame) — an optimistic upper bound on the benchmark's instance-level "
                    "recall, so a non-positive recall_drop is not proof of 'no domain gap'.",
        },
        "clip_stats": clip_stats,
        "benchmark": benchmark,
        "measured_frame_presence_recall": measured,
        "comparison": comparison,
    }

    out_path = out_path or os.path.join(cfg.config_dir, "outputs", "reality_check.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    _print_report(report, out_path)
    return report


def _measure_recall(labels_path: str, detected_by_frame: dict[int, set],
                    violation_names: list[str]) -> dict:
    raw = _load_json(labels_path)
    labels = raw.get("labels", raw)  # allow bare {idx: [...]} too
    per_class = {nm: {"gt_frames": 0, "hits": 0} for nm in violation_names}
    used = 0
    supplied = 0
    skipped: list[int] = []
    for k, gt in labels.items():
        if str(k).startswith("_"):       # allow "_comment" keys in the bare format
            continue
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        supplied += 1
        if idx not in detected_by_frame:  # out of range for the processed stream
            skipped.append(idx)
            continue
        used += 1
        detected = detected_by_frame[idx]
        for nm in violation_names:
            if nm in gt:
                per_class[nm]["gt_frames"] += 1
                if nm in detected:
                    per_class[nm]["hits"] += 1
    out = {
        "metric": "frame_presence_recall",
        "note": "A frame counts as a hit if the class is detected ANYWHERE in it; this is an "
                "optimistic upper bound on the benchmark's instance-level recall.",
        "labels_supplied": supplied,
        "labelled_frames_used": used,
        "skipped_indices": sorted(skipped),
        "per_class": {},
    }
    for nm, c in per_class.items():
        recall = (c["hits"] / c["gt_frames"]) if c["gt_frames"] else None
        out["per_class"][nm] = {"gt_frames": c["gt_frames"], "hits": c["hits"],
                                "frame_presence_recall": round(recall, 4) if recall is not None else None}
    return out


def _compare(clip_stats: dict, benchmark: dict, violation_names: list[str],
             measured: Optional[dict]) -> dict:
    bench_pc = (benchmark or {}).get("per_class", {})
    per_class = {}
    gap_flags = []
    for nm in violation_names:
        cs = clip_stats["per_class"].get(nm, {})
        bench = bench_pc.get(nm, {})
        entry = {
            "benchmark_recall": bench.get("recall"),
            "benchmark_f1": bench.get("f1"),
            "clip_present_rate": cs.get("present_rate", 0.0),
            "clip_mean_confidence": cs.get("mean_confidence", 0.0),
            "clip_detections": cs.get("detections", 0),
        }
        if measured:
            m = measured["per_class"].get(nm, {})
            fpr = m.get("frame_presence_recall")
            entry["measured_frame_presence_recall"] = fpr
            br = bench.get("recall")
            if fpr is not None and br is not None:
                entry["recall_drop"] = round(br - fpr, 4)
                if (br - fpr) > 0.15:
                    gap_flags.append(nm)
        else:
            # behavioural heuristic: a class that benchmarked high but is detected
            # at low confidence or almost never on the clip is a gap signal.
            if cs.get("detections", 0) == 0 and bench.get("recall"):
                gap_flags.append(nm)
        per_class[nm] = entry

    if measured:
        used = measured.get("labelled_frames_used", 0)
        total_gt = sum(measured["per_class"].get(nm, {}).get("gt_frames", 0)
                       for nm in violation_names)
        if used == 0 or total_gt == 0:
            # No labelled frame actually contributed — never claim "holds" on no evidence.
            skipped = measured.get("skipped_indices", [])
            verdict = (f"INCONCLUSIVE: no usable hand-labelled frames "
                       f"({used} of {measured.get('labels_supplied', 0)} labels matched a "
                       f"violation class in range"
                       + (f"; out-of-range indices {skipped}" if skipped else "") + "). "
                       "Label indices are 0-based into the PROCESSED stream (raw_frame // --every). "
                       "No recall claim can be made — fix the indices and re-run.")
        elif gap_flags:
            verdict = ("DOMAIN GAP: frame-presence recall on " + ", ".join(gap_flags) +
                       " is >15 points below the clean benchmark — and frame-presence recall is an "
                       "OPTIMISTIC upper bound, so the true instance-level gap is at least this big. "
                       "A targeted fine-tuning pass on first-person-style data is warranted.")
        else:
            verdict = ("No gross gap: frame-presence recall on the violation classes is within 15 "
                       "points of the clean benchmark on the labelled frames. NOTE: frame-presence "
                       "recall over-counts (one detection scores the whole frame), so treat this as "
                       "'no obvious gap', not a clean pass — label more frames to tighten it.")
    else:
        verdict = ("Behavioural only (no hand-labels supplied). " +
                   ("Some violation class(es) were never detected on this clip: " +
                    ", ".join(gap_flags) + " — likely a domain gap. "
                    if gap_flags else
                    "Violation classes were detected on the clip. ") +
                   "This is detection behaviour, NOT a recall measurement — hand-label "
                   "a handful of frames and re-run with --labels for a real number.")

    return {"violation_classes": per_class, "gap_flags": gap_flags, "verdict": verdict}


def _print_report(report: dict, out_path: str) -> None:
    cs = report["clip_stats"]
    print("\n" + "=" * 64)
    print(" Reality-check — worn-camera vs clean benchmark")
    print("=" * 64)
    print(f"  frames processed     : {cs['frames_processed']}")
    print(f"  frames w/ detection  : {cs['frames_with_detection']} "
          f"({cs['detection_rate']*100:.1f}%)")
    print(f"  mean persons / frame : {cs['mean_persons_per_frame']}")
    print("  violation classes (clip present-rate / mean-conf  vs  benchmark recall):")
    comp = report["comparison"]["violation_classes"]
    for nm, e in comp.items():
        br = e.get("benchmark_recall")
        mr = e.get("measured_frame_presence_recall")
        tail = f"  frame-pres recall {mr*100:.0f}%" if mr is not None else ""
        brs = f"{br*100:.0f}%" if br is not None else "?"
        print(f"    {nm:<10} present {e['clip_present_rate']*100:5.1f}%  "
              f"conf {e['clip_mean_confidence']:.2f}   vs bench recall {brs}{tail}")
    print("-" * 64)
    print("  VERDICT: " + report["comparison"]["verdict"])
    print(f"\nWrote {out_path}")
