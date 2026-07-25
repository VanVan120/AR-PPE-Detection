"""Analyse a clip recorded on a phone — the path that needs no network at all.

A development site rarely has usable WiFi, and carrying a laptop around one is awkward.
So the most robust way to get the system in front of a supervisor is: **record normally
on the phone at the site, then drop the file in afterwards**. No app, no pairing, no
signal.

Produces one folder per clip containing everything needed to review it and send feedback:

    outputs/site/<clip-name>/
        annotated.mp4     the clip with the AR overlay burned in
        report.json       per-worker violations, durations, compliance
        summary.txt       a readable page-long summary
        worst_<n>.jpg     still frames of the worst moments, for a message or a slide

    python -m phase7_mobile.analyze site_visit.mp4
    python -m phase7_mobile.analyze clips/ --arview glasses
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from src.config import load_config                    # noqa: E402
from src.pipeline import SafetyPipeline               # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".webm")


def analyze_clip(path: str, cfg, out_root: str, every: int = 1,
                 max_stills: int = 3, progress: bool = True,
                 on_progress=None) -> dict:
    """Run one clip end to end and write the review bundle. Returns the report dict.

    `on_progress(fraction)` is called as the clip is consumed, so a caller with no
    terminal — the phone app, which shows a progress bar while a just-recorded clip is
    analysed — can report the same thing the console prints.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    # Re-running the same clip reuses this folder. annotated.mp4/report.json are
    # overwritten, but worst_*.jpg are numbered, so a run that finds fewer incidents would
    # otherwise leave stale stills behind and the bundle would mix two runs.
    for old in os.listdir(out_dir):
        if old.startswith("worst_") and old.endswith(".jpg"):
            try:
                os.remove(os.path.join(out_dir, old))
            except OSError:
                pass

    # The tracker must be told the rate it will ACTUALLY be fed at. With --every 3 it sees
    # a third of the frames, so passing the source fps would make `lost_track_buffer`
    # cover three times less real time and drop tracks through short occlusions.
    eff_fps = max(1.0, fps_in / max(1, every))
    pipe = SafetyPipeline(cfg, frame_rate=int(round(eff_fps)), quiet=True)
    writer = None
    t0 = time.perf_counter()
    frame_no = 0
    processed = 0
    # Keep the frames with the most simultaneous violations — the moments actually worth
    # showing someone, rather than an arbitrary first/last frame.
    stills: list = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if every > 1 and (frame_no - 1) % every:
            continue
        elapsed = frame_no / fps_in                      # clip time, not wall-clock
        res = pipe.process(frame, frame_no, elapsed)
        processed += 1

        if writer is None:
            h, w = res.frame.shape[:2]
            writer = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"),
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     max(1.0, fps_in / max(1, every)), (w, h))
        writer.write(res.frame)

        if res.alerts:
            # Keep the worst frames, but never two from the same moment. A sustained
            # violation plateaus at the same alert count for hundreds of frames, and a
            # plain "top N by count, ties by frame number" picks N *consecutive* frames —
            # three near-identical stills of one incident instead of three incidents.
            min_gap = max(1, int(fps_in * 2))          # at least ~2 s apart
            near = [i for i, (_c, f0, _im) in enumerate(stills)
                    if abs(f0 - frame_no) < min_gap]
            if near:
                i = near[0]
                if len(res.alerts) > stills[i][0]:     # better shot of the same moment
                    stills[i] = (len(res.alerts), frame_no, res.frame.copy())
            else:
                stills.append((len(res.alerts), frame_no, res.frame.copy()))
                stills.sort(key=lambda t: (-t[0], t[1]))
                del stills[max_stills:]

        if total and processed % 25 == 0:
            done = 100.0 * frame_no / total
            if progress:
                print(f"\r  {done:5.1f}%  ({frame_no}/{total} frames)", end="", flush=True)
            if on_progress is not None:
                on_progress(min(1.0, frame_no / float(total)))

    cap.release()
    if writer is not None:
        writer.release()
    if progress:
        print("\r" + " " * 40, end="\r")

    pipe.close(elapsed_s=frame_no / fps_in, frame_no=frame_no)
    report = pipe.report()
    report["clip"] = {"file": os.path.abspath(path), "frames": frame_no,
                      "processed": processed, "fps_in": round(fps_in, 2),
                      "seconds": round(frame_no / fps_in, 1),
                      "analysis_seconds": round(time.perf_counter() - t0, 1)}

    for i, (_n, fno, img) in enumerate(stills, start=1):
        cv2.imwrite(os.path.join(out_dir, f"worst_{i}.jpg"), img)

    import json
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(_summary_text(name, report, pipe))
    return report


def _summary_text(name: str, report: dict, pipe: SafetyPipeline) -> str:
    clip = report.get("clip", {})
    lines = [
        "=" * 68,
        f"SITE CLIP REPORT - {name}",
        "=" * 68,
        f"  length            : {clip.get('seconds', 0)} s "
        f"({clip.get('frames', 0)} frames @ {clip.get('fps_in', 0)} fps)",
        f"  analysis took     : {clip.get('analysis_seconds', 0)} s",
        "",
    ]
    v = report.get("violations", {})
    lines.append(f"  unique person-violations : {v.get('unique_violations', 0)}")
    for kind, n in sorted((v.get("by_type") or {}).items()):
        lines.append(f"    - {kind}: {n}")
    lines.append("")
    text = pipe.format_report()
    if text:
        lines.append(text)
    lines += [
        "",
        "WHAT TO SEND BACK",
        "-" * 68,
        "  1. Did the boxes follow the right people?",
        "  2. Was anyone missed, or flagged wrongly? Note the time in the clip.",
        "  3. Did a worker's name/number change when they came back into view?",
        "  4. Was the overlay readable outdoors, and was anything in the way?",
        "  5. Anything you expected it to notice and it did not?",
        "",
        "  Send annotated.mp4 + this file. The timestamps make issues reproducible.",
        "=" * 68,
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Analyse phone clips recorded at a site (no network needed)")
    ap.add_argument("path", help="a video file, or a folder of them")
    ap.add_argument("--config", default=os.path.join(_ROOT, "phase2", "config.yaml"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "site"))
    ap.add_argument("--every", type=int, default=1,
                    help="process every Nth frame (2-3 makes a long clip much faster)")
    ap.add_argument("--arview", default=None,
                    choices=["composite", "seethrough", "glasses"],
                    help="which AR view to burn into the annotated video")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.arview:
        cfg.arview_mode = args.arview

    if os.path.isdir(args.path):
        clips = sorted(os.path.join(args.path, f) for f in os.listdir(args.path)
                       if f.lower().endswith(VIDEO_EXT))
        if not clips:
            raise SystemExit(f"no video files in {args.path}")
    else:
        clips = [args.path]

    failed = []
    for i, clip in enumerate(clips, start=1):
        print(f"[{i}/{len(clips)}] {os.path.basename(clip)}")
        try:
            rep = analyze_clip(clip, cfg, args.out, every=args.every)
        except (SystemExit, RuntimeError, cv2.error) as e:
            # One unreadable or half-copied file must not throw away the whole batch —
            # a folder pulled off a phone very often has exactly one such file.
            print(f"  [!] skipped: {e}")
            failed.append(os.path.basename(clip))
            continue
        w = rep.get("workers", {})
        print(f"  -> {w.get('workers_seen', 0)} worker(s), "
              f"{w.get('total_violation_episodes', 0)} violation episode(s), "
              f"{w.get('total_violation_s', 0)}s unsafe")
        print(f"  -> {os.path.join(args.out, os.path.splitext(os.path.basename(clip))[0])}")
    if failed:
        print()
        print(f"[!] {len(failed)} clip(s) could not be read: {', '.join(failed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
