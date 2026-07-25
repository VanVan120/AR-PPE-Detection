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
from src.videoout import open_writer                  # noqa: E402

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".webm")

# Roughly how many frames to actually run the detector on, when `every` is left to choose
# itself. At ~120 ms a frame on a CPU that is about two minutes of waiting — long enough to
# be worth a progress bar, short enough that nobody assumes it has hung. A 30 s clip at
# 30 fps is 900 frames, so short clips are still analysed in full.
AUTO_FRAME_BUDGET = 900


class ClipError(RuntimeError):
    """A clip that could not be read. A distinct type because the CLI turns it into a
    skipped file and the phone app turns it into a message on screen — and it must not be
    `SystemExit`, which sails straight through `except Exception` in a worker thread and
    leaves the job stuck on 'analysing' for ever."""


def choose_stride(total: int, every: int = 0, budget: int = AUTO_FRAME_BUDGET) -> int:
    """Frames to skip so a long clip finishes in a predictable time. `every>0` wins."""
    if every and every > 0:
        return int(every)
    if total <= 0:
        return 1                                       # unknown length: do not guess
    return max(1, -(-total // max(1, budget)))         # ceil(total / budget)


def analyze_clip(path: str, cfg, out_root: str, every: int = 0,
                 max_stills: int = 3, progress: bool = True,
                 on_progress=None) -> dict:
    """Run one clip end to end and write the review bundle. Returns the report dict.

    `every=0` (the default) picks a frame stride so that even a long clip finishes in a
    predictable time; pass `every=1` to force every frame. `on_progress(fraction)` is
    called as the clip is consumed — with **-1** when the file does not declare its length,
    so a caller showing a bar can switch it to indeterminate rather than leaving it
    frozen at 0% and looking hung.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ClipError(f"could not open video: {path}")
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    # A phone clip with a broken header can report 0, a NaN, or something absurd. `or` does
    # not catch NaN (it is truthy), and a NaN rate would make every timestamp NaN — the
    # report would then be full of nulls with nothing saying why.
    if not fps_in or fps_in != fps_in or not (0.1 <= fps_in <= 480):
        fps_in = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    every = choose_stride(total, every)

    name = os.path.splitext(os.path.basename(path))[0]
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    # Re-running the same clip reuses this folder. report.json is overwritten, but
    # worst_*.jpg are numbered and the annotated video's extension depends on which
    # encoder this build has — so a re-run that finds fewer incidents, or that picks a
    # different codec, would otherwise leave files from the previous run alongside the new
    # ones and the bundle would quietly mix two.
    for old in os.listdir(out_dir):
        if ((old.startswith("worst_") and old.endswith(".jpg"))
                or old.startswith("annotated.")):
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
    video_path = ""
    codec_name, browser_ok = "", True
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
            # NOT a hardcoded mp4v: that is MPEG-4 Part 2, which no browser plays, so the
            # review clip would arrive as a black rectangle on the phone it was made for.
            writer, video_path, codec_name, browser_ok = open_writer(
                os.path.join(out_dir, "annotated"), eff_fps, (w, h))
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

        if processed % 25 == 0:
            if total:
                done = 100.0 * frame_no / total
                if progress:
                    print(f"\r  {done:5.1f}%  ({frame_no}/{total} frames)",
                          end="", flush=True)
                if on_progress is not None:
                    on_progress(min(1.0, frame_no / float(total)))
            else:
                # No frame count in the header — common for a clip streamed off a phone.
                # Report -1 rather than nothing, so a progress bar goes indeterminate
                # instead of sitting frozen at 0% looking like a hang.
                if progress:
                    print(f"\r  {frame_no} frames", end="", flush=True)
                if on_progress is not None:
                    on_progress(-1.0)

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
                      "every": every,
                      "analysis_seconds": round(time.perf_counter() - t0, 1)}
    report["video"] = {"file": os.path.basename(video_path) if video_path else "",
                       "codec": codec_name, "plays_in_browser": bool(browser_ok)}

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
    ]
    every = int(clip.get("every", 1) or 1)
    if every > 1:
        lines.append(f"  NOTE: every {every}th frame was analysed, to keep a clip this "
                     f"long to a sensible")
        lines.append(f"        wait. A violation therefore has to last about {every}x "
                     f"longer before it")
        lines.append("        is reported. Re-run with --every 1 if exact counts matter.")
    vid = report.get("video", {})
    if vid.get("file"):
        lines.append(f"  annotated video   : {vid['file']}  ({vid.get('codec', '?')})")
    if vid.get("file") and not vid.get("plays_in_browser", True):
        lines.append("  NOTE: this build of OpenCV could only write MPEG-4 Part 2, which")
        lines.append("        web browsers and phones do NOT play. Open it in VLC.")
    lines.append("")
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
        f"  Send {vid.get('file') or 'the annotated video'} + this file. The timestamps "
        f"make issues reproducible.",
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
    ap.add_argument("--every", type=int, default=0,
                    help="process every Nth frame. 0 (default) chooses a stride so a long "
                         "clip finishes in a predictable time; 1 forces every frame")
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
        except (ClipError, SystemExit, RuntimeError, cv2.error) as e:
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
