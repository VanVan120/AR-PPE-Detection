#!/usr/bin/env python3
"""Phase 2 — real-time PPE detection, tracking, compliance & simulated AR overlay.

One entry point. Loads the Phase 1 detector and runs it live on a webcam or a
video file: detect -> track (persistent person IDs) -> per-person compliance
(debounced) -> AR heads-up overlay, with live FPS and per-stage latency.

    python run.py                          # live demo on the configured source
    python run.py --check                  # environment / model / source readiness
    python run.py --source data/clips/x.mp4  # run on a specific clip
    python run.py --source 1               # a different webcam index
    python run.py --record                 # also save an annotated session video
    python run.py --reality-check data/clips/firstperson.mp4   # domain-gap report
    python run.py --reality-check clip.mp4 --labels labels.json # measured recall

Live controls:  q / ESC = quit   ·   s = screenshot   ·   r = toggle recording
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# MUST run before anything imports ultralytics. ultralytics latches
# AUTOINSTALL from this variable at IMPORT time (utils/__init__.py), so setting it
# later — e.g. inside src/detector.py — is silently ignored on any path that
# imported ultralytics first (such as `--check`'s dependency probe). Without it,
# loading an exported .onnx on a CUDA machine makes ultralytics pip-install a
# 241 MB onnxruntime-gpu and, when the host's CUDA doesn't match that build,
# break ONNX inference outright. A live safety demo must not reshape its own
# environment mid-run.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

from src.config import (load_config, validate_config, validate_against_model,
                        resolve_device, Config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time PPE detection + tracking + AR overlay")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    p.add_argument("--check", action="store_true", help="run readiness checks and exit")
    p.add_argument("--source", default=None, help="override config source (webcam index or video path)")
    p.add_argument("--record", action="store_true", help="save an annotated session video to outputs/")
    p.add_argument("--no-display", action="store_true", help="headless: process without a window")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = unlimited)")
    p.add_argument("--reality-check", default=None, metavar="CLIP",
                   help="run the domain-gap reality-check on a clip and exit")
    p.add_argument("--labels", default=None, help="hand-label JSON for measured recall (reality-check)")
    p.add_argument("--every", type=int, default=1, help="reality-check: process every Nth frame")
    return p.parse_args(argv)


def _source_value(cfg: Config, override):
    """Resolve the value handed to FrameSource (override wins over config)."""
    if override is None:
        return cfg.resolved_source()
    if str(override).isdigit():
        return int(override)
    return override  # path, relative to CWD


# ---------------------------------------------------------------------------
# --check
# ---------------------------------------------------------------------------
def run_check(cfg: Config, args: argparse.Namespace) -> int:
    print("=" * 64)
    print(" Phase 2 readiness check")
    print("=" * 64)
    ok = True

    v = sys.version_info
    py_ok = (v.major, v.minor) >= (3, 11)
    print(f"[{'ok' if py_ok else 'warn'}] Python {v.major}.{v.minor}.{v.micro}")

    for mod, pip_name in [("ultralytics", "ultralytics"), ("supervision", "supervision"),
                          ("cv2", "opencv-python"), ("numpy", "numpy"), ("yaml", "PyYAML"),
                          ("torch", "torch")]:
        try:
            __import__(mod)
            print(f"[ ok ] import {mod}")
        except Exception as e:
            ok = False
            print(f"[FAIL] import {mod} — install '{pip_name}' ({e})")

    print(f"[ ok ] compute device: {resolve_device(cfg.device_pref)}")

    print("-" * 64)
    for issue in validate_config(cfg):
        print(issue)
        if issue.level == "error":
            ok = False

    # Model load + class-name validation (the placeholder-name trap).
    print("-" * 64)
    try:
        from src.detector import Detector
        det = Detector(cfg.weights_path, cfg.device, cfg.confidence_threshold, cfg.imgsz)
        print(f"[ ok ] model loaded: {os.path.basename(cfg.weights_path)} "
              f"({len(det.names)} classes: {sorted(det.names.values())})")
        for issue in validate_against_model(cfg, det.names):
            print(issue)
            if issue.level == "error":
                ok = False
    except Exception as e:
        ok = False
        print(f"[FAIL] could not load model: {e}")

    # Source open (best-effort; a missing webcam is a warning, not a hard fail).
    print("-" * 64)
    try:
        from src.source import FrameSource, SourceError
        src_val = _source_value(cfg, args.source)
        try:
            fs = FrameSource(src_val, target_fps=cfg.target_fps)
            print(f"[ ok ] source opened: {src_val}  "
                  f"({fs.width}x{fs.height} @ {fs.fps:.0f}fps"
                  f"{', ' + str(fs.frame_count) + ' frames' if not fs.is_live else ', live'})")
            fs.release()
        except SourceError as e:
            print(f"[warn] source not available: {e}")
    except Exception as e:
        print(f"[warn] source check skipped: {e}")

    # Optional features
    if cfg.workid_enabled or cfg.activity_enabled or cfg.identity_enabled:
        print("-" * 64)
    if cfg.identity_enabled:
        if not cfg.identity_appearance:
            print("[ ok ] Worker identity: badge-only (appearance re-ID off)")
        else:
            try:
                from src.reid import build_embedder
                emb = build_embedder(cfg.identity_method, device="cpu")
                print(f"[ ok ] Worker identity: {emb.name} appearance re-ID · "
                      f"threshold {cfg.identity_match_threshold} · "
                      f"names survive track changes and re-entry")
            except Exception as e:
                print(f"[warn] Worker identity check failed: {e}")
        if not cfg.workid_enabled:
            print("       (no ArUco badges configured — workers appear as "
                  "'Worker 1', 'Worker 2', ... Enable workid for real names.)")
    if cfg.workid_enabled:
        try:
            from src.workid import aruco_available
            if aruco_available():
                print(f"[ ok ] Work ID: cv2.aruco present · {cfg.workid_dictionary} · "
                      f"{len(cfg.workid_markers)} worker(s) mapped")
            else:
                print("[warn] Work ID enabled but cv2.aruco missing — "
                      "install opencv-contrib-python (disabled at run time otherwise)")
        except Exception as e:
            print(f"[warn] Work ID check failed: {e}")
    if cfg.activity_enabled:
        if cfg.activity_backend == "placeholder":
            print("[ ok ] Activity: 'placeholder' scaffold (returns 'pending-dataset')")
        elif cfg.activity_backend == "kinetics":
            try:
                __import__("torchvision")
                print("[ ok ] Activity: 'kinetics' generic demo (downloads weights on first run)")
            except Exception:
                print("[warn] Activity backend 'kinetics' needs torchvision (missing)")
        elif cfg.activity_backend == "assembly101":
            ck, ac = cfg.activity_checkpoint, cfg.activity_actions_csv
            if ck and os.path.isfile(ck) and ac and os.path.isfile(ac):
                print(f"[ ok ] Activity: 'assembly101' TAS model ({os.path.basename(ck)}) "
                      "(live uses a STAND-IN extractor unless a TSM extractor is supplied)")
                pm = cfg.activity_procedure_model
                if pm and os.path.isfile(pm):
                    print(f"[ ok ] Activity: order-based mistake detection on ({os.path.basename(pm)})")
                elif pm:
                    print(f"[warn] activity.procedure_model not found: {pm}")
                am = cfg.activity_anticipation_model
                if am and os.path.isfile(am):
                    print(f"[ ok ] Activity: next-step anticipation on ({os.path.basename(am)})")
                elif am:
                    print(f"[warn] activity.anticipation_model not found: {am}")
            else:
                print("[warn] Activity 'assembly101' needs valid activity.checkpoint + "
                      "activity.actions_csv (train one in phase3_activity)")
        else:
            print(f"[warn] Activity: unknown backend '{cfg.activity_backend}'")

    print("=" * 64)
    if ok:
        print("READY — run `python run.py` for the live demo.")
        return 0
    print("NOT READY — resolve the [FAIL] items above.")
    return 1


# ---------------------------------------------------------------------------
# reality-check
# ---------------------------------------------------------------------------
def run_reality(cfg: Config, args: argparse.Namespace) -> int:
    from src.reality_check import run_reality_check
    from src.source import SourceError
    clip = args.reality_check
    try:
        run_reality_check(cfg, clip, every=max(1, args.every),
                          max_frames=args.max_frames, labels_path=args.labels)
        return 0
    except SourceError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[FAIL] reality-check failed: {e}", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# live demo
# ---------------------------------------------------------------------------
def _save_image(image, path: str) -> None:
    import cv2
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(path)


def _tracked_arrays(tracked):
    """(track_ids, boxes) for the persons carrying a tracker id this frame."""
    ids, boxes = [], []
    if len(tracked) == 0 or tracked.tracker_id is None:
        return ids, boxes
    for box, tid in zip(tracked.xyxy, tracked.tracker_id):
        if tid is None:
            continue
        ids.append(int(tid))
        boxes.append([float(v) for v in box])
    return ids, boxes


def _roster_hud(identity, ident_by_track: dict, fc) -> list:
    """Rows for the on-screen WORKERS panel: who is known, who is here, who is unsafe.

    Present workers are listed first (a worker who has stepped out of frame is still
    remembered, but shown dimmed), and each row records whether the name came from a
    badge or from an appearance match only.
    """
    if identity is None:
        return []
    violating_uids = set()
    for person in getattr(fc, "persons", []):
        res = ident_by_track.get(person.tracker_id)
        if res is not None and person.active:
            violating_uids.add(res.uid)
    present = {r.uid for r in ident_by_track.values()}
    rows = []
    for w in identity.roster():
        rows.append({"label": w.label, "badge": w.marker_confirmed,
                     "present": w.uid in present,
                     "violating": w.uid in violating_uids})
    rows.sort(key=lambda r: (not r["present"], not r["violating"], r["label"]))
    return rows


def run_live(cfg: Config, args: argparse.Namespace) -> int:
    import cv2
    from src.source import FrameSource, SourceError
    from src.detector import Detector
    from src.tracker import PersonTracker
    from src.compliance import ComplianceMonitor
    from src.metrics import PerfTracker
    from src import overlay
    from src.eventlog import EventLog

    out_dir = os.path.join(cfg.config_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    # --- model --------------------------------------------------------------
    try:
        print(f"Loading detector '{os.path.basename(cfg.weights_path)}' on {cfg.device}...")
        detector = Detector(cfg.weights_path, cfg.device, cfg.confidence_threshold, cfg.imgsz)
    except Exception as e:
        print(f"[FAIL] could not load weights '{cfg.weights_path}': {e}", file=sys.stderr)
        return 1
    detector.warmup()

    # Resolve config class NAMES against the model's actual class ids.
    person_ids = detector.class_ids_for([cfg.person_class])
    rules_by_id = {}
    for name, rule in cfg.violation_rules.items():
        for cid in detector.class_ids_for([name]):
            rules_by_id[cid] = rule
    if not person_ids:
        print(f"[FAIL] person_class '{cfg.person_class}' is not a model class "
              f"{sorted(detector.names.values())}. Fix config.yaml.", file=sys.stderr)
        return 1
    if not rules_by_id:
        print(f"[FAIL] none of the violation_rules match a model class "
              f"{sorted(detector.names.values())}. Fix config.yaml.", file=sys.stderr)
        return 1
    # At least one rule resolved, so we keep running — but a rule whose name is not a
    # model class would sit silently dead all session. --check hard-fails on it; on a
    # normal run we warn loudly so a typo can't hide a missing safety check.
    for issue in validate_against_model(cfg, detector.names):
        if issue.level == "error":
            print(f"[warn] {issue.message}  (run `python run.py --check`)", file=sys.stderr)

    # --- source -------------------------------------------------------------
    src_val = _source_value(cfg, args.source)
    try:
        source = FrameSource(src_val, target_fps=cfg.target_fps)
    except SourceError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        return 1

    tracker = PersonTracker(person_ids, list(rules_by_id.keys()),
                            frame_rate=int(source.fps) or cfg.target_fps,
                            lost_track_buffer=cfg.lost_track_buffer)
    monitor = ComplianceMonitor(rules_by_id, cfg.debounce_frames, cfg.clear_frames,
                                cfg.association_containment, cfg.lost_track_buffer)
    perf = PerfTracker()

    # --- optional: Work ID, event log, activity recognition -----------------
    binder = None
    if cfg.workid_enabled:
        try:
            from src.workid import WorkIdBinder, aruco_available
            if not aruco_available():
                raise RuntimeError("cv2.aruco unavailable — install opencv-contrib-python")
            binder = WorkIdBinder(cfg.workid_dictionary, cfg.workid_markers,
                                  cfg.workid_containment,
                                  gc_after=max(cfg.lost_track_buffer, 30) * 3,
                                  reacquire_after=cfg.lost_track_buffer)
            print(f"Work ID: ArUco {cfg.workid_dictionary}, {len(cfg.workid_markers)} worker(s) mapped")
        except Exception as e:
            print(f"[warn] Work ID disabled: {e}", file=sys.stderr)

    # Persistent worker identity sits ABOVE the tracker: it survives track-id churn,
    # so a worker's name and violation history follow the person, not the track.
    identity = None
    history = None
    if cfg.identity_enabled:
        try:
            from src.identity import IdentityManager
            from src.reid import build_embedder
            from src.workerlog import WorkerHistory
            embedder = build_embedder(cfg.identity_method, device=cfg.device)
            identity = IdentityManager(
                embedder,
                match_threshold=cfg.identity_match_threshold,
                margin=cfg.identity_margin,
                max_exemplars=cfg.identity_max_exemplars,
                forget_after=cfg.identity_forget_after,
                min_box_height=cfg.identity_min_box_height,
                appearance_enabled=cfg.identity_appearance)
            history = WorkerHistory()
            mode = f"{embedder.name} appearance" if cfg.identity_appearance else "badge only"
            print(f"Worker identity: ON ({mode}, threshold {cfg.identity_match_threshold})")
        except Exception as e:
            print(f"[warn] worker identity disabled: {e}", file=sys.stderr)
            identity, history = None, None

    elog = None
    if cfg.event_log_path:
        try:
            elog = EventLog(cfg.event_log_path)
            print(f"Event log -> {cfg.event_log_path}")
        except Exception as e:
            print(f"[warn] event log disabled: {e}", file=sys.stderr)

    activity_mod = None
    if cfg.activity_enabled:
        try:
            from src.activity import ActivityModule
            activity_mod = ActivityModule(cfg.activity_backend, cfg.activity_clip_len,
                                          cfg.activity_stride, device=cfg.device,
                                          checkpoint=cfg.activity_checkpoint,
                                          actions_csv=cfg.activity_actions_csv,
                                          procedure_model=cfg.activity_procedure_model,
                                          anticipation_model=cfg.activity_anticipation_model)
            note = ("  (SCAFFOLD — returns 'pending-dataset')" if activity_mod.backend == "placeholder"
                    else "  (generic Kinetics demo — NOT construction steps)")
            print(f"Activity: backend '{activity_mod.backend}', clip {cfg.activity_clip_len}"
                  f"@stride {cfg.activity_stride}{note}")
        except Exception as e:
            print(f"[warn] activity recognition disabled: {e}", file=sys.stderr)

    headless = args.no_display
    window = "AR Safety Monitor — Phase 2"
    recording = bool(cfg.save_output_video or args.record)
    writer = None
    saved_path = None
    record_failed = False
    shot_n = 0

    print(f"Source: {cfg.source_display if args.source is None else src_val}  "
          f"({source.width}x{source.height} @ {source.fps:.0f}fps)")
    print(f"Tracking class: {cfg.person_class}   violation rules: "
          f"{[r.class_name for r in rules_by_id.values()]}")
    if not headless:
        print("Controls:  q/ESC = quit   s = screenshot   r = toggle recording")

    def _ensure_writer(frame):
        """Open the writer lazily, with a codec fallback. If no codec is available
        (e.g. a headless OpenCV build without mp4v/FFMPEG), disable recording rather
        than silently no-op every write() and then falsely claim a saved file."""
        nonlocal writer, saved_path, recording, record_failed
        if writer is not None or record_failed:
            return writer
        h, w = frame.shape[:2]
        fps = source.fps if source.fps > 0 else cfg.target_fps
        base = os.path.join(out_dir, "session_video")
        for fourcc_name, path in (("mp4v", base + ".mp4"), ("XVID", base + ".avi")):
            cand = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), fps, (w, h))
            if cand.isOpened():
                writer, saved_path = cand, path
                print(f"Recording -> {path}")
                return writer
            cand.release()
        record_failed = True
        recording = False
        print("[warn] could not open a video writer (no available codec) — recording disabled.",
              file=sys.stderr)
        return None

    t_start = time.perf_counter()
    frame_no = 0
    rc = 0
    prev_activity_res = None      # identity guard: log each mistake once, not every frame
    try:
        for frame in source.frames():
            perf.start_frame()
            frame_no += 1

            with perf.stage("detect"):
                dets = detector.detect(frame)
            with perf.stage("track"):
                tracked, violations = tracker.update(dets)
            with perf.stage("compliance"):
                fc = monitor.update(tracked, violations)

            # `badge_of` is ONLY what an ArUco tag actually confirmed. `worker_of` is the
            # display identity, which may additionally come from an appearance match.
            # They are kept apart so the event log's `identified` flag stays truthful:
            # an appearance guess must never be recorded as a confirmed identification.
            badge_of = {}
            if binder is not None:
                with perf.stage("workid"):
                    badge_of = binder.resolve(frame, tracked)

            worker_of = dict(badge_of)
            ident_by_track = {}
            if identity is not None:
                with perf.stage("identity"):
                    tids, boxes = _tracked_arrays(tracked)
                    # NOTE: `frame` is still the clean camera image here — the overlay is
                    # drawn later, so appearance is never described from our own HUD.
                    ident_by_track = identity.update(frame, tids, boxes,
                                                     marker_labels=badge_of)
                worker_of = {t: r.label for t, r in ident_by_track.items()}
                if history is not None:
                    history.update(frame_no, time.perf_counter() - t_start, fc,
                                   ident_by_track)

            activity_res = None
            if activity_mod is not None:
                with perf.stage("activity"):
                    activity_res = activity_mod.update("ego", frame)
                # A new ActivityResult object means fresh inference this frame (update()
                # returns the cached result otherwise). Log a workflow mistake once, when
                # it first appears — not on every frame the cached result persists.
                if activity_res is not None and activity_res is not prev_activity_res:
                    if getattr(activity_res, "mistake", False) and elog is not None:
                        elog.log_mistake(frame_no, time.perf_counter() - t_start,
                                         activity_res.step, getattr(activity_res, "detail", ""))
                        print(f"[MISTAKE] {getattr(activity_res, 'detail', '') or activity_res.step}")
                    prev_activity_res = activity_res

            # Console + structured event log (fires once per (person, violation)).
            for ev in fc.new_events:
                who = worker_of.get(ev.person_id) or f"Person #{ev.person_id}"
                print(f"[ALERT] {who}: {ev.label} ({ev.severity.upper()})")
                if elog is not None:
                    # badge_of, not worker_of: only a real tag counts as "identified".
                    elog.log_violation(frame_no, time.perf_counter() - t_start,
                                       ev.person_id, badge_of.get(ev.person_id), ev)

            hud = {"fps": perf.live_fps, "stage_ms": perf.live_stage_ms(),
                   "recording": recording, "device": cfg.device, "activity": activity_res,
                   "workers": _roster_hud(identity, ident_by_track, fc)}
            with perf.stage("render"):
                overlay.annotate(frame, fc, hud, worker_of)
                if recording:
                    wr = _ensure_writer(frame)
                    if wr is not None:
                        wr.write(frame)
                if not headless:
                    try:
                        cv2.imshow(window, frame)
                        key = cv2.waitKey(1) & 0xFF
                    except cv2.error:
                        print("[warn] no display available — switching to headless.")
                        headless = True
                        key = 255
                    if key in (ord("q"), 27):           # q or ESC
                        break
                    elif key == ord("s"):
                        shot_n += 1
                        sp = os.path.join(out_dir, f"screenshot_{shot_n:03d}.jpg")
                        _save_image(frame, sp)
                        print(f"Saved {sp}")
                    elif key == ord("r"):
                        recording = not recording
                        print(f"Recording {'ON' if recording else 'OFF'}")

            perf.end_frame()
            if args.max_frames and perf.frames >= args.max_frames:
                print(f"Reached --max-frames {args.max_frames}.")
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"[FAIL] runtime error: {e}", file=sys.stderr)
        rc = 1
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if history is not None:
            # Close any violation still running so its duration is real, not zero.
            history.close(elapsed_s=time.perf_counter() - t_start, frame_no=frame_no)
            print(history.format_report())
            if cfg.identity_report:
                try:
                    history.save_json(cfg.identity_report)
                    print(f"Per-worker report -> {cfg.identity_report}")
                except OSError as e:
                    print(f"[warn] could not write the worker report: {e}", file=sys.stderr)
        if identity is not None:
            s = identity.stats
            print(f"Identity: {len(identity.workers)} worker(s) tracked · "
                  f"{s['appearance_matches']} appearance re-match(es) · "
                  f"{s['promotions']} promoted by badge · {s['new_workers']} new")
        if elog is not None:
            if binder is not None:
                elog.log_bindings(binder.all_bindings())   # reconcile anonymous rows
            elog.close()
        if not headless:
            try:
                import cv2 as _cv2
                _cv2.destroyAllWindows()
            except Exception:
                pass

    # --- summaries ----------------------------------------------------------
    perf.print_summary(cfg.device)
    _print_session_summary(monitor.summary(), binder.all_bindings() if binder else {})
    if elog is not None and elog.enabled:
        mist = f", {elog.mistakes} workflow mistake(s)" if elog.mistakes else ""
        print(f"\nWrote {elog.count} violation event(s){mist} -> {cfg.event_log_path}")
    if saved_path is not None:
        print(f"\nSaved annotated video -> {saved_path}")
    return rc


def _print_session_summary(summary: dict, bindings: "dict | None" = None) -> None:
    bindings = bindings or {}
    print("\n" + "=" * 60)
    print(" Session summary")
    print("=" * 60)
    n_unique = summary.get('unique_violations', 0)
    cap_note = "+ (session cap reached; count truncated)" if summary.get("truncated") else ""
    print(f"  unique person-violations fired : {n_unique} {cap_note}".rstrip())
    by_type = summary.get("by_type", {})
    if by_type:
        print("  by type:")
        for t, n in by_type.items():
            print(f"    - {t}: {n}")
    by_person = summary.get("by_person", {})
    if by_person:
        print("  by worker:" if bindings else "  by person:")
        for pid, types in by_person.items():
            who = bindings.get(pid, f"#{pid}")
            print(f"    - {who}: {', '.join(types)}")


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.check:
        return run_check(cfg, args)
    if args.reality_check:
        return run_reality(cfg, args)
    return run_live(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
