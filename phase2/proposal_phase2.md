# Proposal: Real-Time Video & Tracking — From Detector to AR Monitoring (Phase 2)

## 1. Context

Phase 1 produced a trained construction-safety detector scoring **90+% on all metrics on the clean test set.** That is a strong *static-image* detector — but the project's goal is a worn-camera, real-time AR system, and clean-benchmark accuracy does not automatically survive live, moving, first-person video.

This phase does three things:

1. Runs the trained detector in **real time** on a video source (webcam or clip), with **tracking** so detections are stable per person instead of flickering.
2. Turns raw per-frame detections into **deduplicated safety-violation events** shown in a **simulated AR overlay**.
3. Runs the **reality-check**: quantifies how performance on real-time / first-person footage compares to the clean benchmark — i.e. measures the domain gap *before* anything is built on top.

The webcam/clip here is the laptop stand-in for the eventual glasses camera. The point of this phase is to be deployment-aware: prove the detector works on the kind of video AR actually produces.

## 2. Scope — what to build

A Python tool that:

1. Loads the trained detector and runs inference on a video source: **live webcam OR a video file**.
2. Adds **multi-object tracking** so each person/object keeps a persistent ID across frames.
3. Applies **compliance logic**: collapses per-frame detections into **deduplicated per-person violation events** (e.g. "person #5: NO-Hardhat") with a severity, instead of one alert per frame.
4. Renders a **live annotated overlay** — a simulated AR heads-up view with boxes, IDs, and active safety warnings.
5. **Instruments performance**: live FPS and per-stage latency (detect / track / render).
6. Runs a **reality-check** comparing detection behaviour on a self-recorded first-person clip against the clean benchmark.

## 3. Non-goals — do NOT build these

- ❌ Daily-report / VLM reporting (that is the next phase)
- ❌ Model export or quantization for device (TFLite / CoreML / ONNX / TensorRT) (a later phase)
- ❌ Real AR-glasses hardware or SDK integration
- ❌ Retraining the model — this phase consumes the trained weights as-is. If the reality-check exposes a gap, fixing it (targeted fine-tuning/augmentation) is a separate follow-up, not part of this build.
- ❌ A backend server, database, or cloud sync
- ❌ Multi-camera fusion

If something drifts toward the full system, stop and leave it for a future proposal.

## 4. Tech stack — decisions already made (don't deliberate)

- **Language:** Python 3.11+
- **Inference:** `ultralytics` — load the trained YOLO `.pt` weights from Phase 1.
- **Tracking:** `supervision` (Roboflow) — ByteTrack tracker + annotation utilities.
- **Video / display / overlay:** OpenCV (capture, window, HUD drawing).
- **Config:** YAML (weights path, source, thresholds, class→rule+severity mapping, debounce).
- **Device:** auto-detect CUDA/MPS, fall back to CPU.

## 5. Project structure

```
.
├── proposal_phase2.md
├── README.md
├── requirements.txt
├── config.yaml
├── data/
│   └── clips/                  # self-recorded first-person test videos
├── models/
│   └── best.pt                 # trained detector weights (or point config at runs/.../best.pt)
├── src/
│   ├── config.py               # load + validate config.yaml
│   ├── source.py               # webcam / video-file frame source
│   ├── detector.py             # load trained YOLO, per-frame inference
│   ├── tracker.py              # ByteTrack via supervision, persistent IDs
│   ├── compliance.py           # per-person violation state + dedup + debounce
│   ├── overlay.py              # simulated AR HUD on the live feed
│   ├── metrics.py              # FPS + per-stage latency
│   └── reality_check.py        # run on a clip, log stats, compare to benchmark
├── run.py                      # live demo entry point
└── outputs/
    ├── session_video.mp4       # optional recorded session
    └── reality_check.json      # domain-gap assessment
```

## 6. Build phases

Build and verify one phase before the next. Commit after each.

**Phase 0 — Scaffold + load model.** Structure, `requirements.txt`, `config.yaml`, README skeleton. Install deps. Load the trained weights and run a single-image sanity check — confirm detections on a known test image match Phase 1 behaviour. `run.py --check` verifies: weights load, a video source opens, packages import, device detected.

**Phase 1 — Real-time video pipeline.** Read frames from the configured source (webcam index or video file). Run detection per frame, draw boxes + labels, display the live annotated feed, and show a live **FPS counter**. Respect the confidence threshold from config.

**Phase 2 — Tracking.** Add ByteTrack via `supervision` so each detection gets a persistent track ID across frames. Boxes should stay attached to the same person as they move; IDs should persist through brief occlusions.

**Phase 3 — Compliance logic.** Maintain per-tracked-person violation state. A violation (e.g. `NO-Hardhat`) must **persist for `debounce_frames`** before it fires (kills single-frame flicker), and **clears after `clear_frames`** compliant/absent. Each active violation carries a severity from config. Output is a stable, deduplicated list of "who is currently violating what" — not one alert per frame.

**Phase 4 — Simulated AR overlay.** Render a heads-up display on the live feed: active violations listed/colour-coded by severity, attached to the offending tracked person. This is the "AR glasses view" demonstrated on a laptop. Keep it OpenCV-simple — no UI framework. Add a key to screenshot / toggle recording.

**Phase 5 — Performance instrumentation.** Time each stage (detect / track / render) and report end-to-end FPS. Print a summary on exit (avg/min FPS, per-stage latency). This is the data that later decides whether the model is fast enough for a device.

**Phase 6 — Reality-check (the make-or-break).** Run the detector on a **self-recorded first-person clip** (see prerequisites). Log detection stats across the clip. **Optionally hand-label a handful of frames** to estimate real-world recall on the violation classes, and write `reality_check.json` comparing it to the Phase 1 benchmark numbers. The deliverable is an honest answer to: *does the 90+% hold on worn-camera video, or is there a domain gap to close?*

## 7. Config (`config.yaml`)

> **Note (implementation):** the proposal's example `violation_rules` used placeholder
> names (`NO-Hardhat`, `NO-Safety Vest`, `NO-Mask`). The actual trained model exposes
> `Helmet, No-Helmet, No-Vest, Person, Vest`, so the shipped `config.yaml` maps
> `No-Helmet` (high) and `No-Vest` (medium). There is no mask class in the model.

## 8. Acceptance criteria

- A single command (`python run.py`) runs the trained model **live on a webcam or a video file** with a real-time annotated overlay.
- Each person gets a **stable track ID**; violations are **deduplicated per person**, not per frame, with debounce so they don't flicker.
- A **simulated AR HUD** shows active safety warnings colour-coded by severity.
- **Live FPS and per-stage latency** are reported; runs on CPU (note expected FPS) and uses GPU/MPS if present.
- A **`reality_check.json`** quantifies how first-person-footage performance compares to the Phase 1 benchmark.
- Degrades gracefully: no webcam → clear message; bad weights path → clear message; not a stack trace.

## 9. Stretch goals — only if the above fully works and time remains

- Save an annotated output video of a session to `outputs/`.
- Print a session summary at exit (counts per violation type, per person) — a precursor to the daily report.
- Per-class confidence thresholds.

## 10. What comes next (do NOT build now)

Once real-time + tracking + the reality-check are done: the AI daily safety report (VLM tier), then model export/quantization with on-device latency measurement, then the polished AR-glasses deployment. If Phase 6 reveals a domain gap, a short targeted fine-tuning pass on first-person-style data comes first.
