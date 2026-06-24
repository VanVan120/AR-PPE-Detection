# Phase 2 — Real-Time Video, Tracking & Simulated AR Monitoring

Takes the Phase 1 PPE detector (90+% on the clean test set) and makes it
**deployment-aware**: runs it live on a webcam or clip, tracks each person with a
persistent ID, collapses noisy per-frame detections into **deduplicated,
debounced per-person safety violations**, and draws a **simulated AR heads-up
overlay** — the laptop stand-in for the eventual glasses view. Then it runs the
**reality-check**: how well does that clean-set accuracy survive worn-camera video?

See [proposal_phase2.md](proposal_phase2.md) for the full scope and non-goals.

## Pipeline

```
 webcam / clip ─▶ detect (YOLO) ─▶ track persons (ByteTrack) ─▶ compliance ─▶ AR overlay ─▶ screen / mp4
                    src/detector     src/tracker                 src/compliance   src/overlay
                                                                 (associate + debounce)
                     └────────────────────── src/metrics: FPS + per-stage latency ──────────────────────┘
```

The trained model exposes **5 classes**: `Helmet, No-Helmet, No-Vest, Person, Vest`.
The safety-critical **violations** are `No-Helmet` (high) and `No-Vest` (medium).
A violation box is attributed to the tracked person whose box most contains it, so
alerts read "**Person #5: No hard hat**", deduplicated — not one alert per frame.

> Note: the proposal's example config used placeholder names (`NO-Hardhat`,
> `NO-Safety Vest`, `NO-Mask`). The shipped [config.yaml](config.yaml) uses the
> model's real class names. There is no mask class in this model.

## Setup

```bash
cd phase2
pip install -r requirements.txt          # ultralytics, supervision, opencv, numpy, pyyaml
python run.py --check                     # verify deps, model, class names, source, device
```

For GPU, install a CUDA torch build from https://pytorch.org first. CPU works too
(lower FPS — the perf summary reports the number).

`models/best.pt` is the Phase 1 `best_refined.pt`. Point `weights:` elsewhere in
config.yaml to use a different checkpoint.

## Run

```bash
python run.py                              # live demo on the configured source (webcam 0)
python run.py --source data/clips/walk.mp4 # run on a clip instead
python run.py --source 1                   # a different webcam
python run.py --record                     # also save outputs/session_video.mp4
python run.py --no-display --max-frames 300 --record   # headless render to video
```

**Live controls:**  `q` / `ESC` quit · `s` screenshot · `r` toggle recording.

The HUD shows live FPS, per-stage latency (detect/track/compliance/render), person count,
active-violation counts by severity, and a colour-coded alert list. Each person's
box is **red** (high), **orange** (medium), or **green** (compliant). On exit it
prints a performance summary and a session summary (violations per type / person).

## Reality-check (the make-or-break)

Quantifies the domain gap between worn-camera footage and the clean benchmark.

1. **Record** a short first-person clip: phone at chest/head height, walk through a
   site (or a mock-up) past people with and without hard hats / vests. Save it to
   `data/clips/`.
2. **Behavioural check** (no labels needed):
   ```bash
   python run.py --reality-check data/clips/firstperson.mp4
   ```
   Reports, per violation class, how often it's detected and at what confidence vs
   the Phase 1 benchmark recall — flagging classes that collapse on real video.
3. **Measured recall** (optional, stronger): hand-label a handful of frames and
   re-run. Frame indices are **0-based into the *processed* stream** — with `--every N`
   that is `raw_frame // N` (so with no `--every`, just the raw frame number). Indices
   out of range are warned and ignored, and a run with no usable labels reports
   `INCONCLUSIVE` (never a false "holds"). Label format (frame index → classes present):
   ```json
   { "labels": { "30": ["No-Helmet", "Person"], "60": ["Person"] } }
   ```
   ```bash
   python run.py --reality-check data/clips/firstperson.mp4 --labels labels.json
   ```
   Writes `outputs/reality_check.json` with an honest verdict: does 90+% hold, or
   is a targeted fine-tuning pass on first-person data the next step? (The measured
   number is *frame-presence* recall — an optimistic upper bound on instance recall,
   noted as such in the report.)

A throwaway demo clip can be generated from the Phase 1 dataset with
[tools/make_sample_clip.py](tools/make_sample_clip.py) (it's a slideshow with
synthetic camera motion, **not** real worn-camera footage — use a real recording
for an honest reality-check).

## Config (`config.yaml`)

| key | meaning |
|---|---|
| `weights` | trained detector path (config-relative) |
| `source` | `0` = webcam, or a video path |
| `confidence_threshold` | min detection confidence |
| `imgsz` | inference resolution (multiple of 32); lower (480/320) = faster on CPU |
| `device` | `auto` / `cpu` / `cuda` / `mps` |
| `person_class` | model's person class name (must match the model) |
| `violation_rules` | `class_name → {severity, label}` (names must match the model) |
| `association_containment` | min fraction of a violation box inside a person box to attribute it |
| `debounce_frames` | frames a violation must persist before it fires |
| `clear_frames` | frames absent before an active violation clears |
| `lost_track_buffer` | frames an ID survives through occlusion |
| `save_output_video` | write an annotated session video |

## Outputs

- `outputs/session_video.mp4` — annotated session (with `--record` / `save_output_video`)
- `outputs/screenshot_*.jpg` — `s`-key captures
- `outputs/reality_check.json` — domain-gap assessment

## Troubleshooting

- **No webcam / can't open** → clear message; pass `--source <path>` or a different index.
- **Bad weights path** → clear message at load; fix `weights:` in config.yaml.
- **A rule never fires** → run `python run.py --check`; it validates every rule
  name against the model's actual classes and reports any that can't fire.
- **No display (headless/SSH)** → use `--no-display` (optionally with `--record`).
