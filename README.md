# AI-Empowered Workflow Monitoring for AR-Glasses Inspection

A computer-vision system for a civil-engineering **AR-glasses inspection assistant**. It
does two complementary jobs on a live egocentric video feed:

1. **PPE safety** — detect who is and isn't wearing a **hard hat** and **hi-vis vest**, and
   raise deduplicated per-person violation alerts in real time.
2. **Workflow understanding** — recognise the **assembly step** underway, flag **out-of-order**
   steps, and **anticipate the next** step — the "you're here / next: …" guidance an AR
   assistant surfaces.

Built in three phases, each self-contained and independently runnable:

| Phase | What it is | Where | State |
|---|---|---|---|
| **1 — Detector** | YOLOv8 fine-tuned to **90%+ on every metric, all 5 PPE classes** | repo root | ✅ |
| **2 — Real-time AR** | detector + person tracking + per-person compliance + **AR HUD overlay** | [`phase2/`](phase2/) | ✅ |
| **3 — Workflow understanding** | Assembly101 step recognition + mistake detection + next-step anticipation | [`phase3_activity/`](phase3_activity/) | ✅ |

> Summer-internship project for *AI-Empowered Dynamic Workflow Monitoring for Inspection via
> AR Glasses*.

---

## Quick start

**Prerequisites:** Python **3.11+**. A GPU is optional (everything runs on CPU — just slower).

### Easiest (Windows): double-click `START.bat`
A menu launcher handles everything — no command line needed:

```
[1] First-time setup    installs everything into a private .venv (once)
[2] Readiness check      shows what's installed / missing
[3] Run LIVE demo        webcam
[4] Run demo on a VIDEO  point it at a file
[5] Verify Phase 3       runs the tests (no downloads needed)
```

Run **[1]** once, then **[2]** to confirm it's ready, then **[3]**. On macOS/Linux, or to run
things by hand, use the steps below.

### 1. Verify the code with zero downloads
The Phase 3 logic is unit-tested against synthetic fixtures, so you can confirm it all works
**without any model weights or datasets**:

```bash
pip install numpy                                   # all three below need only numpy
python phase3_activity/tests/test_tas.py            # ALL_TAS True
python phase3_activity/tests/test_mistake.py        # ALL_MISTAKE True
python phase3_activity/tests/test_anticipation.py   # ALL_ANTICIPATION True
pip install torch                                   # the pipeline test also trains a tiny model
python phase3_activity/tests/test_pipeline.py       # ALL_PIPELINE True
```

### 2. Run the live AR safety demo (Phase 2)
Needs the trained detector at `phase2/models/best.pt` (see **[Weights & data](#weights--data)**)
and a webcam or a video file.

```bash
cd phase2
pip install -r requirements.txt
python run.py --check                              # readiness check — tells you exactly what's missing
python run.py                                      # live webcam   (q quit · s screenshot · r record)
python run.py --source data/clips/site.mp4         # or a video file
```

`--check` is the friendliest starting point: it validates Python, dependencies, the model, the
camera/video source, and any optional features, and prints `READY` or the exact items to fix.

### 3. Reproduce the Phase 3 numbers (optional, needs the dataset)
See **[phase3_activity/README.md](phase3_activity/README.md)** for the Assembly101 download and:
```bash
python -m phase3_activity.tas.evaluate      --fold val           # step recognition
python -m phase3_activity.tas.mistake_eval  --procedure assembly # mistake detection
python -m phase3_activity.tas.anticipation  --procedure assembly # next-step anticipation
```

---

## Phase 1 — the detector (results)

Held-out **test split, 4,190 images**, scored with ultralytics' native validation
(`best_refined.pt`, YOLOv8s):

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Helmet | 97.0% | 95.6% | 96.3% | 97.9% | 82.1% |
| No-Helmet | 93.8% | 93.4% | 93.6% | 97.5% | 80.3% |
| No-Vest | 95.8% | 95.9% | 95.9% | 97.6% | 87.3% |
| Person | 96.6% | 97.8% | 97.2% | 98.9% | 90.4% |
| Vest | 97.3% | 97.7% | 97.5% | 99.1% | 89.2% |
| **All (mean)** | **96.1%** | **96.1%** | **96.1%** | **98.2%** | **85.9%** |

**5 / 5 classes clear 90%** on precision, recall, F1, and mAP@50. The blocker to 90% was never
the architecture — it was **data quantity** for the safety-critical *absence* classes (a person
*without* a hard hat). The full prototype that established this (zero-shot vs fine-tune vs VLM,
threshold tuning, fusion) is written up in **[docs/phase1_prototype.md](docs/phase1_prototype.md)**.

```bash
pip install -r requirements.txt
python run.py --check
python eval_ppe.py --model best_refined.pt --dataset-dir data/ppe_download   # per-class P/R/F1/mAP
```

---

## Phase 2 — real-time tracking & AR overlay ([`phase2/`](phase2/))

Takes the trained detector and makes it **deployment-aware**:

```
webcam / clip ─▶ detect (YOLO) ─▶ track persons (ByteTrack) ─▶ compliance ─▶ AR HUD overlay ─▶ screen / mp4
                                                              (debounce + dedup per person)
```

- **Stable per-person track IDs**; violations deduplicated per person, not per frame.
- **A polished AR heads-up overlay** — a status header, corner-bracket person reticles coloured
  by their worst violation (green = compliant), an active-alerts card, a workflow card (Phase 3),
  and a bottom status bar. ~57 FPS on CUDA with per-stage latency reported.
- **Reality-check** — runs on a self-recorded first-person clip and quantifies the domain gap vs
  the Phase 1 benchmark (an honest answer to "does 90%+ survive worn-camera video?").
- **Optional features** (off by default): Work-ID worker badges, a JSONL event log, and the
  Phase 3 activity backend (step recognition + mistake + anticipation). Enable in
  [`phase2/config.yaml`](phase2/config.yaml); `run.py --check` validates each.
- **Graceful** — no webcam / bad weights / headless all degrade with clear messages.

Full details in **[phase2/README.md](phase2/README.md)**.

---

## Phase 3 — workflow understanding ([`phase3_activity/`](phase3_activity/))

Egocentric **procedural-activity** understanding on the **Assembly101** dataset (Sener et al.,
CVPR 2022), following the supervisor's staged plan. All numbers are on the held-out **val** fold:

| Capability | What it does | Result |
|---|---|---|
| **Step recognition** | label every frame with the assembly step (temporal action segmentation) | MoF **40.5**, Edit 31.2, F1@10/25/50 = 32.7 / 29.1 / 21.6 (matches published C2F-TCN, MoF 37.8) |
| **Mistake detection** | learn the expected order, flag out-of-order steps | **100%** recall on injected order violations, 8.2% false positives per transition |
| **Next-step anticipation** | predict the next step from the steps done so far | top-1 15.5% / top-3 27.5%, beating the frequency baseline (11.3 / 23.2) |

The trained model plugs into the Phase 2 seam (`activity.backend: assembly101`), and the mistake /
anticipation models drive the workflow card in the overlay. **Honest scope, methods, and caveats
are documented in [phase3_activity/README.md](phase3_activity/README.md).**

---

## Weights & data

Model weights (`*.pt`) and datasets are **excluded from git** (size), so a fresh `git clone` has
the code but not the large files. To run the parts that need them:

| To run | You need | Where it goes |
|---|---|---|
| Phase 3 **tests** | nothing (synthetic fixtures) | — |
| Phase 2 **live demo** | the trained detector `best.pt` | `phase2/models/best.pt` |
| Phase 1 **eval** | `best_refined.pt` + the PPE dataset | repo root / `data/` |
| Phase 3 **scoring** | Assembly101 features + annotations | `phase3_activity/data/` ([guide](phase3_activity/README.md)) |

- **PPE dataset** — Roboflow `segp-fcn6m/ppe-yezzu-fwbjo` (42k images, CC BY 4.0); train with
  [kaggle_ppe.ipynb](kaggle_ppe.ipynb) → [kaggle_ppe_continue.ipynb](kaggle_ppe_continue.ipynb),
  drop `best_refined.pt` at the root and copy it to `phase2/models/best.pt`.
- **API keys** are read from an environment variable / secret, never hard-coded (see
  [.env.example](.env.example)).

---

## Tech stack
Python 3.11+ · **ultralytics** (YOLOv8) · **supervision** (ByteTrack) · **OpenCV** · PyTorch ·
NumPy · pandas · PyYAML. Phase 3's temporal model is a self-contained MS-TCN; its mistake /
anticipation models are pure-Python (no heavy deps).

## Project structure
```
.
├── README.md                       # ← this file (whole-project overview)
├── run.py · train.py · eval_ppe.py · tune.py · src/   # Phase 1: detector + eval harness
├── docs/phase1_prototype.md        # deep-dive: zero-shot vs fine-tune vs VLM
├── kaggle_ppe*.ipynb · ppe_colab.ipynb                # cloud training
├── phase2/                         # Phase 2: real-time video, tracking, AR overlay
│   ├── run.py · config.yaml · README.md
│   └── src/{detector,tracker,compliance,overlay,workid,eventlog,activity,...}.py
└── phase3_activity/                # Phase 3: Assembly101 workflow understanding
    ├── README.md
    ├── tas/{dataset,model,train,evaluate,procedure,anticipation,visualize,infer_seam}.py
    └── tests/{test_tas,test_mistake,test_anticipation,test_pipeline}.py
```

## Roadmap
- ✅ **Phase 1** — PPE detector @ 90%+ on all metrics, all classes
- ✅ **Phase 2** — real-time tracking + AR overlay + reality-check (+ optional Work-ID / event log)
- ✅ **Phase 3** — workflow understanding: step recognition → mistake detection → anticipation
- ⬜ **Next** — fine-grained anticipation / mistake benchmarks · on-device model export · AR-glasses deployment

## Credits & license
- **PPE dataset:** Roboflow Universe `segp-fcn6m/ppe-yezzu-fwbjo` — **CC BY 4.0**.
- **Assembly101:** Sener et al., *"Assembly101"*, CVPR 2022 — **CC BY-NC 4.0** (non-commercial).
- **Built with:** [ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 and Roboflow
  [supervision](https://github.com/roboflow/supervision).
- **Code license:** see [LICENSE](LICENSE).
