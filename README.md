# Construction-Site PPE Safety Detection — Detector → Real-Time AR Monitoring

A computer-vision system that detects construction-site **PPE compliance** (hard hats
and hi-vis vests) and surfaces safety **violations** in real time. Built in two phases:

1. **Detector (Phase 1)** — fine-tune YOLOv8 on a 42k-image PPE dataset to a strong,
   honest benchmark: **90%+ on every metric for all 5 classes** on a held-out test set.
2. **Real-time AR monitoring (Phase 2)** — run that detector live on video with
   multi-object tracking, deduplicated per-person violation logic, a simulated AR
   heads-up overlay, performance instrumentation, and a worn-camera **reality-check**.

> Summer-internship project for *AI-Empowered Dynamic Workflow Monitoring for Inspection
> via AR Glasses*. Phase 1 lives at the repo root; Phase 2 is a self-contained subproject
> in [`phase2/`](phase2/).

---

## 🎯 Results — Phase 1 detector (`best_refined.pt`, YOLOv8s)

Held-out **test split, 4,190 images**, scored with ultralytics' native validation:

| Class | Precision | Recall | F1 | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Helmet | 97.0% | 95.6% | 96.3% | 97.9% | 82.1% |
| No-Helmet | 93.8% | 93.4% | 93.6% | 97.5% | 80.3% |
| No-Vest | 95.8% | 95.9% | 95.9% | 97.6% | 87.3% |
| Person | 96.6% | 97.8% | 97.2% | 98.9% | 90.4% |
| Vest | 97.3% | 97.7% | 97.5% | 99.1% | 89.2% |
| **All (mean)** | **96.1%** | **96.1%** | **96.1%** | **98.2%** | **85.9%** |

**5 / 5 classes clear 90%** on precision, recall, F1, and mAP@50.

### The data story — why it took 42k images
| Stage | `No-Helmet` F1 | mean mAP@50 |
|---|---|---|
| Zero-shot YOLO-World (open-vocab, no training) | ~0% | ~36% |
| Fine-tune on 717 images | ~63% | ~68% |
| **Fine-tune on 42k images (final)** | **93.6%** | **98.2%** |

The blocker to 90% was never the architecture — it was **data quantity** for the
safety-critical *absence* classes (a person *without* a hard hat). Open-vocabulary
prompts can't ground "absence"; scaling labelled data did. The full prototype that
established this (zero-shot vs fine-tune vs VLM, threshold tuning, fusion) is written up
in **[docs/phase1_prototype.md](docs/phase1_prototype.md)**.

---

## Phase 1 — detector & evaluation harness (repo root)

A reproducible harness that scores detection backends head-to-head on the dataset's own
labels, plus the cloud-training notebooks that produced the final model.

```bash
pip install -r requirements.txt
python run.py --check                                              # readiness check
python eval_ppe.py --model best_refined.pt --dataset-dir data/ppe_download   # per-class P/R/F1/mAP
```

| File | Role |
|---|---|
| [train.py](train.py) | Fine-tune a YOLO detector on the dataset's own labels |
| [eval_ppe.py](eval_ppe.py) | Authoritative per-class P/R/F1/mAP via ultralytics native val |
| [run.py](run.py) | Full prototype: detector (+ optional VLM tier) → annotated images, safety reports, `results.html`, `metrics.json` |
| [src/](src/) | config, loaders, detectors (zero-shot YOLO-World + fine-tuned), evaluator, reporter, fusion |
| [kaggle_ppe.ipynb](kaggle_ppe.ipynb) · [kaggle_ppe_continue.ipynb](kaggle_ppe_continue.ipynb) | Train (30 epochs) → refine (+20 epochs) on Kaggle GPUs |

---

## Phase 2 — real-time tracking & simulated AR ([`phase2/`](phase2/))

Takes the trained detector and makes it **deployment-aware**:

```
webcam / clip ─▶ detect (YOLO) ─▶ track persons (ByteTrack) ─▶ compliance ─▶ AR overlay ─▶ screen / mp4
                                                              (debounce + dedup per person)
```

- **Stable per-person track IDs**; violations deduplicated per person, not per frame.
- **Simulated AR HUD** — severity-coloured boxes (red/orange/green), status panel, live alert list.
- **~57 FPS on CUDA** (detect ~16 ms / track ~1 ms / render ~3 ms), with per-stage latency reported.
- **Reality-check** — runs on a self-recorded first-person clip and quantifies the domain
  gap vs the Phase 1 benchmark (an honest answer to "does 90%+ survive worn-camera video?").
- **Graceful** — no webcam / bad weights / headless all degrade with clear messages.

```bash
cd phase2
pip install -r requirements.txt
python run.py --check
python run.py                                  # live webcam  (q quit · s screenshot · r record)
python run.py --source data/clips/clip.mp4     # or a video file
python run.py --reality-check data/clips/firstperson.mp4
```

Full details in **[phase2/README.md](phase2/README.md)** · scope in [phase2/proposal_phase2.md](phase2/proposal_phase2.md).

---

## Getting the model & data

Model weights (`*.pt`) and the datasets are **excluded from git** (size). To reproduce:

1. **Dataset** — Roboflow `segp-fcn6m/ppe-yezzu-fwbjo` (42k images, CC BY 4.0). Download
   via the `roboflow` SDK; the API key is read from a Kaggle/Colab **Secret** or the
   `ROBOFLOW_API_KEY` environment variable — never hard-coded.
2. **Train** — run [kaggle_ppe.ipynb](kaggle_ppe.ipynb) (30 epochs) then
   [kaggle_ppe_continue.ipynb](kaggle_ppe_continue.ipynb) (+20-epoch refine). Drop the
   resulting `best_refined.pt` at the repo root and copy it to `phase2/models/best.pt`.

## Tech stack
Python 3.11+ · **ultralytics** (YOLOv8) · **supervision** (ByteTrack) · **OpenCV** ·
PyTorch · PyYAML · optional VLM tier (Ollama / Anthropic Claude).

## Project structure
```
.
├── README.md                  # ← you are here (whole-project overview)
├── docs/phase1_prototype.md   # deep-dive: zero-shot vs fine-tune vs VLM (Tracks A/B/C)
├── run.py · train.py · eval_ppe.py · tune.py · compare_models.py · fusion_eval.py
├── src/                       # Phase 1 detector + evaluation harness
├── config.yaml · requirements.txt · .env.example
├── kaggle_ppe*.ipynb · ppe_colab.ipynb   # cloud training (keys via Secret/env var)
└── phase2/                    # Phase 2: real-time video, tracking, AR overlay
    ├── run.py · config.yaml · benchmark.json
    ├── src/{source,detector,tracker,compliance,overlay,metrics,reality_check}.py
    ├── tools/make_sample_clip.py
    └── README.md · proposal_phase2.md
```

## Roadmap
- ✅ **Phase 1** — PPE detector @ 90%+ on all metrics, all classes
- ✅ **Phase 2** — real-time tracking + simulated AR overlay + reality-check
- ⬜ **Phase 3** — AI daily safety report (VLM tier)
- ⬜ **Phase 4** — model export / quantization + on-device latency
- ⬜ **Phase 5** — AR-glasses deployment

## Credits & license
- **Dataset:** Roboflow Universe `segp-fcn6m/ppe-yezzu-fwbjo` — **CC BY 4.0** (attribution required).
- **Built with:** [ultralytics](https://github.com/ultralytics/ultralytics) YOLOv8 and
  Roboflow [supervision](https://github.com/roboflow/supervision).
- **Code:** add a license of your choice (e.g. MIT) before wider distribution.
