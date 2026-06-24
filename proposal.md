# Proposal: Construction Safety Inspection — Approach-Validation Prototype (v2)

## 1. Context

This is the first prototype for an internship project: *AI-Empowered Dynamic Workflow Monitoring for Inspection via AR Glasses*, aimed at helping civil engineers/supervisors monitor worker safety on construction sites and auto-generate reports.

The full system is large. **This prototype is deliberately narrow.** Its job is to answer one question *with real numbers* and prove one loop:

- **Question:** for construction safety detection, which perception approach is good enough to build on — zero-shot open-vocabulary detection (YOLO-World, no training) or a vision-language model (VLM)? Answer it with measured precision/recall against a labelled dataset, not just by eye.
- **Loop:** prove end-to-end `image in → detections → one-line natural-language safety report out`.

Everything else in the project comes later and depends on the answer this prototype produces.

## 2. Data (already downloaded)

The **Roboflow "Construction Site Safety"** dataset (YOLOv8 export, CC BY 4.0) is already downloaded. It ships as a standard YOLO dataset: a `data.yaml` listing class names + split paths, and `train/`, `valid/`, `test/` folders, each containing `images/` and `labels/`.

- Place the unzipped dataset under `data/dataset/`.
- **Read class names from `data.yaml` — do not hardcode them.** The PPE-relevant classes (e.g. `Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Person`, `Mask`) are the ones that matter; the dataset has ~25 classes total, including many vehicles/objects we ignore.
- Use the **`test/` split** as evaluation ground truth, and a small sample of it for quick qualitative review.

This labelled `test/` split removes the need for any hand-labelling.

## 3. Scope — what to build

A Python command-line tool that:

1. Reads images from the dataset (and/or an ad-hoc folder).
2. Runs each image through **two pipelines**: (A) **YOLO-World** open-vocab detector driven by a configurable prompt list, and (B) a **VLM** returning structured safety observations.
3. Generates a **one-line natural-language safety report** per image from the combined detections.
4. Produces a **side-by-side results page** for qualitative review.
5. **Evaluates both pipelines against the dataset's `test/` labels** and reports quantitative metrics — this is the headline output.

## 4. Non-goals — do NOT build these

- ❌ Model training or fine-tuning (run pre-trained / zero-shot models only)
- ❌ AR glasses / HoloLens / hardware integration of any kind
- ❌ Real-time video processing (single still images only)
- ❌ A database or persistence beyond writing output files
- ❌ A polished web app or front-end framework (a simple static results page is the ceiling)
- ❌ Progress monitoring or schedule comparison (a later, separate phase)

If something drifts toward the full system, stop and leave it for a future proposal.

## 5. Tech stack — decisions already made (don't deliberate)

- **Language:** Python 3.11+
- **Open-vocab detection:** `ultralytics` (YOLO-World). Weights auto-download on first run.
- **VLM (local, default):** Ollama running a vision model. `qwen2.5:7b` is text-only and won't work for images — pull a vision model and verify the exact tag via `ollama list` (e.g. `qwen2.5vl`, `llava`, `llama3.2-vision`).
- **VLM (API, optional):** behind an `--api` flag, use the Anthropic or OpenAI vision API instead. Keys from environment variables, never hard-coded.
- **Dataset loading + detection metrics:** `supervision` (Roboflow's library). It loads YOLO datasets (`DetectionDataset.from_yolo`), does IoU matching, and computes mAP / confusion matrix — use it rather than hand-rolling metric code.
- **Image handling / annotation:** OpenCV and/or Pillow (or supervision's annotators).
- **Output:** static `results.html` (plain HTML/CSS) + machine-readable `results.json` + `metrics.json`.
- **Env:** `venv` + `pip` with `requirements.txt` (or `uv`).
- **Device:** auto-detect CUDA/MPS, fall back to CPU (slow is fine).

## 6. Project structure

```
.
├── proposal.md
├── README.md
├── requirements.txt
├── config.yaml
├── .env.example
├── data/
│   └── dataset/                # unzipped Roboflow export (data.yaml, train/, valid/, test/)
├── src/
│   ├── config.py               # load + validate config.yaml, read dataset data.yaml
│   ├── loader.py               # discover images / load the test split + labels
│   ├── detector_yolo.py        # YOLO-World pipeline
│   ├── detector_vlm.py         # VLM pipeline (Ollama + optional API)
│   ├── reporter.py             # detections -> one-line safety report
│   ├── evaluator.py            # metrics vs ground-truth labels (NEW)
│   ├── compare.py              # qualitative side-by-side summary
│   └── render.py               # write results.html + results.json + metrics.json
├── run.py                      # single entry point
└── outputs/
    ├── yolo_world/             # annotated images
    ├── vlm/                    # raw + parsed VLM responses
    ├── results.html
    ├── results.json
    └── metrics.json
```

## 7. Build phases

Build and verify one phase before the next. Commit after each.

**Phase 0 — Scaffold.** Structure, `requirements.txt`, `config.yaml`, `.env.example`, README skeleton. Install deps. Add `run.py --check`: verify Python version, packages importable, Ollama reachable + a vision model present, and the dataset exists under `data/dataset/` with a readable `data.yaml` and a `test/` split. Report clearly what's missing.

**Phase 1 — YOLO-World pipeline.** Load images from the dataset's `test/images/` (and optionally an ad-hoc folder). Run YOLO-World with the configurable prompt list. Output detections (label, confidence, bbox) and save annotated copies. Respect the confidence threshold from config.

**Phase 2 — VLM pipeline.** Send each image to the vision model with a prompt asking for a **JSON array** of safety observations `{type, description, severity}` (severity = low/medium/high). Parse defensively (strip prose/code fences). Save raw + parsed. `--api` swaps Ollama for the API path with the same contract.

**Phase 3 — Report generation.** Combine both pipelines' findings into one concise safety line per image, e.g. *"3 workers detected; 1 without a hard hat near machinery — medium risk."* Use the VLM to phrase it, or a template if the VLM is off. Must work even if one pipeline is skipped.

**Phase 4 — Qualitative results page.** `results.html` showing, per image side by side: the annotated YOLO-World image, the VLM observations, and the report line. Plus `results.json`.

**Phase 5 — Quantitative evaluation (the headline).** Evaluate against the dataset's `test/` labels:
- **YOLO-World (detection metrics):** map each prompt to a dataset class via `prompt_to_class` in config (prompts with no matching class are detected but not scored). Use `supervision` to IoU-match predictions to ground-truth boxes at IoU 0.5 and compute per-class precision, recall, F1, and mAP@50, plus a confusion matrix.
- **VLM (image-level metrics):** the VLM emits text, not boxes, so score it at the image level on the core violation classes — e.g. is `NO-Hardhat` / `NO-Safety Vest` present in the image's labels, and did the VLM flag it? Report per-class image-level precision/recall/F1.
- Write everything to `metrics.json` and surface a summary table at the top of `results.html`.

**Phase 6 — Docs.** `README.md`: prerequisites, setup, where to put the dataset, run commands, local↔API switch, known limitations, and a short **"How to read the metrics to decide whether zero-shot is enough or fine-tuning is needed"** section.

## 8. Config (`config.yaml`)

Keep all knobs here. The prompt list and the prompt→class mapping are meant to be iterated on.

```yaml
dataset_dir: "data/dataset"
eval_split: "test"
sample_for_review: 20          # how many test images to show in results.html
confidence_threshold: 0.25
pipelines: ["yolo_world", "vlm"]

safety_prompts:
  - "person wearing a hard hat"
  - "person without a hard hat"
  - "person wearing a high-visibility safety vest"
  - "person without a safety vest"
  - "person"
  - "person standing near an unprotected edge"
  - "exposed rebar"

# Map each prompt to a dataset class (read exact names from data.yaml). null = detect but don't score.
prompt_to_class:
  "person wearing a hard hat": "Hardhat"
  "person without a hard hat": "NO-Hardhat"
  "person wearing a high-visibility safety vest": "Safety Vest"
  "person without a safety vest": "NO-Safety Vest"
  "person": "Person"
  "person standing near an unprotected edge": null
  "exposed rebar": null
```

## 9. Acceptance criteria

- A single command (`python run.py`) produces: annotated images, per-image safety reports, a side-by-side `results.html`, and **`metrics.json` with YOLO-World precision/recall/F1/mAP@50 (per class) and VLM image-level precision/recall/F1, all against the dataset's `test/` labels.**
- Runs on CPU (slow is fine), auto-uses GPU/MPS if present.
- Degrades gracefully: Ollama down → skip VLM with a warning; dataset missing → clear instruction, not a stack trace.
- Prompts, thresholds, and prompt→class mapping change in `config.yaml` with no code edits.
- README lets a new person set it up and run from scratch.

## 10. Stretch goals — only if the above fully works and time remains

- Sample frames from a short video clip and run the same pipelines per frame.
- A minimal live webcam demo using the YOLO-World pipeline.

Optional; do not start at the expense of the core.

## 11. What comes next (do NOT build now)

Once the metrics are in: read YOLO-World's per-class precision/recall to decide whether zero-shot is good enough or whether fine-tuning on the dataset's `train/` split is warranted. After that, the AR capture layer and a fuller reporting pipeline. (Hand-labelling is no longer needed — the dataset provides the ground truth.)
