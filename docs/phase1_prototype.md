# Construction Safety Inspection — Approach-Validation Prototype

A deliberately narrow prototype for the internship project *AI-Empowered Dynamic
Workflow Monitoring for Inspection via AR Glasses*. It answers **one question with
real numbers** and proves **one loop**:

- **Question:** for construction-site safety detection, which perception approach
  is good enough to build on — zero-shot open-vocabulary detection
  (**YOLO-World**, no training) or a **vision-language model (VLM)**? Answered with
  measured precision/recall/F1/mAP against the dataset's labelled `test/` split.
- **Loop:** `image in → detections → one-line natural-language safety report out`.

It runs each image through **two pipelines**, writes a **side-by-side
`results.html`** for qualitative review, and — the headline — writes
**`metrics.json`** with quantitative metrics against the ground-truth labels.

```
image ──▶ (A) YOLO-World  ─┐
       └▶ (B) VLM          ├─▶ one-line safety report  ─▶ results.html
                           └─────────────────────────────▶ metrics.json (vs test/ labels)
```

---

## 1. Prerequisites

- **Python 3.11+** (tested on 3.13).
- **The dataset** — Roboflow "Construction Site Safety" (YOLOv8 export, CC BY 4.0),
  unzipped under `data/dataset/` (see [§3](#3-where-to-put-the-dataset)).
- **For the local VLM (default):** [Ollama](https://ollama.com) running, with a
  **vision** model pulled. `qwen2.5:7b` is **text-only and will not work on
  images** — pull a vision model and confirm the tag with `ollama list`:
  ```bash
  ollama pull qwen2.5vl:7b        # or: llava:7b, llama3.2-vision, moondream
  ollama list                      # copy the exact tag into config.yaml (ollama_model)
  ```
- **For the optional API VLM (`--api`):** an `ANTHROPIC_API_KEY` in the environment.
- **GPU is optional.** The code auto-detects CUDA / Apple MPS and falls back to CPU
  (slower, but works).

## 2. Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU note:** `ultralytics` pulls in a default `torch`. For CUDA/MPS acceleration,
> install the matching torch build first from <https://pytorch.org> (e.g. the cu126
> wheels), then `pip install -r requirements.txt`. YOLO-World weights and a small
> CLIP text-encoder download automatically on first run.

Copy the env template if you'll use `--api`:

```bash
cp .env.example .env            # then set ANTHROPIC_API_KEY
```

## 3. Where to put the dataset

Unzip the Roboflow YOLOv8 export so the layout is:

```
data/dataset/
├── data.yaml                  # class names + split paths (read automatically)
├── test/   ├── images/  └── labels/    # ← used as evaluation ground truth
├── valid/  ├── images/  └── labels/
└── train/  ├── images/  └── labels/
```

Class names are **read from `data.yaml`** — nothing is hardcoded. The PPE-relevant
classes (`Hardhat`, `NO-Hardhat`, `Safety Vest`, `NO-Safety Vest`, `Person`) are the
ones that matter; the dataset's ~25 classes include vehicles/objects we ignore.

## 4. Run

First, verify everything is in place:

```bash
python run.py --check
```

This checks the Python version, that packages import, that Ollama is reachable with
a vision model, and that the dataset exists with a readable `data.yaml` and `test/`
split — reporting clearly what's missing.

Then run the whole thing:

```bash
python run.py                  # full run on the test split (local Ollama VLM)
```

Useful flags:

| Command | What it does |
|---|---|
| `python run.py` | Full run on the `test/` split: both pipelines, reports, `results.html`, `metrics.json`. |
| `python run.py --check` | Readiness check only (no processing). |
| `python run.py --api` | Use the **Anthropic** vision API for the VLM instead of Ollama. |
| `python run.py --limit 8` | Quick run on the first 8 images (for a fast sanity check). |
| `python run.py --images path/to/folder` | Run the pipelines on an **ad-hoc** image folder (no labels → no evaluation). |
| `python run.py --config other.yaml` | Use a different config file. |

### Local ↔ API switch

- **Local (default):** uses Ollama at `ollama_host` with `ollama_model` from
  `config.yaml`. No flag needed.
- **API:** add `--api`. Uses `api_provider`/`api_model` (default Claude
  `claude-opus-4-8`) with the key from `ANTHROPIC_API_KEY`. The VLM contract is
  identical (a JSON array of `{type, description, severity}` observations).

## 5. Outputs

Everything is written under `outputs/`:

```
outputs/
├── results.html        # side-by-side review: annotated image + VLM observations + report,
│                       #   with the quantitative metrics tables at the top  ← open this
├── results.json        # every per-image record (detections, observations, report)
├── metrics.json        # the headline quantitative metrics (vs test/ labels)
├── yolo_world/         # annotated images (boxes coloured: red=violation, green=ok, orange=unscored)
└── vlm/                # raw + parsed VLM responses, one JSON per image
```

Open `outputs/results.html` in any browser — it's a static page, no server needed.

## 6. Configuration (`config.yaml`)

All knobs live in `config.yaml`; **prompts, thresholds, and the prompt→class
mapping change there with no code edits.** Key fields:

- `safety_prompts` — the open-vocabulary classes YOLO-World looks for.
- `prompt_to_class` — maps each prompt to an exact dataset class (read names from
  `data.yaml`). `null` = the prompt is detected/shown but **not** scored.
- `confidence_threshold` — operating point for shown detections and the per-class
  precision/recall/F1 + confusion matrix.
- `yolo_conf_floor` — low floor YOLO-World runs at so **mAP@50** can integrate the
  full curve (display still uses `confidence_threshold`).
- `yolo_model` — `yolov8s-worldv2.pt` by default; `…m/l/x-worldv2.pt` are more
  accurate and slower.
- `ollama_model` / `api_model` — the vision models for each VLM path.
- `vlm_eval_classes`, `vlm_violation_keywords`, `vlm_negative_keywords` — control
  how a VLM observation is counted as "flagging" a violation class at the image
  level.

## 7. How to read the metrics — is zero-shot enough, or is fine-tuning needed?

`metrics.json` (and the table at the top of `results.html`) is the whole point.

**YOLO-World — detection metrics** (per scored class, IoU 0.5):

- **Recall** = of the real PPE-violation boxes, how many were found. Low recall on
  `NO-Hardhat` / `NO-Safety Vest` means **missed hazards** — the dangerous failure
  mode for a safety tool.
- **Precision** = of the boxes it flagged, how many were correct. Low precision
  means **false alarms** that erode trust.
- **AP@50 / mAP@50** = precision/recall integrated across all confidence levels —
  the standard single-number summary of detector quality.
- **Confusion matrix** = where predictions land (incl. confusing one class for
  another, and the `background` row/col for false positives / missed boxes).

**VLM — image-level metrics** on the core violation classes: did the image contain
`NO-Hardhat` / `NO-Safety Vest`, and did the VLM flag it? Reported as per-class
precision/recall/F1 over images.

**Decision guide:**

| What the numbers show | Read it as |
|---|---|
| Violation-class **recall ≳ 0.8** at acceptable precision | Zero-shot is likely **good enough to build on** for those classes. |
| **High recall, low precision** | Tune `confidence_threshold` / iterate `safety_prompts`; the signal is there. |
| **Low recall** that the prompt list can't lift | Zero-shot is **not enough** → **fine-tune** YOLO-World on the dataset's `train/` split for those classes. |
| VLM clearly beats / loses to YOLO-World on the violation classes | Tells you which pipeline to centre the next phase on. |

> Notes on reading the table: a per-class precision of **`—`** means the detector
> made **no predictions** for that class at the operating point (different from a
> low precision, which means it predicted but was often wrong). Macro averages are
> taken over the classes that actually occur in the ground truth (`support > 0`).

Per the project plan, the **next phase** uses exactly these per-class numbers to
decide zero-shot vs. fine-tuning — then the AR capture layer and a fuller reporting
pipeline. Hand-labelling is not needed; the dataset provides the ground truth.

## 8. Closing the gap — Tracks A & B

The prototype's verdict was clear: zero-shot **Person** detection works, but the
PPE classes have poor recall and the safety-critical **violation** classes
(`NO-Hardhat` / `NO-Safety Vest`) score ~0 — an open-vocabulary prompt can't ground
the *absence* of an object. Two follow-up tracks act on that finding. Both reuse
the **same evaluation harness**, so every number stays comparable.

### Track B — squeeze zero-shot with no training

The headline scores YOLO-World at one global `confidence_threshold` (0.25). But
each class has a different precision/recall trade-off: `Hardhat` scores P=100% /
R=3.6% at 0.25 while its AP@50 is ~70% — most of its recall sits *just below* the
threshold. `tune.py` recovers it by choosing the **F1-optimal threshold per class
on the validation split** (never on test), then reporting the gain on held-out test.

```bash
python tune.py                 # writes outputs/tuning/{thresholds.json, tuning.html, pr_*.png}
```

Result on the held-out test split: **macro F1 18.3% → 37.2% with zero training**
(`Hardhat` F1 7.0% → 76.3%, `Safety Vest` 6.3% → 34.2%). The tuned thresholds are
already pasted into `config.yaml` under `per_class_thresholds`, so `python run.py`
applies them; clear that map to revert to the single global threshold.

Crucially, **the violation classes stay at ~0** — no threshold recovers recall the
detector never produced. A prompt-phrasing experiment confirms the ceiling:

```bash
python prompts_experiment.py   # tests positive-concept phrasings on the valid split
```

Reframing a negation as a positive concept ("a bare head", "a worker in a plain
t-shirt") lifts the violation classes from 0 to only ~10–25% best-F1 at **terrible
precision (10–26%)**. Zero-shot simply cannot do violation detection here — which
is the case for Track A.

> `--reuse-vlm` lets `python run.py` re-score detection changes (e.g. tuned
> thresholds, a fine-tuned model) **without** re-running the slow VLM, by reusing
> the cached responses in `outputs/vlm/`. A full re-score drops from ~35 min to seconds.

### Track A — fine-tune a detector on the labelled data

The proper fix for the violation classes is training. `train.py` fine-tunes a
standard YOLO detector on the dataset's own `train/` + `valid/` labels (it writes a
corrected, absolute-path `data.yaml` so ultralytics finds the splits):

```bash
python train.py                # -> models/finetuned.pt  (config: finetune_* knobs)
python compare_models.py       # zero-shot vs fine-tuned on test, same harness
```

`compare_models.py` writes `outputs/comparison/comparison.html` — a side-by-side
per-class table (mAP@50 is the fair, threshold-independent headline) plus annotated
example pairs from each model. To run the **whole** prototype (reports, VLM,
`results.html`) on the fine-tuned model instead of zero-shot, set in `config.yaml`:

```yaml
detector_backend: finetuned    # default: yolo_world
```

**Result on the held-out test split (same harness, mAP@50 is the fair headline):**

| | Zero-shot YOLO-World | Fine-tuned | |
|---|---|---|---|
| **mAP@50** | 35.6% | **68.3%** | ~1.9× |
| **macro F1** | 37.2% | **73.0%** | ~2× |
| Hardhat (F1) | 76.3%¹ | 80.6% | +4.3 |
| Safety Vest (F1 / AP50) | 34.2% / 26.2% | **74.6% / 75.7%** | big |
| **NO-Hardhat** (F1 / AP50) | 0% / 0% | **63.2% / 50.9%** | 0 → usable |
| **NO-Safety Vest** (F1 / AP50) | —² / 0% | **65.4% / 59.1%** | 0 → usable |
| Person (F1) | 75.3% | 81.1% | — |

¹ zero-shot Hardhat shown at its Track-B tuned threshold. ² `—` = the zero-shot
detector made no predictions for that class at all.

**Verdict:** fine-tuning roughly **doubles** mAP@50 and macro F1, and — the decisive
point — turns the two safety-critical **violation** classes from *unusable* (0)
into *deployable* detectors (F1 ~63–65%). This is the concrete evidence that, for
construction PPE compliance, **zero-shot perception is a scaffold but fine-tuning
is required**. See `outputs/comparison/comparison.html` for the side-by-side table
and example detections.

#### Pushing the fine-tuned model further — what helps and what doesn't

Three levers were measured on the held-out test split. The two negatives are as
informative as the win — they show *where the ceiling actually is*:

| Lever | Result | Why |
|---|---|---|
| Per-class threshold tuning (`tune.py --backend finetuned`) | **no gain** (73.0→72.3% F1) | the fine-tuned model is already well-calibrated; 0.25 is near-optimal (unlike zero-shot) |
| Bigger model — yolov8m, 2.3× params (`train.py --base-model yolov8m.pt`) | **no gain** (mAP 68.3→66.8%) | only 521 training images — **data-limited, not capacity-limited**; the bigger model slightly overfits |
| **Test-time augmentation** (`tta: true`) | **mAP 68.3 → 70.8%** ✅ | merges predictions over augmented views; the reliable free lever in a small-data regime |

So the accuracy ceiling here is set by **data quantity**, not model size or operating
point. The one lever that helps without more data is TTA (config `tta: true`,
~2–3× slower inference). The clear path to a further jump is **more labelled data**,
not a heavier model.

### Track C — fuse the fine-tuned detector with the VLM

With a strong detector in hand, the VLM's role changes. It is no longer the better
violation detector (the fine-tuned model beats it: image-level violation F1 **78.8%
vs 25.4%**), so the fusion keeps the **detector authoritative** on the scored
classes and uses the VLM for what it uniquely adds:

```bash
python fusion_eval.py          # -> outputs/fusion/fusion.html + fusion.json
```

`src/fusion.py` reconciles the two per image:

- both agree → **confirmed** violation (grounded + corroborated)
- detector only → **grounded** violation (high precision)
- VLM only → **review** flag (a *possible* detector miss — surfaced, not blindly trusted)
- VLM observations with no detector class → **context hazards** (unprotected edge,
  exposed rebar) — the VLM's genuine value-add

**Image-level violation detection on test (macro over NO-Hardhat / NO-Safety Vest):**

| Decision rule | Precision | Recall | F1 |
|---|---|---|---|
| detector-only | 80.8% | 76.9% | **78.8%** |
| VLM-only | 24.2% | 26.7% | 25.4% |
| fusion **OR** (either) | 47.8% | **87.8%** | 61.9% |
| fusion **AND** (both) | 76.4% | 15.7% | 25.9% |

**Honest verdict:** once the detector is fine-tuned, **detector-only has the best F1**
— naively OR-ing in a noisy VLM *lowers* F1. Fusion's value is therefore not a magic
F1 win but: (1) **fusion-OR as a high-recall "don't miss a violation" safety mode**
(recall 76.9% → **87.8%**, accepting more false alarms for human review), and (2) the
**reconciliation report** that keeps the detector authoritative while surfacing VLM
review-flags and the context hazards the detector is blind to. See
`outputs/fusion/fusion.html`.

## 9. Known limitations

- **Single still images only** — no video/real-time (a later, separate phase).
- **VLM scoring is image-level**, not boxes — the VLM emits text, so it can't be
  IoU-matched; it's scored on whether it flags a violation class per image. The
  keyword mapping (`vlm_*` config) is a reasonable heuristic, not exhaustive.
- **VLM speed** — local 7B vision models take seconds per image; a full 82-image
  run can take many minutes on CPU. "Slow is fine"; use `--limit` for quick checks.
- **Zero-shot prompts matter** — YOLO-World results depend on the prompt wording;
  `safety_prompts` is meant to be iterated on.
- **Still not built (by design):** AR/hardware, real-time video, databases, or a
  heavy web front-end. Fine-tuning *was* a prototype non-goal but is now included as
  the **Track A** follow-up ([§8](#8-closing-the-gap--tracks-a--b)) once the metrics
  showed it was needed.

## 10. Project structure

```
.
├── proposal.md
├── README.md
├── requirements.txt
├── config.yaml              # all knobs (prompts, thresholds, mapping, per-class + finetune)
├── .env.example
├── data/dataset/            # the Roboflow export (you place it here)
├── src/
│   ├── config.py            # load + validate config.yaml, read dataset data.yaml
│   ├── loader.py            # discover images / load any split + labels
│   ├── detector_yolo.py     # YOLO-World (zero-shot) pipeline
│   ├── detector_finetuned.py# fine-tuned detector + build_detector() backend factory
│   ├── detector_vlm.py      # VLM pipeline (Ollama + optional Anthropic API)
│   ├── reporter.py          # detections -> one-line safety report
│   ├── evaluator.py         # metrics vs ground-truth labels (per-class thresholds aware)
│   ├── tune.py              # per-class threshold curves + F1-optimal operating points
│   ├── fusion.py            # Track C: reconcile detector + VLM into one assessment
│   ├── compare.py           # per-image record assembly + summary
│   └── render.py            # write results.html + results.json + metrics.json
├── run.py                   # single entry point (--reuse-vlm, detector_backend)
├── tune.py                  # Track B: per-class threshold tuning (valid -> test)
├── prompts_experiment.py    # Track B: prompt-phrasing experiment
├── train.py                 # Track A: fine-tune a detector on train+valid
├── compare_models.py        # Track A: zero-shot vs fine-tuned, head-to-head
├── fusion_eval.py           # Track C: detector+VLM fusion eval + reports
└── outputs/                 # generated (incl. tuning/, comparison/, train/, fusion/)
```

## 11. License / data

The dataset is Roboflow "Construction Site Safety" under **CC BY 4.0**. Model
weights (YOLO-World, CLIP) download from their respective sources on first run.
