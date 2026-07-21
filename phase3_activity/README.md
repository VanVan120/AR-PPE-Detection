# Phase 3 — Activity / Step Recognition (Assembly101 Temporal Action Segmentation)

**The supervisor-confirmed priority:** *step / action recognition first*; next-action
anticipation and mistake detection come **after** this reaches high accuracy. This
directory builds that step-recognition track on **Assembly101**, whose three official
benchmarks line up exactly with that staged plan:

1. **Temporal Action Segmentation (TAS)** — step/action recognition ← *we are here*
2. **Action anticipation** — later
3. **Mistake detection** — later (Assembly101 ships correct/mistake/correction labels)

We train/evaluate on Assembly101's **precomputed 2048-D TSM features**, so no
raw-video CNN training is needed — a single consumer GPU is enough. The trained model
plugs into the existing Phase 2 seam (`phase2/src/activity.py`'s `infer(clip)`).

> Data spec below is **source-verified** against the official repo
> `assembly-101/assembly101-temporal-action-segmentation`. Items that could not be
> confirmed from a primary source are marked **[UNCERTAIN]** — trust the download over
> this doc where they disagree.

---

## Status (milestones)

| # | Milestone | State |
|---|---|---|
| **M1** | Data-format loader + evaluation metrics, unit-tested (no data needed) | ✅ **done** — `tas/dataset.py`, `tas/metrics.py`, `tests/test_tas.py` (10/10) |
| **M2a** | Self-contained MS-TCN baseline + full train/eval/scoring pipeline, smoke-tested end-to-end on a synthetic fixture | ✅ **done** — `tas/{model,train,evaluate,postprocess,torch_dataset}.py`, `tests/test_pipeline.py` |
| **M2b** | Train/eval on the real downloaded features | 🟢 **loader validated on real data**; sanity subset learns (val MoF 17→26 in 12 epochs). Full 200-epoch run = a long compute job; official C2F-TCN checkpoint still an option for the exact sanity target |
| **M3** | `assembly101` backend wired into `phase2/src/activity.py` (`infer(clip)`): bridge (`tas/infer_seam.py`), checkpoint I/O, config keys, `--check`, tested | ✅ **done** (live uses a stand-in extractor; offline scoring is the correct path) |

Run the tests (no data needed):
```bash
python phase3_activity/tests/test_tas.py        # ALL_TAS True       (loader + metrics)
python phase3_activity/tests/test_pipeline.py   # ALL_PIPELINE True  (model + train/eval)
```

---

## Data access (do this to unblock M2 — it's gated + large, so start now)

You need **two separate downloads** from **two different places**. Total license:
**CC BY-NC 4.0** (attribution required, non-commercial). Everything lands under
`phase3_activity/data/` (gitignored).

### A. Coarse annotations — small (~a few MB), on Google Drive (NOT in git)
The GitHub repo `assembly-101/assembly101-annotations` ships only READMEs; the actual
files are on Google Drive.
1. Open `github.com/assembly-101/assembly101-annotations`, read the README for the
   Google Drive link (root folder `1QoT-hIiKUrSHMxYBKHvWpW9Z9aCznJB7`; coarse folder
   `1sS3JNB6efBF9Payci9kpZVE4c0nybbb5`).
2. Download the **`coarse-annotations/`** tree into `data/coarse-annotations/`. You need:
   `actions.csv`, `coarse_splits/` (6 files), `coarse_labels/` (per-sequence GT).
   *(Fine-grained annotations are NOT needed for TAS — they're for the later
   anticipation phase.)*

### B. TSM features — large + gated, on Hugging Face
1. Go to `huggingface.co/datasets/cvml-nus/assembly101`, log in, and accept the terms
   ("agree to share your contact information"). *(Legacy Google Drive access via
   `assembly-101/assembly101-download-scripts` is being phased out.)*
2. **Download ONE fixed RGB view to start (~34 GB)**, e.g. `C10095_rgb`, into
   `data/TSM_features/C10095_rgb/`. The full set is ~403 GB across 16 views — you do
   **not** need all of it to stand up the pipeline. The loader skips views that aren't
   present, so a single view runs end-to-end. **[UNCERTAIN]** whether the *canonical
   benchmark number* is single- or multi-view — validate against the sanity target below.

Expected layout (all gitignored):
```
phase3_activity/data/
  coarse-annotations/{actions.csv, coarse_splits/, coarse_labels/}
  C10095_rgb/                         # one per-view LMDB (data.mdb + lock.mdb)
```
No `statistic_input.pkl` needed — the loader reads each annotation's own frame range
directly from its `coarse_labels` file. The LMDB view can sit directly under
`data/` (default) or under `data/TSM_features/` (pass `--features-root`).

---

## Data-format contract (what `tas/dataset.py` implements)

- **Features:** one LMDB per **view** under `TSM_features/<view>/`. Per-frame key
  `"{sequence}/{view}/{view}_{frame:010d}.jpg"`, value = raw bytes of a **2048-D
  float32** vector. All frame indices are at **30 fps**.
- **Classes:** `actions.csv` (header) → **202** coarse classes (11 verbs × 61 objects
  valid combos; 171 are tail classes). The model head is always 202.
- **Labels:** `coarse_labels/<assembly|disassembly>_<core>.txt`, **TAB-separated**, no
  header, 3 cols `start end action_cls` (segments; **end-exclusive**), absolute frame
  numbers. (This release spells `disassembly` correctly — no `disassebly` quirk.)
- **Splits:** `coarse_splits/{train,val,test}_coarse_{assembly,disassembly}.txt`. Each
  line's **first field is the `coarse_labels` filename** (`assembly_<core>.txt`); the
  LMDB feature sequence is the bare **`<core>`** (assembly + disassembly are two
  annotations over one recording). Folds: `train`, `train_val`, `val`. **Test GT is
  withheld** — report on **val**. The loader is annotation-centric: each sample loads
  features over its own GT frame range, so no `statistic_input.pkl` is required.
- **Aggregation:** max-pool features over non-overlapping **20-frame** chunks, capped
  at **1200** pooled steps; upsample predictions back to frame length before scoring.

## Evaluation metrics (`tas/metrics.py`)

Three metrics, three aggregation schemes (a uniform average will **not** match the
reference), all with `bg_class=()` — **no background class is excluded** for Assembly101:

- **MoF / frame accuracy** — global per-frame micro-average.
- **Segmental Edit** — per-video macro-average (mean of per-video normalized edit).
- **F1@{10,25,50}** — global per-segment micro-average (accumulate tp/fp/fn, one
  precision/recall/F1 at the end; each GT segment matched at most once).

---

## M2 — training / evaluation

**Implemented now (M2a): a self-contained MS-TCN baseline** (`tas/model.py`) with the
full train → eval → score pipeline (`tas/train.py`, `tas/evaluate.py`,
`tas/postprocess.py`, `tas/torch_dataset.py`), smoke-tested end-to-end on a synthetic
fixture. Once the data is downloaded and `statistic_input.pkl` is built, run on the
real features (batch size 1 over variable-length videos):

```bash
python -m phase3_activity.tas.train    --data-root phase3_activity/data --view C10095_rgb
python -m phase3_activity.tas.evaluate --ckpt phase3_activity/models/mstcn_best.pt --fold val
```
Lower `--num-f-maps` / `--num-layers` if VRAM is tight (the temporal model is tiny —
LMDB I/O and sequence length dominate, not VRAM). Start on one view.

**For the canonical val sanity target** — `F1@10/25/50 = 33.3 / 28.6 / 20.6, Edit
31.7, MoF 37.8` — use the official **C2F-TCN + pretrained checkpoint** (M2b): its
value *is* the checkpoint, which only loads into the official module. The loader,
metrics, and train/eval harness here are model-agnostic, so C2F-TCN drops in without
changing them; the MS-TCN baseline trains on the real features directly and should
land a few F1 below C2F-TCN. **[UNCERTAIN]** exact VRAM / wall-clock — time one epoch
first. Avoid ASFormer as a first baseline under tight VRAM.

## M3 — wired into the Phase 2 seam ✅

`tas/infer_seam.py::Assembly101Recognizer` implements the `infer(clip)` contract and
is registered as the `assembly101` backend in `phase2/src/activity.py::build_recognizer`
(no change to `_ClipBuffer` / `ActivityModule` behaviour). Enable it from Phase 2 once
you have a trained checkpoint:
```yaml
# phase2/config.yaml
activity:
  enabled: true
  backend: "assembly101"
  checkpoint: "path/to/phase3_activity/models/mstcn_best.pt"
  actions_csv: "path/to/phase3_activity/data/coarse-annotations/actions.csv"
```
`python run.py --check` validates both paths.

**Caveat (why offline is the correct path):** Assembly101 models consume **TSM
features**, not raw clips, so a *live* backend needs an on-device TSM extractor.
**[NOT SOURCE-CONFIRMED]** the exact TSM backbone/pretraining, so the recognizer
defaults to a **stand-in** ResNet-50 extractor — the seam runs end-to-end but live
labels are a wiring demo (like `kinetics`), not meaningful steps. Supply a matching
TSM extractor via `Assembly101Recognizer(extractor=...)` for real live accuracy, or —
the correct path today — use the model for **offline scoring** (`tas/evaluate.py`).
(The clip buffer already snapshots clean frames before the overlay is drawn, so a
model sees correct egocentric pixels.)

## Roadmap hooks (same features, same seam)

- **Anticipation:** reuse the LMDB reader + `chunk_maxpool`; switch to fine-grained
  annotations (1380 classes); predict the action at *t+Δ*. `infer(clip)` unchanged.
- **Mistake detection:** `ActivityResult.mistake` already exists; compare the
  recognized/anticipated step stream against an expected assembly-order model; a
  deviation sets `mistake=True`, which maps to a new `phase2/src/eventlog.py` event.

---

## Open / uncertain items (flagged, not invented)
- Single-view vs full 16-view for the *canonical* benchmark number — loader supports
  single-view; validate against the sanity target.
- VRAM / wall-clock are engineering estimates, not published — measure first.
- TSM backbone identity/pretraining — irrelevant offline; matters only for a live extractor.
- Test split is not locally scorable (GT withheld) — all local numbers are val-fold.

## Attribution
Assembly101: Sener et al., *"Assembly101: A Large-Scale Multi-View Video Dataset for
Understanding Procedural Activities"*, CVPR 2022. Data under CC BY-NC 4.0 (non-commercial).
