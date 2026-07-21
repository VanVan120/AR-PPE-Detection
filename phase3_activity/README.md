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
| **M1** | Data-format loader + evaluation metrics, unit-tested (no data needed) | ✅ **done** — `tas/dataset.py`, `tas/metrics.py`, `tests/test_tas.py` (10/10 pass) |
| **M2** | Download data → reproduce the official C2F-TCN val numbers with the pretrained checkpoint → then train | ⏳ blocked on the data download |
| **M3** | Wire an `assembly101` backend into `phase2/src/activity.py` (`infer(clip)`) | ⏳ after M2 |

Run the M1 tests:
```bash
python phase3_activity/tests/test_tas.py       # ALL_TAS True
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

### Then build the frame-span index once
```bash
python -c "from phase3_activity.tas.dataset import build_statistic_input as b; \
b('phase3_activity/data/TSM_features', ['C10095_rgb'], 'phase3_activity/data/statistic_input.pkl')"
```

Expected layout (all gitignored):
```
phase3_activity/data/
  coarse-annotations/{actions.csv, coarse_splits/, coarse_labels/}
  TSM_features/C10095_rgb/            # one per-view LMDB to start
  statistic_input.pkl
```

---

## Data-format contract (what `tas/dataset.py` implements)

- **Features:** one LMDB per **view** under `TSM_features/<view>/`. Per-frame key
  `"{sequence}/{view}/{view}_{frame:010d}.jpg"`, value = raw bytes of a **2048-D
  float32** vector. All frame indices are at **30 fps**.
- **`statistic_input.pkl`:** `{video: {view: [min_frame, max_frame]}}`, span **inclusive**.
- **Classes:** `actions.csv` (header) → **202** coarse classes (11 verbs × 61 objects
  valid combos; 171 are tail classes). The model head is always 202.
- **Labels:** `coarse_labels/<assembly|disassembly>_<seq>.txt`, **TAB-separated**, no
  header, 3 cols `start end action_cls` (segments; **end-exclusive**). GT-filename
  quirk to replicate: `disassembly` → `disassebly`.
- **Splits:** `coarse_splits/{train,val,test}_coarse_{assembly,disassembly}.txt`; folds
  `train`, `train_val` (train+val), `val`. **Test GT is withheld** — not locally
  scorable; report on **val**.
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

## M2 — model (when data is on disk)

**Recommended baseline: C2F-TCN** (the official Assembly101 TAS baseline). Fastest
correct path: download its **pretrained checkpoint** and run inference → score first
(no training). Then train with the official config, lowering **batch size 20 → 4–6**
to fit ~8 GB VRAM (the only change that matters — the temporal model is tiny; LMDB
I/O and sequence length dominate).

**Sanity target (official repo, val fold):**
`F1@10/25/50 = 33.3 / 28.6 / 20.6, Edit 31.7, MoF 37.8`. If a run lands here (± a
point), the pipeline is correct. **[UNCERTAIN]** exact VRAM / wall-clock — time one
short epoch before committing.

Avoid ASFormer as the first baseline under tight VRAM (attention memory scales badly
with Assembly101's long sequences). MS-TCN++ is a lighter fallback but is **not** in
the official repo (needs porting).

## M3 — wire into the Phase 2 seam

Add a third backend to `phase2/src/activity.py::build_recognizer` — no change to the
`infer(clip)` contract, `_ClipBuffer`, or `ActivityModule`:
```python
if backend == "assembly101":
    return Assembly101Recognizer(ckpt=..., device=device)
```
**Caveat:** Assembly101 models consume **TSM features**, not raw clips, so a *live*
backend needs an on-device TSM feature extractor in front of C2F-TCN. **[NOT
SOURCE-CONFIRMED]** the exact TSM backbone/pretraining. For now: use the Assembly101
model for **offline** video scoring; keep the live AR path on `placeholder`/`kinetics`
until a feature extractor is validated. (The buffer already snapshots clean frames
before the overlay is drawn, so a trained model sees correct egocentric pixels.)

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
