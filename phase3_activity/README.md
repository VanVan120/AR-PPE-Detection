# Phase 3 — Activity / Step Recognition (Assembly101 Temporal Action Segmentation)

**The supervisor-confirmed priority:** *step / action recognition first*; next-action
anticipation and mistake detection come **after** this reaches high accuracy. This
directory builds that step-recognition track on **Assembly101**, whose three official
benchmarks line up exactly with that staged plan:

1. **Temporal Action Segmentation (TAS)** — step/action recognition ✅ *done*
2. **Mistake detection** — ✅ *done* (assembly-**order** deviation model; see below)
3. **Action anticipation** — ✅ *done* (next-**step** anticipation; see below)

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
| **M2b** | Train/eval on the real downloaded features | ✅ **done** — full run trained; **val MoF 40.5, Edit 31.2, F1@10/25/50 = 32.7 / 29.1 / 21.6** over 120 val videos, *above* the published C2F-TCN reference (MoF 37.8) |
| **M3** | `assembly101` backend wired into `phase2/src/activity.py` (`infer(clip)`): bridge (`tas/infer_seam.py`), checkpoint I/O, config keys, `--check`, tested | ✅ **done** (live uses a stand-in extractor; offline scoring is the correct path) |
| **M4** | **Mistake detection** — learned assembly-order model (`tas/procedure.py`), honest injected-fault eval (`tas/mistake_eval.py`), wired into the seam + event log | ✅ **done** — 100% recall on injected order violations, 6.6% per-transition FP on real val |
| **M5** | **Next-step anticipation** — transition model (`tas/anticipation.py`) with baselines + honest top-k eval, wired into the seam + overlay hint | ✅ **done** — top-1 15.5% / top-3 27.5% on real val, beating the frequency baseline (11.3 / 23.2) |
| **M6** | **Workflow monitor demo** (`tas/demo.py`) — all three capabilities replayed on one step stream, plus the GT-vs-predicted stream comparison | ✅ **done** — runs with **zero downloads**; see *Seeing it run* below |

Run the tests (no data needed):
```bash
python phase3_activity/tests/test_tas.py           # ALL_TAS True          (loader + metrics)
python phase3_activity/tests/test_mistake.py       # ALL_MISTAKE True      (order model + monitor)
python phase3_activity/tests/test_anticipation.py  # ALL_ANTICIPATION True (next-step model)
python phase3_activity/tests/test_pipeline.py      # ALL_PIPELINE True     (model + train/eval + seam)
python phase3_activity/tests/test_demo.py          # ALL_DEMO True         (workflow-monitor replay)
```

---

## Seeing it run (start here)

The tests above prove the logic; **this** shows the product. `tas/demo.py` replays a
step stream through all three capabilities at once — recognised step → is it out of
order? → what comes next? — exactly as the live Phase 2 seam consumes it.

```bash
# zero downloads: the two small learned models are committed (phase3_activity/models/)
python -m phase3_activity.tas.demo --source sample --inject-fault

# with the coarse annotations (2.6 MB): replay a real held-out recording
python -m phase3_activity.tas.demo --source gt --index 0 --inject-fault

# the full pipeline: the TRAINED model predicts the steps, which then drive the monitor
python -m phase3_activity.tas.demo --source model --index 0
```

`--source auto` (the default) picks the best available: `model` → `gt` → `sample`.
Sample output (a real held-out build with an injected fault):

```
  [11] attach wheel                             next: demonstrate functionality 32%, ...
  [12] demonstrate functionality          <-#1  next: attach cabin 7%, screw wheel 6%, ...
  [13] attach door                              next: attach transport cabin 22%, ...
  [14] attach roof                        <-top3 next: attach bumper 10%, attach cabin 9%, ...
       *** MISTAKE (order_violation): 'attach roof' should precede: demonstrate
           functionality (which already occurred)
  ...
  mistakes flagged    : 1  -> order_violation
  injected fault      : CAUGHT
```

`<-#1` marks a step that the anticipation model had ranked **first before it happened**.
Anticipation is scored with no lookahead: the prediction at position *i* uses only
`steps[:i]`. In `--source sample` the stream is generated *by* the model, so no accuracy
is reported (it would be circular) — only the mechanism is shown.

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

**A self-contained MS-TCN baseline** (`tas/model.py`) with the full train → eval → score
pipeline (`tas/train.py`, `tas/evaluate.py`, `tas/postprocess.py`,
`tas/torch_dataset.py`), smoke-tested end-to-end on a synthetic fixture and **trained to
completion on the real features** (results below). No `statistic_input.pkl` is needed —
the loader reads each annotation's own frame range. Once the data is downloaded, run
(batch size 1 over variable-length videos):

```bash
python -m phase3_activity.tas.train    --data-root phase3_activity/data --view C10095_rgb
python -m phase3_activity.tas.evaluate --ckpt phase3_activity/models/mstcn_best.pt --fold val
# visualise predictions vs ground truth as a step-timeline PNG (-> outputs/):
python -m phase3_activity.tas.visualize --ckpt phase3_activity/models/mstcn_best.pt --fold val --count 4
```
Lower `--num-f-maps` / `--num-layers` if VRAM is tight (the temporal model is tiny —
LMDB I/O and sequence length dominate, not VRAM). Start on one view.

`tas/visualize.py` renders GT vs predicted step ribbons (same colour per step — a
vertical colour-match means correct) plus a green/red agreement strip and the
per-sequence MoF/Edit/F1 — a demoable picture of the model at work.

### Measured result (M2b — the trained model)

| | MoF | Edit | F1@10 | F1@25 | F1@50 |
|---|---|---|---|---|---|
| **this MS-TCN baseline** (120 val videos, view `C10095_rgb`) | **40.5** | **31.2** | **32.7** | **29.1** | **21.6** |
| published C2F-TCN reference | 37.8 | 31.7 | 33.3 | 28.6 | 20.6 |

Reproduce with `python -m phase3_activity.tas.evaluate --ckpt
phase3_activity/models/mstcn_best.pt --fold val`.

**Read this honestly.** The baseline lands *above* the reference on MoF and F1@50 and
within ~0.6 on the rest, which says the loader / metrics / training harness are correct —
that is what the comparison is for. It is **not** a claim of beating C2F-TCN: this is a
**single view** (`C10095_rgb`), and it is **[UNCERTAIN]** whether the published number is
single- or multi-view, so the two columns may not be strictly like-for-like. Treat it as
"reference-quality, pipeline validated", not SOTA.

The loader, metrics and train/eval harness are model-agnostic, so the official **C2F-TCN
+ pretrained checkpoint** drops in without changing them if you want the exact published
figure. Avoid ASFormer as a first baseline under tight VRAM.

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

## M4 — mistake detection (assembly-order deviation) ✅

The *canonical* Assembly101 mistake benchmark is defined on the **fine-grained**
annotations (each fine action tagged correct / mistake / correction), a separate gated
download we don't have. Rather than block on it — and because it maps better onto the
real "dynamic workflow monitoring" goal — we detect **order deviations from the learned
workflow**, which needs only the coarse annotations already on disk (no 34 GB features),
so it covers **every** training assembly.

**How it works** (`tas/procedure.py`):
- **Learn** an expected-order model from the training step sequences. For each ordered
  pair (a, b), measure how often `a` precedes `b` across videos containing both; if that
  is near-deterministic (≥ `precedence_tau`, default 0.99) with enough support
  (≥ `min_support`, default 20), `a` is a real **prerequisite** of `b`. Genuinely
  interchangeable steps sit near 50/50 and never become constraints — the key to a low
  false-positive rate. On the real **assembly** train fold this distils 16 constraints
  such as `attach interior → attach cabin`, `attach base → screw chassis`,
  `attempt to attach cabin → attach cabin`.
- **Detect online** (`MistakeMonitor`): a constraint `a → b` is violated only when
  **both** steps are actually performed **and** in the wrong order — detectable as the
  stream arrives (when a step is entered, if a step it must precede has *already*
  happened). A step that is simply **absent** (part of a different assembly) is never
  mistaken for a skipped prerequisite.

```bash
# build the order model (uses only coarse annotations — no features):
python -m phase3_activity.tas.procedure  --data-root phase3_activity/data --procedure assembly
# honest evaluation on the held-out val fold (injected order violations vs clean):
python -m phase3_activity.tas.mistake_eval --data-root phase3_activity/data --procedure assembly
```

**Measured on the real val fold** (model learned on train, evaluated on val — no leakage):
- **Recall = 100%** on injected order violations (95/95 — swap a constraint pair → all caught).
- **Per-transition false-positive rate = 6.6%** (59 flags / 892 transitions) — >93% of
  clean step transitions are correctly left un-flagged. The sequence-level clean-flag rate
  (35%, 21/60) is a *conservative* ceiling: inspection shows most such flags are genuine
  **rework** (attach→detach→re-attach) or interchangeable orderings, not detector errors.
- **Precision 81.9%** over the mixed clean/perturbed set.

> **Corrected number.** This was previously published as **8.2%**. That figure divided by
> the count of *distinct* steps rather than the number of transitions the monitor actually
> judged; when a step recurs (rework) the denominator is too small and the rate is
> overstated. Fixed in `mistake_eval.py` — the detector is slightly *better* than was
> claimed, not worse. `tests/test_demo.py` now guards the convention.

**The number that matters for deployment — recogniser noise roughly doubles it.** All of
the above scores *ground-truth* step streams, so it measures the **order model alone**. Run
the same order model on the streams the **trained recogniser actually produces** and the
false-flag rate rises, because a recogniser that oscillates (`inspect toy` → `screw chassis`
→ `inspect toy` …) manufactures out-of-order transitions that never happened:

| step stream | per-transition flag rate | sequences with ≥1 flag |
|---|---|---|
| ground truth (order model alone) | 6.6% | 35% |
| **trained recogniser (whole pipeline)** | **11.4%** | **50%** |

*(60 assembly val sequences, `python -m phase3_activity.tas.demo --scan 999`. The GT column
reproduces `mistake_eval` exactly — two independent code paths agreeing.)* The fix for that
gap is **temporal smoothing of the recognised step stream**, not a looser order model —
loosening the constraints would cost the 100% recall. Worth stating plainly to anyone
reading the 6.6%: that is the ceiling, and 11.4% is today's end-to-end reality.

**Honest scope:** this measures *order-violation* detection (its own claim), **not** the
fine-grained mistake benchmark. It flags out-of-order performed steps; detecting a
genuinely *skipped* step needs a per-assembly plan / bill-of-materials (absence alone is
ambiguous). Both are stated so no one mistakes this for the canonical number.

**Wired into Phase 2** (same seam, honest live caveat): set `activity.procedure_model`
to the built JSON and the `assembly101` recognizer sets `ActivityResult.mistake` +
`detail` online; `run.py` logs a `workflow_mistake` event (once per out-of-order step)
to the event log. Live labels still depend on the stand-in TSM extractor caveat above —
offline replay of a recognized/GT step stream is the meaningful path today.

## M5 — next-action anticipation (next-step) ✅

The *canonical* Assembly101 anticipation benchmark predicts the **fine-grained** action
(1380 classes) at a future time *t+Δ*, which needs the fine-grained annotations (the
same gated download we don't have). As with mistake detection, we build the version
that runs on the coarse annotations already on disk and maps onto the AR use case:
**given the steps done so far, predict the next step** ("next: attach cabin").

**How it works** (`tas/anticipation.py`): a first-order Markov transition model over the
coarse steps with three workflow-aware refinements a raw bigram lacks — **done-masking**
(each step is done ~once, so completed steps are excluded), **back-off interpolation**
(transition ⊕ marginal, so unseen transitions degrade gracefully) and a **cold-start**
start-step distribution. Precedence **feasibility masking** is available but *off by
default* — it HURTS here because assembly ordering is loose, so strict prerequisites
prune valid next-steps (an honest, measured finding).

```bash
python -m phase3_activity.tas.anticipation --data-root phase3_activity/data --procedure assembly --examples 10
```

**Measured on the real val fold** (learned on train, no leakage; 952 predictions):

| predictor | top-1 | top-3 |
|---|---|---|
| marginal (frequency, done-masked) | 11.3% | 23.2% |
| bigram (transition, done-masked)  | 15.2% | 27.5% |
| **full (interpolated, shipped)**  | **15.5%** | **27.5%** |

Absolute accuracy is modest **by nature** — next-step over 157 loosely-ordered assembly
steps is genuinely hard (many parts can be attached next in any order). The model still
beats the frequency baseline by ~4 points and, as the worked examples show, nails the
strongly-ordered transitions a workflow assistant most needs (`attach interior →
attach cabin` 30%, `attach body → screw chassis` 17%) while staying appropriately
uncertain on interchangeable ones. **Measured realistic upside:** conditioning on the
known product / work-order (which a real inspection *does* know) lifts this to top-1
17% / top-3 29% — a natural next step, left out of the shipped runtime model because the
live ego stream isn't told the product.

**Wired into Phase 2:** set `activity.anticipation_model`; the `assembly101` recognizer
tracks the recognised-step history and fills `ActivityResult.next_steps`, which the
overlay renders as a `next: …` hint. Same live stand-in-extractor caveat as above.

## Roadmap hooks (same features, same seam)

- **Anticipation:** ✅ implemented above (`tas/anticipation.py`, next-step). Upgrades:
  condition on the known product (measured +2 top-3), or move to the fine-grained *t+Δ*
  benchmark (1380 classes) once those annotations are downloaded — the LMDB reader +
  `chunk_maxpool` carry over, target shifted forward by Δ.
- **Mistake detection:** ✅ implemented above (`tas/procedure.py`). Next upgrade would be
  the fine-grained correct/mistake/correction benchmark once those annotations are
  downloaded — the loader/metrics pattern here carries over.

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
