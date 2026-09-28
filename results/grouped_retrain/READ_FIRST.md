# READ FIRST — five things in this bundle that mislead if read at face value

Written for the agent drafting v3. Everything below was checked against the files in this
bundle and the ultralytics 8.4.75 source.

## 1. Training time is 11.22 h — not the 0.53 h in `run_info.json`

`run_info.json`'s `hours`, `started` and `ended` describe **only the final session**, and the
`time` column in `results.csv` **restarts at every resume**. Summed across sessions:

| stage | sessions (epochs → hours) | total |
|---|---|---:|
| stage 1 | 1–23 → 3.74, 24–40 → 2.74, 41–50 → 1.60 | **8.08 h** |
| stage 2 | 1–5 → 0.79, 6–17 → 1.89, 18–20 → 0.47 | **3.14 h** |
| | | **11.22 h** |

Epoch time only; excludes evaluation and the gaps between sessions. (`REPORT_local_train.md`
§4's "≈ 12.8 h" was a pre-run projection.)

## 2. How the run was interrupted

Five sessions, each resumed from the previous `last.pt`:

| session | covered | ended by |
|---|---|---|
| 1 | stage 1, epochs 1–23 | Windows Update reboot, 2026-09-24 02:46 |
| 2 | stage 1, epochs 24–40 | stopped deliberately |
| 3 | stage 1, epochs 41–50 + eval; stage 2, epochs 1–5 | BSOD, bugcheck `0x0A`, 17:14 |
| 4 | stage 2, epochs 6–17 | stopped deliberately |
| 5 | stage 2, epochs 18–20 + eval | completed, 21:13 |

An earlier handover said "four sessions, interrupted twice by Windows Update". That was wrong:
one Windows Update reboot, one BSOD, two deliberate stops.

## 3. A deviation from "same recipe" that the methods section must disclose

ultralytics 8.4.75 does **not** save early-stopping state in a checkpoint — `EarlyStopping` is
rebuilt fresh at `engine/trainer.py:373`, and only `best_fitness` is restored on resume. So
every resume restarted the patience counter, and **neither stage ever stopped early**.

Replaying ultralytics' own rule (`utils/torch_utils.py`: improvement is `fitness >
best_fitness`; stop when `epoch − best_epoch ≥ patience`) over the recorded validation curves:

| | the recipe, uninterrupted, would have… | actual run | Δ val mAP@50-95 |
|---|---|---|---:|
| stage 1 (patience 12) | stopped after epoch **28**, best epoch **16** (0.48888) | 50 epochs, best epoch **31** (0.49215) | +0.0033 |
| stage 2 (patience 10) | stopped after epoch **11**, best epoch **1** (0.47300)\* | 20 epochs, best epoch **13** (0.47414) | +0.0011 |

\* Replayed on the stage-2 trajectory actually run, which started from the epoch-31 weights.
A fully uninterrupted pipeline would have started stage 2 from epoch 16, so that exact
counterfactual cannot be measured — nor can the counterfactual checkpoints' **test** scores,
because `best.pt` was overwritten each time a better epoch arrived.

What this does and does not change:

- Checkpoints were still **selected on validation fitness**, never on the test set.
- The effect on validation is **≤ 0.33 pp** mAP@50-95 — two orders of magnitude smaller than
  the 27-point drop the paper reports. No conclusion moves.
- "Stage 2 is worse than stage 1" holds in **both** the actual run and the replay.
- It **is** still a deviation: both stages trained more epochs than the recipe's early stopping
  allows. "Like-for-like reproduction" should carry that qualifier, not be stated bare.

A methods sentence that says exactly this:

> Training was interrupted four times and resumed from the most recent checkpoint. Because the
> training framework does not persist early-stopping state across resumes, both stages ran
> their full epoch budgets (50 and 20) instead of stopping early as an uninterrupted run would
> have (after epochs 28 and 11). Checkpoints were selected on validation mAP@50-95 throughout;
> replaying the early-stopping rule on the recorded validation curves shows the selected
> checkpoints differ in validation mAP@50-95 by at most 0.0033.

## 4. `run_info.json` says `"dirty": true` — harmless

`git status` saw three untracked task-brief files. No tracked file differed from commit
`d32da26` while training ran: the only later edit (`tools/ckpt_train_record.py`) was made at
21:22, nine minutes after the run finished, and the training script does not import it.

## 5. Work ID vs the detector

The Work ID **measurements** are independent of this retrain: `reid_eval.py` scores an
injected-occlusion protocol on **synthetic** sequences with ground truth known by construction
(`phase5_workid/README.md` lines 67–70 and 116–122). Keep them.

But in deployment the identity layer consumes the detector's **Person** boxes, and Person
dropped too: **98.9 → 84.87 mAP@50, recall 97.8 → 80.92** — roughly one person in five missed
per frame, which the tracker's coasting has to bridge. Do not make end-to-end, on-site Work ID
claims that assume the old Person accuracy.

---

## Contents

| path | what |
|---|---|
| `eval_stage2/eval_stage2.{json,md}` | **the headline.** Four runs — `full_640`, `one_per_source_640`, `full_480`, `full_320` — overall and per class: precision, recall, F1, mAP@50, mAP@50-95 |
| `eval_stage1/eval_stage1.{json,md}` | the same four runs for stage 1 |
| `eval_stage*/eval_splits/` | the exact image lists each run scored (4,174 full / 584 one-per-source) |
| `stage1/`, `stage2/` `results.csv` | per-epoch training curves — **mind §1: `time` resets per session** |
| `stage1/`, `stage2/` `args.yaml` | the recipe as ultralytics received it (batch 16, nbs 96/48, workers 2) |
| `run_info.json` | versions, GPU, commit, split sha256, Part B values — **mind §1 and §4** |
| `split_summary.json` | per-split images, source groups, per-class instances; sha256 `60d0437e…` |
| `REPORT_local_train.md` | environment, recipe refit to 6 GB, smoke and resume tests |

Weights (`best_grouped.pt`, 22.5 MB) were removed deliberately.
