# Revision R1: new evidence for the Frontiers revision (manuscript 2019025)

**Prepared by:** Claude Code running **Claude Opus 5.5** (model ID `claude-opus-5-5`), working
autonomously from 19 November 2026. **Branch:** `revision-r1`, created from `main` at `978778e`.
**Unchanged:** tag `v1.0.1-paper` and its GitHub release were not touched.

<!-- SUMMARY: written last -->

## 0. Before the parts

### 0.1 Why CMOT was dropped

The task asked for the CMOT construction tracking dataset (Yan et al., 2024). Its annotations are
on GitHub, and were cloned to `data/cmot/repo` at commit `a289e87`. They match the task exactly:

- 40 worker sequences (19 train, 2 val, 19 test) with 30,399 frames;
- 121 worker IDs, of which 16 are ever labelled no-helmet, in 10 sequences.

The frames, however, are only on Alipan, and Alipan needs a mainland-China phone number to sign
in. That is not possible from Malaysia. On 19 November the plan therefore changed:

- **CMOT is dropped.** The annotations stay on disk but are not used.
- **Part C becomes MOT17.** The real-video identity test (Part D) now runs on the 7 MOT17 train
  sequences.
- **Part E becomes synthetic.** The CMOT violation-episode analysis is replaced by a synthetic
  timing test of the compliance logic.

My own attempt to query the Alipan share's public metadata was refused by Claude Code's safety
check, and I did not try to get round it.

### 0.2 Code changes to existing scripts

The only existing script that changed is `tools/train_grouped.py`. It gained three options:

- **`--seed N` and `--no-plots`.** Their defaults reproduce the old behaviour byte for byte:
  `model.train` receives identical arguments, in identical key order. This was checked against
  `978778e` with a stub `YOLO`, for both stages, before and after the third option was added.
- **`--faithful-resume`** (off by default; it only acts together with `--resume`). It was added after
  seed 1 crashed and had to be resumed (see Part A). ultralytics 8.4.75 has two resume flaws:
  - it does not save early-stopping state, so the patience counter restarts at the resume epoch;
  - its resume path closes mosaic without resetting the data-loader workers.

  The option fixes both at `on_train_start`:
  - early stopping gets back the best epoch from the stage's own `results.csv`, together with the
    checkpoint's `best_fitness`;
  - after a late resume the loader is reset, exactly as the uninterrupted path does at
    `trainer.py:417`.

  A unit test confirmed that the restored state equals that of an uninterrupted run replayed over
  the same recorded epochs.
- **`run_info.json`** gains no fields unless an option is given.

All other new code is in `tools/revision/`. Nothing under `phase2/src/` or `phase*/` changed.

### 0.3 GPU incident before Part A

The first launch of seed 1 (19 Nov, 17:28) ran at 15 s per iteration instead of about 0.3 s.

- **Cause:** the GPU was held at its minimum clock, 210 MHz, by a false power reading. The sensor
  read 86–94 W even with the GPU idle, against a 75.8 W limit.
- **Action:** that attempt was stopped after 28 iterations, and its log is kept in
  `partA/attempt0_gpu_power_capped/`.
- **Fix:** after a restart, a 30 s load test (`tools/revision/gpu_check.py`, results in
  `partA/gpu_check/`) gave a median SM clock of 2,550 MHz at 85 W and 25.0 TFLOPS fp16. The idle
  reading was a sane 2.3 W.
- **Relaunch:** seed 1 was started again from scratch at 17:56.

## Part E (replaces the CMOT episodes): timing of the compliance logic at the deployed settings

**What was run.** `tools/revision/compliance_timing.py` builds phase2's `ComplianceMonitor` exactly
as `SafetyPipeline` does (`phase2/src/pipeline.py:55-58, 87-89`). The values come from
`phase2/config.yaml`: debounce 5, clear 15, containment 0.30, lost-track buffer 30. Nothing in
`phase2/src` was changed.

- **Sequences.** The input comes from the Dataset D generator (`phase5_workid.reid_eval.make_sequence`):
  4 workers in "similar" kit and 900 frames, i.e. 30 s at the generator's nominal 30 fps. Its
  ground-truth person boxes and ground-truth track IDs are used, with no tracker.
- **Episodes.** Worker 0 violates for 1 s, worker 1 for 3 s and worker 2 for 10 s. Worker 3 never
  violates and serves as the control.
  - Each episode starts at a random frame in 360–420, after every worker's scripted occlusion and
    re-entry (frames 315–342), so one continuous track ID carries it.
  - The evidence is a No-Helmet box inside the person's helmet band (containment 1.0). Nothing is
    injected for anyone outside an episode.
- **Missing evidence.** Each episode frame's evidence is dropped independently with p = 0, 0.2 or
  0.4. The same draw is used at both rates, so the two rates are paired.
- **Rates.** The monitor is updated every frame (30 fps) and every 4th frame (7.5 Hz, "about 7
  updates per second").
- **Seeds.** 20, numbered 100–119. Box geometry does not depend on the seed; the seed drives the
  dropouts and the episode start times.

**Definitions.**
- **Alert delay:** the time of the frame whose monitor update first reports the alert, minus the
  episode start.
- **Clear delay:** the time of the first update after the episode end at which the violation is no
  longer active, minus the episode end. The episode end is the first compliant frame.
- **Split:** two or more alerts on that track between the episode start and its clear.
- **False alert:** any alert on the control worker or outside an episode window.

| rate | evidence missing | episode | alerted | alert delay, s: median [min–max] | clear delay, s: median [min–max] |
|---|---:|---:|---:|---|---|
| 30 fps | 0% | 1 s | 20/20 | 0.133 [0.133–0.133] | 0.467 [0.467–0.467] |
| 30 fps | 0% | 3 s | 20/20 | 0.133 [0.133–0.133] | 0.467 [0.467–0.467] |
| 30 fps | 0% | 10 s | 20/20 | 0.133 [0.133–0.133] | 0.467 [0.467–0.467] |
| 30 fps | 20% | 1 s | 20/20 | 0.200 [0.133–0.600] | 0.467 [0.367–0.467] |
| 30 fps | 20% | 3 s | 20/20 | 0.217 [0.133–0.400] | 0.467 [0.367–0.467] |
| 30 fps | 20% | 10 s | 20/20 | 0.200 [0.133–0.467] | 0.467 [0.367–0.467] |
| 30 fps | 40% | 1 s | **16/20** | 0.333 [0.133–0.667] | 0.467 [0.333–0.467] |
| 30 fps | 40% | 3 s | 20/20 | 0.383 [0.133–1.533] | 0.433 [0.400–0.467] |
| 30 fps | 40% | 10 s | 20/20 | 0.517 [0.133–0.900] | 0.467 [0.333–0.467] |
| 7.5 Hz | 0% | 1 s | 20/20 | 0.600 [0.533–0.633] | 1.900 [1.867–1.967] |
| 7.5 Hz | 0% | 3 s | 20/20 | 0.583 [0.533–0.633] | 1.917 [1.867–1.967] |
| 7.5 Hz | 0% | 10 s | 20/20 | 0.583 [0.533–0.633] | 1.917 [1.867–1.967] |
| 7.5 Hz | 20% | 1 s | **11/20** | 0.767 [0.533–0.967] | 1.900 [1.667–1.967] |
| 7.5 Hz | 20% | 3 s | 20/20 | 0.817 [0.533–2.367] | 1.900 [1.667–1.967] |
| 7.5 Hz | 20% | 10 s | 20/20 | 0.933 [0.533–2.200] | 1.900 [1.667–1.967] |
| 7.5 Hz | 40% | 1 s | **5/20** | 0.833 [0.633–0.867] | 1.900 [1.833–1.967] |
| 7.5 Hz | 40% | 3 s | **18/20** | 1.050 [0.700–2.000] | 1.900 [1.567–1.967] |
| 7.5 Hz | 40% | 10 s | 20/20 | 2.300 [0.867–6.600] | 1.850 [1.400–1.933] |

The delay columns cover alerted episodes only.
- **Splits:** no episode split into two or more alerts, in any condition.
- **False alerts:** none in any condition, i.e. 0 per worker-minute over 40 worker-minutes per
  condition (4 workers × 30 s × 20 seeds).
- **Clearing:** every alerted episode cleared.

**What the frame-based rules mean in seconds.** The monitor counts updates, not seconds.
- **Raise.** An alert fires on the 5th evidence update, 4 update intervals after the first one: 0.133 s
  at 30 fps and 0.53 s at 7.5 Hz. Where the episode start falls relative to the sampling grid adds
  up to one interval, which gives the observed 0.53–0.63 s. The brief's "5 frames ≈ 0.17 s at 30 fps"
  counts 5 intervals; the code fires one interval earlier.
- **Missed frames.** A missed evidence frame lowers the evidence streak by one instead of resetting it.
  Missing evidence therefore delays alerts rather than preventing them, unless the episode is short.
  At 7.5 Hz a 1 s episode gives only 7–8 updates, so with 20% or 40% of the evidence missing it
  raised an alert in only 55% and 25% of cases.
- **Clear.** A violation clears on the 15th consecutive update without evidence while the person is
  still tracked. That is 14 intervals after the first compliant frame: 0.47 s at 30 fps and about
  1.9 s at 7.5 Hz. It is 0.5 s and 2.0 s after the last evidence frame.

**Caveats.**
- **Missing evidence was independent per frame.** A real detector's misses are correlated (a turned
  head, a dark frame), which gives longer gaps and therefore longer delays and possible splits.
  This test covers only the compliance logic.
- **Track breaks were not exercised.** Ground-truth track IDs were used, so a tracker ID break could
  not occur. The code map shows that compliance state is keyed on the tracker ID: a ByteTrack ID
  break restarts the debounce and gives a second alert, even when the Work ID layer re-links the
  worker. This was verified with a probe during the code mapping.
- **Recurrence was not exercised.** The evidence streak is uncapped and is not reset when a violation
  clears (`phase2/src/compliance.py:115, 124-126`). A recurrence soon after a long episode therefore
  re-fires on its first evidence frame, without the 5-frame debounce. There is one episode per
  worker here, so this was not exercised; the manuscript's "raised once seen in five frames" needs
  that qualifier.

**Command, versions, run time.**
- **Command:** `.venv\Scripts\python.exe -B tools\revision\compliance_timing.py`
- **Versions:** Python 3.13.3, numpy 2.5.1, supervision 0.29.1.
- **Run time:** 65 s.
- **Results:** `partE/episodes.csv` (one row per episode) and `partE/summary.json`.

## Part F.3: what a "false merge" is, and how ambiguous matches are handled

Both files are unchanged between `v1.0.1-paper` (`978778e`) and this branch; both were last
changed in `0fd1e04` (2 September 2026).

**The false-merge definition.** It is in `phase5_workid/reid_eval.py`, `_score()`, lines 300–329:

```python
    for w, uid in assign_log:                       # reid_eval.py:310
        u = remap(uid)
        uid_by_worker.setdefault(w, set()).add(u)
        if uid_owner.setdefault(u, w) != w:         # :313
            false_merges += 1                       # :314
    ...
        "false_merge_rate": 100.0 * false_merges / max(len(assign_log), 1),   # :325
```

The log it reads is built in `evaluate()`, lines 366–371. On every frame, every track that has an
identity and belongs to a real worker (phantoms excluded) adds one entry:

```python
        for t in tracks:
            res = out.get(t.track_id)
            if res is None or t.worker < 0:             # :368  (still in probation, or a phantom)
                continue
            assign_log.append((t.worker, res.uid))      # :370
```

What this means:
- **Ownership.** A uid is "owned" by the first ground-truth worker logged with it.
- **What counts.** Every later person-frame in which that uid is given to a different worker counts
  as one false merge.
- **Denominator.** The rate divides by all identity assignments in the log: every person-frame that
  had an identity. It does not divide by re-entries.
- **One error counts many times.** A single wrong re-identification that persists for N frames adds N
  false merges.
- **Dilution.** Every correct frame of every worker dilutes the rate. With 4 workers present for 78
  of 90 frames, a sequence has about 300 assignments, so 12.5% for one seed means roughly 39
  person-frames under another worker's uid.
- **Offline pass.** For `enhanced+offline`, the end-of-session merge map is applied to every uid
  first (`remap`, built by `_remap_from`, which follows merge chains). The offline pass is
  therefore scored on exactly the same log.
- **Same rule elsewhere.** The pipeline head-motion table (`pipeline_eval`, lines 459–478) uses the
  identical rule.
- **Same rule in this revision.** Part F.1 reuses `_score` unchanged, and Part F.2 calls it directly.

**How ambiguous matches are handled.** This is in `phase2/src/identity.py`, `_decide()`, lines 552–566,
the rule shared by live matching and consolidation:

```python
        if best < threshold:                         # identity.py:560
            return None, best
        if len(scores) > 1:
            second = scores[int(order[1])]
            if (best - second) < margin:             # :564
                return None, best              # ambiguous -> refuse to guess   (:565)
        return uids[best_i], best
```

When no worker is returned, the track becomes a new worker. This is in `update()`, pass 3, lines
283–284:

```python
                if uid is None:
                    uid = self._new_worker(None)
```

Confirmed against the code:
- **Appearance matches.** If the best known worker is below `match_threshold` (0.62), or beats the
  runner-up by less than `margin` (0.04 live, `config.yaml` lines 82–83), the matcher does not
  guess: it opens a new ID.
- **Consolidation.** The end-of-session pass uses the same rule with a wider margin,
  `consolidate_margin = max(margin, 0.10)` (line 176).

There are two qualifications, both in `_choose()`, lines 522–550, and both active in the deployed
configuration because the gate is on (`config.yaml` line 95):

1. **Position rule (lines 542–549).** If appearance is ambiguous but exactly one known worker could
   physically be there, that worker is accepted by position (`source='position'`). It must also be
   the most similar of all candidates and score at least `gate_floor` (0.45). This is a decision on
   position, not a refusal to guess. In identical PPE it can name the wrong person: a probe during
   the code mapping showed a returning identical-PPE worker whose badge was never readable being
   given another worker's identity this way (see Part F.2).
2. **Decisive-appearance override (lines 531–536).** A candidate vetoed by the gate is still
   accepted if its similarity is at least 0.85 and leads the runner-up by at least 0.20
   (`OVERRIDE_SIM`, `OVERRIDE_MARGIN`, lines 136–137). It needs at least two candidates.

**Merges cannot be undone.**
- `_merge_into` removes the source worker.
- `WorkerHistory.merge` deletes the source record.
- No `unmerge`, split or undo exists anywhere in `phase2/src`.
- The badge binder never overwrites a configured name on a living track (`workid.py:99-100`).

A wrong merge or a wrong name, once made, therefore stays for the rest of the session. Part F.2
tests this empirically.

## Part B: detector results at the deployed threshold, and bootstrap intervals

**What was run.** `tools/revision/threshold_eval.py` works in two stages, so inference runs only once.

1. **predict.** One ultralytics validation pass per (checkpoint, split):
   - **Settings:** imgsz 640, conf 0.001, batch 16, rect batches, on the CPU (the GPU was training).
   - **Two NMS settings from one forward pass:** IoU 0.6, as the brief specifies, and IoU 0.7,
     ultralytics' default and the setting of the published Table 3.
   - **Cache:** each image's predictions, ground truth and ultralytics' own true-positive matrix
     (`data/revision_r1_cache/partB/`, not committed).
   - **Jobs:** `best_grouped.pt` and `best_grouped_stage1.pt` on the grouped valid (4,163 images) and
     test (4,174) splits, and `best_refined.pt` on the published test split (4,190).
   - **Weights:** read from their local copies, after checking that each SHA-256 equals the
     `v1.0.1-paper` release asset.
2. **analyze.** Thresholds and bootstrap are computed from the cache.

**Checks that passed before anything was reported** (`partB_results.json`, `checks`):
- **Cache vs printed metrics.** For all 5 jobs, the mAP and P/R computed from the cache equal what the
  ultralytics validator printed, with a difference of 0.
- **Cache vs published Table 3.** At NMS 0.7 the cache reproduces the published, GPU-computed
  `full_640` figures:
  - stage 2: P 70.78 / R 71.35 / mAP@50 71.11 / mAP@50-95 46.24, against 70.78 / 71.34 / 71.11 / 46.24;
  - stage 1: mAP@50 74.47 / mAP@50-95 48.34, against 74.47 / 48.33.
  - The largest difference is 0.07 points, from CPU against GPU arithmetic.
- **Bootstrap code.** The weighted mAP used for the bootstrap equals ultralytics' `ap_per_class`
  exactly at unit weights, for every cache.

### B.1–B.2 Precision, recall and F1 at IoU 0.5 for three confidence thresholds

The three rules:
- **(a)** conf 0.35, the threshold the system uses.
- **(b)** the F1-optimal confidence chosen on **valid** and applied to test.
- **(c)** the F1-optimal confidence on test itself, which is the benchmark-style figure ultralytics prints.

"F1-optimal" is ultralytics' own rule: one confidence shared by all classes, at the maximum of the
box-smoothed mean-F1 curve on a 1,000-point grid. For (a) and (b), predictions above the threshold
(strictly, as ultralytics' NMS and the deployed detector do) are re-matched image by image with
ultralytics' matching rule. For (c), the re-matched values equal ultralytics' interpolated values to
0.1 points.

**`best_grouped.pt` (stage 2, the paper's checkpoint), grouped test, NMS IoU 0.6.**
Each cell is precision / recall / F1, in %.

| class | (a) conf 0.35 | (b) conf 0.163 (valid-optimal) | (c) conf 0.104 (test-optimal) |
|---|---|---|---|
| Helmet | 85.5 / 85.7 / 85.6 | 81.2 / 88.7 / 84.8 | 78.8 / 89.8 / 83.9 |
| **No-Helmet** | **50.7 / 40.7 / 45.2** | 45.6 / 48.3 / 46.9 | 42.4 / 51.8 / 46.7 |
| No-Vest | 74.0 / 47.6 / 58.0 | 70.7 / 54.7 / 61.6 | 69.2 / 57.9 / 63.1 |
| Person | 86.4 / 74.8 / 80.2 | 83.8 / 79.5 / 81.6 | 82.5 / 81.5 / 82.0 |
| Vest | 87.2 / 71.5 / 78.6 | 83.4 / 77.3 / 80.3 | 81.8 / 80.1 / 80.9 |
| mean | 76.8 / 64.1 / 69.8 (macro F1 69.5) | 72.9 / 69.7 / 71.3 (71.0) | 70.9 / 72.2 / 71.6 (71.3) |

**`best_grouped_stage1.pt`, grouped test, NMS IoU 0.6.**

| class | (a) conf 0.35 | (b) conf 0.234 (valid-optimal) | (c) conf 0.174 (test-optimal) |
|---|---|---|---|
| Helmet | 85.4 / 87.4 / 86.4 | 82.2 / 89.4 / 85.6 | 80.1 / 90.4 / 84.9 |
| **No-Helmet** | **52.3 / 45.8 / 48.9** | 49.4 / 53.6 / 51.4 | 47.0 / 57.0 / 51.5 |
| No-Vest | 72.3 / 55.7 / 62.9 | 70.1 / 60.7 / 65.1 | 68.9 / 63.5 / 66.1 |
| Person | 85.0 / 78.5 / 81.6 | 83.2 / 81.6 / 82.4 | 82.0 / 83.5 / 82.8 |
| Vest | 85.7 / 78.0 / 81.6 | 82.7 / 80.6 / 81.6 | 81.1 / 82.3 / 81.7 |
| mean | 76.1 / 69.1 / 72.4 (72.3) | 73.5 / 73.2 / 73.3 (73.2) | 71.8 / 75.4 / 73.6 (73.4) |

"Mean" is the macro mean of P and R, with the F1 of those means. The mean of the per-class F1 values
is in brackets.

**At NMS IoU 0.7 (Table 3's setting), stage 2:**
- **(a)** mean 76.4 / 64.1 / 69.7, with No-Helmet 50.7 / 40.7 / 45.2.
- **(b)** conf 0.174: mean 72.7 / 69.3 / 70.9, with No-Helmet 45.7 / 47.6 / 46.6.
- **(c)** conf 0.121: mean 70.8 / 71.3 / 71.1. This is exactly the published Table 3 P and R.
- **Difference from NMS 0.6:** at most 0.9 points in any per-class cell and 0.4 in any mean.

Every cell, for both checkpoints and both NMS settings, is in `partB/thresholds_per_class.csv`.

**Reading.**
- **No-Helmet at the deployed threshold.** At conf 0.35 the deployed detector finds about 4 in 10
  No-Helmet instances on unseen source photographs: recall 40.7%. Half of its No-Helmet boxes are
  correct (precision 50.7%).
- **Lowering the threshold.** Using the threshold that validation prefers (about 0.16–0.17) raises
  No-Helmet recall to 48% at 46% precision. The benchmark-style (c) figure is the most favourable for
  recall, because its threshold is chosen on the test set itself.

### B.3 Bootstrap intervals (1,000 resamples, seed 0, 95% percentile intervals)

- **Cluster bootstrap.** Resamples source groups (the `group` column of
  `tools/splits/ppe_grouped_split.csv`). Each drawn group brings all its images, with their
  multiplicity.
- **Image bootstrap.** Resamples images.
- **Paired draws.** The same draws are used for every checkpoint scored on the same test set.
  This makes the stage 2 vs stage 1 comparison paired.

Shown at NMS 0.7, the Table 3 setting. At NMS 0.6 every interval moves by 0.2 points or less; see
`partB_results.json`.

| checkpoint | test set (images / source groups) | mAP@50 | cluster 95% CI | image 95% CI | mAP@50-95 | cluster 95% CI | image 95% CI |
|---|---|---:|---|---|---:|---|---|
| `best_grouped.pt` (stage 2) | grouped (4,174 / 584) | 71.1 | [67.8, 74.7] | [70.0, 72.4] | 46.2 | [43.6, 49.2] | [45.4, 47.1] |
| `best_grouped_stage1.pt` | grouped (4,174 / 584) | 74.5 | [71.2, 78.1] | [73.4, 75.7] | 48.3 | [45.8, 51.5] | [47.5, 49.2] |
| `best_refined.pt` | published (4,190 / 2,370) | 98.2 | [97.8, 98.8] | [98.0, 98.6] | 85.9 | [84.5, 86.9] | [85.3, 86.4] |

- **Paired stage 2 − stage 1 (cluster):** mAP@50 −3.3 [−4.2, −2.4], mAP@50-95 −2.1 [−2.7, −1.5].
- **Paired stage 2 − stage 1 (image):** mAP@50 −3.3 [−3.7, −2.9].
- **Standard errors for stage 2's mAP@50:** 1.8 points (cluster) and 0.6 points (image).

**Reading.**
- **Width.** The 584 source groups, not the 4,174 images, are the independent units. Resampling images
  understates the uncertainty about threefold. An honest interval for the paper's 71.1 is
  **[67.8, 74.7]**.
- **Stage 2 vs stage 1.** Stage 2 is worse than stage 1 in essentially every resample. The paired
  interval excludes 0, so "stage 2 is worse" is not noise.
- **The published test split.** Its 4,190 images fall into 2,370 source groups, yet its interval is
  narrow and high. The intervals measure sampling noise only; they cannot remove the leakage that
  inflates that 98.2.

The raw bootstrap samples are in `partB/bootstrap_samples.csv`.

### B.4 When the deployed settings were set

`confidence_threshold: 0.35`, `debounce_frames: 5` and `clear_frames: 15` were introduced in the very
first project commit, `68a82db` (24 June 2026, 20:25 +08:00).
- **Where:** in both `phase2/config.yaml` (lines 15, 36, 37) and the code defaults in
  `phase2/src/config.py` (lines 27, 36, 37).
- **History:** `git log -G` shows no later commit changing any of them. The only later commit touching
  `debounce_frames` in `config.py` is `0fd1e04` (2 September), which only adds a check that probation
  stays below the debounce.
- **Timing:** the dataset audit's re-split came on 22 September (`7d82fd4`), three months later. The
  deployed thresholds therefore predate the audit and cannot have been tuned on the grouped test
  split.

### B.5 Supplementary check: the live detector's own inference path at conf 0.35

The live system does not run ultralytics' validator. `phase2/src/detector.py` calls
`model.predict(frame, conf=0.35, imgsz=640)`. That path uses single-label NMS at IoU 0.7, letterboxes
each image separately, and returns boxes in original pixels.

`threshold_eval.py deployed` runs exactly that call on every grouped-test image with `best_grouped.pt`.
It matches the boxes to the YOLO label files at IoU 0.5, using the same matching rule as above.

| class | live predict path, P / R / F1 (TP / FP / FN) | validation-style cache, NMS 0.7, P / R / F1 (TP / FP / FN) |
|---|---|---|
| Helmet | 85.3 / 85.6 / 85.4 (7,411 / 1,280 / 1,248) | 85.2 / 85.7 / 85.4 (7,421 / 1,292 / 1,238) |
| **No-Helmet** | **52.9 / 41.8 / 46.7** (420 / 374 / 584) | 50.7 / 40.7 / 45.2 (409 / 398 / 595) |
| No-Vest | 73.5 / 48.8 / 58.7 (2,243 / 810 / 2,351) | 73.9 / 47.7 / 58.0 (2,190 / 773 / 2,404) |
| Person | 84.9 / 76.0 / 80.2 (6,602 / 1,170 / 2,082) | 86.1 / 74.8 / 80.1 (6,496 / 1,049 / 2,188) |
| Vest | 85.3 / 72.5 / 78.4 (1,660 / 285 / 631) | 86.3 / 71.5 / 78.2 (1,638 / 259 / 653) |
| mean | 76.4 / 64.9 (macro F1 69.9) | 76.4 / 64.1 (macro F1 69.4) |

The two paths agree to within 2.2 points in every cell and within 0.8 in the means. The B.1–B.2
conf-0.35 figures can therefore stand for the live system: on unseen source photographs, the deployed
detector finds about 42% of No-Helmet instances, and about 53% of its No-Helmet boxes are correct.

- **Command:** `.venv-gpu\Scripts\python.exe tools\revision\threshold_eval.py deployed --threads 2`
- **Run time:** 15.7 min on the CPU.
- **Result file:** `partB/deployed_path_grouped_s2_test.json`

### Deviations, notes, versions and run times

- **Deviations and notes.**
  - **NMS IoU.** The brief's NMS IoU of 0.6 is reported. IoU 0.7 is also reported from the same
    forward pass, because Table 3 used ultralytics' default of 0.7 and the bootstrap should be
    centred on the published numbers.
  - **Cache source.** Predictions were cached inside ultralytics' validator: multi-label NMS, rect
    batches. This makes the cache reproduce the published evaluation exactly. The live detector uses
    `model.predict`, which has single-label NMS and per-image letterboxing. A supplementary pass
    measures that path directly (B.5 below).
  - **Hardware.** Inference ran on the CPU. It differs from the published GPU numbers by at most 0.07
    points.
  - **Weights.** The release staging folder named in an earlier handover no longer exists. The
    weights were read from `paper_materials/retrain/` and the repository root after their SHA-256
    values were checked against the release.
- **Surprise.** By default, `phase2/config.yaml` loads `phase2/models/best.pt`, which is byte-identical
  to **`best_refined.pt`**, the model trained on the leaky split, not to `best_grouped.pt`. Anyone
  running the shipped app gets the leaky-split model unless they swap the file.
- **Commands.**
  - `.venv-gpu\Scripts\python.exe tools\revision\threshold_eval.py predict --threads 4`, with
    `CUDA_VISIBLE_DEVICES=-1`.
  - `... threshold_eval.py analyze`.
- **Versions.** Python 3.13.3, torch 2.13.0+cu130 (on the CPU), ultralytics 8.4.75, numpy 2.5.2,
  OpenCV 5.0.0.
- **Run times.**
  - Prediction: 14.3 + 12.6 + 12.4 + 11.0 + 11.1 = 61.4 min, on 4 threads while the GPU trained.
  - Analysis, including 12,000 bootstrap mAP evaluations: 179 s.

## Part F.1: held-out seeds and configurations never used in tuning (R1.3)

### F.1.1 How the parameters were chosen

This was established from git history, the documents and every Claude Code transcript on this laptop.
Two independent investigators did the search, and a third agent verified every claim against the
evidence and refuted 12 of them. The full record is in `partF1/provenance.json`.

| parameter (deployed value) | set in | how it was chosen, per the record | seeds |
|---|---|---|---|
| `match_threshold` 0.62 | `76d9da5`, 25 Jul 2026, this laptop | The only reconstructable tuning run: `python -m phase5_workid.reid_eval --sweep` on the first-version layer (no probation, no gate), decided on the "similar" scenario ("highest recall with zero false merges"). 0.60, 0.62 and 0.70 tie exactly on that sweep, so the rule does not single out 0.62. | 0, 1, 2 (inferred from the CLI defaults and an exact rerun; the original output is lost) |
| `margin` 0.04 | `76d9da5` | No sweep or rationale recorded beyond "refuse to guess". Every later run held it at 0.04. | none recorded |
| `probation_frames` 3 | `0fd1e04`, 2 Sep 2026, made in another clone and pulled here on 21 Sep | A constraint: it must stay below `debounce_frames` 5. Only ever measured at 3 (ablation rows). | 0–2, and 0–9 for the 10-seed confirmation |
| gate on | `0fd1e04` | A design decision stated in the spec, presented with the `--ablation` table. The order of events is not recorded. | 0–2, and 0–9 |
| `gate_speed` 0.03 per frame (0.9 body heights per s) | `0fd1e04` | "Chosen by a sweep" according to a single source, the design spec; the README defers to it and the manuscript copies it. No command, value grid, seeds or output survives, and the commit message does not mention a sweep. | unknown |
| `gate_slack` 0.5, `gate_horizon` 90 frames (3 s), `gate_floor` 0.45, override 0.85 / 0.20, `consolidate_margin` 0.10 | `0fd1e04` | Listed as defaults; no record of how they were chosen. | none recorded |

**Held-out seeds.**
- **Disjoint from every recorded run.** Every recorded or documented Work ID run before this revision
  (tuning, evaluation, cross-check replays, unit tests) used seeds 0–9. Seeds 100–119 are disjoint
  from those, and so are their derived random streams.
- **Parameters frozen.** `git diff 0fd1e04 81daca4` is empty for every identity file.
- **First use.** Seeds 100–119 first appear on 19 November 2026.
- **Caveat on the record.** Held-out status can only be stated relative to the record. The Phase 9
  session left no logs on this laptop.
- **Caveat on what a seed changes.** In this generator a seed changes only the pixels: background
  texture, lighting and noise. Box geometry, spacing, track IDs and re-entry frames are the same for
  every seed. At the default configuration, held-out seeds therefore re-test appearance noise on the
  geometry the gate was designed on; they do not test whether the gate generalises.
- **Timing.** The runs below started about 9 minutes after the provenance check began, not after it
  finished. No parameter was changed at any point, so they are valid held-out runs.

**Were the "unseen" configurations really unseen?** None of the seven is recorded as having been used
to choose a parameter value. Several were run before, though:
- **Never run before this revision:** `--gap 6`, `--gap 18`, `--gap 24` and `--workers 3`. Unit tests
  use 3 workers, but only as harness checks.
- **`--workers 6`:** not unseen. It is a published stress test, run on seeds 0–2, and was re-run in
  the October cross-check.
- **`--frames 120`:** run before only together with `--relocate --gap 45`.
- **`--motion 4`:** run before only through `--pipeline`, the head-motion table with real ByteTrack.
- **Which ones test the gate:** a gate-geometry replay shows that only `--gap 24` and `--workers 6`
  put the gate outside the geometry it was designed on. A second vacated worker becomes spatially
  plausible in 2 of 4 re-entries at gap 24 and in 5 of 6 with 6 workers. The other five mainly re-test
  appearance under new pixels.

### F.1.2 Results: 20 held-out seeds (100–119), parameters frozen

**What was run.** The brief's command `python -m phase5_workid.reid_eval --ablation --seed 100 --seeds 20`
was run with each configuration's flag added. That gives 8 CLI runs; their printed tables are in
`partF1/cli/`.

The CLI prints only 20-seed means. `tools/revision/reid_heldout.py` therefore makes the same
`make_sequence` → `_fresh` → `evaluate` calls one seed at a time, with the same arguments as the CLI.
This gives the per-seed values in `partF1/per_seed.csv`. All 120 printed CLI lines (8 configurations
× 15 rows) match the wrapper's 20-seed aggregates exactly (`partF1/check_cli_vs_wrapper.json`).

**Definitions.**
- **Re-ID recall:** forced re-entries reunited with the right worker.
- **False merges:** the paper's definition from `_score()`; see Part F.3.
- **Spread across seeds:** shown as mean ± SD [95% CI of the mean, Student t with 19 df, clipped to
  0–100] (lowest–highest seed).
- **IDF1:** an optional output of the wrapper, standard IDF1 on per-frame (true worker, uid) pairs.
  A track in probation has no uid and counts as a miss. The offline row is scored after the
  consolidation merges. Nothing in `phase5_workid/` was changed to get it.

**Distinct kit.** In every configuration and every row, "distinct" gives 100% recall and 0 false
merges on all 20 seeds. Its IDF1 is 100 for the baseline and 96.9–98.1 when probation is on, because
probation costs 2 frames per new track. It is left out of the table below. All 5 ablation rows,
including `+probation` and `+gate`, are in `partF1/summary.json`.

| configuration | scenario | row | re-entries recovered | re-ID recall, % | false merges, % | IDF1, % |
|---|---|---|---:|---|---|---|
| default (gap 12, 4 workers, 90 frames) | similar | baseline | 70/80 | 87.5 ± 17.2 [79.4, 95.6] (50.0–100.0) | 0.75 ± 3.37 [0.00, 2.33] (0.00–15.06) | 94.5 ± 7.8 |
| default (gap 12, 4 workers, 90 frames) | similar | enhanced | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| default (gap 12, 4 workers, 90 frames) | similar | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| default (gap 12, 4 workers, 90 frames) | uniform | baseline | 3/80 | 3.8 ± 12.2 [0.0, 9.5] (0.0–50.0) | 9.76 ± 10.60 [4.80, 14.72] (0.00–28.85) | 56.2 ± 5.4 |
| default (gap 12, 4 workers, 90 frames) | uniform | enhanced | 78/80 | 97.5 ± 11.2 [92.3, 100.0] (50.0–100.0) | 0.62 ± 2.80 [0.00, 1.93] (0.00–12.50) | 96.2 ± 5.1 |
| default (gap 12, 4 workers, 90 frames) | uniform | enhanced+offline | 78/80 | 97.5 ± 11.2 [92.3, 100.0] (50.0–100.0) | 0.62 ± 2.80 [0.00, 1.93] (0.00–12.50) | 96.2 ± 5.1 |
| --gap 6 | similar | baseline | 70/80 | 87.5 ± 19.0 [78.6, 96.4] (25.0–100.0) | 1.52 ± 6.79 [0.00, 4.69] (0.00–30.36) | 95.0 ± 7.8 |
| --gap 6 | similar | enhanced | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.6 ± 0.0 |
| --gap 6 | similar | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.6 ± 0.0 |
| --gap 6 | uniform | baseline | 8/80 | 10.0 ± 12.6 [4.1, 15.9] (0.0–25.0) | 14.11 ± 16.66 [6.31, 21.91] (0.00–55.95) | 61.0 ± 6.0 |
| --gap 6 | uniform | enhanced | 78/80 | 97.5 ± 11.2 [92.3, 100.0] (50.0–100.0) | 0.67 ± 3.00 [0.00, 2.08] (0.00–13.44) | 96.4 ± 5.3 |
| --gap 6 | uniform | enhanced+offline | 78/80 | 97.5 ± 11.2 [92.3, 100.0] (50.0–100.0) | 0.67 ± 3.00 [0.00, 2.08] (0.00–13.44) | 96.4 ± 5.3 |
| --gap 18 | similar | baseline | 66/80 | 82.5 ± 21.6 [72.4, 92.6] (25.0–100.0) | 1.22 ± 5.43 [0.00, 3.76] (0.00–24.31) | 92.0 ± 9.8 |
| --gap 18 | similar | enhanced | 79/80 | 98.8 ± 5.6 [96.1, 100.0] (75.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 96.6 ± 2.5 |
| --gap 18 | similar | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.1 ± 0.0 |
| --gap 18 | uniform | baseline | 6/80 | 7.5 ± 11.8 [2.0, 13.0] (0.0–25.0) | 12.29 ± 12.69 [6.35, 18.23] (0.00–35.76) | 58.0 ± 5.4 |
| --gap 18 | uniform | enhanced | 74/80 | 92.5 ± 24.5 [81.0, 100.0] (0.0–100.0) | 1.64 ± 5.29 [0.00, 4.11] (0.00–21.32) | 94.0 ± 10.4 |
| --gap 18 | uniform | enhanced+offline | 74/80 | 92.5 ± 24.5 [81.0, 100.0] (0.0–100.0) | 2.28 ± 7.03 [0.00, 5.57] (0.00–24.26) | 94.0 ± 10.4 |
| --gap 24 | similar | baseline | 76/80 | 95.0 ± 10.3 [90.2, 99.8] (75.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.8 ± 4.5 |
| --gap 24 | similar | enhanced | 78/80 | 97.5 ± 7.7 [93.9, 100.0] (75.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 95.9 ± 3.0 |
| --gap 24 | similar | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 96.9 ± 0.0 |
| --gap 24 | uniform | baseline | 5/80 | 6.2 ± 11.1 [1.1, 11.4] (0.0–25.0) | 6.59 ± 7.54 [3.06, 10.12] (0.00–21.97) | 59.9 ± 4.4 |
| --gap 24 | uniform | enhanced | 8/80 | 10.0 ± 18.8 [1.2, 18.8] (0.0–75.0) | 4.35 ± 6.89 [1.13, 7.58] (0.00–25.00) | 59.9 ± 7.4 |
| --gap 24 | uniform | enhanced+offline | 26/80 | 32.5 ± 30.5 [18.2, 46.8] (0.0–100.0) | 11.55 ± 13.86 [5.07, 18.04] (0.00–43.55) | 68.7 ± 12.4 |
| --workers 3 | similar | baseline | 52/60 | 86.7 ± 22.7 [76.0, 97.3] (33.3–100.0) | 0.83 ± 3.73 [0.00, 2.58] (0.00–16.67) | 93.8 ± 10.6 |
| --workers 3 | similar | enhanced | 60/60 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --workers 3 | similar | enhanced+offline | 60/60 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --workers 3 | uniform | baseline | 0/60 | 0.0 ± 0.0 [0.0, 0.0] (0.0–0.0) | 9.19 ± 9.48 [4.75, 13.62] (0.00–20.09) | 55.1 ± 0.0 |
| --workers 3 | uniform | enhanced | 58/60 | 96.7 ± 10.3 [91.9, 100.0] (66.7–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 96.0 ± 4.2 |
| --workers 3 | uniform | enhanced+offline | 60/60 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --workers 6 | similar | baseline | 116/120 | 96.7 ± 8.7 [92.6, 100.0] (66.7–100.0) | 0.33 ± 1.48 [0.00, 1.02] (0.00–6.62) | 98.7 ± 3.4 |
| --workers 6 | similar | enhanced | 120/120 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --workers 6 | similar | enhanced+offline | 120/120 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --workers 6 | uniform | baseline | 5/120 | 4.2 ± 7.4 [0.7, 7.6] (0.0–16.7) | 6.83 ± 5.14 [4.42, 9.23] (0.00–15.81) | 59.1 ± 3.2 |
| --workers 6 | uniform | enhanced | 9/120 | 7.5 ± 10.1 [2.8, 12.2] (0.0–33.3) | 4.08 ± 5.73 [1.40, 6.76] (0.00–17.57) | 59.3 ± 3.6 |
| --workers 6 | uniform | enhanced+offline | 26/120 | 21.7 ± 20.3 [12.2, 31.2] (0.0–66.7) | 18.30 ± 14.09 [11.71, 24.89] (0.00–41.67) | 64.8 ± 8.1 |
| --frames 120 | similar | baseline | 73/80 | 91.2 ± 12.2 [85.5, 97.0] (75.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 96.3 ± 5.2 |
| --frames 120 | similar | enhanced | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 98.1 ± 0.0 |
| --frames 120 | similar | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 98.1 ± 0.0 |
| --frames 120 | uniform | baseline | 1/80 | 1.2 ± 5.6 [0.0, 3.9] (0.0–25.0) | 12.41 ± 9.65 [7.89, 16.92] (0.00–26.85) | 56.0 ± 2.2 |
| --frames 120 | uniform | enhanced | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 98.1 ± 0.0 |
| --frames 120 | uniform | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 98.1 ± 0.0 |
| --motion 4 | similar | baseline | 71/80 | 88.8 ± 19.0 [79.9, 97.6] (25.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 94.8 ± 8.7 |
| --motion 4 | similar | enhanced | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --motion 4 | similar | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --motion 4 | uniform | baseline | 5/80 | 6.2 ± 13.8 [0.0, 12.7] (0.0–50.0) | 4.20 ± 8.10 [0.41, 7.99] (0.00–28.85) | 57.5 ± 6.0 |
| --motion 4 | uniform | enhanced | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |
| --motion 4 | uniform | enhanced+offline | 80/80 | 100.0 ± 0.0 [100.0, 100.0] (100.0–100.0) | 0.00 ± 0.00 [0.00, 0.00] (0.00–0.00) | 97.4 ± 0.0 |

**Reading.**
- **Default configuration, held-out seeds.** The published identical-PPE result holds:
  - enhanced recall 97.5% (78 of 80 re-entries), with 0.62% false merges;
  - baseline recall 3.8% (3 of 80), with 9.8% false merges.
  The false merges are concentrated in single seeds: the enhanced layer's worst seed has 12.5%.
- **Configurations inside the tuned geometry.** With `--gap 6`, `--workers 3`, `--frames 120` and
  `--motion 4`, enhanced identical-PPE recall stays between 96.7% and 100%.
- **`--gap 18`.** Enhanced recall is 92.5%, and one seed scored 0%.
- **The two configurations that move the gate outside the geometry it was designed on:**
  - `--gap 24`: enhanced identical-PPE recall falls to **10.0% (8 of 80)**.
  - `--workers 6`: it falls to **7.5% (9 of 120)**.
  - In both, end-of-session consolidation recovers some re-entries (32.5% and 21.7%), but raises
    false merges to 11.6% and 18.3%.

  The identical-PPE result therefore depends on the scene geometry: workers spaced about one body
  height apart, returning within about 0.6 s. It does not hold outside that geometry.

**Commands, versions, run time.**
- **CLI runs.** `.venv\Scripts\python.exe -m phase5_workid.reid_eval --ablation --seed 100 --seeds 20`,
  then the same with each of `--gap 6`, `--gap 18`, `--gap 24`, `--workers 3`, `--workers 6`,
  `--frames 120` and `--motion 4`. About 2 min each.
- **Per-seed wrapper.** `.venv\Scripts\python.exe tools\revision\reid_heldout.py --idf1`. 16.5 min.
- **Versions.** Python 3.13.3, numpy 2.5.1, OpenCV 5.0.0.
- **CPU.** Single-threaded, while the GPU was training.

## Part F.2: badges inside the identity pipeline (R2.2)

**Question.** The paper measures badge (ArUco) read range in isolation (`phase5_workid/badge_eval.py`). Reviewer 2 asks what happens when badges are only *sometimes* readable inside the identity pipeline:

- Does a worker get named, and how quickly?
- Does a worker ever get someone else's name?
- Are anonymous fragments created while the badge was hidden folded back into the named worker, together with their violation history?
- Can a wrong merge be undone?

**What was run.** Synthetic workers wearing real ArUco badges were passed through the deployed per-frame pipeline: `phase2/src/pipeline.py` `SafetyPipeline`, built from `phase2/config.yaml`. The stages are ByteTrack → ComplianceMonitor → WorkIdBinder (crop search on) → IdentityManager → live merges → WorkerHistory, then `close()` with end-of-session consolidation.

- **Detector:** the only part replaced. An oracle returns the true person boxes and injected No-Helmet head boxes.
- **Workers:** Alice Tan (marker 0) and Bob Lim (marker 6), in *identical* PPE (both `SIMILAR[0]`), 320 px tall, re-rendered every frame with `render_badged_figure`.
- **Runs:** 30 fps, 20 seeds (100-119) per scenario, 160 runs in all.
- **Scenarios:** the brief's four (a-d), plus four variants added after review. Two close coverage gaps (b_close, d_noreturn), one is a control (c_named) and one is a perspective sensitivity check (c_persp).

Settings were read back from the constructed objects and are the shipped ones:

| component | settings |
|---|---|
| ByteTrack | fr 30, max_time_lost 30, det_thresh 0.35 |
| ComplianceMonitor | debounce 5, clear 15 |
| WorkIdBinder | containment 0.5, crop 320 px, reacquire 30, gc 90 |
| IdentityManager | match 0.62, margin 0.04, probation 3, gate on, gate horizon 3 s, 0.9 body heights/s, floor 0.45; consolidation 0.62 / 0.10 |
| WorkerHistory | absence tolerance 15 |

| scenario | what happens (timings drawn per seed) |
|---|---|
| **a** | Alice's badge is readable for 2.5-3.5 s, hidden for 4.1-6.0 s (median 5.1 s) while she stays in view, then readable again. Bob is in view throughout, badge readable. No-Helmet: Alice 2 s inside the hidden period; Bob 1.5 s. |
| **b** | Both are named. Both leave. Alice is away exactly 60 frames (ByteTrack keeps a lost id for ~31 frames) and returns to her spot with her badge hidden while Bob is still away. Her badge becomes readable 3-4 s later. Bob returns 0.7-2 s after her. No-Helmet: Alice 1 s that clears before her badge is read, plus 2 s that is still open when it is read; Bob 1.5 s. |
| **c** | Both enter with their badges turned away and walk past each other 0.2-1 s later (6 px/frame; boxes overlap for ~25 frames, max IoU 0.84-0.90), then stand at each other's spot for 6 s. Badges are hidden in seeded random segments, independently per worker: hidden for 1-3 s from entry, then alternately shown and hidden for 0.5-3 s each. Hidden share: median 0.56, range 0.36-0.69. No-Helmet: each worker 2 s after the crossing. |
| **d** | Alice's badge is never readable; Bob's always is. Both leave once. Alice returns (after 1.7-3.9 s away) while Bob is still away, so the badge-confirmed lookalike is a candidate for her. No-Helmet: Bob 1.5 s before leaving; Alice 2 s after returning. |
| b_close | As b, but Alice leaves again (60 frames) before her badge is read on the re-entry track, then returns readable. The fragment's track has ended, so only `close()` can fold it. |
| c_named | Control (the first design of c): badges hidden i.i.d. per frame (p 0.5), and the workers stand 2-3 s before crossing, so both are named before the crossing. |
| c_persp | As c (same timings, hiding and render seeds), but the worker further back (Bob) is drawn 300 px tall instead of 320 (6 % smaller, consistent perspective). |
| d_noreturn | As d (same timings and renders until then), but Bob does not come back, so he is a consolidation candidate for Alice's re-entry record. |

### F.2.1 Main findings

1. **A hidden badge costs nothing while the worker keeps the same track (a).**
   - The binder's label is sticky and the identity layer keeps the track's uid.
   - Over 20 seeds: 0 wrong names and 0 false merges in 13,418 assignments.
   - Alice's violation during the hidden period is in her named record in 20/20 seeds.
2. **Fragments created while the badge was hidden are folded back correctly, with their violation history.**
   - (b) All 15 re-entry fragments were merged live into Alice's named record on the frame her badge was read: 0.00-0.03 s after it became readable again. All 30 of their episodes landed in her named record at that moment, including 15 still open; the open ones were re-keyed and kept running.
   - (b_close) When the fragment's track ended before any read, nothing happened live. `close()` then folded 18/18 fragments, and their 18/18 episodes, into her named record.
3. **Identical PPE produces wrong names, through three different paths.**
   - **Appearance or position at re-entry.** The identity layer gave never-badged or hidden-badge Alice the absent lookalike's name, "Bob Lim":
     - (b) 2/20 seeds, by appearance (similarity 0.99), 95 person-frames;
     - (d) 4/20 seeds, 2 by appearance and 2 by the gate's lone-plausible-position rule, 231 person-frames.
   - **The badge binder in a crossing.** The binder gave Alice's (front) badge to Bob's overlapping, smaller box. Bob's still-anonymous track was then named "Alice Tan" for the rest of the session:
     - (c) 1/20 seeds; the boxes differed by 1 px;
     - (c_persp) 5/20 seeds, with a consistently smaller back box.
   - **End-of-session consolidation.** In (d_noreturn), 1/20 seeds: never-badged Alice's re-entry record was folded into "Bob Lim".
4. **A wrong merge is never undone.** All 23 wrong-merge events (in 17 of the 160 runs) were still merged at the end of the session.
   - In 6 events the identity layer stopped misattributing *going forward*, and only because Bob came back and his badge re-claimed his name.
   - The misattributed frames and violation episodes stayed in the wrong record in every case.
   - In (c)/(c_persp), Bob's own badge was decoded 70-113 times after the error and ignored each time.
   - No method in `phase2/src` can split a record (F.2.2.5).

### F.2.2 Results

#### F.2.2.1 Time until named and until the badge is first read

The table gives seconds from the worker's first appearance, as median [min-max] over the seeds in which the event happened, and the number of seeds in which it never happened.

- **Named:** the first frame whose IdentityResult label equals the worker's name.
- **Read:** the first frame the binder's `_assign` gives the worker's own marker to the worker's own track.
- **Floor:** 0.033 s, ByteTrack's one-frame activation lag.

| scenario | worker | named: median [range] (s) | never named | first read: median [range] (s) | never read |
|---|---|---|---|---|---|
| a | Alice Tan | 0.03 (all 20) | 0/20 | 0.03 (all 20) | 0/20 |
| a | Bob Lim | 0.03 (all 20) | 0/20 | 0.03 (all 20) | 0/20 |
| b | Alice Tan | 0.03 (all 20) | 0/20 | 0.03 (all 20) | 0/20 |
| b | Bob Lim | 0.03 (all 20) | 0/20 | 0.03 (all 20) | 0/20 |
| c | Alice Tan | 2.08 [1.03-3.00] | 0/20 | 2.08 [1.03-3.00] | 0/20 |
| c | Bob Lim | 2.00 [1.00-2.93] | 1/20 | 2.00 [1.00-2.93] | 0/20 |
| d | Alice Tan (badge never readable) | — | 20/20 | — | 20/20 |
| d | Bob Lim | 0.03 (all 20) | 0/20 | 0.03 (all 20) | 0/20 |
| b_close | Alice Tan | 0.03 [0.03-0.07] | 0/20 | 0.03 [0.03-0.07] | 0/20 |
| b_close | Bob Lim | 0.03 (all 20) | 0/20 | 0.03 (all 20) | 0/20 |
| c_named | Alice Tan | 0.07 [0.03-0.27] | 0/20 | 0.07 [0.03-0.27] | 0/20 |
| c_named | Bob Lim | 0.07 [0.03-0.20] | 0/20 | 0.07 [0.03-0.20] | 0/20 |
| c_persp | Alice Tan | 2.08 [1.03-3.00] | 0/20 | 2.08 [1.03-3.00] | 0/20 |
| c_persp | Bob Lim | 1.90 [1.00-2.93] (15 named) | 5/20 | 2.00 [1.00-2.93] | 0/20 |
| d_noreturn | Alice Tan (badge never readable) | — | 20/20 | — | 20/20 |
| d_noreturn | Bob Lim | 0.03 (all 20) | 0/20 | 0.03 (all 20) | 0/20 |

Re-entries are timed from the re-entry frame. "Commit" is the identity decided when probation ended on the new track (source:label).

| scenario | re-entry | named after: median [range] (s) | read after: median [range] (s) | commit at re-entry |
|---|---|---|---|---|
| b | Alice (badge hidden) | 3.47 [0.10-3.97] | 3.68 [3.07-4.00] | new anonymous "Worker 1" ×15; appearance → own name ×3; appearance → **"Bob Lim" ×2** |
| b | Alice: read after the badge is shown again | | 0.00 [0.00-0.03] | |
| b | Bob (badge readable) | 0.03 (all 20) | 0.03 (all 20) | marker → own name ×20 |
| d | Alice (badge never readable) | never (20/20) | never | new "Worker 2" ×14; appearance → her own first record ×2; appearance → **"Bob Lim" ×2**; position → **"Bob Lim" ×2** |
| d | Bob | 0.03 (all 20) | 0.03 (all 20) | marker ×20 |
| b_close | Alice, 2nd visit (badge hidden, leaves before any read) | 5.00 [0.10-5.50] (named only on the 3rd visit, except 2 appearance matches) | 5.03 [4.53-5.50] | new "Worker 1" ×18; appearance → own name ×2 |
| b_close | Alice, 3rd visit (readable) | 0.03 [0.03-0.07] | 0.03 [0.03-0.07] | marker ×20 |
| d_noreturn | Alice | never (20/20) | never | identical to d (same renders up to here) |

In c and c_persp, naming is a badge read on an anonymous track that is renamed in place, so named and read coincide.

The exceptions are the seeds in which Bob was never named (c: 1, c_persp: 5). There his badge *was* decoded and assigned to his track (first read 2.3-2.9 s after entry, 70-113 such frames per run), but the binder discarded each read: a configured name ("Alice Tan", wrongly bound before) is never overwritten on a living track.

#### F.2.2.2 Wrong-name assignments

A wrong name is a person-frame whose label is the *other* worker's configured name. An event is one stretch of wrong naming on one ByteTrack id.

| scenario | wrong person-frames | events | seeds with any | worker affected | source at onset | binder: wrong bindings / mis-assigned reads (to the smaller box) | frames with one uid on two tracks |
|---|---|---|---|---|---|---|---|
| a | 0 | 0 | 0/20 | — | — | 0 / 0 | 0 |
| b | 95 | 2 | 2/20 | Alice as "Bob Lim" | appearance ×2 | 0 / 0 | 0 |
| c | 226 | 1 | 1/20 | Bob as "Alice Tan" | marker ×1 | 1 / 1 (1) | 226 |
| d | 231 | 4 | 4/20 | Alice as "Bob Lim" | appearance ×2, position ×2 | 0 / 0 | 0 |
| b_close | 0 | 0 | 0/20 | — | — | 0 / 0 | 0 |
| c_named | 0 | 0 | 0/20 | — | — | 0 / 2 (2) | 0 |
| c_persp | 1,116 | 5 | 5/20 | Bob as "Alice Tan" | marker ×5 | 5 / 37 (37) | 1,116 |
| d_noreturn | 472 | 4 | 4/20 | Alice as "Bob Lim" | appearance ×2, position ×2 | 0 / 0 | 0 |

How each kind arises:

- **Appearance or position at re-entry (b, d, d_noreturn).** Alice re-enters while Bob is away, so both records are candidates, and with identical PPE the appearance scores are near-equal.
  - Usually the matcher refuses to guess and opens a new record.
  - In b (2 seeds) and d (2 seeds), Bob's gallery happened to beat Alice's by the 0.04 margin (similarity 0.97-0.99).
  - In d (2 seeds), Alice's own earlier record had been gone longer than the 3 s gate horizon (3.17 and 3.27 s). It then carried no position information, so Bob was the lone plausible neighbour and the position rule picked him (`identity.py:542-549`; similarity 0.95 and 1.00).
  - The wrong name ended when Bob came back and his badge took the name back (`identity.py:443-446`), after 44-85 frames. With Bob visible again and therefore excluded, Alice's track was re-matched by appearance to her own record: her named record in b, her earlier anonymous "Worker 1" in d.
  - In d_noreturn Bob never comes back, so the same 4 seeds carried the wrong name to the end of the session (118 frames each).
- **The binder in a crossing (c, c_persp).**
  - `WorkIdBinder._assign` gives a tag that lies inside two person boxes to the *smaller* box (`workid.py:131-139`). It refuses only if the areas are within 5 % *and* the containments are within 0.05.
  - During the crossing, Bob (behind) has a full-extent box that covers Alice's visible badge. In c both figures are 320 px, so which box is smaller depends on a 1 px `int()` rounding of the rendered width. In c_persp Bob's box is always smaller.
  - All 1 (c), 2 (c_named) and 37 (c_persp) mis-assigned reads went to the strictly smaller box.
  - A mis-assigned read on a track already named (c_named: 2 reads; c_persp: 3 more seeds) is discarded by stickiness. On a still-anonymous track (c: 1 seed; c_persp: 5) it binds "Alice Tan" for good:
    - Bob's anonymous record is merged live into Alice's (`identity.py:437-440`);
    - both tracks then carry the same uid for the rest of the session: the binder keeps Alice's own track bound too, and pass 1 of `IdentityManager.update` (`identity.py:219-237`) does not check whether a uid was already claimed that frame;
    - Bob's later violation goes into Alice's record.
  - This is conditional on full-extent (amodal) boxes. A real detector's box for a half-hidden person may not reach the front worker's badge. The finding is the rule's behaviour under that geometry, not a measured rate for deployed boxes.
- **Consolidation (d_noreturn, 1 seed, 101).** See F.2.2.3; consolidation renames no live frame, so it does not appear in this table.

Outcomes of the crossings, split by how many tracks the binder had not yet named when the boxes first overlapped:

| scenario | tracks unbound at first overlap | seeds | seeds with a wrong name | wrong person-frames | false merges live / final | binder mis-assigned reads |
|---|---|---|---|---|---|---|
| c | 0 | 4 | 0 | 0 | 0 / 0 | 0 |
| c | 1 | 5 | 0 | 0 | 0 / 0 | 0 |
| c | 2 | 11 | 1 | 226 | 226 / 264 | 1 |
| c_named | 0 | 20 | 0 | 0 | 0 / 0 | 2 |
| c_persp | 0 | 4 | 0 | 0 | 0 / 0 | 9 |
| c_persp | 1 | 5 | 2 | 446 | 446 / 552 | 13 |
| c_persp | 2 | 11 | 3 | 670 | 670 / 821 | 15 |

No ByteTrack id ever covered both workers, in any of the 160 runs, and every worker kept one id through every crossing. The oracle boxes move smoothly, so the errors above come from the binder and the identity layer, not from tracker id switches.

#### F.2.2.3 False merges (the paper's definition)

The paper's definition is `phase5_workid.reid_eval._score` on the (true worker, uid) log of every identified person-frame. An assignment is a false merge when its uid is already owned by a different worker (the first logged worker owns it). The rate is false merges divided by all assignments, pooled over the 20 seeds. Three bases are scored:

- **live:** uids as output each frame;
- **after consolidation:** remapped through `_remap_from(consolidation merges)`, the paper's "+offline" row;
- **final:** remapped through `_remap_from(live take_merges + consolidation merges)`, i.e. the records the WorkerHistory report ends with.

| scenario | live: false merges / assignments (%) | seeds | after consolidation (%) | seeds | final (%) | seeds | re-ID recovered (live / consol. / final) |
|---|---|---|---|---|---|---|---|
| a | 0 / 13,418 (0.00) | 0 | 0 (0.00) | 0 | 0 (0.00) | 0 | — |
| b | 95 / 10,687 (0.89) | 2 | 95 (0.89) | 2 | 95 (0.89) | 2 | 23/40 / 23/40 / 38/40 |
| c | 226 / 11,039 (2.05) | 1 | 226 (2.05) | 1 | 264 (2.39) | 1 | — |
| d | 231 / 9,599 (2.41) | 4 | 231 (2.41) | 4 | 231 (2.41) | 4 | 22/40 / 36/40 / 36/40 |
| b_close | 0 / 11,142 (0.00) | 0 | 0 (0.00) | 0 | 0 (0.00) | 0 | 24/60 / 60/60 / 60/60 |
| c_named | 0 / 13,414 (0.00) | 0 | 0 (0.00) | 0 | 0 (0.00) | 0 | — |
| c_persp | 1,116 / 11,028 (10.12) | 5 | 1,116 (10.12) | 5 | 1,373 (12.45) | 5 | — |
| d_noreturn | 472 / 5,963 (7.92) | 4 | 590 (9.89) | 5 | 590 (9.89) | 5 | 2/20 / 2/20 / 2/20 |

- **Per-seed rates.** The median is 0.00 % in every scenario. The maxima per seed are b 9.26 %, c 42.56 % live / 49.72 % final, d 17.28 %, c_persp 42.45 % / 49.64 %, and d_noreturn 45.38 %.
- **Final above live (c, c_persp).** The live merge also moves the frames Bob's anonymous record had *before* the error (38-57 frames) into Alice's record.
- **d_noreturn seed 101.** Consolidation folded Alice's anonymous re-entry record (118 frames, and her No-Helmet episode) into "Bob Lim".
  - Appearance was not decisive: Bob scored 0.986 and her own first record 0.950, short of the 0.10 consolidation margin.
  - Her own first record had been gone 3.13 s, beyond the 3 s gate horizon, so it carried no position information. Bob, gone 2.53 s, was the lone plausible neighbour, and the position rule chose him (`identity.py:386-394` → `_choose`, `identity.py:542-549`). This was confirmed with a scratch probe of `_choose` during `close()`.
  - In seed 100 both candidates were plausible and consolidation refused.
  - Consolidation never merges records whose presence intervals overlap (`identity.py:376`). In d, Bob is present at the same time as each of Alice's records, so that rule excluded him in every seed.

#### F.2.2.4 Anonymous records: relabelled or merged at reacquisition (i) and at consolidation (ii), with their violation episodes

Only a record's anonymous period is counted: the frames output under it while it was "Worker N", and the episodes opened in that period.

- **Fragment:** a second record of a worker whose badge is read in the run.
- **Initial:** the worker's first record, committed while the badge was hidden.
- **Never-badged:** a record of Alice in d / d_noreturn.

| scenario | kind | records | (i) at reacquisition | (ii) at end of session | resolved by | episodes (open at reacq.) | in the owner's named record: at reacq. / at end |
|---|---|---|---|---|---|---|---|
| a | — | 0 | — | — | — | — | — |
| b | fragment | 15 | merged live into the right name 15 | correct name 15 | live merge 15 | 30 (15 open) | 30 / 30 |
| c | initial | 40 | renamed in place, right name 39; **merged live into the wrong name 1** | correct name 39; **wrong name 1** | reacquisition 40 | 0 | — |
| d | never-badged | 34 | not resolved 34 | merged into her own earlier record 14; separate 20 | consolidation 14; none 20 | 19 | (no named record) |
| b_close | fragment | 18 | **not resolved live 18** (the read happened on a later track) | correct name 18 | **consolidation 18** | 18 | 0 / **18** |
| c_named | initial | 5 | renamed in place 5 | correct name 5 | reacquisition 5 | 0 | — |
| c_persp | initial | 40 | renamed in place 35; **merged live into the wrong name 5** | correct 35; **wrong 5** | reacquisition 40 | 0 | — |
| d_noreturn | never-badged | 34 | not resolved 34 | separate 33; **merged into "Bob Lim" 1** | consolidation 1; none 33 | 16 | (no named record) |

(i) **At reacquisition.**
- A hidden-badge fragment whose track is still alive when the badge is read is merged live (`take_merges` → `WorkerHistory.merge`) before that frame is folded in. Its episodes are in the named record from that frame on: 30/30, including the 15 that were still open. The open ones were re-keyed and kept running.
- A worker's *first* record, committed while the badge was hidden, is renamed in place on the first read: same uid, so its history never moves.

(ii) **At consolidation.**
- A fragment whose track ended before any read is not touched live; `close()` folded 18/18 into the right named record, with 18/18 episodes.
- Records of the never-badged worker were merged only into her own earlier record (d: 14) or into the lookalike's named record (d_noreturn: 1). The rest stay separate anonymous records in the report.
- The duplicate-episode branch of `WorkerHistory.merge` (both records with the same violation open) was never reached: 0 episodes absorbed.

**Violation history.** All 340 injected windows were detected (`windows.csv`). Each was detected 0.13 s after it started (debounce 5), except one b window, detected after 0.07 s because its debounce counter carried over from the previous window.

| scenario | episodes | in the worker's own named record | in the right person's record | in the OTHER worker's named record |
|---|---|---|---|---|
| a | 40 | 40 | 40 | 0 |
| b | 61 (one window split in two) | 59 | 59 | 2 (Alice's, in "Bob Lim") |
| c | 40 | 39 | 39 | 1 (Bob's, in "Alice Tan") |
| d | 43 (three windows split) | 20 (Bob's) | 39 | 4 (Alice's, in "Bob Lim") |
| b_close | 40 | 40 | 40 | 0 |
| c_named | 40 | 40 | 40 | 0 |
| c_persp | 40 | 35 | 35 | 5 (Bob's, in "Alice Tan") |
| d_noreturn | 40 | 20 (Bob's) | 35 | 5 (Alice's, in "Bob Lim") |

A window is split when the worker's track changes uid mid-violation, for example when Bob's badge took his name back. The part before the switch stays in the wrong record.

#### F.2.2.5 Is a wrong merge ever undone? No.

Once a record has absorbed another worker's frames, through a wrong match, a wrong badge binding, or a live or consolidation merge, nothing in the pipeline separates them again.

**Empirically:**
- There were 23 wrong-merge events in 17 runs: b 2, c 2, d 4, c_persp 10, d_noreturn 5.
- At the end of the session, 23/23 were still merged; the final record held both workers' person-frames.
- 6/23 were corrected *going forward*: later frames of the misattributed worker went to a record it owns. That happened in b (2) and d (4), only because Bob came back and his badge took his name back. Even then, the earlier frames and episodes stayed where they were.

**Invariants, checked on every frame of all 160 runs (0 violations of each):**
- a WorkerHistory record's `frames_seen` never decreased;
- an IdentityManager worker's `frames_seen` never decreased;
- a uid merged away live was never output again;
- an episode changed record only along a merge made in that same frame, and never returned to a record it had left.

**In the code:**
- `phase2/src` has no unmerge, split, undo, revert or rollback method. A regex over every `def` found none, and the class method lists are in `summary.json` `code_evidence`.
- `IdentityManager._merge_into` (`identity.py:712`) pops the source worker (`identity.py:714`) and keeps no copy. Its only "undo" (`identity.py:718`) restores the source when the destination does not exist, which is not a reversal.
- `WorkerHistory.merge` (`workerlog.py:200`) moves the episodes and deletes the source record (`workerlog.py:257`). When both records had the same violation open, it also deletes the later episode (`workerlog.py:251`).
- `WorkIdBinder.resolve` never overwrites a configured name on a living track (`workid.py:99-100`). This is why Bob's own badge, decoded 70-113 times after the error in c/c_persp, never corrected his name.
- The pipeline applies merges one way only: `pipeline.py:160-161` (live) and `pipeline.py:213-217` (consolidation).

**Consequence.** A wrong name or wrong merge, made live or at consolidation, stays in the session's report: frames, time span and violation episodes are attributed to the wrong worker for good.

### F.2.3 Command, versions, run time

Run from the repository root in Git Bash:

```
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=-1 .venv/Scripts/python.exe -B tools/revision/badge_identity_eval.py > paper_materials/revision_r1/partF2/run.log 2>&1
```

- **Recorded command** (`summary.json` `meta.command`): `C:\Users\user\Desktop\SummerInternProject\.venv\Scripts\python.exe -B tools/revision/badge_identity_eval.py`, cwd = repository root. Defaults: seeds 100-119, all 8 scenarios, `--nms-iou 0.7`, `--drop-occluded 0`, output folder `paper_materials/revision_r1/partF2`. `main()` refuses to run without `-B`, and `meta.python_dont_write_bytecode` is `true`.
- **Run time:** 1,543.5 s (25.7 min), 2026-11-19 20:54:26 → 21:20:09, CPU only, 2 threads, one process. Per scenario, in seconds:

  | a | b | c | d | b_close | c_named | c_persp | d_noreturn |
  |---|---|---|---|---|---|---|---|
  | 242.2 | 189.1 | 194.7 | 181.5 | 204.1 | 229.6 | 188.5 | 112.3 |

- **Versions:** Python 3.13.3, numpy 2.5.1, OpenCV 5.0.0 (`cv2.aruco.ArucoDetector`), supervision 0.29.1 (ByteTrack), Windows 11 (10.0.26200).
- **Threads:** cv2 2; OMP/MKL/OPENBLAS 2; `CUDA_VISIBLE_DEVICES=-1`.
- **Code state:** git HEAD 5f58ce1 on branch revision-r1. `phase2/` and `phase5_workid/` are unmodified (git status clean, no new `.pyc`). The script is `tools/revision/badge_identity_eval.py`, sha256 `fd0352818591281d4c5fce71f5428b300939b4af9ac144f0f62eed3f07a57077`. It was untracked during the run and is committed with this section, unchanged.
- **Reproducibility:** an earlier full run with the same plans and seeds (20:19-20:44) gave a byte-identical per-frame log, and identical fragments, episodes, windows and wrong-merge tables. The rerun was made only to change two counting definitions (see Deviations 9).

### F.2.4 Design decisions

1. **The deployed pipeline, not a re-implementation.**
   - The pipeline is built from `phase2/config.yaml` with three overrides only: `workid.enabled = True` (it ships `false`), `device = cpu`, and `workid.markers = {0: "Alice Tan", 6: "Bob Lim"}`.
   - `src.detector.Detector` is replaced in-process by a stub before the pipeline is constructed. pipeline.py imports it inside `__init__`, so the real `reset()`, `process()` and `close()` run unchanged, and `identity.consolidate` runs inside `close()`.
   - The harness asserts that the binder, identity layer and history were built; with `quiet=True` the pipeline would silently disable them on any construction error.
2. **Observation without interference.**
   - `FrameResult` carries no uids, so the harness wraps instance methods of the pipeline's own objects: `tracker.update`, `monitor.update`, `binder.resolve`, `binder._assign`, `identity.update`, `identity.take_merges`, `identity.consolidate`, `history.update` and `history.merge`.
   - Every wrapper calls the original with the same arguments and returns its result unchanged.
   - Consolidation merges are recorded separately from live merges and asserted equal to what `consolidate()` returned.
3. **Oracle detector.**
   - Person boxes (class 3) are the full extent of each rendered figure, at confidence 0.9 (reid_eval's oracle value).
   - No-Helmet boxes (class 1, confidence 0.9) sit inside the helmet band of the violating worker's box: top 22 % of the height, inner 60 % of the width.
   - Ground truth travels in `Detections.data["gt"]`, which ByteTrack preserves, so each tracker id maps exactly to a true worker.
   - The deployed detector's class-wise NMS is emulated. `Detector.detect` passes no `iou`, so ultralytics' default 0.7 applies, and a box is suppressed when IoU > 0.7. Oracle scores are all equal, so the less-occluded (front) box survives, and a suppressed person's head box is withheld with it. This removed 1-4 back-box frames per crossing run and never a violation frame (checked).
4. **Scenes.**
   - `render_badged_figure` (`phase5_workid/badge_eval.py`) is called fresh for every worker in every frame, with a per-(seed, scene, frame, worker) seed. Every frame is therefore an independent read trial, as in badge_eval's read-rate measurement.
   - The figure is 320 px tall with a 10 cm badge (DICT_4X4_50). Its person-box patch is lifted with `_draw_worker`'s own `int()` rounding and pasted at the scenario position on one background per seed (`reid_eval._background(np.random.default_rng(seed))`). The worker further back (feet higher in the image) is pasted first.
   - A badge is hidden by overpainting the returned card box, plus 1 px, with the outfit's helmet colour.
   - Markers 0 and 6 share `SIMILAR[0]`, so the PPE is identical (asserted).
   - A variant that shares its parent's timings also shares its render seeds (c_persp with c; d_noreturn with d), so it differs only in what the variant changes.
5. **Per-frame ground truth** (`per_frame.csv.gz`), one row per person-frame:
   - the true worker and ByteTrack id, and the box;
   - whether the badge was shown (not overpainted) and whether it was visible (also not covered by a figure in front);
   - whether the worker's own marker was decoded anywhere in the frame;
   - the marker `_assign` gave this track (`read_mid`), the binder label (`in_badge_of`, `badge_label`), and the IdentityResult (uid, label, source, similarity);
   - the debounced No-Helmet state, and whether the box was withheld or NMS-suppressed.
6. **Definitions.** These are in `summary.json` `definitions`. The wrong-merge analysis also records, per event, any forward correction and which workers the final record holds, besides the frame-level invariants in F.2.2.5.
7. **Checks.** Each run carries scenario premise checks (`summary.json` `checks`). All passed in all runs except one deliberately loose premise in c/c_persp: "some track unbound at first overlap" held in 16/20 seeds. In seeds 103, 106, 107 and 117 both badges had been shown and read before the boxes met. Those seeds are kept, reported, and shown separately in the crossing table.

### F.2.5 Changes made after three internal reviews (each defect was verified before fixing)

The three reviewers, A, B and C, were independent checking agents, not the journal's reviewers.

| defect (reviewer.item) | verdict | change |
|---|---|---|
| A.1: the oracle reports two Person boxes with IoU > 0.7, which the deployed NMS cannot output | real (`detector.py:124-130` passes no `iou`; ultralytics `iou: 0.7`, `agnostic_nms: False`) | class-wise NMS at 0.7 emulated by default (`--nms-iou`, 0 = off) |
| A.2: the binder mis-assignment in (c) comes from harness geometry | real (an amodal back box covers the front badge; `_assign`'s smallest-box rule; 134/135 px widths from `int()`) | reported as conditional on amodal boxes (`summary.json` `caveats.crossings` and F.2.2.2); `c_persp` sensitivity variant; per-seed count of mis-assigned reads that went to the smaller box (all did) |
| A.3 / C.2: smoke runs launched without `-B` | real (the recorded command lacked `-B`; no .pyc was written) | `main()` refuses to run without `-B`; the smoke and full runs used it (recorded) |
| B.1 (major): (c) cannot produce an identity error, since both tracks are bound ~2 frames after entry, long before the crossing | real (reproduced) | (c) redesigned (badges turned away on entry, walk 0.2-1 s after entering, seeded contiguous hiding segments); the first design kept as the labelled control `c_named`; outcomes split by tracks unbound at first overlap |
| B.2: the smallest-box tie is decided by a random 1 px | real | caveat + `c_persp` (back worker 300 px, feet on the same line) |
| B.3: consolidation never exercised for a badge-identified worker's fragment; (d)'s consolidation fixed by the plan | real | `b_close` (the fragment's track ends before any read) and `d_noreturn` (Bob is a real consolidation candidate) |
| B.4: no violation open at merge time | real | (b): Alice's second No-Helmet window starts 25 frames before her badge becomes readable and ends 35 frames after; 15 episodes were open at the live merge |
| C.1 (major): the fragment-episode check counts episodes opened after an in-place rename | real (reproduced on the reviewer's c103 output) | only the anonymous period counts (frames and episodes); new kind `initial_anonymous`; per-record `episodes_opened_after_rename`; label-vs-frame consistency check (0 mismatches) |
| C, note: `moved_back_by_end` judges by name and could mislead | accepted | replaced by `final_record_workers` / `still_merged_at_end` and the frame-level invariants |
| B.1 (optional): reid_eval's handover re-entry | not done | see Deviations 3 |

### F.2.6 Deviations from the brief

1. **(c) hiding pattern.** The brief says "hide each badge independently at random in about half the frames, seeded".
   - The main (c) hides in seeded random *contiguous* segments, independent per worker: hidden 1-3 s from entry, then alternately shown and hidden for 0.5-3 s. The hidden share has a median of 0.56 (range 0.36-0.69).
   - Per-frame coin flips get a badge read within a frame or two. The sticky binding then fixes every identity outcome before the crossing, which the control `c_named` (literal per-frame hiding, share 0.45-0.55) shows: 0 errors in 20/20.
   - To make the crossing happen while tracks may still be unnamed, the workers enter with badges turned away and start walking 0.2-1 s after entering. The brief does not specify this timing.
2. **Extra scenarios.** Besides a-d there are b_close, c_named, c_persp and d_noreturn. They are reported separately and are not mixed into a-d.
3. **"As in reid_eval's swap idea."** The swap is implemented as the brief's parenthetical describes: the workers walk past each other with overlapping boxes and end at each other's spot. reid_eval's `make_sequence(swap=True)` handover (a returning worker re-enters at a spot another absent worker vacated) was **not** implemented.
4. **c_persp height.** It draws the back worker 300 px tall instead of 320. It is a sensitivity variant only; all other scenarios use 320 px for both workers.
5. **(b) violations.** Alice has two windows instead of one: one clears before her badge is read, the other is still open when it is read. This was added so the open-episode path of `WorkerHistory.merge` is tested.
6. **Oracle geometry.**
   - Boxes are full-extent (amodal) with no detector jitter or misses, apart from the emulated NMS. ByteTrack never swapped ids, so tracker-induced identity errors are not represented. Binder outcomes in the crossings are conditional on amodal boxes (F.2.2.2).
   - "Leaving view" means the figure vanishes in place, like an occlusion, and the worker re-enters at the same spot.
7. **"Actually read."** The per-frame log keeps the brief's `tid in badge_of` (`in_badge_of`, `badge_label`). However, "time until the badge is first read" uses the binder's `_assign` (own marker given to own track that frame), for two reasons:
   - `badge_of` includes sticky labels, and in the crossings it includes *wrong* ones. In c seed 119 and c_persp seeds 102, 112, 114, 118 and 119, Bob's track entered `badge_of` as "Alice Tan" 12-46 frames before his own badge was first decoded on it. His own name never entered `badge_of`.
   - In the other 274 of 280 worker-runs with any read, the first `_assign` read equals the first frame with the worker's own name in `badge_of`.
8. **Floor on time until named.** It includes ByteTrack's one-frame activation lag (0.033 s), so 0.03 s is the floor.
9. **Wrong-name events.** An event is a stretch of identity outputs on one ByteTrack id with the same wrong label; frames without an identity output for that worker do not split it.
   - The first full run used "consecutive frames". That split each crossing error in two at the 3-4-frame NMS suppression of Bob's box.
   - "Binder wrong bindings" likewise became distinct (worker, track, wrong name) instead of onsets.
   - Both definitions were changed after the first full run and everything was rerun. Every pipeline output was byte-identical; only these two counts changed (c: 2→1 event; c_persp: 10→5).
   - No threshold or scenario parameter was changed after seeing results.
10. **Not exercised.**
    - The duplicate-episode branch of `WorkerHistory.merge`.
    - Violation episodes in the anonymous period of (c)'s initial records, because the violations are injected after the crossing.
    - The handover re-entry (Deviation 3).

### F.2.7 Output files (`paper_materials/revision_r1/partF2/`)

- `summary.json`: meta (command, versions, threads, git, settings read back, run time), definitions, caveats, code evidence, and per-scenario summaries.
- `per_seed.csv`: one row per (scenario, seed), 160 rows.
- Per-object tables: `fragments.csv`, `episodes.csv`, `windows.csv`, `wrong_names.csv`, `wrong_merges.csv`.
- `per_frame.csv.gz`: per-frame ground truth and pipeline state.
- `runs.jsonl`: per-run detail.
- `run.log`: console log.

## Part D (on MOT17 instead of CMOT): identity on real video with real detector boxes

### D.1 Data

**Source.** `MOT17.zip` was downloaded from https://motchallenge.net/data/MOT17.zip (5,860,214,001
bytes, SHA-256 `7b486bb2af5cfedbfabc020511ff933a3edbf12182ce4985eec1a210f20f9e06`) into `data/mot17/`
(gitignored). The download took 37 min. Only the 7 train sequences of the FRCNN variant were
extracted; all variants share the same images and ground truth.

| sequence | camera | fps | frames | resolution | pedestrian IDs (considered) |
|---|---|---:|---:|---|---:|
| MOT17-02 | static | 30 | 600 | 1920×1080 | 62 |
| MOT17-04 | static | 30 | 1,050 | 1920×1080 | 83 |
| MOT17-05 | moving | 14 | 837 | 640×480 | 133 |
| MOT17-09 | static | 30 | 525 | 1920×1080 | 26 |
| MOT17-10 | moving | 30 | 654 | 1920×1080 | 57 |
| MOT17-11 | moving | 30 | 900 | 1920×1080 | 75 |
| MOT17-13 | moving | 25 | 750 | 1920×1080 | 110 |
| **total** | | | **5,316** | | **546** |

**Ground truth.** It is the official `gt.txt`, copied unchanged into TrackEval's MotChallenge2DBox
layout (`tools/revision/mot17_prepare.py`). TrackEval's own preprocessing applies the standard
handling:
- only pedestrians are evaluated;
- tracker boxes matched to distractor classes are removed (person on vehicle, static person,
  distractor, reflection);
- zero-marked rows are ignored.

None of these rules were written by me.

**About 7 updates per second.** Every round(fps/7)-th frame is kept: every 4th at 30 and 25 fps,
every 2nd at 14 fps, giving 7.5, 6.25 and 7 Hz. The ground truth is the same rows restricted to the
kept frames and renumbered 1…K, and only those frames are scored.

### D.2 Detections, cached once

`tools/revision/mot17_detect.py` ran on the CPU, because the GPU was training. Settings: imgsz 640,
conf 0.10, every frame, person class only, NMS IoU 0.7 (ultralytics' default, as in phase2's
detector).

- **(i) `best_grouped.pt`.** Class 3 "Person". 36,050 boxes, 200 ms per frame.
- **(ii) COCO-pretrained `yolov8s.pt`.** The weights training started from (SHA-256 `1f47a78b…`),
  class 0. 102,467 boxes, 115 ms per frame.

The same pass stores each box's 128-D native appearance feature for BoT-SORT. It uses the same forward
pre-hook that ultralytics registers for `with_reid: True, model: auto`.

**Checks.**
- The hooked pass returns detections identical to a plain `predict`.
- An independent reviewer confirmed that the cache's conf > 0.35 subset is bit-identical to what
  phase2's own `Detector.detect` passes to ByteTrack (14 frames per detector).

**Person detection at conf 0.35, IoU 0.5** (TrackEval CLEAR counts, standard distractor handling):

| detector | recall | precision |
|---|---:|---:|
| `best_grouped.pt` | **19.7%** | 91.4% |
| COCO `yolov8s.pt` | **43.1%** | 89.7% |

### D.3 Trackers

All trackers replay the same cached detections, each with its own default settings; nothing was tuned.
The code is in `tools/revision/mot17_track.py`.

1. **ByteTrack as in our pipeline.** `phase2/src/tracker.py` `PersonTracker`, using supervision 0.29.1
   `sv.ByteTrack`.
   - **Settings:** `frame_rate = round(fps / stride)` (the clip analyser's convention) and
     `lost_track_buffer` 30. That keeps a lost track about 1 s at any rate.
   - **Input:** the pipeline only ever feeds ByteTrack boxes with conf > 0.35, its detector
     threshold, so that is what this run gets.
   - **IDs:** raw ByteTrack track IDs.
2. **ByteTrack + Work ID layer.** The same tracker plus phase2's `IdentityManager`.
   - **Settings:** `identity_kwargs(cfg, frame_rate)` (match 0.62, margin 0.04, probation 3, gate
     on, histogram embedder), badges off.
   - **Clock:** clip time `now_s = frame / fps`, the clip analyser's own convention.
   - **IDs:** the stable worker uid. Ghost (predicted) boxes are never emitted.
   - **Probation:** a new track gets its uid on the 3rd update it appears in, so its first 2 boxes
     (more if it vanishes during probation) have no uid. The app shows these as "Person #id". Three
     variants are reported:
     - provisional IDs (1,000,000 + track ID), exactly what the app displays;
     - **`_uidonly`**: those boxes dropped;
     - **`_offline`**: the uids after the end-of-session consolidation.
   - **Control.** To compare like with like, **`_probfilt`** is ByteTrack with each track's first 2
     boxes dropped, a probation-matched control. An independent reviewer showed it is needed: dropping
     those boxes alone lowers ID switches and IDs per person.
3. **BoT-SORT with ReID.** ultralytics 8.4.75, the version that trained our detector, installed in
   `.venv-eval`.
   - **Settings:** `botsort.yaml` with `with_reid: True` and `model: auto`, i.e. the detector's own
     128-D features.
   - **Replay:** on the cached boxes and features, with the original frame for its sparse-optical-flow
     camera-motion compensation, built exactly as `model.track` builds it (`BOTSORT(args=cfg)`).
   - **Lost tracks:** kept for 30 updates at any frame rate.
   - **Verified:** an independent reviewer showed that this replay equals a real `model.track` run.
4. **StrongSORT and Deep OC-SORT.** boxmot 26.0.0 (in `.venv-eval`), `StrongSort()` and `DeepOcSort()`
   with their default configs.
   - **ReID weights:** boxmot's default `osnet_x0_25_msmt17.pt` (SHA-256 `6f57607f…`), fetched by
     boxmot from its catalogue during a smoke test on 19 November. All runs then used that local copy.
   - **Lost tracks:** `max_age` 30.

**Oracle boxes.** Trackers 1 and 2 also ran on the MOT17 ground-truth pedestrian boxes (class 1,
considered flag 1), given confidence 0.9. This is the value `reid_eval` gives synthetic boxes;
ByteTrack's association gate depends on IoU × confidence.

**About 7 Hz.** Trackers 1 and 2 also ran at about 7 updates per second, on both detectors and on the
oracle boxes.

**Sensitivity runs (`__feedall`).** Trackers 1 and 2 fed every cached box (conf ≥ 0.10). Nothing was
tuned on these runs.

**Effective input thresholds.** These differ even though every tracker receives the same cache:
- our pipeline uses conf > 0.35;
- BoT-SORT 8.4.75 starts tracks at ≥ 0.25 and, with `fuse_score`, can never match a box below 0.25 in
  its low-score stage, so in effect it uses ≥ 0.25;
- StrongSORT keeps conf ≥ 0.6 (its `min_conf`; the `det_thresh` 0.3 it logs is misleading);
- Deep OC-SORT keeps conf > 0.5.

### D.4 Metrics

The metrics come from TrackEval (git `12c8791`) in `tools/revision/mot17_eval.py`: HOTA, DetA, AssA,
IDF1, IDSW and MOTA, per sequence and combined (TrackEval's `COMBINED_SEQ`). They are also combined
separately over the static and the moving-camera sequences, because the overall figure is weighted by
box count and MOT17-04 alone holds about 42% of the ground-truth boxes. TrackEval still uses
`np.float`, `np.int` and `np.bool`, which NumPy 1.24 removed. The script restores them as aliases of
`float`, `int` and `bool` before importing it; TrackEval itself is unmodified.

Two counts are added from TrackEval's own preprocessed data, using per-frame Hungarian matching at
IoU ≥ 0.5:
- **IDs per person:** the mean number of predicted IDs per ground-truth identity.
- **Merged IDs:** predicted IDs that cover two or more ground-truth identities.

**Independent review.** Three reviewers checked Part D before this section was written: faithfulness,
metrics and anomalies. Their scripts, logs and findings are in `partD/review/`.
- **Reproduced:** they reproduced every phase2 run line for line through the real `SafetyPipeline`,
  six TrackEval runs and the detector P/R, by independent computation.
- **Fixed after their review:**
  - the probation-matched control was added;
  - the Work ID clock now follows the clip analyser (an exact-tie float comparison at the 3.0 s gate
    horizon had made 3 of 56 sequence-runs depend on the time origin; combined figures moved ≤ 0.25);
  - the timing was redone under equal, sequential conditions;
  - two run-record fields were corrected;
  - the oracle runs were added to the batch script.

### D.5 Results

TrackEval metrics in %, except the counts (IDSW, merged IDs) and IDs per person. "Combined" is TrackEval's `COMBINED_SEQ` over the 7 sequences.

**Native rate, COCO `yolov8s.pt` detections, 7 sequences combined.**

| tracker | HOTA | DetA | AssA | IDF1 | IDSW | MOTA | IDs per person | merged IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. ByteTrack as in our pipeline (fed conf > 0.35) | 39.8 | 34.0 | 46.9 | 47.2 | 541 | 37.5 | 2.35 | 167 |
| 2. ByteTrack + Work ID, uid as ID (probation boxes under provisional IDs) | 39.1 | 33.9 | 45.4 | 45.9 | 1,048 | 37.1 | 3.39 | 211 |
| 2. ByteTrack + Work ID, probation boxes dropped (`_uidonly`) | 39.1 | 33.2 | 46.4 | 46.2 | 312 | 37.1 | 1.78 | 169 |
| 2. … + end-of-session consolidation (`_offline_uidonly`) | 38.4 | 33.2 | 44.8 | 44.5 | 302 | 37.1 | 1.75 | 167 |
| control: 1 with each track's first 2 boxes dropped (`_probfilt`) | 39.4 | 33.3 | 47.0 | 47.0 | 416 | 37.1 | 2.11 | 147 |
| 3. BoT-SORT + ReID (ultralytics 8.4.75) | 41.2 | 36.9 | 46.3 | 49.4 | 692 | 39.9 | 2.43 | 188 |
| 4. StrongSORT (boxmot 26.0.0) | 35.3 | 25.2 | 49.7 | 39.7 | 175 | 27.9 | 1.53 | 74 |
| 4. Deep OC-SORT (boxmot 26.0.0) | 37.1 | 28.1 | 49.2 | 43.3 | 188 | 31.3 | 1.58 | 95 |
| sensitivity: 1 fed every cached box (conf ≥ 0.10) | 41.0 | 36.9 | 46.1 | 49.0 | 496 | 40.4 | 2.26 | 177 |
| sensitivity: 2 (`_uidonly`) fed every cached box | 39.9 | 36.2 | 44.3 | 47.5 | 357 | 40.1 | 1.88 | 181 |

**Native rate, our `best_grouped.pt` detections, 7 sequences combined.**

| tracker | HOTA | DetA | AssA | IDF1 | IDSW | MOTA | IDs per person | merged IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1. ByteTrack as in our pipeline (fed conf > 0.35) | 22.7 | 15.1 | 34.3 | 24.4 | 325 | 17.2 | 2.26 | 87 |
| 2. ByteTrack + Work ID, uid as ID (probation boxes under provisional IDs) | 22.2 | 15.1 | 32.8 | 23.6 | 669 | 16.9 | 3.31 | 124 |
| 2. ByteTrack + Work ID, probation boxes dropped (`_uidonly`) | 22.2 | 14.5 | 34.1 | 23.5 | 171 | 16.8 | 1.69 | 95 |
| 2. … + end-of-session consolidation (`_offline_uidonly`) | 21.8 | 14.5 | 33.0 | 22.8 | 166 | 16.8 | 1.65 | 89 |
| control: 1 with each track's first 2 boxes dropped (`_probfilt`) | 22.4 | 14.6 | 34.7 | 24.0 | 250 | 16.8 | 2.07 | 68 |
| 3. BoT-SORT + ReID (ultralytics 8.4.75) | 24.0 | 17.0 | 34.3 | 26.8 | 302 | 19.1 | 2.09 | 85 |
| 4. StrongSORT (boxmot 26.0.0) | 19.1 | 10.5 | 35.0 | 19.3 | 110 | 12.4 | 1.64 | 49 |
| 4. Deep OC-SORT (boxmot 26.0.0) | 19.3 | 10.7 | 34.9 | 19.4 | 137 | 12.6 | 1.73 | 49 |
| sensitivity: 1 fed every cached box (conf ≥ 0.10) | 24.7 | 17.8 | 34.5 | 27.2 | 309 | 20.2 | 2.20 | 96 |
| sensitivity: 2 (`_uidonly`) fed every cached box | 24.2 | 17.2 | 34.2 | 26.5 | 175 | 19.8 | 1.69 | 100 |

**Ground-truth boxes (oracle), native rate and about 7 Hz.**

| tracker | HOTA | DetA | AssA | IDF1 | IDSW | MOTA | IDs per person | merged IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native: ByteTrack as in our pipeline | 97.2 | 99.4 | 95.0 | 96.6 | 160 | 99.3 | 1.27 | 87 |
| native: control: ByteTrack, first 2 boxes of each track dropped | 96.4 | 98.4 | 94.5 | 96.3 | 108 | 98.4 | 1.18 | 72 |
| native: ByteTrack + Work ID (provisional IDs) | 90.9 | 99.4 | 83.2 | 88.0 | 668 | 98.9 | 2.19 | 153 |
| native: ByteTrack + Work ID (`_uidonly`) | 90.9 | 98.4 | 83.9 | 88.2 | 100 | 98.4 | 1.16 | 138 |
| native: ByteTrack + Work ID (`_offline_uidonly`) | 89.6 | 98.4 | 81.5 | 86.2 | 100 | 98.4 | 1.15 | 138 |
| ~7 Hz: ByteTrack as in our pipeline | 87.2 | 93.6 | 81.3 | 85.0 | 1,483 | 89.5 | 3.64 | 416 |
| ~7 Hz: control: ByteTrack, first 2 boxes of each track dropped | 85.0 | 88.3 | 81.8 | 84.7 | 993 | 85.6 | 2.86 | 308 |
| ~7 Hz: ByteTrack + Work ID (provisional IDs) | 82.2 | 93.6 | 72.1 | 78.5 | 2,061 | 87.5 | 4.68 | 542 |
| ~7 Hz: ByteTrack + Work ID (`_uidonly`) | 81.9 | 88.0 | 76.3 | 80.2 | 941 | 85.6 | 2.67 | 298 |
| ~7 Hz: ByteTrack + Work ID (`_offline_uidonly`) | 80.9 | 88.0 | 74.4 | 78.9 | 935 | 85.6 | 2.56 | 260 |

**About 7 Hz, detector boxes.**

| tracker | HOTA | DetA | AssA | IDF1 | IDSW | MOTA | IDs per person | merged IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coco: ByteTrack | 37.9 | 32.1 | 45.2 | 44.7 | 523 | 34.4 | 2.53 | 183 |
| coco: control: ByteTrack `_probfilt` | 36.8 | 29.7 | 45.8 | 43.6 | 336 | 32.3 | 2.11 | 135 |
| coco: ByteTrack + Work ID (`_uidonly`) | 36.1 | 29.6 | 44.2 | 41.7 | 261 | 32.5 | 1.80 | 135 |
| grouped: ByteTrack | 21.0 | 13.7 | 32.5 | 22.0 | 262 | 15.3 | 2.24 | 82 |
| grouped: control: ByteTrack `_probfilt` | 20.1 | 12.1 | 33.4 | 20.6 | 176 | 13.8 | 1.98 | 61 |
| grouped: ByteTrack + Work ID (`_uidonly`) | 19.3 | 12.1 | 31.0 | 19.0 | 134 | 13.8 | 1.69 | 75 |

**Static (02, 04, 09) vs moving camera (05, 10, 11, 13), native rate, COCO detections and oracle.**

| tracker | HOTA | DetA | AssA | IDF1 | IDSW | MOTA | IDs per person | merged IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static: ByteTrack | 39.2 | 32.4 | 47.8 | 45.8 | 198 | 35.8 | 2.62 | 53 |
| static: BoT-SORT + ReID | 40.6 | 35.7 | 46.6 | 48.2 | 327 | 39.2 | 3.20 | 76 |
| static: StrongSORT | 32.7 | 21.8 | 49.3 | 35.3 | 87 | 23.8 | 1.83 | 25 |
| static: Work ID (`_uidonly`) | 38.9 | 31.9 | 47.7 | 45.1 | 130 | 35.6 | 2.00 | 59 |
| static: oracle ByteTrack | 99.4 | 99.9 | 99.0 | 99.3 | 0 | 99.9 | 1.00 | 5 |
| static: oracle Work ID (`_uidonly`) | 93.9 | 99.4 | 88.7 | 91.6 | 0 | 99.4 | 1.00 | 35 |
| moving: ByteTrack | 40.6 | 36.7 | 45.4 | 49.5 | 343 | 40.5 | 2.23 | 114 |
| moving: BoT-SORT + ReID | 42.0 | 39.0 | 45.7 | 51.4 | 365 | 41.3 | 2.10 | 112 |
| moving: StrongSORT | 39.3 | 31.1 | 50.0 | 46.5 | 88 | 35.0 | 1.40 | 49 |
| moving: Work ID (`_uidonly`) | 39.4 | 35.5 | 44.2 | 48.0 | 182 | 39.7 | 1.67 | 110 |
| moving: oracle ByteTrack | 93.2 | 98.5 | 88.1 | 92.0 | 160 | 98.4 | 1.39 | 82 |
| moving: oracle Work ID (`_uidonly`) | 85.3 | 96.6 | 75.3 | 82.1 | 100 | 96.7 | 1.23 | 103 |

**Oracle at about 7 Hz, static vs moving camera.**

| tracker | HOTA | DetA | AssA | IDF1 | IDSW | MOTA | IDs per person | merged IDs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| static: ByteTrack | 99.0 | 99.4 | 98.6 | 99.2 | 7 | 99.4 | 1.06 | 9 |
| static: Work ID (`_uidonly`) | 93.8 | 97.7 | 90.2 | 92.7 | 5 | 97.7 | 1.04 | 30 |
| moving: ByteTrack | 66.0 | 85.0 | 51.3 | 62.4 | 1,476 | 74.5 | 4.87 | 407 |
| moving: Work ID (`_uidonly`) | 60.1 | 73.8 | 49.0 | 59.1 | 936 | 67.5 | 3.51 | 268 |

### D.6 What the results say

1. **On MOT17, detection limits everything, and our detector is the weaker one.**
   - **Recall.** At conf 0.35, `best_grouped.pt` finds 19.7% of MOT17 pedestrians, against 43.1% for
     the COCO weights it was fine-tuned from.
   - **Every tracker on our detections scores low:** HOTA 19–25.
   - **It depends on size.** An independent reviewer's breakdown of visible pedestrians by box height
     (`partD/review/anomalies/det_bins.py`):

     | detector | < 75 px | 75–100 px | 100–150 px | 150–200 px | 200–300 px | ≥ 300 px |
     |---|---|---|---|---|---|---|
     | `best_grouped.pt` | ≈ 0% | 5% | 14% | 20% | 51% | 80% |
     | COCO `yolov8s.pt` | 3–11% | 43% | 55% | 61% | 93% | 98% |

   - **Not resolution.** At imgsz 1280 its recall on visible pedestrians rose only from 30.2% to
     32.2%, on 107 sampled frames.
   - **Unseen identities.** It never detects 272 of the 546 pedestrian identities in any frame
     (COCO: 133).

   This is a domain gap: the PPE training set is close-up construction photographs, not street scenes
   with small, distant people. For comparing trackers, the COCO detections are the informative input.

2. **Against standard trackers, our pipeline's ByteTrack is competitive; the ranking is set by the
   input threshold, not by association.**
   - **On COCO detections:**
     - BoT-SORT with ReID scores best: HOTA 41.2, IDF1 49.4.
     - The pipeline's ByteTrack scores HOTA 39.8, IDF1 47.2.
   - **Same boxes, same result.** Given the same boxes (`__feedall`), ByteTrack scores HOTA 41.0 and
     IDF1 49.0. Its association matches BoT-SORT's: AssA 46.1 against 46.3.
   - **Why BoT-SORT's ReID barely matters.** It can only rescue a pair whose boxes already overlap at
     IoU ≥ 0.5 (its `proximity_thresh`), so it is not a long-range re-identification method.
   - **StrongSORT and Deep OC-SORT** have the fewest ID switches (175 and 188) and the highest AssA
     (49.7 and 49.2), but the lowest DetA (25.2 and 28.1) and HOTA (35.3 and 37.1).
     - **Why DetA is low:** their default thresholds (0.6 and 0.5) and 3-hit confirmation discard many
       boxes.
     - **Why AssA looks high:** it is partly a selection effect, because it is computed on fewer,
       more confident detections.

3. **The Work ID layer trades fewer ID switches for more identity merges, and on balance it lowers
   identity quality on MOT17.** Compared with the probation-matched ByteTrack control:
   - **In all six conditions** (3 inputs × 2 rates), HOTA, AssA and IDF1 are lower.
   - **It does what it was built for:** fewer ID switches and less fragmentation:
     - COCO: 312 against 416 switches, 1.78 against 2.11 IDs per person;
     - our detector: 171 against 250, 1.69 against 2.07;
     - oracle: 100 against 108, 1.16 against 1.18.
   - **But it puts different people under one ID** more often: merged IDs are 169 against 147 (COCO),
     95 against 68 (our detector) and 138 against 72 (oracle).
   - **Ground-truth boxes, net effect:** AssA falls from 94.5 to 83.9 and IDF1 from 96.3 to 88.2.
   - **The re-links are mostly wrong.** An instrumented replay by the reviewer
     (`partD/review/anomalies/oracle_relinks.py`):
     - on oracle boxes, only **10 of 141** re-links joined the same person and 131 joined different
       people (median appearance similarity 0.79 for the wrong ones, 0.895 for the right ones);
     - on COCO detections, 80 of 319 joined the same person, 139 a different one, and 100 could not be
       scored.
   - **End-of-session consolidation** adds a few more merges and lowers IDF1 slightly.
   - **Static cameras.** With ground-truth boxes ByteTrack alone makes **0** ID switches (1.00 IDs per
     person). The identity layer can only add errors there: AssA 88.7 against 99.0, and 35 merged IDs
     against 5.

   Three caveats:
   - **The oracle is harder than real use.** 23.6% of oracle boxes have visibility below 0.25 and 14.6%
     run off the frame, which no detector would produce; 68 of the 131 wrong re-links were decided on
     such a heavily occluded box.
   - **MOT17's own labels split some people.** For example MOT17-02 IDs 14 and 76–80, so "merged IDs"
     slightly overstates errors.
   - **Small people cannot be embedded.** The histogram embedder skips boxes under 48 px tall, so such
     people cannot be re-identified by appearance at all.

   On real street video, the identity layer's appearance matching (colour histograms, threshold 0.62,
   margin 0.04) is not discriminative enough to re-link people safely. This is consistent with the
   synthetic identical-PPE results in F.1.

4. **About 7 updates per second breaks tracking with a moving camera.**
   - **Combined, the effect looks small:** COCO ByteTrack HOTA goes from 39.8 to 37.9.
   - **Moving cameras collapse:** even with ground-truth boxes, ByteTrack's HOTA on the moving-camera
     sequences falls from 93.2 to **66.0**, with 1,476 ID switches. On static cameras it stays at 99.0.
   - **Cause:** the pipeline's ByteTrack has no camera-motion compensation. At 7.5 Hz a panning camera
     moves the boxes too far between updates for IoU association.
   - **No rescue from Work ID:** the identity layer does not recover it (HOTA 60.1).

5. **CPU time.** Measured one run at a time, with 1 PyTorch and 1 OpenCV thread, on the COCO
   detections of MOT17-02, -05 and -11 (2,337 frames), with `tools/revision/run_partD_timing.sh`.

   | tracker | boxes given to it | boxes per frame | ms per frame, 3 sequences | MOT17-02 (1920×1080, static) | MOT17-05 (640×480, moving) | MOT17-11 (1920×1080, moving) |
   |---|---|---:|---:|---:|---:|---:|
   | ByteTrack as in our pipeline | conf > 0.35 | 7.9 | **1.09** | 1.21 | 1.01 | 1.09 |
   | ByteTrack + Work ID: the ByteTrack call | conf > 0.35 | 7.9 | 1.27 | 1.41 | 1.10 | 1.33 |
   | ByteTrack + Work ID: the identity layer | | | 0.71 | 0.70 | 0.71 | 0.73 |
   | ByteTrack + Work ID: total | | | **1.98** | 2.11 | 1.80 | 2.05 |
   | BoT-SORT + ReID (ultralytics 8.4.75) | all cached (≥ 0.10) | 14.9 | **22.8** (camera-motion compensation 20.8) | 24.7 | 16.9 | 27.1 |
   | StrongSORT (boxmot 26.0.0) | all cached (≥ 0.10) | 14.9 | **151.1** | 253.2 | 89.7 | 140.1 |
   | StrongSORT, pre-filtered | conf > 0.6 | 6.0 | **73.5** | 78.7 | 54.7 | 87.6 |
   | Deep OC-SORT (boxmot 26.0.0) | all cached (≥ 0.10) | 14.9 | **153.7** | 256.6 | 94.2 | 140.5 |
   | Deep OC-SORT, pre-filtered | conf > 0.5 | 6.7 | **79.7** | 89.1 | 60.3 | 91.5 |

   - **What is timed.** Only the tracker call (and, for Work ID, the identity layer, timed
     separately). Detection is cached and image decoding is excluded, but anything a tracker does
     with the image is included: BoT-SORT's sparse optical flow, the OSNet crops of StrongSORT and
     Deep OC-SORT, and the colour histograms of the identity layer.
   - **Our pipeline costs about 2 ms per frame** for tracking and identity together. The ByteTrack
     call is the same code in both runs, so its 1.09 against 1.27 ms is run-to-run variation (or
     cache effects of the identity layer running in between), not extra work.
   - **BoT-SORT** spends 91% of its time on camera-motion compensation over the full frame. Its ReID
     features come with the detector's forward pass (here read from the cache), so they add nothing.
   - **StrongSORT and Deep OC-SORT** spend almost all their time embedding boxes with OSNet on one CPU
     thread: 37–78 times the 2 ms of our pipeline's tracking and identity together. Given only the
     boxes they keep anyway (pre-filtered above their own thresholds), they take about half as long.
   - **Same tracks.** The pre-filtered runs give byte-identical tracks to the unfiltered ones. Every
     timing run's tracks are byte-identical to the main runs' on these three sequences (15 of 15
     files), so the timing conditions changed nothing but the clock.
   - **Hardware.** AMD Ryzen 5 7640HS (6 cores, 12 threads), 15.3 GB RAM, Windows 11. The GPU training
     job (Part A) ran at the same time, so absolute times carry that load; the ratios are the
     informative part.
   - **When.** All seven runs ran one after another, 21:20–21:49 on 19 November. The first attempt at
     the four boxmot runs crashed after its first sequence: the boxmot wrapper in `mot17_track.py`
     changes directory so that boxmot finds its local weights, which broke a relative output path. The path was made absolute and those four runs were repeated in
     full.

6. **Other things noticed.**
   - **Box geometry.** Our detector's person boxes sit a median 8 px above MOT17's box tops. This is a
     labelling-convention difference, and it lowers localisation (and so DetA) independently of
     identity.
   - **Detector-only rows.** In `combined_metrics.csv` and `per_sequence_metrics.csv` the
     `detections_conf035` rows only give detector recall and precision; their other columns are blank
     on purpose.
   - **A fragility in the deployed code.** `IdentityManager._gap` compares elapsed seconds with a strict
     `>` against exactly 3.0 s. Gaps of exactly the horizon (42 frames at 14 fps, 75 at 25, 90 at 30)
     then flip with floating-point rounding. In clip analysis this makes per-sequence Work ID results
     depend on the time origin: up to 2.3 IDF1 on one sequence, at most 0.25 combined. With live
     wall-clock time it hardly matters. The code was not changed.

### D.7 Deviations, versions, commands, run times

- **Deviations.**
  - **Dataset.** CMOT was replaced by MOT17 (see 0.1).
  - **Detection hardware.** Detection ran on the CPU, because the GPU was training.
  - **BoT-SORT environment.** BoT-SORT ran in `.venv-eval` with ultralytics 8.4.75 and `lap`. The
    ultralytics trackers need `lap`, which is missing from `.venv` and `.venv-gpu`, and those must not
    be changed.
  - **Run batches.** The external trackers' outputs are from the first batch (19:37–20:22). The phase2
    and oracle runs are from the second batch (21:05–21:13), after the clock-origin fix, which does not
    affect the other trackers.
  - **CPU time.** CPU time in the run records is informational only. The comparable figures are the
    separate timing runs.
  - **Provisional IDs.** The provisional-ID handling of probation boxes, and the `_uidonly`,
    `_offline` and `_probfilt` variants, are reporting choices; all are reported.
- **Versions.**
  - `.venv`: Python 3.13.3, supervision 0.29.1, numpy 2.5.1, OpenCV 5.0.0.
  - `.venv-eval`: Python 3.13.3, torch 2.15.0+cpu, ultralytics 8.4.75, boxmot 26.0.0, TrackEval git
    `12c8791`, numpy 2.5.4, scipy 1.18.1, OpenCV 4.11.0.86, lap 0.5.14. Its full `pip freeze` is in
    `environments/venv_eval_pip_freeze.txt`.
  - Detection used `.venv-gpu` on the CPU: torch 2.13.0+cu130, ultralytics 8.4.75.
- **Commands.**
  1. Download: `powershell -File tools\revision\download_mot17.ps1`.
  2. Prepare: `.venv\Scripts\python.exe tools\revision\mot17_prepare.py`.
  3. Detect: `.venv-gpu\Scripts\python.exe tools\revision\mot17_detect.py --threads 4`, with
     `CUDA_VISIBLE_DEVICES=-1`.
  4. Track and score: `bash tools/revision/run_partD.sh`, then
     `STREAMS=phase2 bash tools/revision/run_partD.sh`.
  5. Timing: `bash tools/revision/run_partD_timing.sh`.
- **Run times.**
  - download: 37 min;
  - detection: 18 + 9 min;
  - phase2 tracker runs: 5–74 s each;
  - BoT-SORT: about 3 min per detector;
  - StrongSORT and Deep OC-SORT: 7–13 min per detector;
  - TrackEval scoring with all variants and camera subsets: 487 s.
  - timing runs (sequential): 4 s for ByteTrack to 6.3 min for Deep OC-SORT, 29 min in all.
- **Results.**
  - `partD/combined_metrics.csv`: every run, combined and by camera type.
  - `partD/per_sequence_metrics.csv`.
  - `partD/trackeval/`: TrackEval's own summaries and detailed CSVs.
  - `partD/runs/`: one record per run, with command, versions, inputs and timing.
  - `partD/timing/`.
  - `partD/review/`.
  - The tracker MOT files and the detection caches stay in `data/revision_r1_cache/partD/`.
