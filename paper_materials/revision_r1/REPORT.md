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
