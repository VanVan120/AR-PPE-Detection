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

The only existing script that changed is `tools/train_grouped.py`. It gained `--seed N` and
`--no-plots`. Their defaults reproduce the old behaviour byte for byte:

- `model.train` receives identical arguments, in identical key order. This was checked against
  `978778e` with a stub `YOLO`, for both stages.
- `run_info.json` gains no fields unless the options are given.

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
