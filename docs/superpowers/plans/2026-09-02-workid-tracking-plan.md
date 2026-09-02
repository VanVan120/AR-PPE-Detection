# Work ID tracking enhancement — implementation plan

Spec: [2026-09-02-workid-tracking-design.md](../specs/2026-09-02-workid-tracking-design.md)

Baseline captured before any change (scratchpad `baseline/`): re-ID table
100/75/8 recall, 0/0/9.6 false merge; head-motion table; all suites green.

Each step: test first, implement, run the suite, then re-run the baseline benchmark and
diff it against the captured output.

1. **`WorkerHistory.merge`** — episodes, frames, open episodes, badge flag.
2. **Identity: probation** — `probation_frames`, pending buffer, mean descriptor,
   marker commits immediately, GC of abandoned pending tracks.
3. **Identity: gating** — last box + global shift per worker, plausibility, spatial veto,
   position-assisted rule, `source="position"`.
4. **Identity: merges** — badge naming an anonymous record merges it; `take_merges()`.
5. **Identity: consolidate** — presence intervals, non-overlap constraint, greedy merge.
6. **Identity: ghosts** — `coast_frames`, `ghosts()`.
7. **Harness** — delayed scoring, `--ablation`, `--phantoms`, `--relocate`, pipeline
   comparison; pin the baseline rows in tests.
8. **Wiring** — config keys + validation, `SafetyPipeline`, `run.py`, overlay ghost
   drawing, phone `people` ghost rows + `app.js` dashed boxes.
9. **Badge range** — `WorkIdBinder.crop_detect`, `badge_eval.py`, tests.
10. **Real-footage harness** — `badge_gt_eval.py` scorer + CLI, tests.
11. **Measure** — ablation, phantoms, relocate, pipeline, badge tables.
12. **Docs** — README, phase5 README, phase2 README config table, USER_GUIDE, config.yaml.
13. **Verify** — every suite, baseline diff, `run.py --check` still loads config.
