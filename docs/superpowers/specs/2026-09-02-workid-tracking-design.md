# Work ID tracking enhancement — design

**Date:** 2026-09-02
**Trigger:** supervisor feedback — *"enhance the tracking effects of the Work ID of workers;
no need to develop AI glasses anymore"*. Last engineering effort before the paper
(Frontiers in Virtual Reality / Augmented Reality).

## Goal

Make a worker's identity stick to the right person on screen and in the safety report:
fewer phantom workers, names that do not flip, a label that stays on the person through a
short occlusion. Every mechanism must be measurable as an ablation row against the numbers
already published in the repo, and **those baseline numbers must reproduce exactly**.

Out of scope: any further AR-glasses work (Phase 6 is frozen as a finished design study),
replacing the tracker, a trained person re-ID network.

## Baseline that must not change

`python -m phase5_workid.reid_eval` (histogram, 3 seeds, 4 workers, gap 12, thr 0.62):

| scenario | re-ID recall | false merge | frag | workers |
|---|---|---|---|---|
| distinct | 100% | 0.0% | 1.00 | 4.0/4 |
| similar | 75% | 0.0% | 1.25 | 5.0/4 |
| uniform | 8% | 9.6% | 1.92 | 7.0/4 |

`--pipeline` head-motion table: ByteTrack 1.00/1.25/1.50/1.75/1.75 ids per worker at
0/4/8/12/20 px, identity 1.00/1.00/1.00/1.00/1.75, false merge 0/0/0/0/29.8%.

All existing test suites stay green.

## Where identity breaks today (from the code)

1. `IdentityManager.update` pass 3 commits a match or a brand-new worker on a track's
   **first** frame. A one-frame false detection or a half-visible person becomes a
   permanent "Worker N".
2. The gallery match is **appearance only**: where the worker was last seen and how long
   ago is ignored, so identical PPE is unresolvable even when only one worker could
   physically be there.
3. **No late correction**: a track bound to a duplicate worker is never merged back, even
   in the offline clip path where the whole video is known. A badge that names a track
   already holding an anonymous record strands that record's history.
4. The name pill **vanishes with the track** during a short loss.
5. ArUco markers are detected on the full frame, so a badge on a distant helmet is below
   the detector's resolution.

## Design

### 1. Probation before a worker exists (`probation_frames`)

A track not bound to a worker collects embeddings for `probation_frames` consecutive
sightings before anything is decided. During probation it is absent from the identity
output, so the overlay shows the anonymous track id, and neither the roster nor the
history sees it. On commit, the L2-normalised **mean** of the collected embeddings is
matched, and every collected embedding enters the gallery. A marker read on a track in
probation commits immediately (markers stay authoritative).

Default in the manager: `1` (today's behaviour, keeps benchmarks). Config default: `3`,
below `debounce_frames` (5) so no violation can fire before a worker exists.

### 2. Spatio-temporal gating with ego-motion compensation (`gate`)

Each worker remembers its last box and the accumulated **global image shift** at that
time. The shift is the per-frame median displacement of tracks that continued from the
previous frame — a cheap camera-motion estimate that costs nothing extra. A lost worker's
predicted position is `last_centre + (shift_now − shift_then)`.

A candidate is *plausible* for a new track if the new centre is within
`h · (gate_slack + gate_speed · dt)` of the prediction (`h` = mean box height,
`dt` = frames since last seen) and the height ratio is within 0.5–2.0. Only workers seen
within `gate_horizon` frames carry position information; older ones are treated as
"anywhere".

Decision rule for an unbound track with embedding `e`:

1. **Spatial veto** — recently-seen workers that are implausible are removed from the pool.
2. **Appearance rule (unchanged)** — best in the pool must clear `match_threshold` and beat
   the runner-up by `margin`.
3. **Position-assisted rule (new)** — otherwise, if exactly **one** recently-seen worker is
   plausible and its similarity is at least `gate_floor`, accept it (`source="position"`).
4. Else a new worker.

Two plausible workers in identical kit are still refused: the margin rule decides among
them and the position rule needs exactly one. Refusing to guess is preserved.

A spatial veto yields to **decisive** appearance (best similarity ≥ 0.85 and ≥ 0.20
ahead of every other known worker): a distinctly dressed worker who genuinely reappears
far away is kept, while identical kit can never be decisive and stays vetoed.

Defaults: `gate_slack 0.5`, `gate_speed 0.03` heights/frame (about walking pace; the
config expresses it as 0.9 heights/second and the pipeline divides by the frame rate),
`gate_horizon 90` frames (config: 3 s), `gate_floor 0.45`. The rate was chosen by a
sweep: faster growth lets neighbours a body-height apart become plausible within half a
second, after which appearance noise decides between identical people. Manager default:
gate **off**; config default: on.

### 3. Merges that carry history

`WorkerHistory.merge(src_uid, dst_uid)` folds one record into another: episodes,
frames, first/last seen, `marker_confirmed` OR-ed, open episodes re-keyed. The identity
manager reports merges through `take_merges()` and both entry points apply them.

Online: when a badge names a track that currently holds an **anonymous** record and the
name already exists, the anonymous record is merged into the named one instead of being
stranded. A badge-confirmed record is never merged away.

Offline (`consolidate()`, called once at session close, and by the clip analyser): greedy
agglomeration of anonymous records into any record whose presence intervals **never
overlap** with theirs (two records visible at the same moment are provably different
people), applying the same decision rule as online — centroid similarity with threshold
and margin, or a single spatially plausible neighbour at the fragment boundary. Because
its mistakes are permanent in the report, it demands a wider runner-up margin (0.10)
than the live matcher. This is the second pass the recorded-clip workflow was missing.

### 4. Coasting labels (`coast_frames`)

`ghosts()` returns, for workers not visible this frame but seen within `coast_frames`,
the predicted (shift-compensated) box, the label and the age. The composite overlay draws
a dashed, dimmed reticle with the name; the phone app receives them in `people` rows
flagged `ghost: true` and draws them dashed. Predicted boxes are always visibly different
from detected ones. Manager default `0` (off); config `coast_seconds 0.7`.

### 5. Badge read range (`workid.crop_detect`)

For every track without a binding, the binder also crops the upper part of the person
box, upscales it to at least `crop_min_height` px, runs ArUco detection there, and maps the
corners back. Results are merged with the full-frame pass (dedupe by marker id). Measured
by `phase5_workid/badge_eval.py`, which renders real ArUco markers onto the helmet band of
the synthetic figures at decreasing sizes and reports the read rate with and without the
crop pass.

### 6. Evaluation

`reid_eval.py`:
- scores a re-entry at the **first frame its new track receives an identity** (identical to
  today for probation 1, so the baseline reproduces);
- the default output shows baseline / enhanced / enhanced+offline; `--ablation` adds
  +probation and +gate on their own, per scenario;
- `--phantoms N` injects short spurious tracks; `--relocate` makes returning workers
  re-enter elsewhere; `--swap` is the handover case (a returning worker re-enters in
  another absent worker's spot), the adversarial case for position gating;
- `--pipeline` reports baseline and enhanced identity columns side by side, with the
  enhanced layer's coverage (share of person-frames labelled).

`phase5_workid/badge_gt_eval.py`: **badge-as-ground-truth** protocol for real footage. One
detector pass; the binder's badge reads label tracks; a second, appearance-only identity
manager is scored against those labels with the same three metrics. No manual labelling.
The scorer is unit-tested; the CLI needs weights and a clip.

### 7. Documentation

README roadmap: Phase 6 frozen, phone app is the delivery target, new Phase 9 section with
the ablation table. `phase5_workid/README.md`: mechanisms, tables, honest limits, the
real-footage protocol. `USER_GUIDE.md`: updated weaknesses. `config.yaml`: new keys.

## Interfaces

- `IdentityManager(embedder, ..., probation_frames=1, gate=False, gate_slack=0.5,
  gate_speed=0.03, gate_horizon=90, gate_floor=0.45, coast_frames=0,
  consolidate_threshold=None, consolidate_margin=0.10, gate_speed_s=None,
  gate_horizon_s=None, coast_s=None)` — the `*_s` twins are per second and take over
  whenever `update()` is given `now_s`, which every real entry point supplies (a live
  loop rarely processes every camera frame, so frame-based knobs would be wrong in real
  time).
- `IdentityManager.update(frame, track_ids, boxes, marker_labels, now_s=None)` — tracks
  in probation are absent from the result. Co-visibility (a worker seen on or after the
  frame the new track appeared) excludes that worker outright; it is a proof, not a veto,
  and nothing overrides it. The position-assisted rule additionally requires the lone
  plausible worker to be the appearance argmax over everyone known, so a stranger who
  steps into a vacated spot is not matched.
- `IdentityManager.take_merges() -> list[(src, dst)]`, `consolidate() -> list[(src, dst)]`,
  `ghosts() -> list[dict]`.
- `IdentityResult.source` gains `"position"`.
- `WorkerHistory.merge(src_uid, dst_uid)`.
- `WorkIdBinder(..., crop_detect=True, crop_min_height=320)`.
- `SafetyPipeline` wires all of the above; `run.py` mirrors it.

## Testing

Unit tests per mechanism in `phase2/tests/test_identity.py`, `test_workerlog.py`, the
ghost-overlay check in `phase2/tests/test_arview.py`, the ghost-row check in
`phase8_phoneapp/tests/test_phoneapp.py`, `phase5_workid/tests/test_reid_eval.py`
(delayed scoring, options, baseline pinned), and the new
`phase5_workid/tests/test_badge_eval.py` (which also covers the binder's crop path) and
`test_badge_gt_eval.py`. Benchmark non-regression is checked by diffing the baseline rows
against the captured pre-change output.
