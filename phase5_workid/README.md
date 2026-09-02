# Phase 5 — Worker identity tracking (Work ID)

**The ask:** *"the detection is really good but only on PPE. If you can focus on the
tracking on the workers' Work ID, it will be fantastic."* — and, one round later:
*"enhance the tracking effects of the Work ID of workers; no need to develop AI glasses
anymore."* The second round is the **tracking enhancement** section further down; its
numbers supersede the first-round table, which is kept as the baseline every mechanism is
measured against.

Phase 2 already tracked people, but only as `tracker_id` — pure motion continuity.
ByteTrack retires that id the moment a worker is occluded past `lost_track_buffer` or
leaves frame, and the same person returns as a different number. Everything keyed on it
resets: the name, the debounced compliance state, the violation history. On screen that
reads as "Person 3" becoming "Person 11" after walking behind a van, and it is why a
per-worker safety record was impossible.

Phase 5 adds a **persistent worker identity above the tracker**, so the name and the
history follow the *person*.

---

## What was actually missing (and why Jian saw only PPE)

Work ID was already fully implemented — the ArUco binder, a printable tag generator, name
pills in the overlay, README section, all wired into `run.py`. Two things kept it
invisible:

1. `phase2/config.yaml` shipped with `workid.enabled: false`.
2. Even switched on, it needed **printed ArUco tags**, which a person evaluating from
   existing footage does not have.

So the fix could not just be "flip the flag" — that repeats the same outcome. Phase 5
makes worker identity work **with zero props**, while keeping the badge authoritative
when one is present.

---

## How identity is decided

Two signals, strict precedence (`phase2/src/identity.py`):

1. **ArUco badge — authoritative.** A visible tag *is* the worker, with a real name.
   Appearance never overrides it.
2. **Appearance — bridging.** No tag visible: match a compact appearance descriptor
   (`phase2/src/reid.py`) against the gallery of known workers to recover who they were.

Three invariants make the fusion safe:

- **Stable uid.** A worker's uid is assigned once and never changes. Names change — an
  anonymous `Worker 2` becomes `Alice Tan` when a badge is finally read — and the
  violation history stays attached across that rename instead of being orphaned.
- **Mutual exclusion.** One worker cannot be two people in one frame, so a lookalike can
  never duplicate an identity.
- **Refuse to guess.** A match must clear an absolute similarity threshold *and* beat the
  runner-up by a margin. Otherwise the person becomes a new worker. Inventing a spurious
  worker is a cheap, visible error; silently merging two people is an expensive, invisible
  one that moves a violation onto the wrong person's record.

Two descriptors ship: a banded HSV **colour histogram** (default — no model, no download,
~0.1 ms/crop) and an opt-in torchvision **ResNet-18** embedder (`identity.method: deep`,
downloads ImageNet weights on first use).

---

## Measured: does it actually re-identify?

`reid_eval.py` runs an **injected-occlusion protocol** — the same self-supervised shape as
the Phase 3 injected-fault evaluation. Every worker is removed for a gap of frames and
returns under a **new track id**, exactly as ByteTrack behaves after a long occlusion.
Ground truth is known by construction.

```bash
python -m phase5_workid.reid_eval            # all three scenarios
python -m phase5_workid.reid_eval --sweep    # the threshold trade-off
```

**4 workers, 12-frame gaps, 3 seeds, threshold 0.62 (the shipped default):**

| scenario | re-ID recall | false merge | fragmentation | identities / true |
|---|---|---|---|---|
| **distinct** clothing | **100%** | 0.0% | 1.00 | 4.0 / 4 |
| **similar** — same issued vest, personal helmet | **75%** | **0.0%** | 1.25 | 5.0 / 4 |
| **uniform** — identical PPE | **8%** | 9.6% | 1.92 | 7.0 / 4 |

- **re-ID recall** — forced re-entries reunited with the right worker. Higher is better.
- **false merge** — assignments given *another* worker's identity. This is the dangerous
  error, so it is reported separately and never averaged into a headline.
- **fragmentation** — distinct identities per true worker; 1.00 is perfect.

### The finding that matters

**Appearance re-ID collapses when everyone wears the same PPE** — 8% recall, 9.6% false
merges, nearly 2 identities per worker. That is not a bug to be tuned away; it is the
physical limit of appearance matching, and it is *the* argument for the ArUco badge on a
real site where uniform kit is the norm. It also vindicates the hybrid design: appearance
carries the demo and any footage where people are distinguishable, and the badge carries
the site.

### Why the default threshold is 0.62

From `--sweep` on the realistic `similar` scenario:

| threshold | re-ID recall | false merge |
|---|---|---|
| 0.40 | 83% | 5.0% |
| 0.50–0.55 | 75% | 5.0% |
| **0.60–0.70** | **75%** | **0.0%** |
| 0.80–0.90 | 67% | 0.0% |

0.62 is the highest recall that still produces **zero** false merges. Dropping to 0.40
buys 8 points of recall at the cost of a 5% chance of attributing a violation to the wrong
person — a bad trade for a safety record.

### Honest limits

- **The sequences are synthetic.** The repo has no annotated multi-person video, and
  grading re-ID against a tracker's own output would be circular. Generated sequences give
  exact ground truth and run anywhere, but they are rendered figures, not a site. **These
  numbers measure the matcher, not site performance.**
- The `distinct` scenario is easy — 100% at every threshold tested — so it validates
  plumbing, not discrimination. `similar` is the informative one.
- No claim is made about MOTA/IDF1 or any person-re-ID benchmark. The claim is exactly
  the metric reported: recovery of a worker's identity across an induced track break.
- The default descriptor is colour-based, so it is sensitive to strong lighting change.
  `identity.method: deep` is available and untested at scale here.

---

## Making the identity stick — the tracking enhancement

The first-round layer decided a track's identity on the frame it appeared, from
appearance alone, and never revisited it. Reading the code against the phone-app finding
(*40 frames of a 7-person clip produced 15 named workers*) gave five concrete causes, and
each became a mechanism with an ablation row. Design record:
[`docs/superpowers/specs/2026-09-02-workid-tracking-design.md`](../docs/superpowers/specs/2026-09-02-workid-tracking-design.md).

| cause | mechanism | config key |
|---|---|---|
| a worker was created on a track's **first frame**, so a one-frame false detection or a half-visible person became a permanent "Worker N" | **probation** — a track must be seen `probation_frames` times before it can become or match a worker; its descriptors are pooled meanwhile | `identity.probation_frames` |
| matching was **appearance only**; identical PPE was unresolvable even when only one worker could physically be there | **spatio-temporal gating** — a recently-seen worker who could not have reached the spot is vetoed even if the colours agree; a lone worker who could be there is accepted on weaker appearance. Camera pan is compensated from the other tracks' motion | `identity.gate*` |
| a track bound to a duplicate record was **never merged back**, and a badge naming a track that held an anonymous record stranded that record's history | **merges that carry history** — live: the badge folds the anonymous record into the named one; offline: an end-of-session pass merges anonymous fragments whose presence intervals never overlap | `identity.consolidate` |
| the name pill **vanished with the track** for the few frames of an occlusion | **coasting** — a lost worker keeps a dashed, dimmed, predicted box with a `~` before the name, then expires | `identity.coast_seconds` |
| badges were searched on the **full frame** only, so a distant helmet's badge was below the detector's resolution | **crop detection** — for every person not yet bound, the head-and-torso region is upscaled and searched again | `workid.crop_detect` |

Two rules keep the gate honest. Two plausible lookalikes are still refused (the margin rule
decides among them, and the position rule needs exactly one candidate). And a spatial veto
yields to *decisive* appearance — best similarity ≥ 0.85 and 0.20 ahead of every other
known worker — so a distinctly dressed worker who genuinely reappears far away is kept,
while identical kit can never be decisive.

### Measured: the same protocol, mechanism by mechanism

`python -m phase5_workid.reid_eval --ablation` — 4 workers, 12-frame gaps, 3 seeds,
threshold 0.62. `baseline` is the first-round layer and reproduces the table above exactly;
`enhanced` is what `config.yaml` now ships; `+offline` adds the end-of-session pass.

| scenario | config | re-ID recall | false merge | frag | identities / true |
|---|---|---|---|---|---|
| distinct | every configuration | **100%** | 0.0% | 1.00 | 4.0 / 4 |
| similar | baseline | 75% | 0.0% | 1.25 | 5.0 / 4 |
| similar | +probation | **100%** | 0.0% | 1.00 | 4.0 / 4 |
| similar | +gate | 75% | 0.0% | 1.25 | 5.0 / 4 |
| similar | **enhanced** (and +offline) | **100%** | **0.0%** | **1.00** | **4.0 / 4** |
| uniform | baseline | 8% | 9.6% | 1.92 | 7.0 / 4 |
| uniform | +probation | 8% | 9.7% | 1.92 | 7.0 / 4 |
| uniform | +gate | 67% | 13.8% | 1.33 | 4.3 / 4 |
| uniform | **enhanced** (and +offline) | **100%** | **0.0%** | **1.00** | **4.0 / 4** |

The ablation is the finding. Probation alone fixes the *similar* case (a descriptor pooled
over three looks is enough to separate different helmets) and does nothing for identical
kit. The gate alone lifts identical kit to 67% but *adds* false merges, because a single
first-frame descriptor is too noisy to choose between two plausible neighbours. Together,
the pooled descriptor decides only among workers who could physically be there, and the
identical-PPE row goes from 8% recall with 9.6% false merges to **100% with none** — on
this protocol, where a worker returns close to where they vanished.

**Ten seeds** (`--seeds 10`, 40 re-entries per row) say the same: similar 80% → 100%,
uniform 2% recall / 11.3% false merge → 100% / 0.0%.

Read the identical-PPE row for what it is. On this protocol the four workers stand about
1.1 body heights apart and each returns close to where they vanished, so at the 12-frame
gap the gate radius (0.86 body heights) contains exactly one vacated spot — position can
decide what appearance cannot. The gate's growth rate was chosen by a sweep on this
protocol (the design record says so). The rows below are the conditions that break that
assumption, measured on purpose.

### Where it still breaks — measured on purpose

| condition | baseline | enhanced | note |
|---|---|---|---|
| **6 workers shoulder to shoulder, identical PPE** (`--workers 6`) | 0% / 11.7% | 17% / 11.1% (+offline: 33% / 19.4%) | people closer than the gate radius in identical kit: position cannot separate them, and the offline pass then over-merges. **The badge case.** |
| **identical workers swap places** (`--swap`: each returns in the spot another absent worker just left) | distinct 100% · similar 75% · uniform 0% / 9.6% | distinct 75% · similar 67% / 4.2% · **uniform 0% / 52.7%** | position is *actively misleading*: the spot's previous occupant is the only plausible candidate and everyone looks alike, so the newcomer is confidently given the wrong identity. **The one measured condition where the enhanced layer is worse than the baseline**, and the strongest argument for the badge wherever workers in identical kit rotate positions. |
| **re-entering somewhere else after 1.5 s** (`--relocate --gap 45 --frames 120`) | distinct 100% · similar 75% · uniform 0% / 12.0% | distinct 83% · similar 83% · uniform 17% / 17.1% (+offline 28.9%) | position information has decayed; only decisive appearance rescues a distant return, and identical kit has none. The offline pass makes the uniform row worse, not better |
| **6 spurious 1–2 frame tracks** (`--phantoms 6`) | 7.0 / 7.7 / 9.3 identities for 4 workers | **4.0 / 4** in every scenario | probation, as intended |
| **head motion** (`--pipeline`, same-vest kit) | 1.00 ids/worker up to 12 px/frame; 1.75 and 29.8% false merge at 20 | identical up to 12 px; 27.0% at 20; labels 95.5–97.4% of person-frames (probation leaves each track's first frames unlabelled, the baseline labels 100%) | no regression; the pan compensation does not rescue the gallery once blur has destroyed it |

The six-worker and swap rows are the ones to carry into the site: identical PPE **and** a
crowd, or identical PPE **and** people changing places, is exactly where appearance and
position both run out — and where the gate can be confidently wrong rather than
uncertain. That is the argument for the badge the first round already made, now with the
failure mode named.

### Badge read range

`python -m phase5_workid.badge_eval` — a real ArUco marker (`cv2.aruco`) on the synthetic
figure's helmet, sized as a 10 cm printed badge, anti-aliased and placed at a random
sub-pixel offset and scale per trial (a marker pasted at integer sizes aligns with the
pixel grid at some sizes and not others, and the read rate then oscillates with height —
an artifact the first version of this table had), with noise and blur, 20 trials per row:

| person height (px) | badge (px) | full frame | + crop pass |
|---|---|---|---|
| 420 | 24.7 | 100% | 100% |
| 320 | 18.8 | 75% | **100%** |
| 260 | 15.3 | 70% | **90%** |
| 220 | 12.9 | 45% | **65%** |
| 180 | 10.6 | 5% | **25%** |
| 150 | 8.8 | 10% | 10% |
| ≤ 120 | ≤ 7.1 | 0% | 0% |

The crop pass adds 15–25 points of read rate across the 11–19 px range and nothing below
9 px, where the marker's modules are under two pixels. On a phone frame 640 px wide a
10 cm badge is roughly 8–9 px at 5 m, so a reliable read moves out by about half a metre
(from ~19 px to ~15 px) and occasional reads reach a metre further. A 15 cm badge
(`--badge-cm 15`) reads at 95% where the 10 cm one reads at 25%: a larger printed badge is
still the cheapest improvement of all.

### Real footage, without labelling anyone

Every number above is synthetic, and the READMEs say so. `badge_gt_eval.py` is the path to
a number from a real clip: workers wear the printed badges, **the badge reads are the
ground truth**, and the appearance-only identity layer — which never sees them — is scored
on whether it keeps the same worker uid on the same badge name across track breaks. One
detector pass, the same three metrics, baseline and enhanced side by side:

```bash
python -m phase5_workid.badge_gt_eval site_visit.mp4 --out outputs/badge_gt.json
```

It needs the detector weights and a clip of three to five people wearing the tags from
`phase2/tools/make_worker_tags.py`. Only badge-labelled frames are scored, so a small or
often-hidden badge gives a sparse score, not a bad one. The scorer is unit-tested
(`phase5_workid/tests/test_badge_gt_eval.py`); the clip is the missing piece.

### Cost

Probation and gating are bookkeeping on top of the existing 0.6 ms/frame; the crop pass
costs one small detection per *unbound* person per frame — measured at +2.5 ms/frame for
six unbound persons at 480p, and nothing for a person once a badge has named them, so a
site where nobody wears a badge pays it every frame; the offline pass runs once, at
session end.

---

## Per-worker safety report

Identity makes the report possible (`phase2/src/workerlog.py`). Violations are stored as
**episodes** (start → end), not counters, so durations are real:

```
====================================================================
PER-WORKER SAFETY REPORT
====================================================================
worker                 seen  events   unsafe  compliant
--------------------------------------------------------------------
Alice Tan                47s      2      12s        74%
    - No-Helmet: 1
    - No-Vest: 1
Worker 3 *               31s      1       4s        91%
    - No-Vest: 1
--------------------------------------------------------------------
2 worker(s), 1 confirmed by badge, 3 violation episode(s), 16s unsafe in total
* = identity from appearance only (no badge seen) — treat the name as provisional
```

The `*` is deliberate: an inspector must never mistake a provisional appearance match for
a confirmed identification. The same distinction is enforced in the JSONL event log, whose
`identified` flag is set **only** by a real badge — an appearance guess is never recorded
as a confirmed ID.

Set `identity.report: outputs/workers.json` in `phase2/config.yaml` for the machine-readable
version.

---

## Running it

```bash
# see the measurement (no camera, no weights, no downloads)
python -m phase5_workid.reid_eval               # baseline vs enhanced
python -m phase5_workid.reid_eval --ablation    # each mechanism on its own
python -m phase5_workid.reid_eval --phantoms 6  # with false detections
python -m phase5_workid.reid_eval --workers 6   # a crowd in identical kit
python -m phase5_workid.reid_eval --swap        # workers changing places (the gate's limit)
python -m phase5_workid.badge_eval              # badge read range

# tests
python phase2/tests/test_identity.py        # ALL_IDENTITY True
python phase2/tests/test_workerlog.py       # ALL_WORKERLOG True
python phase5_workid/tests/test_reid_eval.py # ALL_REID_EVAL True
python phase5_workid/tests/test_badge_eval.py    # ALL_BADGE_EVAL True
python phase5_workid/tests/test_badge_gt_eval.py # ALL_BADGE_GT True

# live / on a video — identity is ON by default
cd phase2 && python run.py --source ../clip.mp4
```

For real names, print tags and enable badges:

```bash
cd phase2
python tools/make_worker_tags.py     # printable ArUco tags from config.yaml
# then set workid.enabled: true and map marker ids -> names
```

Print the tags at a constant physical size (~8–10 cm) and mount them on the helmet or
vest. Any worker without a tag still gets a persistent `Worker N` identity.

## Configuration (`phase2/config.yaml`)

| key | meaning |
|---|---|
| `identity.enabled` | persistent worker identity on/off (default **on**) |
| `identity.appearance` | `false` → badge-only (names only where a tag is read) |
| `identity.method` | `histogram` (no download) or `deep` (torchvision ResNet-18) |
| `identity.match_threshold` | min cosine similarity to accept a re-identification (0.62) |
| `identity.margin` | best must beat the runner-up by this, else stay unsure |
| `identity.forget_after` | frames an **unbadged** worker is remembered (badged ones are kept) |
| `identity.report` | optional path for the per-worker JSON safety report |
| `identity.probation_frames` | sightings before a track can become or match a worker (3; keep it below `debounce_frames`) |
| `identity.gate` | spatio-temporal gating on re-identification (on) |
| `identity.gate_slack` / `.gate_speed` / `.gate_horizon` | position tolerance in body heights (0.5), its growth per second gone (0.9, about walking pace), and the seconds after which position says nothing (3) |
| `identity.gate_floor` | minimum similarity for a position-assisted match (0.45) |
| `identity.coast_seconds` | how long a lost worker keeps a dashed, predicted label (0.5) |
| `identity.consolidate` | end-of-session merge of anonymous fragments that never overlapped in time (on) |
| `workid.crop_detect` / `.crop_min_height` | search upscaled person crops for badges (on), and the crop height they are upscaled to (320) |
