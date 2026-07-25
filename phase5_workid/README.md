# Phase 5 — Worker identity tracking (Work ID)

**The ask:** *"the detection is really good but only on PPE. If you can focus on the
tracking on the workers' Work ID, it will be fantastic."*

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
python -m phase5_workid.reid_eval

# tests
python phase2/tests/test_identity.py        # ALL_IDENTITY True
python phase2/tests/test_workerlog.py       # ALL_WORKERLOG True
python phase5_workid/tests/test_reid_eval.py # ALL_REID_EVAL True

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
