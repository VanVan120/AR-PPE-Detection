# Phase 6 — AR-glasses readiness

Phases 1–5 built a system that runs on a laptop showing a webcam feed. This phase closes
the gap between that and something a headset can actually wear. The models don't change;
the **render target** and the **head motion** do.

```bash
python -m phase6_arview.preview            # see all three views side by side
python -m phase5_workid.reid_eval --pipeline  # head-motion measurement
```

---

## 1. The render target

`overlay.py` composites the HUD **onto the camera image**. That is correct for a monitor
and for **video-passthrough** headsets (Quest 3, Vision Pro), where the camera feed *is*
the display. It is wrong for **optical see-through** glasses (HoloLens, Xreal, Rokid),
where the wearer looks through the lens and the display only *adds* light:

| | monitor / passthrough | optical see-through |
|---|---|---|
| camera image | drawn | **must not be drawn** — it would paint video over reality |
| dark translucent cards | the standard way to make text readable | **invisible** — near-black emits no light |
| frame edges | all visible | **outside the lens** — a ~30–50° FOV sees only the middle |
| text size | monitor-legible is fine | needs scaling up for low angular resolution |

`phase2/src/arview.py` adds a `seethrough` renderer that respects all four: bright strokes
and text on black, confined to a **safe zone**, with larger glyphs. Workers outside the
lens FOV are not silently dropped — they get a **chevron** on the boundary pointing at
them, and unsafe ones appear in a short `OFF-VIEW` list. Workers *inside* the FOV are
deliberately **not** repeated in that list; they already carry a label on their body, and
display space is the scarcest resource on a headset.

The layer is **hard-clipped to the lens FOV** as a final step. A display physically cannot
emit outside its own field of view, so the invariant is enforced by construction rather
than trusting every draw call — stroke width and anti-aliasing both spill past a clamped
coordinate, which is exactly the bug the test suite caught.

`simulate_glasses()` blends the layer additively over the dimmed world and crops to the
FOV. `cv2.add` is a reasonable model of an additive combiner: graphics **brighten** the
scene and can never occlude it, which is precisely why dark UI does not work.

### Modes

```bash
cd phase2
python run.py --arview composite    # HUD on the camera image (monitor / passthrough)
python run.py --arview seethrough   # graphics on black, for an optical lens
python run.py --arview glasses      # what the wearer would see
```

Or set `arview.mode` in `config.yaml`, along with `fov_ratio` (how much of the camera
frame the lens covers), `scale` (text/stroke size) and `show_fov` (outline the lens FOV on
the composite view, so you can see what a headset would lose).

Measured render cost on the test clip: composite **8.2 ms**, seethrough **4.5 ms**
(less to draw), glasses **10.3 ms**.

---

## 2. Head motion — the measured part

A head-mounted camera translates the whole scene every frame. That is not what ByteTrack's
constant-velocity motion model expects, so it splits one worker across several
`tracker_id`s. The Phase 5 identity layer exists to glue those back together, and
`reid_eval.py --pipeline` measures how much of the damage it actually repairs — real
ByteTrack feeding the real identity manager, scored against known ground truth.

**4 workers, `similar` outfits (same issued vest, personal helmets), 3 seeds:**

| head motion (px/frame) | ByteTrack ids per worker | **+ identity layer** | false merge |
|---|---|---|---|
| 0 (tripod) | 1.00 | **1.00** | 0.0% |
| 4 | 1.25 | **1.00** | 0.0% |
| 8 | 1.50 | **1.00** | 0.0% |
| 12 | 1.75 | **1.00** | 0.0% |
| 20 | 1.75 | 1.75 | **29.8%** |

1.00 means nobody was ever lost. Two things to take from this:

- **The identity layer completely absorbs head-motion fragmentation up to ~12 px/frame.**
  ByteTrack alone degrades to 1.75 identities per worker — each person's safety record cut
  into pieces — while the end-to-end system holds at a perfect 1.00.
- **It has a breaking point, and the failure is ugly.** At 20 px/frame the appearance
  gallery itself stops coping: fragmentation returns *and* false merges jump to ~30%,
  meaning roughly a third of assignments hand a worker someone else's identity. Past that
  speed the system should be trusted less, not more. This is the honest operating limit.

### What was measured and rejected

An earlier version of the head-motion test appeared to show re-ID accuracy **improving**
with motion. That was an artifact, not a finding: the synthetic figures carry per-pixel
noise, motion blur smooths it away, and colour histograms are largely blur-invariant to
begin with. It is reported here rather than quietly deleted because it is the kind of
result that looks like good news and isn't. The `--pipeline` measurement above replaced
it, because it measures the mechanism head motion actually breaks — tracker association —
rather than descriptor robustness.

The `motion` parameter is genuine pixels-per-frame of peak displacement; the sinusoid
amplitudes are chosen so that `d/df` peaks at exactly that value. It is **not** calibrated
to degrees per second for any specific headset.

---

## Honest limits

- **Synthetic sequences.** No annotated multi-person video is available, so ground truth
  is generated. These numbers measure the matcher and the tracker association, **not site
  performance**. Rendered figures are not people.
- **The glasses simulation is a design-review aid**, not photometry. It ignores per-eye
  offset, lens distortion, vergence, and the real transmittance curve of a visor.
- **No hardware was used.** Nothing here has run on a headset. `fov_ratio` is a
  configurable guess until someone measures the target device.
- **Latency is not compensated.** Phase 4 measured ~16 FPS at 480 px on CPU; at ~60 ms per
  frame, boxes lag the world during head motion and will visibly trail a moving worker.
  Forward-predicting boxes by the measured pipeline latency is the obvious next step and
  is **not** implemented.
- **No ego-motion compensation.** It was deliberately not built: the measurement above
  shows the identity layer already absorbs fragmentation across the useful range, so
  compensating ByteTrack's Kalman state would add complexity for no measured gain below
  12 px/frame. Above it, the binding constraint is the appearance gallery, not the tracker.

## Tests

```bash
python phase2/tests/test_arview.py       # ALL_ARVIEW True
```

Covers the properties that decide whether a design is displayable at all: the layer is
mostly black (transparent), **nothing** is drawn outside the lens safe zone, an unsafe
worker outside the FOV still produces an indicator, a visible worker is not duplicated in
the OFF-VIEW list, and graphics add light rather than occluding the world.
