"""Revision R1, Part F.2 (R2.2): ArUco badges inside the identity pipeline (synthetic).

What runs
---------
Synthetic workers wearing REAL ArUco badges are passed through the DEPLOYED per-frame pipeline,
phase2/src/pipeline.py `SafetyPipeline`: ByteTrack -> ComplianceMonitor -> WorkIdBinder (crop
search on) -> IdentityManager -> live merges -> WorkerHistory, and `close()` at the end of the
session (end-of-session consolidation). The pipeline is built from phase2/config.yaml with two
changes only: `workid.enabled = True` (it ships False) and `device = "cpu"`; the badge names
are set to {0: "Alice Tan", 6: "Bob Lim"} (markers 0 and 6 share SIMILAR[0], i.e. identical
PPE). Everything else is at the shipped values, read back from the constructed objects into
summary.json.

The YOLO detector is the only part replaced. `src.detector.Detector` is swapped for a stub
in-process before `SafetyPipeline()` is constructed (pipeline.py imports it inside __init__),
so the real reset()/process()/close() run unchanged. The stub returns the ground-truth person
boxes (class 3 Person, confidence 0.9) and injected No-Helmet head boxes (class 1, conf 0.9,
the helmet band of the person box, as in Part E's compliance_timing.py). Per-frame outputs are
captured by wrapping INSTANCE methods of the pipeline's own objects (tracker.update,
monitor.update, binder.resolve, binder._assign, identity.update, identity.take_merges,
identity.consolidate, history.update, history.merge); every wrapper calls the original and
returns its result unchanged. Nothing under phase2/ or phase5_workid/ is modified.

The oracle never reports a pair of boxes the deployed detector could not output: Detector.detect
calls model.predict without `iou`/`agnostic_nms`, so ultralytics applies class-wise NMS at IoU 0.7
(suppress when IoU > 0.7). `--nms-iou 0.7` (default) emulates it: of two same-class boxes with
IoU > 0.7 only the less-occluded (front) one is reported (all oracle scores are equal, so the
visible fraction stands in for the score); a suppressed person's head box is withheld too.
`--nms-iou 0` turns the emulation off. Person boxes are full-extent (amodal): a worker behind
another keeps the box of their whole figure.

Scenes
------
Each frame is composed on a background shared by the whole sequence
(`reid_eval._background(np.random.default_rng(seed))`). For every worker in view a FRESH figure
is rendered with `badge_eval.render_badged_figure(height, marker, 10 cm, DICT_4X4_50,
seed=<per seed/scene/frame/worker>)` (height 320 px except one sensitivity variant), its
person-box patch is lifted (using the same int() rounding `_draw_worker` paints with, so a box
is 134 or 135 px wide at 320 px) and pasted at the scenario's position; workers further back
(feet higher in the image) are pasted first. A badge is HIDDEN by overpainting the returned
card box (+1 px) with the outfit's helmet colour before pasting. Re-rendering every frame gives
independent read trials, as in badge_eval's read-rate measurement (320 px: 85 % full frame,
99 % with the crop pass, README). A badge is "visible" when it is not overpainted and not
covered by a patch pasted in front of it. Positions sway gently (+/-15 px, +/-6 px).

Scenarios (30 fps, 10 empty lead-in frames, timings drawn per seed from the stated ranges)
------------------------------------------------------------------------------------------
Main scenarios (the brief's a-d):
  a  Alice's badge readable, then hidden for 4-6 s (turned away) while she stays in view, then
     readable again. Bob (identical PPE) stays in view at another spot, badge readable.
     No-Helmet: Alice for 2 s inside the hidden period; Bob for 1.5 s early on.
  b  Both named (badges readable); both leave; Alice is away exactly 60 frames (ByteTrack
     keeps a lost id ~31 frames at 30 fps) and returns to her spot with the badge HIDDEN while
     Bob is still away, so both lookalikes are candidates; her badge becomes readable 3-4 s
     later. Bob returns 0.7-2 s after her, badge readable. No-Helmet: Alice twice after her
     re-entry -- 1 s that has cleared before her badge is read, and 2 s that is still OPEN
     when it is read (it starts 25 frames before and ends 35 frames after); Bob 1.5 s after
     his return.
  c  Alice and Bob (identical PPE) enter with their badges turned away and start walking past
     each other 0.2-1 s later at 6 px/frame (boxes overlap, max IoU ~0.8), then stand at each
     other's starting spot for 6 s. Each badge is hidden in random contiguous segments, drawn
     independently per worker (seeded): hidden from entry for 1-3 s, then alternately shown
     and hidden for 0.5-3 s each. So a track is usually still unnamed when the boxes overlap.
     No-Helmet: each for 2 s after the crossing, at different times.
  d  Alice's badge is NEVER readable (always overpainted); Bob's always is. Both leave and
     return once: Bob is away 3-5 s and Alice returns (after 1.5-4 s away) while he is still
     away, so a badge-confirmed lookalike is a candidate for her re-identification.
     No-Helmet: Bob for 1.5 s before he leaves; Alice for 2 s after she returns.
Variants (added after review):
  b_close     as b, but Alice leaves again (60 frames) before her badge is ever read on the
              re-entry track and comes back readable: that fragment's track has ended, so only
              the end-of-session consolidation (pipe.close) can fold it. Bob returns while the
              fragment is in view (as in b).
  c_named     control, the first design of (c): badges hidden i.i.d. per frame (p = 0.5) and
              the workers stand 2-3 s before walking, so both are named before the crossing.
  c_persp     as c (same timings, hiding and render seeds), but the worker further back (Bob)
              is drawn 300 px tall instead of 320 (about 6 % smaller, consistent perspective;
              feet on the same line), so his box is always the smaller one.
  d_noreturn  as d (same timings and render seeds until then), but Bob leaves and does not
              come back, so the badge-confirmed lookalike is a candidate when consolidate()
              decides about Alice's re-entry record.
`--drop-occluded F` (default off) withholds a person box from the tracker while less than F
of it is unoccluded; it only matters in the crossings.

Measured per (scenario, seed); summarised over seeds 100-119 per scenario
--------------------------------------------------------------------------
  * time until named: seconds from a worker's first appearance to the first frame its
    IdentityResult label equals its true name; separately, time until its badge is first
    actually read (its own marker decoded and assigned to its own track by the binder that
    frame, captured from binder._assign). For re-entries (b, b_close, d, d_noreturn) also from
    the re-entry frame.
  * wrong-name assignments: person-frames whose label is ANOTHER worker's configured name;
    events = maximal stretches of a worker's identity outputs on one ByteTrack id with the
    same wrong label (frames without an identity output for that worker do not split one).
  * false merges, the paper's definition: phase5_workid.reid_eval._score is called on the log
    of (true worker, uid) for every identified person-frame (frame order, then worker order).
    An assignment is a false merge when its uid is already owned by a different true worker
    (first logged owns it); rate = false merges / all assignments. Three remaps:
    `live` (uids as output each frame, as reid_eval's live rows), `consolidated` (through
    _remap_from(consolidation merges), as the paper's +offline row) and `final` (through live
    take_merges AND consolidation merges, i.e. the records the WorkerHistory report ends with).
  * anonymous records ('Worker N'), by kind -- `fragment`: a SECOND record of a worker whose
    badge is read in the run (created after the worker already had a uid, e.g. on a re-entry
    with the badge hidden); `initial_anonymous`: the worker's FIRST record, created because
    the badge was hidden when the track was committed; `anonymous_identity`: a record of a
    worker whose badge is never read. Only the record's ANONYMOUS period counts: its frames
    and the violation episodes opened while it was anonymous. (i) at reacquisition: renamed
    in place / merged live (take_merges) / not resolved, and whether into the right name;
    whether those episodes are in a record carrying the owner's name right after that frame.
    (ii) after pipe.close(): the final record's name, and where the episodes ended up.
  * wrong merges and undo: an assignment of a uid owned by another worker, or a merge (live
    or consolidation) of records owned by different workers. Checked empirically: forward
    correction (later frames of that worker under a uid it owns -- not an undo), whether the
    final record still holds both workers' frames, and invariants over every frame (history
    records' and identity workers' frames_seen never shrink; episodes move only along merges
    and never return; a merged-away uid is never output again), plus the code (no
    unmerge/split method; summary.json cites lines).
  * scenario checks that each scenario exercises what it claims (summary.json "checks").

Outputs (--out, default paper_materials/revision_r1/partF2): per_seed.csv, fragments.csv,
episodes.csv, windows.csv, wrong_names.csv, wrong_merges.csv, per_frame.csv.gz, runs.jsonl
(written as runs finish) and summary.json (command, versions, settings, run time).
Must be launched with `python -B` (main() refuses otherwise), so no .pyc lands in phase2/src.

    .venv\\Scripts\\python.exe -B tools\\revision\\badge_identity_eval.py
    .venv\\Scripts\\python.exe -B tools\\revision\\badge_identity_eval.py --nseeds 2 --out <dir>
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True          # never write .pyc into phase2/src (also run with -B)
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import argparse  # noqa: E402
import csv  # noqa: E402
import datetime  # noqa: E402
import gzip  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from typing import Dict, List, Optional, Tuple  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# phase2 must win over the repo root: the root has its own, unrelated `src` package.
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "phase2"))
import src as _phase2_src  # noqa: E402

assert os.path.dirname(os.path.abspath(_phase2_src.__file__)) == os.path.join(ROOT, "phase2", "src"), \
    f"'src' resolved to {_phase2_src.__file__}, not phase2/src"

import cv2  # noqa: E402

cv2.setNumThreads(2)
import numpy as np  # noqa: E402
import supervision as sv  # noqa: E402

import src.detector as _det_mod  # noqa: E402
from phase5_workid.badge_eval import render_badged_figure  # noqa: E402
from phase5_workid.reid_eval import (FRAME_H, FRAME_W, SIMILAR, _background, _iou,  # noqa: E402
                                     _remap_from, _score)

CONFIG = os.path.join(ROOT, "phase2", "config.yaml")
DEFAULT_OUT = os.path.join(ROOT, "paper_materials", "revision_r1", "partF2")

FPS = 30
HEIGHT = 320                 # px; badge read rate 85 % full frame / 99 % with crop (README)
HEIGHT_BACK_PERSP = 300      # c_persp only: the worker further back, ~6 % smaller
BADGE_CM = 10.0
DICTIONARY = "DICT_4X4_50"
CONF = 0.9                   # oracle box confidence, as reid_eval._detections
NMS_IOU = 0.7                # ultralytics default `iou` (cfg/default.yaml); Detector passes none
LEAD_IN = 10                 # empty frames before anyone enters
MODEL_NAMES = {0: "Helmet", 1: "No-Helmet", 2: "No-Vest", 3: "Person", 4: "Vest"}  # data.yaml
PERSON, NO_HELMET = 3, 1
WORKERS = ({"tag": "A", "name": "Alice Tan", "marker": 0},
           {"tag": "B", "name": "Bob Lim", "marker": 6})       # 0 % 6 == 6 % 6: identical PPE
NW = len(WORKERS)
SPOT = (110.0, 600.0)        # left x of each worker's standing spot
FEET = (445.0, 420.0)        # feet line: Alice slightly nearer the camera (pasted in front)
PHASE = (0.0, 40.0)
SWAY_X, SWAY_Y = 15.0, 6.0   # gentle sway, px
WALK_SPEED = 6.0             # px/frame in (c): ~1 m/s for a 320 px (1.7 m) figure
ABSENCE_B = 60               # frames away in (b); ByteTrack drops a lost id after ~31
HIDE_P = 0.5                 # c_named: probability a badge is hidden in a frame
SEG_ENTRY = (30, 91)         # c / c_persp: badge hidden from entry for 1-3 s (frames, [lo, hi))
SEG = (15, 91)               # ... then alternately shown / hidden for 0.5-3 s each
SEED0, NSEEDS = 100, 20
MAIN_SCENARIOS = ("a", "b", "c", "d")
VARIANTS = ("b_close", "c_named", "c_persp", "d_noreturn")
SCENARIOS = MAIN_SCENARIOS + VARIANTS
VARIANT_OF = {"b_close": "b", "c_named": "c", "c_persp": "c", "d_noreturn": "d"}
# Render-seed stream per scenario. A variant that shares its parent's timings shares its
# render seeds too, so it differs from the parent only in what the variant changes. c_named
# keeps the stream the first (c) design used.
RENDER_KEY = {"a": 0, "b": 1, "c_named": 2, "d": 3, "b_close": 4, "c": 5, "c_persp": 5,
              "d_noreturn": 3}
SCEN_TITLE = {
    "a": "badge readable, hidden while the worker stays in view, readable again",
    "b": "occlusion and re-entry (60 frames away) with the badge hidden, readable later",
    "c": "two workers in identical PPE walk past each other soon after entering; badges hidden "
         "in random segments (independently per worker, about half the frames)",
    "d": "one badge never readable, alongside a badge-identified worker in identical PPE",
    "b_close": "as b, but the worker leaves again before the badge is read on the re-entry "
               "track; that fragment can only be folded at end-of-session consolidation",
    "c_named": "control (first design of c): badges hidden i.i.d. per frame, both workers "
               "named before the crossing",
    "c_persp": "as c, with the worker further back drawn 300 px tall (consistent perspective)",
    "d_noreturn": "as d, but the badge-identified worker leaves and does not return",
}


# --------------------------------------------------------------------------- stub detector
class StubDetector:
    """Stands in for src.detector.Detector: same constructor arity, `.names`,
    `class_ids_for`, `warmup`, `detect`. Returns whatever the harness put in `.next`."""

    def __init__(self, *args, **kwargs):
        self.names = dict(MODEL_NAMES)
        self.next = sv.Detections.empty()

    def class_ids_for(self, names):
        wanted = set(names)
        return [i for i, n in self.names.items() if n in wanted]

    def warmup(self):
        pass

    def detect(self, frame):
        return self.next


_det_mod.Detector = StubDetector          # before any SafetyPipeline() is constructed
from src.config import load_config  # noqa: E402
from src.pipeline import SafetyPipeline  # noqa: E402


# --------------------------------------------------------------------------- scenario plans
@dataclass
class Plan:
    scenario: str
    n: int
    present: np.ndarray            # (NW, n+1) bool, frame index 1..n
    xf: np.ndarray
    yf: np.ndarray
    shown: np.ndarray              # badge NOT overpainted
    violating: np.ndarray          # No-Helmet evidence injected
    windows: List[Tuple[int, int, int]] = field(default_factory=list)   # (worker, start, end)
    reentries: List[Tuple[int, int]] = field(default_factory=list)      # (worker, frame)
    keys: Dict[str, int] = field(default_factory=dict)
    heights: Tuple[int, ...] = (HEIGHT,) * NW                            # figure height, px

    def violate(self, k: int, a: int, b: int) -> None:
        assert self.present[k, a:b].all(), (self.scenario, k, a, b)
        self.violating[k, a:b] = True
        self.windows.append((k, a, b))

    def first(self, k: int) -> Optional[int]:
        idx = np.flatnonzero(self.present[k])
        return int(idx[0]) if len(idx) else None


def _base_plan(scen: str, n: int, heights: Tuple[int, ...] = (HEIGHT,) * NW) -> Plan:
    f = np.arange(n + 1, dtype=float)
    P = Plan(scen, n, np.zeros((NW, n + 1), bool), np.zeros((NW, n + 1)), np.zeros((NW, n + 1)),
             np.ones((NW, n + 1), bool), np.zeros((NW, n + 1), bool), heights=tuple(heights))
    for k in range(NW):
        P.xf[k] = SPOT[k] + SWAY_X * np.sin((f + PHASE[k]) / 18.0)
        P.yf[k] = FEET[k] - heights[k] + SWAY_Y * np.cos((f + PHASE[k]) / 21.0)   # top of box
    return P


def _on(arr: np.ndarray, k: int, a: int, b: int, value: bool = True) -> None:
    arr[k, max(1, a):min(b, arr.shape[1])] = value          # frames [a, b)


def plan_a(seed: int) -> Plan:
    rng = np.random.default_rng([seed, 0xA])
    first = LEAD_IN + 1
    h0 = first + int(rng.integers(75, 106))                  # readable for 2.5-3.5 s
    h1 = h0 + int(rng.integers(120, 181))                    # hidden for 4-6 s
    n = h1 + 90                                              # readable again for 3 s
    P = _base_plan("a", n)
    for k in range(NW):
        _on(P.present, k, first, n + 1)
    _on(P.shown, 0, h0, h1, False)
    va = h0 + int(rng.integers(15, 31))
    P.violate(0, va, va + 60)
    P.violate(1, first + 30, first + 75)
    P.keys = {"first": first, "hide_start": h0, "hide_end": h1}
    return P


def plan_b(seed: int) -> Plan:
    rng = np.random.default_rng([seed, 0xB])
    first = LEAD_IN + 1
    a_out = first + int(rng.integers(80, 111))               # Alice leaves (absent from a_out)
    a_in = a_out + ABSENCE_B                                 # ... and is back 60 frames later
    h1 = a_in + int(rng.integers(90, 121))                   # badge hidden until h1 (3-4 s)
    n = h1 + 90
    b_out = a_out + int(rng.integers(-20, 21))               # Bob leaves around the same time
    b_in = a_in + int(rng.integers(20, 61))                  # ... and returns after her
    P = _base_plan("b", n)
    _on(P.present, 0, first, a_out)
    _on(P.present, 0, a_in, n + 1)
    _on(P.present, 1, first, b_out)
    _on(P.present, 1, b_in, n + 1)
    _on(P.shown, 0, a_in, h1, False)
    P.violate(0, a_in + 8, a_in + 38)                       # cleared before the badge is read
    P.violate(0, h1 - 25, h1 + 35)                          # still OPEN when it is read
    P.violate(1, b_in + 20, b_in + 65)
    P.reentries = [(0, a_in), (1, b_in)]
    P.keys = {"first": first, "alice_out": a_out, "alice_in": a_in, "hide_end": h1,
              "bob_out": b_out, "bob_in": b_in}
    return P


def plan_b_close(seed: int) -> Plan:
    """As b, but Alice leaves again before her badge is read on the re-entry track."""
    rng = np.random.default_rng([seed, 0xB, 2])
    first = LEAD_IN + 1
    a_out = first + int(rng.integers(80, 111))               # Alice leaves (named before)
    a_in = a_out + ABSENCE_B                                 # back 60 frames later, badge hidden
    a_out2 = a_in + int(rng.integers(75, 106))               # leaves again 2.5-3.5 s later
    a_in2 = a_out2 + ABSENCE_B                               # back 60 frames later, readable
    n = a_in2 + 90
    b_out = a_out + int(rng.integers(-20, 21))               # Bob leaves around the same time
    b_in = a_in + int(rng.integers(20, 61))                  # ... returns while she is in view
    P = _base_plan("b_close", n)
    _on(P.present, 0, first, a_out)
    _on(P.present, 0, a_in, a_out2)
    _on(P.present, 0, a_in2, n + 1)
    _on(P.present, 1, first, b_out)
    _on(P.present, 1, b_in, n + 1)
    _on(P.shown, 0, a_in, a_out2, False)                     # hidden for the whole 2nd visit
    P.violate(0, a_in + 10, a_in + 70)
    P.violate(1, b_in + 20, b_in + 65)
    P.reentries = [(0, a_in), (1, b_in), (0, a_in2)]
    P.keys = {"first": first, "alice_out": a_out, "alice_in": a_in, "alice_out2": a_out2,
              "alice_in2": a_in2, "bob_out": b_out, "bob_in": b_in}
    return P


def _walk_past(P: Plan, w0: int) -> int:
    """Alice and Bob walk past each other from frame w0 and stop at each other's spot."""
    dist = SPOT[1] - SPOT[0]
    f = np.arange(P.n + 1, dtype=float)
    prog = np.clip((f - w0) * WALK_SPEED, 0.0, dist)
    P.xf[0] = SPOT[0] + prog + SWAY_X * np.sin((f + PHASE[0]) / 18.0)
    P.xf[1] = SPOT[1] - prog + SWAY_X * np.sin((f + PHASE[1]) / 18.0)
    return w0 + int(np.ceil(dist / WALK_SPEED))


def _crossing(seed: int, scen: str, heights: Tuple[int, ...]) -> Plan:
    """c / c_persp: walk past each other soon after entering; badges hidden in segments."""
    rng = np.random.default_rng([seed, 0xC, 2])
    first = LEAD_IN + 1
    w0 = first + int(rng.integers(6, 31))                    # walk 0.2-1 s after entering
    w1 = w0 + int(np.ceil((SPOT[1] - SPOT[0]) / WALK_SPEED))
    n = w1 + 180                                             # 6 s at the swapped spots
    P = _base_plan(scen, n, heights)
    assert _walk_past(P, w0) == w1
    for k in range(NW):
        _on(P.present, k, first, n + 1)
    P.shown[:, :] = True
    for k in range(NW):                                      # independent, seeded segments
        hr = np.random.default_rng([seed, 0xC, 3, k])
        t, hidden, dur = first, True, int(hr.integers(*SEG_ENTRY))   # turned away on entry
        while t <= n:
            if hidden:
                _on(P.shown, k, t, t + dur, False)
            t += dur
            hidden = not hidden
            dur = int(hr.integers(*SEG))
    va = w1 + 15 + int(rng.integers(0, 16))
    vb = w1 + 95 + int(rng.integers(0, 16))
    P.violate(0, va, va + 60)
    P.violate(1, vb, vb + 60)
    P.keys = {"first": first, "walk_start": w0, "walk_end": w1}
    return P


def plan_c(seed: int) -> Plan:
    return _crossing(seed, "c", (HEIGHT, HEIGHT))


def plan_c_persp(seed: int) -> Plan:
    return _crossing(seed, "c_persp", (HEIGHT, HEIGHT_BACK_PERSP))


def plan_c_named(seed: int) -> Plan:
    """The first design of (c), kept as a control: i.i.d. hiding, named before crossing."""
    rng = np.random.default_rng([seed, 0xC])
    hide_rng = np.random.default_rng([seed, 0xC, 1])
    first = LEAD_IN + 1
    w0 = first + int(rng.integers(60, 91))                   # stand 2-3 s, then walk
    w1 = w0 + int(np.ceil((SPOT[1] - SPOT[0]) / WALK_SPEED))
    n = w1 + 180                                             # 6 s at the swapped spots
    P = _base_plan("c_named", n)
    assert _walk_past(P, w0) == w1
    for k in range(NW):
        _on(P.present, k, first, n + 1)
    P.shown = hide_rng.random((NW, n + 1)) >= HIDE_P
    P.shown[:, 0] = True
    va = w1 + 15 + int(rng.integers(0, 16))
    vb = w1 + 95 + int(rng.integers(0, 16))
    P.violate(0, va, va + 60)
    P.violate(1, vb, vb + 60)
    P.keys = {"first": first, "walk_start": w0, "walk_end": w1}
    return P


def _plan_d(seed: int, scen: str, bob_returns: bool) -> Plan:
    rng = np.random.default_rng([seed, 0xD])
    first = LEAD_IN + 1
    b_out = first + int(rng.integers(90, 121))               # Bob (badge readable) leaves
    b_abs = int(rng.integers(90, 151))                       # away 3-5 s
    b_in = b_out + b_abs
    a_in = b_out + int(rng.integers(30, b_abs - 29))         # Alice returns while he is away
    a_abs = min(int(rng.integers(45, 121)), a_in - (first + 30))   # she was away 1.5-4 s
    a_out = a_in - a_abs
    n = (max(a_in, b_in) if bob_returns else a_in) + 120
    P = _base_plan(scen, n)
    _on(P.present, 0, first, a_out)
    _on(P.present, 0, a_in, n + 1)
    _on(P.present, 1, first, b_out)
    if bob_returns:
        _on(P.present, 1, b_in, n + 1)
    P.shown[0, :] = False                                    # Alice's badge: never readable
    P.violate(0, a_in + 10, a_in + 70)
    P.violate(1, first + 30, first + 75)
    P.reentries = [(0, a_in), (1, b_in)] if bob_returns else [(0, a_in)]
    P.keys = {"first": first, "alice_out": a_out, "alice_in": a_in, "bob_out": b_out}
    if bob_returns:
        P.keys["bob_in"] = b_in
    return P


def plan_d(seed: int) -> Plan:
    return _plan_d(seed, "d", True)


def plan_d_noreturn(seed: int) -> Plan:
    return _plan_d(seed, "d_noreturn", False)


PLANS = {"a": plan_a, "b": plan_b, "c": plan_c, "d": plan_d, "b_close": plan_b_close,
         "c_named": plan_c_named, "c_persp": plan_c_persp, "d_noreturn": plan_d_noreturn}


# --------------------------------------------------------------------------- frame composition
def render_seed(seed: int, scen: str, f: int, k: int) -> int:
    key = RENDER_KEY[scen]
    assert 0 <= f < 4096 and 0 <= k < 4 and 0 <= key < 8
    return ((int(seed) * 8 + key) * 4096 + int(f)) * 4 + int(k)


def head_box(b):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    return [x1 + 0.2 * w, y1, x2 - 0.2 * w, y1 + 0.22 * h]     # inside the helmet band


def _intersects(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def _visible_fraction(box, fronts) -> float:
    x1, y1, x2, y2 = (int(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    if not fronts:
        return 1.0
    m = np.ones((y2 - y1, x2 - x1), bool)
    for fx1, fy1, fx2, fy2 in fronts:
        ix1, iy1 = max(x1, int(fx1)) - x1, max(y1, int(fy1)) - y1
        ix2, iy2 = min(x2, int(fx2)) - x1, min(y2, int(fy2)) - y1
        if ix2 > ix1 and iy2 > iy1:
            m[iy1:iy2, ix1:ix2] = False
    return float(m.mean())


def compose(P: Plan, f: int, bg: np.ndarray, seed: int, drop_occluded: float,
            nms_iou: float = NMS_IOU):
    scene = bg.copy()
    # Depth order by the feet line (lower in the image = nearer): back to front.
    order = sorted((k for k in range(NW) if P.present[k, f]),
                   key=lambda k: (P.yf[k, f] + P.heights[k], k))
    info: Dict[int, dict] = {}
    for k in order:                                          # back to front
        marker = WORKERS[k]["marker"]
        rs = render_seed(seed, P.scenario, f, k)
        fr, pbox, card = render_badged_figure(P.heights[k], marker, BADGE_CM, DICTIONARY, seed=rs)
        ix1, iy1, ix2, iy2 = (int(v) for v in pbox)          # _draw_worker's own rounding
        patch = fr[iy1:iy2, ix1:ix2].copy()
        ph, pw = patch.shape[:2]
        cx1 = max(0, int(np.floor(card[0])) - 1 - ix1)      # card box + 1 px, patch coords
        cy1 = max(0, int(np.floor(card[1])) - 1 - iy1)
        cx2 = min(pw, int(np.ceil(card[2])) + 1 - ix1)
        cy2 = min(ph, int(np.ceil(card[3])) + 1 - iy1)
        shown = bool(P.shown[k, f])
        if not shown:
            patch[cy1:cy2, cx1:cx2] = np.asarray(SIMILAR[marker % len(SIMILAR)].helmet, np.uint8)
        x = int(np.clip(round(float(P.xf[k, f])), 0, FRAME_W - pw))
        y = int(np.clip(round(float(P.yf[k, f])), 0, FRAME_H - ph))
        scene[y:y + ph, x:x + pw] = patch
        info[k] = {"box": [x, y, x + pw, y + ph], "card": [x + cx1, y + cy1, x + cx2, y + cy2],
                   "shown": shown, "violating": bool(P.violating[k, f]), "render_seed": rs}
    for i, k in enumerate(order):
        fronts = [info[j]["box"] for j in order[i + 1:]]
        info[k]["visible_frac"] = _visible_fraction(info[k]["box"], fronts)
        info[k]["badge_occluded"] = any(_intersects(info[k]["card"], b) for b in fronts)
        info[k]["badge_visible"] = info[k]["shown"] and not info[k]["badge_occluded"]
        info[k]["detected"] = drop_occluded <= 0 or info[k]["visible_frac"] >= drop_occluded
        info[k]["nms_suppressed"] = False
    if nms_iou > 0:
        # The deployed detector's class-wise NMS (ultralytics default iou 0.7, Detector passes
        # none): of two Person boxes with IoU > nms_iou only one survives. Oracle scores are
        # all equal, so the less-occluded (then the front) box is taken as the higher-scoring
        # one. A suppressed person's head box is withheld with it.
        cand = sorted((k for k in order if info[k]["detected"]),
                      key=lambda k: (-info[k]["visible_frac"], -order.index(k)))
        kept: List[int] = []
        for k in cand:
            if any(_iou(info[k]["box"], info[j]["box"]) > nms_iou for j in kept):
                info[k]["detected"], info[k]["nms_suppressed"] = False, True
            else:
                kept.append(k)
        heads = [k for k in cand if info[k]["detected"] and info[k]["violating"]]
        kept_h: List[int] = []
        for k in heads:                                      # same rule for No-Helmet boxes
            if any(_iou(head_box(info[k]["box"]), head_box(info[j]["box"])) > nms_iou
                   for j in kept_h):
                info[k]["head_nms_suppressed"] = True
            else:
                kept_h.append(k)
    xyxy, cls, gt = [], [], []
    for k in order:
        if not info[k]["detected"]:
            continue
        xyxy.append(info[k]["box"])
        cls.append(PERSON)
        gt.append(k)
        if info[k]["violating"] and not info[k].get("head_nms_suppressed", False):
            xyxy.append(head_box(info[k]["box"]))
            cls.append(NO_HELMET)
            gt.append(k)
    if xyxy:
        dets = sv.Detections(xyxy=np.asarray(xyxy, np.float32).reshape(-1, 4),
                             confidence=np.full(len(xyxy), CONF, np.float32),
                             class_id=np.asarray(cls, int), data={"gt": np.asarray(gt, int)})
    else:
        dets = sv.Detections.empty()
    return scene, dets, info


# --------------------------------------------------------------------------- capture
class Capture:
    """Wraps instance methods of one SafetyPipeline's objects; every wrapper calls the
    original with the same arguments and returns its result unchanged."""

    def __init__(self, pipe: SafetyPipeline):
        self.pipe = pipe
        self.frame = 0
        self.in_close = False
        self.live_merges: List[Tuple[int, str, str, str]] = []   # (frame, src, dst, dst label)
        self.close_merges: List[Tuple[str, str]] = []
        self.consolidation: Optional[List[Tuple[str, str]]] = None
        self.merge_calls: List[Tuple[object, str, str, str, bool]] = []
        self.ep: Dict[int, dict] = {}            # id(Episode) -> origin info (keeps it alive)
        self.absorbed: Dict[int, int] = {}       # id(dropped duplicate) -> id(kept episode)
        self.holders: List[Tuple[int, Dict[int, str]]] = []
        self.rec_labels: List[Tuple[int, Dict[str, str]]] = []
        self.rec_frames: List[Tuple[int, Dict[str, int]]] = []
        self.id_frames: List[Tuple[int, Dict[str, int]]] = []    # identity workers' frames_seen
        self.final_holders: Dict[int, str] = {}
        self.final_labels: Dict[str, str] = {}
        self.dup_uid_frames: List[int] = []
        self.decoded_counts: Counter = Counter()
        self.badge_label_counts: Counter = Counter()
        self.n_workers_before_close = 0
        self.stats_before_close: dict = {}
        self._reset_frame()

        tr, mon, bnd, idm, hist = pipe.tracker, pipe.monitor, pipe.binder, pipe.identity, pipe.history
        o_tr, o_mon, o_res, o_asg = tr.update, mon.update, bnd.resolve, bnd._assign
        o_upd, o_take, o_cons = idm.update, idm.take_merges, idm.consolidate
        o_hupd, o_hmerge = hist.update, hist.merge

        def tracker_update(detections):
            tracked, violations = o_tr(detections)
            self.worker_of_tid, self.tid_of_worker = {}, {}
            gt = tracked.data.get("gt") if len(tracked) else None
            if gt is not None and tracked.tracker_id is not None:
                for tid, k in zip(tracked.tracker_id, gt):
                    self.worker_of_tid[int(tid)] = int(k)
                    self.tid_of_worker[int(k)] = int(tid)
            return tracked, violations

        def monitor_update(tracked, violations):
            fc = o_mon(tracked, violations)
            self.fc = fc
            self.active = {int(p.tracker_id): {a.class_name for a in p.active} for p in fc.persons}
            return fc

        def binder_resolve(frame, tracked):
            out = o_res(frame, tracked)
            self.badge_of = dict(out)
            self.badge_label_counts.update(out.values())
            return out

        def binder_assign(M, P, cont, track_ids, marker_ids):
            out = o_asg(M, P, cont, track_ids, marker_ids)
            self.decoded = [int(m) for m in marker_ids]
            self.decoded_counts.update(set(self.decoded))
            self.assign = {int(t): int(m) for t, m in out.items()}
            return out

        def identity_update(frame, track_ids, boxes, marker_labels=None, now_s=None):
            out = o_upd(frame, track_ids, boxes, marker_labels=marker_labels, now_s=now_s)
            self.ident = dict(out)
            return out

        def take_merges():
            m = o_take()
            if self.in_close:
                self.close_merges.extend(m)
            else:
                for s, d in m:
                    w = idm.workers.get(d)
                    self.live_merges.append((self.frame, s, d, w.label if w else ""))
            return m

        def consolidate():
            m = o_cons()
            self.consolidation = list(m)
            return m

        def history_update(frame_no, elapsed_s, frame_compliance, identity_by_track):
            o_hupd(frame_no, elapsed_s, frame_compliance, identity_by_track)
            for rec in hist.records.values():
                for e in rec.episodes:
                    if id(e) in self.ep:
                        continue
                    ws = sorted({self.worker_of_tid.get(int(p.tracker_id), -1)
                                 for p in frame_compliance.persons
                                 if p.tracker_id in identity_by_track
                                 and identity_by_track[p.tracker_id].uid == rec.uid
                                 and any(a.class_name == e.class_name for a in p.active)})
                    self.ep[id(e)] = {"obj": e, "uid": rec.uid, "label": rec.label,
                                      "frame": int(frame_no), "workers": ws,
                                      "class": e.class_name}
            return None

        def history_merge(src_uid, dst_uid):
            before = []
            for u in (src_uid, dst_uid):
                rec = hist.records.get(u)
                if rec is not None:
                    before.extend(rec.episodes)
            ok = o_hmerge(src_uid, dst_uid)
            if ok:
                kept_ids = {id(e) for e in hist.records[dst_uid].episodes}
                for e in before:
                    if id(e) not in kept_ids:                 # dropped as a duplicate
                        kept = hist._open.get((dst_uid, e.class_name))
                        if kept is not None:
                            self.absorbed[id(e)] = id(kept)
            self.merge_calls.append(("close" if self.in_close else self.frame, src_uid, dst_uid,
                                     "consolidation" if self.in_close else "live", bool(ok)))
            return ok

        tr.update, mon.update = tracker_update, monitor_update
        bnd.resolve, bnd._assign = binder_resolve, binder_assign
        idm.update, idm.take_merges, idm.consolidate = identity_update, take_merges, consolidate
        hist.update, hist.merge = history_update, history_merge

    def _reset_frame(self):
        self.worker_of_tid: Dict[int, int] = {}
        self.tid_of_worker: Dict[int, int] = {}
        self.fc = None
        self.active: Dict[int, set] = {}
        self.badge_of: Dict[int, str] = {}
        self.decoded: List[int] = []
        self.assign: Dict[int, int] = {}
        self.ident: dict = {}

    def begin(self, f: int):
        self.frame = f
        self._reset_frame()

    def _snapshot(self):
        hold, labels, fseen = {}, {}, {}
        for uid, rec in self.pipe.history.records.items():
            labels[uid] = rec.label
            fseen[uid] = rec.frames_seen
            for e in rec.episodes:
                hold[id(e)] = uid
        return hold, labels, fseen

    def end(self, f: int):
        uids = [r.uid for r in self.ident.values()]
        if len(uids) != len(set(uids)):
            self.dup_uid_frames.append(f)
        hold, labels, fseen = self._snapshot()
        self.holders.append((f, hold))
        self.rec_labels.append((f, labels))
        self.rec_frames.append((f, fseen))
        self.id_frames.append((f, {u: w.frames_seen for u, w in self.pipe.identity.workers.items()}))

    def before_close(self):
        self.n_workers_before_close = len(self.pipe.identity.workers)
        self.stats_before_close = dict(self.pipe.identity.stats)
        self.in_close = True

    def after_close(self):
        self.in_close = False
        hold, labels, _ = self._snapshot()
        self.final_holders, self.final_labels = hold, labels

    def holder(self, epid: int, snap: Dict[int, str]) -> Optional[str]:
        seen = set()
        while epid not in snap:
            if epid in seen or epid not in self.absorbed:
                return None
            seen.add(epid)
            epid = self.absorbed[epid]
        return snap[epid]


# --------------------------------------------------------------------------- one run
def build_pipeline():
    cfg = load_config(CONFIG)
    cfg.workid_enabled = True
    cfg.device_pref = "cpu"
    cfg.workid_markers = {w["marker"]: w["name"] for w in WORKERS}
    pipe = SafetyPipeline(cfg, frame_rate=FPS, render=False, quiet=True)
    assert pipe.binder is not None, "WorkIdBinder was not built (silently disabled)"
    assert pipe.identity is not None and pipe.history is not None, "identity layer disabled"
    assert isinstance(pipe.detector, StubDetector)
    assert pipe.person_ids == [PERSON] and NO_HELMET in pipe.rules_by_id
    assert pipe.binder.crop_detect and pipe.binder.crop_min_height == 320
    return pipe


def settings_of(pipe: SafetyPipeline) -> dict:
    b, i, h, m = pipe.binder, pipe.identity, pipe.history, pipe.monitor
    bt = pipe.tracker.tracker
    return {
        "config_file": os.path.relpath(CONFIG, ROOT).replace("\\", "/"),
        "overrides": {"workid.enabled": True, "device": "cpu",
                      "workid.markers": {str(w["marker"]): w["name"] for w in WORKERS}},
        "frame_rate": pipe._frame_rate,
        "bytetrack": {"lost_track_buffer": pipe.cfg.lost_track_buffer,
                      "max_time_lost": getattr(bt, "max_time_lost", None),
                      "track_activation_threshold": getattr(bt, "track_activation_threshold", None),
                      "det_thresh": getattr(bt, "det_thresh", None),
                      "minimum_matching_threshold": getattr(bt, "minimum_matching_threshold", None),
                      "minimum_consecutive_frames": getattr(bt, "minimum_consecutive_frames", None)},
        "compliance": {"debounce_frames": m.debounce_frames, "clear_frames": m.clear_frames,
                       "containment": m.containment_thresh,
                       "rules_by_id": {str(k): v.class_name for k, v in m.rules_by_id.items()}},
        "binder": {"dictionary": pipe.cfg.workid_dictionary, "markers": dict(b.markers),
                   "containment": b.containment, "crop_detect": b.crop_detect,
                   "crop_min_height": b.crop_min_height, "reacquire_after": b._reacquire_after,
                   "gc_after": b._gc_after},
        "identity": {k: getattr(i, k) for k in (
            "match_threshold", "margin", "probation_frames", "gate", "gate_slack", "gate_speed",
            "gate_horizon", "gate_speed_s", "gate_horizon_s", "gate_floor", "coast_frames",
            "coast_s", "consolidate_threshold", "consolidate_margin", "max_exemplars",
            "forget_after", "min_box_height", "refresh_every", "appearance_enabled")},
        "embedder": type(i.embedder).__name__,
        "history": {"absence_tolerance": h.absence_tolerance},
        "identity_consolidate": bool(pipe.cfg.identity_consolidate),
        "scene": {"figure_height_px": HEIGHT, "c_persp_back_figure_height_px": HEIGHT_BACK_PERSP,
                  "badge_cm": BADGE_CM, "box_confidence": CONF, "nms_iou_emulated": NMS_IOU,
                  "frame": [FRAME_W, FRAME_H], "lead_in_frames": LEAD_IN, "spots_x": SPOT,
                  "feet_y": FEET, "sway_px": [SWAY_X, SWAY_Y], "walk_speed_px_per_frame": WALK_SPEED,
                  "absence_b_frames": ABSENCE_B, "hide_p_c_named": HIDE_P,
                  "c_hidden_segment_frames": {"entry_hidden": list(SEG_ENTRY),
                                              "then_alternating": list(SEG)}},
    }


def run_one(scen: str, seed: int, drop_occluded: float, nms_iou: float = NMS_IOU):
    t0 = time.time()
    P = PLANS[scen](seed)
    pipe = build_pipeline()
    cap = Capture(pipe)
    bg = _background(np.random.default_rng(seed))
    rows: List[dict] = []
    for f in range(1, P.n + 1):
        scene, dets, info = compose(P, f, bg, seed, drop_occluded, nms_iou)
        pipe.detector.next = dets
        cap.begin(f)
        pipe.process(scene, f, f / FPS)
        cap.end(f)
        for k in sorted(info):
            g = info[k]
            tid = cap.tid_of_worker.get(k, -1)
            res = cap.ident.get(tid) if tid != -1 else None
            rows.append({
                "scenario": scen, "seed": seed, "frame": f, "t_s": round(f / FPS, 4), "worker": k,
                "tid": tid, "x1": g["box"][0], "y1": g["box"][1], "x2": g["box"][2],
                "y2": g["box"][3], "shown": g["shown"], "badge_occluded": g["badge_occluded"],
                "badge_visible": g["badge_visible"], "visible_frac": round(g["visible_frac"], 3),
                "detected": g["detected"], "nms_suppressed": g["nms_suppressed"],
                "violating_gt": g["violating"],
                "decoded_own": WORKERS[k]["marker"] in cap.decoded,
                "read_mid": cap.assign.get(tid, -1) if tid != -1 else -1,
                "in_badge_of": tid in cap.badge_of, "badge_label": cap.badge_of.get(tid, ""),
                "uid": res.uid if res is not None else "",
                "label": res.label if res is not None else "",
                "source": res.source if res is not None else "",
                "similarity": round(float(res.similarity), 4) if res is not None else "",
                "nohelmet_active": "No-Helmet" in cap.active.get(tid, ()),
            })
    cap.before_close()
    pipe.close(P.n / FPS, P.n)
    cap.after_close()
    report = pipe.report()
    assert cap.consolidation is not None, "consolidate() did not run inside close()"
    assert list(cap.close_merges) == list(cap.consolidation)
    out = analyse(scen, seed, P, rows, cap, pipe, report)
    out["row"]["run_s"] = round(time.time() - t0, 1)
    return out, pipe


# --------------------------------------------------------------------------- analysis
def _t(f, ref) -> Optional[float]:
    return None if f is None or ref is None else round((f - ref) / FPS, 4)


def analyse(scen: str, seed: int, P: Plan, rows: List[dict], cap: Capture,
            pipe: SafetyPipeline, report: dict) -> dict:
    name = {k: WORKERS[k]["name"] for k in range(NW)}
    tag = {k: WORKERS[k]["tag"] for k in range(NW)}
    marker = {k: WORKERS[k]["marker"] for k in range(NW)}
    worker_of_name = {v: k for k, v in name.items()}
    configured = set(worker_of_name)
    rows = sorted(rows, key=lambda r: (r["frame"], r["worker"]))
    R = {k: [r for r in rows if r["worker"] == k] for k in range(NW)}
    row = {"scenario": scen, "seed": seed, "frames": P.n}
    row.update({f"plan_{key}": v for key, v in P.keys.items()})

    def first_frame(rs, pred, start=0):
        return next((r["frame"] for r in rs if r["frame"] >= start and pred(r)), None)

    # ---- timing ----------------------------------------------------------------------
    times = {}
    for k in range(NW):
        first = P.first(k)
        fn = first_frame(R[k], lambda r: r["label"] == name[k])
        fr = first_frame(R[k], lambda r: r["read_mid"] == marker[k])
        t = {"first_frame": first, "named_frame": fn, "read_frame": fr,
             "t_named_s": _t(fn, first), "t_first_read_s": _t(fr, first)}
        for j, re_f in enumerate(sorted(rf for wk, rf in P.reentries if wk == k)):
            sx = "" if j == 0 else str(j + 1)                # b_close: Alice re-enters twice
            fn2 = first_frame(R[k], lambda r: r["label"] == name[k], re_f)
            fr2 = first_frame(R[k], lambda r: r["read_mid"] == marker[k], re_f)
            commit = next((r for r in R[k] if r["frame"] >= re_f and r["uid"]), None)
            t.update({f"reentry{sx}_frame": re_f, f"t_named_after_reentry{sx}_s": _t(fn2, re_f),
                      f"t_read_after_reentry{sx}_s": _t(fr2, re_f),
                      f"reentry{sx}_commit_frame": commit["frame"] if commit else None,
                      f"reentry{sx}_commit_label": commit["label"] if commit else None,
                      f"reentry{sx}_commit_source": commit["source"] if commit else None,
                      f"reentry{sx}_commit_similarity": commit["similarity"] if commit else None})
        if scen == "b" and k == 0:
            h1 = P.keys["hide_end"]
            t["t_read_after_shown_s"] = _t(first_frame(R[k], lambda r: r["read_mid"] == marker[k], h1), h1)
        present = len(R[k])
        t.update({
            "frames_present": present,
            "frames_tracked": sum(r["tid"] != -1 for r in R[k]),
            "frames_identified": sum(bool(r["uid"]) for r in R[k]),
            "frames_named_correct": sum(r["label"] == name[k] for r in R[k]),
            "frames_anonymous": sum(r["label"].startswith("Worker ") for r in R[k]),
            "frames_badge_visible": sum(r["badge_visible"] for r in R[k]),
            "frames_badge_read": sum(r["read_mid"] == marker[k] for r in R[k]),
            "frames_own_marker_decoded": sum(r["decoded_own"] for r in R[k]),
            "frames_decoded_not_bound_to_own_track": sum(r["decoded_own"] and r["read_mid"] != marker[k]
                                                         for r in R[k]),
            "frames_in_badge_of": sum(r["in_badge_of"] for r in R[k]),
            "bytetrack_ids": sorted({r["tid"] for r in R[k] if r["tid"] != -1}),
            "uids": sorted({r["uid"] for r in R[k] if r["uid"]}),
        })
        times[k] = t
        for key in ("t_named_s", "t_first_read_s", "t_named_after_reentry_s",
                    "t_read_after_reentry_s", "reentry_commit_source", "reentry_commit_label",
                    "t_named_after_reentry2_s", "t_read_after_reentry2_s",
                    "reentry2_commit_source", "reentry2_commit_label", "t_read_after_shown_s",
                    "frames_present", "frames_named_correct", "frames_anonymous",
                    "frames_badge_visible", "frames_badge_read",
                    "frames_decoded_not_bound_to_own_track"):
            if key in t:
                row[f"{tag[k]}_{key}"] = t[key]
        row[f"{tag[k]}_bytetrack_ids"] = " ".join(map(str, t["bytetrack_ids"]))
        row[f"{tag[k]}_uids"] = " ".join(t["uids"])

    # ---- wrong names (identity label = ANOTHER worker's configured name) ---------------
    # An event is a maximal stretch of a worker's identity outputs on ONE ByteTrack id that all
    # carry the same wrong label. Frames in which the worker gets no identity output at all
    # (box withheld by the NMS emulation, probation) neither end nor extend it; an output with
    # another label, or a new track, ends it.
    wrong_events, binder_wrong = [], 0
    for k in range(NW):
        cur = None
        for r in R[k]:
            if not r["uid"]:
                continue
            wrong = r["label"] in configured and r["label"] != name[k]
            if wrong and cur is not None and cur["wrong_label"] == r["label"] and cur["tid"] == r["tid"]:
                cur["end"], cur["frames"] = r["frame"], cur["frames"] + 1
                continue
            if cur is not None:
                wrong_events.append(cur)
                cur = None
            if wrong:
                cur = {"scenario": scen, "seed": seed, "worker": name[k],
                       "wrong_label": r["label"], "start": r["frame"], "end": r["frame"],
                       "frames": 1, "source_at_onset": r["source"], "uid": r["uid"],
                       "tid": r["tid"], "badge_label_at_onset": r["badge_label"]}
        if cur is not None:
            wrong_events.append(cur)
        binder_wrong += sum(r["badge_label"] in configured and r["badge_label"] != name[k]
                            for r in R[k])
        row[f"{tag[k]}_wrong_name_frames"] = sum(e["frames"] for e in wrong_events
                                                 if e["worker"] == name[k])
        row[f"{tag[k]}_wrong_name_events"] = sum(1 for e in wrong_events if e["worker"] == name[k])
    row["wrong_name_frames"] = sum(e["frames"] for e in wrong_events)
    row["wrong_name_events"] = len(wrong_events)
    row["binder_wrong_label_frames"] = binder_wrong
    # Binder mis-assignments: a track assigned ANOTHER worker's marker this frame. Logged with
    # the geometry that decides _assign's smallest-box rule (workid.py:119-139).
    by_frame: Dict[int, Dict[int, dict]] = {}
    for r in rows:
        by_frame.setdefault(r["frame"], {})[r["worker"]] = r

    def _area(r) -> int:
        return (r["x2"] - r["x1"]) * (r["y2"] - r["y1"])

    worker_of_marker = {m: k for k, m in marker.items()}
    mis = [r for r in rows if r["read_mid"] in worker_of_marker
           and r["read_mid"] != marker[r["worker"]]]
    mis_smaller = 0
    for r in mis:
        o = by_frame[r["frame"]].get(worker_of_marker[r["read_mid"]])
        mis_smaller += o is not None and _area(r) < _area(o)
    row["binder_misassigned_reads"] = len(mis)
    row["binder_misassigned_reads_to_smaller_box"] = mis_smaller
    # Binder bindings are per track and sticky: count distinct (worker, track, wrong name).
    row["binder_wrong_bindings"] = len({(r["worker"], r["tid"], r["badge_label"]) for r in rows
                                        if r["badge_label"] in configured
                                        and r["badge_label"] != name[r["worker"]]})
    row["duplicate_uid_frames"] = len(cap.dup_uid_frames)

    # ---- false merges: reid_eval._score, three remaps ---------------------------------
    assign_log = [(r["worker"], r["uid"]) for r in rows if r["uid"]]
    pairs = []
    for k, re_f in P.reentries:
        pre = None
        for r in R[k]:
            if r["frame"] < re_f and r["uid"]:
                pre = r["uid"]
        new = next((r["uid"] for r in R[k] if r["frame"] >= re_f and r["uid"]), None)
        pairs.append((pre, new))
    live_pairs = [(s, d) for (_f, s, d, _l) in cap.live_merges]
    cons = list(cap.consolidation or [])
    remap_cons, remap_final = _remap_from(cons), _remap_from(live_pairs + cons)
    n_after = len(pipe.identity.workers)
    fm = {"live": _score(assign_log, pairs, cap.n_workers_before_close, None),
          "consolidated": _score(assign_log, pairs, n_after, remap_cons),
          "final": _score(assign_log, pairs, n_after, remap_final)}
    for v, s in fm.items():
        row[f"fm_{v}_false_merges"] = s["false_merges"]
        row[f"fm_{v}_assignments"] = s["assignments"]
        row[f"fm_{v}_rate_pct"] = round(s["false_merge_rate"], 3)
        row[f"fm_{v}_fragmentation"] = round(s["fragmentation"], 3)
        if pairs:
            row[f"fm_{v}_reid_recovered"] = s["recovered"]
            row[f"fm_{v}_reentries"] = s["reentries"]
    row["live_merges"] = " ".join(f"{s}>{d}@{f}" for f, s, d, _l in cap.live_merges)
    row["consolidation_merges"] = " ".join(f"{s}>{d}" for s, d in cons)
    row["n_live_merges"], row["n_consolidation_merges"] = len(cap.live_merges), len(cons)
    row["uids_created"] = pipe.identity._uid_seq
    row["workers_before_close"], row["workers_after_close"] = cap.n_workers_before_close, n_after
    for key, v in cap.stats_before_close.items():
        row[f"stat_{key}"] = v

    # ---- ownership helpers -----------------------------------------------------------
    owner_first: Dict[str, int] = {}
    for w, u in assign_log:
        owner_first.setdefault(u, w)

    def final_label(uid: str) -> Tuple[str, str]:
        fu = remap_final(uid)
        if fu in cap.final_labels:
            return fu, cap.final_labels[fu]
        w = pipe.identity.workers.get(fu)
        return fu, (w.label if w is not None else "")

    def record_owner(final_uid: str, label: str) -> Optional[int]:
        """A named record belongs to the name; an anonymous one to its majority worker."""
        if label in worker_of_name:
            return worker_of_name[label]
        c = Counter(w for w, u in assign_log if remap_final(u) == final_uid)
        return c.most_common(1)[0][0] if c else None

    def label_at(frame: int, uid: Optional[str]) -> str:
        if uid is None:
            return ""
        return cap.rec_labels[frame - 1][1].get(uid, "")

    # ---- anonymous records ('Worker N'): fragments, initial records, unbadged ----------
    # Only a record's ANONYMOUS period is attributed to it: the person-frames output under
    # the uid while its label was 'Worker N', and the episodes opened while it was anonymous.
    # After an in-place rename the same uid IS the named record; what happens then is the
    # named worker's own history, not a fragment's.
    uid_first: Dict[str, dict] = {}
    for r in rows:
        if r["uid"] and r["uid"] not in uid_first:
            uid_first[r["uid"]] = r
    first_uid_of = {k: next((r["uid"] for r in R[k] if r["uid"]), None) for k in range(NW)}
    read_ever = {k: times[k]["frames_badge_read"] > 0 for k in range(NW)}
    fragments = []
    bookkeeping_mismatch = 0
    for F, r0 in uid_first.items():
        if r0["label"] in configured:
            continue
        rs = [r for r in rows if r["uid"] == F and r["label"] not in configured]
        cnt = Counter(r["worker"] for r in rs)
        owner = cnt.most_common(1)[0][0]
        created = r0["frame"]
        probation = [r for r in R[owner] if created - 3 < r["frame"] <= created]
        rename = next(((r["frame"], r["label"]) for r in rows
                       if r["uid"] == F and r["label"] in configured), None)
        lm = next(((f, d, lab) for (f, s, d, lab) in cap.live_merges if s == F), None)
        cm = next((d for s, d in cons if s == F), None)
        if rename is not None:
            at_i = "renamed_correct" if rename[1] == name[owner] else "renamed_wrong"
            t_res = rename[0]
        elif lm is not None:
            dst_ok = (lm[2] == name[owner] if lm[2] in configured
                      else owner_first.get(lm[1]) == owner)
            at_i = "merged_live_correct" if dst_ok else "merged_live_wrong"
            t_res = lm[0]
        else:
            at_i, t_res = "not_resolved_live", None
        fu, flab = final_label(F)
        if flab in configured:
            at_end = "named_correct" if flab == name[owner] else "named_wrong"
        else:
            fo = record_owner(fu, flab)
            at_end = ("anonymous_separate" if fu == F else
                      ("anonymous_merged_same_worker" if fo == owner else
                       "anonymous_merged_other_worker"))
        eps_all = [(epid, e) for epid, e in cap.ep.items() if e["uid"] == F]
        eps = [(epid, e) for epid, e in eps_all if e["label"] not in configured]  # anonymous period
        # The label test and the frame test must agree (rename / merge happen at t_res).
        bookkeeping_mismatch += sum(1 for _, e in eps_all
                                    if (e["label"] not in configured)
                                    != (t_res is None or e["frame"] < t_res))
        ep_i = sum(1 for epid, e in eps if t_res is not None
                   and label_at(t_res, cap.holder(epid, cap.holders[t_res - 1][1])) == name[owner])
        ep_end = sum(1 for epid, e in eps
                     if cap.final_labels.get(cap.holder(epid, cap.final_holders) or "", "") == name[owner])
        kind = ("anonymous_identity" if not read_ever[owner] else
                "initial_anonymous" if first_uid_of[owner] == F else "fragment")
        fragments.append({
            "scenario": scen, "seed": seed, "uid": F, "first_label": r0["label"],
            "kind": kind,
            "owner": name[owner], "workers": " ".join(f"{name[w]}:{c}" for w, c in sorted(cnt.items())),
            "contaminated": len(cnt) > 1, "created_frame": created,
            "owner_badge_visible_in_probation": any(r["badge_visible"] for r in probation),
            "owner_badge_read_before_creation": any(r["read_mid"] == marker[owner]
                                                    for r in R[owner] if r["frame"] <= created),
            "frames": len(rs), "at_reacquisition": at_i, "resolved_frame": t_res,
            "renamed_to": rename[1] if rename else "", "live_merge_dst": lm[1] if lm else "",
            "live_merge_dst_label": lm[2] if lm else "", "consolidation_dst": cm or "",
            "final_uid": fu, "final_label": flab, "at_end": at_end,
            "resolved_by": ("reacquisition" if t_res is not None else
                            ("consolidation" if cm is not None else "none")),
            "episodes": len(eps), "episodes_open_at_reacquisition": sum(
                1 for _, e in eps if t_res is not None and (e["obj"].end_frame is None
                                                            or e["obj"].end_frame >= t_res)),
            "episodes_in_named_record_at_reacquisition": ep_i,
            "episodes_in_named_record_at_end": ep_end,
            "episodes_opened_after_rename": len(eps_all) - len(eps),
        })
    row["anonymous_record_bookkeeping_mismatch"] = bookkeeping_mismatch
    for kind_, tag_ in (("fragment", "fragments"), ("initial_anonymous", "initial_anonymous")):
        sel = [x for x in fragments if x["kind"] == kind_]
        row[tag_] = len(sel)
        row[f"{tag_}_resolved_live_correct"] = sum(x["at_reacquisition"] in ("renamed_correct", "merged_live_correct") for x in sel)
        row[f"{tag_}_resolved_live_wrong"] = sum(x["at_reacquisition"] in ("renamed_wrong", "merged_live_wrong") for x in sel)
        row[f"{tag_}_named_correct_at_end"] = sum(x["at_end"] == "named_correct" for x in sel)
        row[f"{tag_}_named_wrong_at_end"] = sum(x["at_end"] == "named_wrong" for x in sel)
        row[f"{tag_}_anonymous_at_end"] = sum(x["at_end"].startswith("anonymous") for x in sel)
    row["anonymous_identities_unbadged"] = sum(x["kind"] == "anonymous_identity" for x in fragments)

    # ---- episodes and injected violation windows -------------------------------------
    episodes = []
    for epid, e in cap.ep.items():
        ep = e["obj"]
        ow = e["workers"][0] if len(e["workers"]) == 1 else None
        hu = cap.holder(epid, cap.final_holders)
        hl = cap.final_labels.get(hu or "", "")
        ho = record_owner(hu, hl) if hu else None
        episodes.append({
            "scenario": scen, "seed": seed, "class": e["class"],
            "origin_worker": name[ow] if ow is not None and ow >= 0 else "?",
            "origin_uid": e["uid"], "origin_label": e["label"], "origin_frame": e["frame"],
            "start_s": round(ep.start_s, 4), "end_s": round(ep.end_s, 4) if ep.end_s is not None else "",
            "truncated": ep.truncated, "absorbed_as_duplicate": epid in cap.absorbed,
            "final_uid": hu or "", "final_label": hl,
            "final_record_owner": name[ho] if ho is not None else "",
            "correct_record": ho is not None and ho == ow,
            "in_named_record": ow is not None and ow >= 0 and hl == name[ow],
            "in_other_workers_named_record": hl in configured and ow is not None and hl != name.get(ow),
        })
    row["episodes"] = len(episodes)
    row["episodes_correct_record"] = sum(x["correct_record"] for x in episodes)
    row["episodes_in_named_record"] = sum(x["in_named_record"] for x in episodes)
    row["episodes_in_other_workers_named_record"] = sum(x["in_other_workers_named_record"] for x in episodes)
    windows = []
    for k, a, b in P.windows:
        mine = [x for x in episodes if x["origin_worker"] == name[k]
                and a <= x["origin_frame"] <= b + pipe.monitor.clear_frames]
        mine.sort(key=lambda x: x["origin_frame"])
        windows.append({
            "scenario": scen, "seed": seed, "worker": name[k], "start_frame": a, "end_frame": b,
            "detected": bool(mine), "episodes": len(mine),
            "alert_delay_s": _t(mine[0]["origin_frame"], a) if mine else None,
            "final_label": mine[0]["final_label"] if mine else "",
            "final_correct_record": all(x["correct_record"] for x in mine) if mine else None,
            "final_in_named_record": all(x["in_named_record"] for x in mine) if mine else None,
        })
    row["windows_detected"] = sum(w["detected"] for w in windows)
    row["windows"] = len(windows)

    # ---- wrong merges and whether anything is ever undone ----------------------------
    # The final record holding a uid's person-frames is remap_final(uid). A wrong merge is
    # undone only if the two workers' frames end in DIFFERENT final records; with no split
    # operation that cannot happen, and the invariants below check the records directly.
    final_mix: Dict[str, Counter] = {}
    for w, u in assign_log:
        final_mix.setdefault(remap_final(u), Counter())[w] += 1

    def _mix(fu: str) -> str:
        return " ".join(f"{name[w]}:{c}" for w, c in sorted(final_mix.get(fu, Counter()).items()))

    wrong_merges = []
    for k in range(NW):
        by_uid: Dict[str, List[int]] = {}
        for r in R[k]:
            if r["uid"] and owner_first.get(r["uid"]) != k:
                by_uid.setdefault(r["uid"], []).append(r["frame"])
        for U, fr_list in by_uid.items():
            last = max(fr_list)
            fwd = next((r["frame"] for r in R[k] if r["frame"] > last and r["uid"]
                        and owner_first.get(r["uid"]) == k), None)
            fu, flab = final_label(U)
            mixed = final_mix.get(fu, Counter())
            wrong_merges.append({
                "scenario": scen, "seed": seed, "kind": "assignment",
                "misattributed_worker": name[k], "from_uid": U, "into_uid": U,
                "into_owner": name[owner_first[U]], "first_frame": min(fr_list), "last_frame": last,
                "frames": len(fr_list), "forward_corrected_frame": fwd, "final_uid": fu,
                "final_label": flab, "final_record_workers": _mix(fu),
                "still_merged_at_end": mixed[k] > 0 and mixed[owner_first[U]] > 0})
    for f, s, d in [(f, s, d) for f, s, d, _l in cap.live_merges] + [("close", s, d) for s, d in cons]:
        os_, od = owner_first.get(s), owner_first.get(d)
        if os_ is None or od is None or os_ == od:
            continue
        fu, flab = final_label(d)
        mixed = final_mix.get(fu, Counter())
        wrong_merges.append({
            "scenario": scen, "seed": seed,
            "kind": "live_merge" if f != "close" else "consolidation_merge",
            "misattributed_worker": name[os_], "from_uid": s, "into_uid": d, "into_owner": name[od],
            "first_frame": f, "last_frame": "", "frames": sum(1 for w, u in assign_log if u == s),
            "forward_corrected_frame": None, "final_uid": fu, "final_label": flab,
            "final_record_workers": _mix(fu),
            "still_merged_at_end": mixed[os_] > 0 and mixed[od] > 0})
    # empirical separation checks: records never lose frames, episodes move only along merges
    shrank = 0
    for (f0, a0), (f1, a1) in zip(cap.rec_frames, cap.rec_frames[1:]):
        shrank += sum(1 for u, v in a0.items() if u in a1 and a1[u] < v)
    id_shrank = 0
    for (f0, a0), (f1, a1) in zip(cap.id_frames, cap.id_frames[1:]):
        id_shrank += sum(1 for u, v in a0.items() if u in a1 and a1[u] < v)
    merged_at = {s: f for f, s, _d, _l in cap.live_merges}
    reappeared = sum(1 for r in rows if r["uid"] in merged_at and r["frame"] > merged_at[r["uid"]])
    merges_at: Dict[object, list] = {}
    for fr_, s, d, _ph, ok in cap.merge_calls:
        if ok:
            merges_at.setdefault(fr_, []).append((s, d))
    unexplained, returned = 0, 0
    snaps = cap.holders + [("close", cap.final_holders)]
    hist_of: Dict[int, List[str]] = {}
    for (f0, h0), (f1, h1) in zip(snaps, snaps[1:]):
        along = _remap_from(merges_at.get(f1, []))          # merges made while producing f1
        for epid in cap.ep:
            u0, u1 = cap.holder(epid, h0), cap.holder(epid, h1)
            if u0 is None or u1 is None or u0 == u1:
                continue
            if along(u0) != u1:
                unexplained += 1
            seq = hist_of.setdefault(epid, [u0])
            if u1 in seq:
                returned += 1
            seq.append(u1)
    row["wrong_merge_events"] = len(wrong_merges)
    row["wrong_merges_forward_corrected"] = sum(w["forward_corrected_frame"] is not None for w in wrong_merges)
    row["wrong_merges_still_merged_at_end"] = sum(bool(w["still_merged_at_end"]) for w in wrong_merges)
    row["records_frames_seen_decreased"] = shrank
    row["identity_frames_seen_decreased"] = id_shrank
    row["merged_uid_output_again"] = reappeared
    row["episode_moves_not_via_merge"] = unexplained
    row["episodes_returned_to_earlier_record"] = returned

    # ---- scenario checks: does the run exercise what the scenario claims? ------------
    checks, info = {}, {}
    tids = {k: times[k]["bytetrack_ids"] for k in range(NW)}
    tid_workers: Dict[int, set] = {}
    for r in rows:
        if r["tid"] != -1:
            tid_workers.setdefault(r["tid"], set()).add(r["worker"])
    info["bytetrack_ids_covering_two_workers"] = sum(len(v) > 1 for v in tid_workers.values())
    info["frames_box_withheld"] = sum(not r["detected"] for r in rows)
    info["frames_box_nms_suppressed"] = sum(bool(r["nms_suppressed"]) for r in rows)
    info["own_reads_with_badge_partly_covered"] = sum(
        1 for r in rows if r["read_mid"] == marker[r["worker"]] and r["shown"] and not r["badge_visible"])
    # Premises that hold in every scenario: an overpainted badge is never read; a violation
    # is never injected on a frame whose person box was withheld (NMS / drop-occluded); the
    # anonymous-record bookkeeping (label test vs frame test) agrees.
    checks["no_read_while_overpainted"] = not any(
        r["read_mid"] == marker[r["worker"]] and not r["shown"] for r in rows)
    checks["no_violation_frame_withheld"] = not any(r["violating_gt"] and not r["detected"] for r in rows)
    checks["anonymous_record_bookkeeping_consistent"] = bookkeeping_mismatch == 0

    def reads(k, a, b):
        return sum(1 for r in R[k] if a <= r["frame"] < b and r["read_mid"] == marker[k])

    def tids_in(k, a, b):
        return {r["tid"] for r in R[k] if a <= r["frame"] < b and r["tid"] != -1}

    if scen in ("c", "c_persp", "c_named"):
        both = [(r0, r1) for r0, r1 in zip(R[0], R[1])]
        assert all(a["frame"] == b["frame"] for a, b in both)
        ious = [_iou([a["x1"], a["y1"], a["x2"], a["y2"]], [b["x1"], b["y1"], b["x2"], b["y2"]])
                for a, b in both]
        hid = {k: float(np.mean([not r["shown"] for r in R[k]])) for k in range(NW)}
        f_ov = next((a["frame"] for (a, b), v in zip(both, ious) if v > 0), None)
        unbound = ([k for k in range(NW) if not by_frame[f_ov][k]["in_badge_of"]]
                   if f_ov is not None else [])
        checks["boxes_overlap"] = max(ious) >= 0.5
        checks["positions_swapped"] = (R[0][0]["x1"] < R[1][0]["x1"]) and (R[0][-1]["x1"] > R[1][-1]["x1"])
        info.update({"max_box_iou": round(max(ious), 3), "overlap_frames": sum(v > 0 for v in ious),
                     "first_overlap_frame": f_ov,
                     "tracks_unbound_at_first_overlap": len(unbound),
                     "unbound_at_first_overlap": " ".join(name[k] for k in unbound),
                     "hidden_share_A": round(hid[0], 3), "hidden_share_B": round(hid[1], 3),
                     "badge_visible_share_A": round(float(np.mean([r["badge_visible"] for r in R[0]])), 3),
                     "badge_visible_share_B": round(float(np.mean([r["badge_visible"] for r in R[1]])), 3),
                     "bytetrack_ids_A": len(tids[0]), "bytetrack_ids_B": len(tids[1])})
        if scen == "c_named":
            checks["badges_hidden_about_half"] = all(0.4 <= v <= 0.6 for v in hid.values())
            checks["both_named_before_crossing"] = all(
                times[k]["named_frame"] is not None and times[k]["named_frame"] < P.keys["walk_start"]
                for k in range(NW))
        else:
            checks["some_track_unbound_at_first_overlap"] = len(unbound) > 0
    if scen == "a":
        h0, h1 = P.keys["hide_start"], P.keys["hide_end"]
        checks["subject_in_view_throughout"] = bool(P.present[0, P.first(0):].all())
        checks["subject_single_bytetrack_id"] = len(tids[0]) == 1
        checks["badge_not_visible_while_hidden"] = all(not r["badge_visible"] for r in R[0] if h0 <= r["frame"] < h1)
        checks["no_read_while_hidden"] = reads(0, h0, h1) == 0
        checks["read_before_hidden"] = reads(0, 0, h0) > 0
        checks["read_after_hidden"] = reads(0, h1, P.n + 1) > 0
        info["hidden_s"] = round((h1 - h0) / FPS, 3)
    elif scen == "b":
        a_out, a_in, h1 = P.keys["alice_out"], P.keys["alice_in"], P.keys["hide_end"]
        before, after = tids_in(0, 0, a_out), tids_in(0, a_in, P.n + 1)
        checks["absence_is_60_frames"] = a_in - a_out == ABSENCE_B
        checks["new_bytetrack_id_on_return"] = bool(before) and bool(after) and not (before & after)
        checks["companion_absent_at_reentry"] = not P.present[1, a_in:a_in + 6].any()
        checks["badge_not_visible_while_hidden"] = all(not r["badge_visible"] for r in R[0] if a_in <= r["frame"] < h1)
        checks["no_read_while_hidden"] = reads(0, a_in, h1) == 0
        checks["read_before_leaving"] = reads(0, 0, a_out) > 0
        checks["read_after_shown_again"] = reads(0, h1, P.n + 1) > 0
        info["alice_ids_before_after"] = [sorted(before), sorted(after)]
        info["fragment_episodes_open_at_reacquisition"] = sum(
            x["episodes_open_at_reacquisition"] for x in fragments if x["kind"] == "fragment")
    elif scen == "b_close":
        a_out, a_in, a_out2, a_in2 = (P.keys[x] for x in ("alice_out", "alice_in", "alice_out2", "alice_in2"))
        v1, v2, v3 = tids_in(0, 0, a_out), tids_in(0, a_in, a_out2), tids_in(0, a_in2, P.n + 1)
        checks["absences_are_60_frames"] = a_in - a_out == ABSENCE_B and a_in2 - a_out2 == ABSENCE_B
        checks["new_bytetrack_id_on_each_return"] = (all(bool(v) for v in (v1, v2, v3))
                                                     and not (v1 & v2) and not (v2 & v3) and not (v1 & v3))
        checks["companion_absent_at_reentry"] = not P.present[1, a_in:a_in + 6].any()
        checks["companion_in_view_during_fragment_visit"] = bool(P.present[1, a_in:a_out2].any())
        checks["badge_not_visible_during_second_visit"] = all(
            not r["badge_visible"] for r in R[0] if a_in <= r["frame"] < a_out2)
        checks["no_read_during_second_visit"] = reads(0, a_in, a_out2) == 0
        checks["read_before_leaving"] = reads(0, 0, a_out) > 0
        checks["read_after_third_entry"] = reads(0, a_in2, P.n + 1) > 0
        info["alice_ids_per_visit"] = [sorted(v1), sorted(v2), sorted(v3)]
    elif scen in ("d", "d_noreturn"):
        a_out, a_in, b_out = (P.keys[x] for x in ("alice_out", "alice_in", "bob_out"))
        checks["unbadged_marker_never_decoded"] = cap.decoded_counts.get(marker[0], 0) == 0
        checks["unbadged_name_never_bound"] = cap.badge_label_counts.get(name[0], 0) == 0
        checks["unbadged_never_read"] = reads(0, 0, P.n + 1) == 0
        checks["badged_worker_read"] = reads(1, 0, P.n + 1) > 0
        checks["unbadged_returns_while_badged_away"] = not P.present[1, a_in]
        alice_new = not (tids_in(0, 0, a_out) & tids_in(0, a_in, P.n + 1))
        info.update({"alice_away_s": round((a_in - a_out) / FPS, 3),
                     "bob_away_at_alice_return_s": round((a_in - b_out) / FPS, 3)})
        if scen == "d":
            b_in = P.keys["bob_in"]
            checks["new_bytetrack_ids_on_return"] = alice_new and not (
                tids_in(1, 0, b_out) & tids_in(1, b_in, P.n + 1))
            info["bob_away_s"] = round((b_in - b_out) / FPS, 3)
        else:
            checks["new_bytetrack_ids_on_return"] = alice_new
            checks["badged_worker_never_returns"] = not P.present[1, b_out:].any()
    row["checks_passed"] = all(checks.values())
    row["checks_failed"] = " ".join(k for k, v in checks.items() if not v)
    for key, v in info.items():
        row[f"info_{key}"] = v if not isinstance(v, list) else json.dumps(v)

    final_records = [{"uid": w["uid"], "worker": w["worker"], "badge": w["identified_by_badge"],
                      "frames_seen": w["frames_seen"], "episodes": len(w["episodes"])}
                     for w in report["workers"]["per_worker"]]
    return {"row": row, "times": {name[k]: v for k, v in times.items()}, "fm": fm,
            "fragments": fragments, "episodes": episodes, "windows": windows,
            "wrong_names": wrong_events, "wrong_merges": wrong_merges, "checks": checks,
            "info": info, "frames": rows, "final_records": final_records,
            "identity_stats_after_close": report.get("identity_stats", {}),
            "live_merges": cap.live_merges, "consolidation_merges": cons,
            "duplicate_uid_frames": cap.dup_uid_frames}


# --------------------------------------------------------------------------- summary
def _summ(v) -> Optional[dict]:
    v = [float(x) for x in v if x is not None]
    if not v:
        return None
    return {"n": len(v), "median": round(statistics.median(v), 4), "min": round(min(v), 4),
            "max": round(max(v), 4), "mean": round(statistics.mean(v), 4)}


def summarise(results: List[dict], scen: str) -> dict:
    sel = [r for r in results if r["row"]["scenario"] == scen]
    S: dict = {"scenario": scen, "title": SCEN_TITLE[scen], "runs": len(sel),
               "seeds": [r["row"]["seed"] for r in sel]}
    S["time_to_named_s"], S["time_to_first_badge_read_s"], S["reentry"] = {}, {}, {}
    for w in WORKERS:
        nm = w["name"]
        T = [r["times"][nm] for r in sel]
        S["time_to_named_s"][nm] = {"summary": _summ([t["t_named_s"] for t in T]),
                                    "never_named": sum(t["t_named_s"] is None for t in T)}
        S["time_to_first_badge_read_s"][nm] = {"summary": _summ([t["t_first_read_s"] for t in T]),
                                               "never_read": sum(t["t_first_read_s"] is None for t in T)}
        for sx in ("", "2"):
            if not any(f"reentry{sx}_frame" in t for t in T):
                continue
            S["reentry"][nm + (f" (re-entry {sx})" if sx else "")] = {
                "time_to_named_after_reentry_s": _summ([t.get(f"t_named_after_reentry{sx}_s") for t in T]),
                "never_named_after_reentry": sum(t.get(f"t_named_after_reentry{sx}_s") is None for t in T),
                "time_to_badge_read_after_reentry_s": _summ([t.get(f"t_read_after_reentry{sx}_s") for t in T]),
                "commit_at_reentry": dict(Counter(
                    f"{t.get(f'reentry{sx}_commit_source')}:"
                    f"{'own name' if t.get(f'reentry{sx}_commit_label') == nm else (t.get(f'reentry{sx}_commit_label') or 'none')}"
                    for t in T)),
            }
        if scen == "b" and nm == WORKERS[0]["name"]:
            S["reentry"][nm]["time_to_badge_read_after_shown_again_s"] = _summ(
                [t.get("t_read_after_shown_s") for t in T])
        S.setdefault("frames_named_correct_share", {})[nm] = round(
            sum(t["frames_named_correct"] for t in T) / max(1, sum(t["frames_present"] for t in T)), 4)
        S.setdefault("badge_read_frame_share", {})[nm] = round(
            sum(t["frames_badge_read"] for t in T) / max(1, sum(t["frames_present"] for t in T)), 4)
        S.setdefault("badge_read_share_of_visible_frames", {})[nm] = round(
            sum(t["frames_badge_read"] for t in T) / max(1, sum(t["frames_badge_visible"] for t in T)), 4)
        S.setdefault("own_badge_decoded_but_not_bound_to_own_track_frames", {})[nm] = sum(
            t["frames_decoded_not_bound_to_own_track"] for t in T)
    rows = [r["row"] for r in sel]
    wn = [e for r in sel for e in r["wrong_names"]]
    S["wrong_names"] = {
        "person_frames_total": sum(x["wrong_name_frames"] for x in rows),
        "events_total": len(wn),
        "seeds_with_any": sum(x["wrong_name_frames"] > 0 for x in rows),
        "per_seed_frames": _summ([x["wrong_name_frames"] for x in rows]),
        "by_worker": {w["name"]: {"frames": sum(x[f"{w['tag']}_wrong_name_frames"] for x in rows),
                                  "events": sum(x[f"{w['tag']}_wrong_name_events"] for x in rows)}
                      for w in WORKERS},
        "source_at_onset": dict(Counter(e["source_at_onset"] for e in wn)),
        "binder_wrong_label_frames": sum(x["binder_wrong_label_frames"] for x in rows),
        "binder_wrong_bindings": sum(x["binder_wrong_bindings"] for x in rows),
        "binder_misassigned_reads": sum(x["binder_misassigned_reads"] for x in rows),
        "binder_misassigned_reads_to_smaller_box": sum(x["binder_misassigned_reads_to_smaller_box"] for x in rows),
        "duplicate_uid_frames": sum(x["duplicate_uid_frames"] for x in rows),
    }
    S["false_merges"] = {}
    for v in ("live", "consolidated", "final"):
        fm = [r["fm"][v] for r in sel]
        tot, asg = sum(x["false_merges"] for x in fm), sum(x["assignments"] for x in fm)
        ent = {"false_merges_total": tot, "assignments_total": asg,
               "pooled_rate_pct": round(100.0 * tot / max(1, asg), 4),
               "per_seed_rate_pct": _summ([x["false_merge_rate"] for x in fm]),
               "seeds_with_any": sum(x["false_merges"] > 0 for x in fm),
               "fragmentation": _summ([x["fragmentation"] for x in fm])}
        if any(x["reentries"] for x in fm):
            ent["reid_recovered"] = sum(x["recovered"] for x in fm)
            ent["reentries"] = sum(x["reentries"] for x in fm)
        S["false_merges"][v] = ent
    def _anon(kind: str) -> dict:
        fs = [f for r in sel for f in r["fragments"] if f["kind"] == kind]
        return {
            "count": len(fs), "seeds_with_any": len({f["seed"] for f in fs}),
            "created_with_owner_badge_visible_in_probation": sum(f["owner_badge_visible_in_probation"] for f in fs),
            "contaminated": sum(f["contaminated"] for f in fs),
            "at_reacquisition": dict(Counter(f["at_reacquisition"] for f in fs)),
            "at_end_of_session": dict(Counter(f["at_end"] for f in fs)),
            "resolved_by": dict(Counter(f["resolved_by"] for f in fs)),
            "episodes": sum(f["episodes"] for f in fs),
            "episodes_open_at_reacquisition": sum(f["episodes_open_at_reacquisition"] for f in fs),
            "episodes_in_named_record_at_reacquisition": sum(f["episodes_in_named_record_at_reacquisition"] for f in fs),
            "episodes_in_named_record_at_end": sum(f["episodes_in_named_record_at_end"] for f in fs),
            "episodes_opened_after_rename": sum(f["episodes_opened_after_rename"] for f in fs),
        }
    S["fragments"] = _anon("fragment")
    S["initial_anonymous_records"] = _anon("initial_anonymous")
    S["anonymous_identities_of_never_badged_worker"] = _anon("anonymous_identity")
    eps = [e for r in sel for e in r["episodes"]]
    wins = [w for r in sel for w in r["windows"]]
    S["violation_history"] = {
        "windows_injected": len(wins), "windows_detected": sum(w["detected"] for w in wins),
        "alert_delay_s": _summ([w["alert_delay_s"] for w in wins]),
        "episodes_per_window": dict(Counter(w["episodes"] for w in wins)),
        "episodes_total": len(eps),
        "episodes_in_correct_record": sum(e["correct_record"] for e in eps),
        "episodes_in_own_named_record": sum(e["in_named_record"] for e in eps),
        "episodes_in_other_workers_named_record": sum(e["in_other_workers_named_record"] for e in eps),
        "episodes_absorbed_as_duplicate": sum(e["absorbed_as_duplicate"] for e in eps),
        "by_worker_final_label": {w["name"]: dict(Counter(e["final_label"] for e in eps
                                                          if e["origin_worker"] == w["name"]))
                                  for w in WORKERS},
    }
    wm = [w for r in sel for w in r["wrong_merges"]]
    S["wrong_merges"] = {
        "events": len(wm), "by_kind": dict(Counter(w["kind"] for w in wm)),
        "seeds_with_any": sum(x["wrong_merge_events"] > 0 for x in rows),
        "forward_corrected": sum(w["forward_corrected_frame"] is not None for w in wm),
        "still_merged_at_end_of_session": sum(bool(w["still_merged_at_end"]) for w in wm),
        "history_records_frames_seen_ever_decreased": sum(x["records_frames_seen_decreased"] for x in rows),
        "identity_workers_frames_seen_ever_decreased": sum(x["identity_frames_seen_decreased"] for x in rows),
        "merged_away_uid_output_again": sum(x["merged_uid_output_again"] for x in rows),
        "episode_moves_not_via_a_merge": sum(x["episode_moves_not_via_merge"] for x in rows),
        "episodes_returned_to_an_earlier_record": sum(x["episodes_returned_to_earlier_record"] for x in rows),
    }
    if scen in ("c", "c_persp", "c_named"):
        # Outcomes split by how many tracks the binder had NOT yet named when the boxes first
        # overlapped -- the condition the crossing is about.
        strata: Dict[str, dict] = {}
        for r in sel:
            key = str(r["info"].get("tracks_unbound_at_first_overlap"))
            s_ = strata.setdefault(key, {"seeds": [], "wrong_name_frames": 0, "wrong_name_events": 0,
                                         "seeds_with_wrong_name": 0, "false_merges_live": 0,
                                         "false_merges_final": 0, "assignments": 0,
                                         "binder_wrong_bindings": 0, "binder_misassigned_reads": 0})
            s_["seeds"].append(r["row"]["seed"])
            s_["wrong_name_frames"] += r["row"]["wrong_name_frames"]
            s_["wrong_name_events"] += r["row"]["wrong_name_events"]
            s_["seeds_with_wrong_name"] += r["row"]["wrong_name_frames"] > 0
            s_["false_merges_live"] += r["fm"]["live"]["false_merges"]
            s_["false_merges_final"] += r["fm"]["final"]["false_merges"]
            s_["assignments"] += r["fm"]["live"]["assignments"]
            s_["binder_wrong_bindings"] += r["row"]["binder_wrong_bindings"]
            s_["binder_misassigned_reads"] += r["row"]["binder_misassigned_reads"]
        S["by_tracks_unbound_at_first_overlap"] = dict(sorted(strata.items()))
    S["checks"] = {"all_passed": all(x["checks_passed"] for x in rows),
                   "failures": [{"seed": x["seed"], "failed": x["checks_failed"]}
                                for x in rows if not x["checks_passed"]],
                   "per_check_passed": {k: sum(bool(r["checks"][k]) for r in sel)
                                        for k in (sel[0]["checks"] if sel else {})}}
    infos: Dict[str, list] = {}
    for r in sel:
        for k, v in r["info"].items():
            infos.setdefault(k, []).append(v)
    S["info"] = {k: (_summ(v) if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
                     else v) for k, v in infos.items()}
    S["live_merges_total"] = sum(x["n_live_merges"] for x in rows)
    S["consolidation_merges_total"] = sum(x["n_consolidation_merges"] for x in rows)
    S["uids_created"] = _summ([x["uids_created"] for x in rows])
    S["run_s"] = round(sum(x["run_s"] for x in rows), 1)
    return S


# --------------------------------------------------------------------------- provenance
def _find_lines(rel: str, patterns: List[str]) -> List[dict]:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    return [{"file": rel, "pattern": p,
             "lines": [{"line": i + 1, "text": ln.strip()} for i, ln in enumerate(lines) if p in ln]}
            for p in patterns]


def code_evidence() -> dict:
    from src.identity import IdentityManager
    from src.workerlog import WorkerHistory
    from src.workid import WorkIdBinder
    defs = []
    pat = re.compile(r"def\s+\w*(unmerge|split|undo|revert|rollback|unbind|separate)\w*\s*\(", re.I)
    src_dir = os.path.join(ROOT, "phase2", "src")
    for fn in sorted(os.listdir(src_dir)):
        if fn.endswith(".py"):
            with open(os.path.join(src_dir, fn), encoding="utf-8") as fh:
                for i, ln in enumerate(fh.read().splitlines()):
                    if pat.search(ln):
                        defs.append({"file": f"phase2/src/{fn}", "line": i + 1, "text": ln.strip()})
    return {
        "methods": {c.__name__: sorted(n for n, _ in inspect.getmembers(c, inspect.isfunction))
                    for c in (IdentityManager, WorkerHistory, WorkIdBinder)},
        "method_definitions_matching_unmerge_split_undo_revert": defs,
        "quoted_lines": (
            _find_lines("phase5_workid/reid_eval.py", [
                "if uid_owner.setdefault(u, w) != w:", "false_merges += 1",
                "\"false_merge_rate\": 100.0 * false_merges / max(len(assign_log), 1),",
                "def _remap_from"])
            + _find_lines("phase2/src/identity.py", [
                "def _merge_into", "src = self.workers.pop(src_uid, None)",
                "# nothing to merge into; undo",
                "if current is not None and current.uid != uid and not current.marker_confirmed:",
                "self._merge_into(current.uid, uid)", "self._merge_into(src_uid, dst_uid)"])
            + _find_lines("phase2/src/workerlog.py", [
                "def merge(self, src_uid: str, dst_uid: str) -> bool:",
                "del self.records[src_uid]", "dst.episodes.remove(drop)"])
            + _find_lines("phase2/src/workid.py", [
                "if current is not None and current in self._configured and label != current:",
                "cand.append((float(p_area[j]), -c, j))",
                "cand.sort()                                  # smallest box, then highest containment",
                "if abs(a0 - a1) <= 0.05 * max(a0, a1, 1.0) and abs(nc0 - nc1) < 0.05:"])
            + _find_lines("phase2/src/detector.py", [
                "_to_bgr(frame), conf=self.conf, imgsz=self.imgsz,"])
            + _find_lines("phase2/src/pipeline.py", [
                "for src, dst in self.identity.take_merges():",
                "merges = self.identity.consolidate()", "self.history.merge(src, dst)"])),
    }


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception as e:                                   # noqa: BLE001
        return f"unavailable: {e}"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


EMPTY_FIELDS = {
    "fragments.csv": ["scenario", "seed", "uid", "first_label", "kind", "owner", "workers",
                      "contaminated", "created_frame", "frames", "at_reacquisition",
                      "resolved_frame", "at_end", "resolved_by", "episodes",
                      "episodes_in_named_record_at_reacquisition", "episodes_in_named_record_at_end"],
    "wrong_names.csv": ["scenario", "seed", "worker", "wrong_label", "start", "end", "frames",
                        "source_at_onset", "uid", "tid", "badge_label_at_onset"],
    "wrong_merges.csv": ["scenario", "seed", "kind", "misattributed_worker", "from_uid", "into_uid",
                         "into_owner", "first_frame", "last_frame", "frames",
                         "forward_corrected_frame", "final_uid", "final_label",
                         "final_record_workers", "still_merged_at_end"],
}


def _write_csv(path: str, rows: List[dict]) -> None:
    keys: List[str] = [] if rows else list(EMPTY_FIELDS.get(os.path.basename(path), []))
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=keys, restval="")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if v is None else v) for k, v in r.items()})


DEFINITIONS = {
    "time_to_named_s": "first frame whose IdentityResult label equals the worker's configured name, "
                       "minus the frame the worker first appears, / 30. Includes ByteTrack's 1-frame "
                       "activation lag. A name obtained by appearance/position matching counts.",
    "time_to_first_badge_read_s": "first frame in which the worker's OWN marker is decoded and assigned "
                                  "to the worker's own ByteTrack id by WorkIdBinder._assign (captured), "
                                  "minus the first-appearance frame, / 30. 'in_badge_of' (tid in badge_of) "
                                  "is also logged per frame; it includes sticky labels.",
    "wrong_name": "a person-frame whose IdentityResult label is ANOTHER worker's configured name; an "
                  "event is a maximal stretch of that worker's identity outputs on one ByteTrack id that all "
                  "carry the same wrong label (frames with no identity output for the worker -- box "
                  "withheld by the NMS emulation, probation -- neither end nor extend it).",
    "false_merge": "reid_eval._score on (true worker, uid) per identified person-frame, frame order then "
                   "worker order: an assignment whose (remapped) uid is already owned by a different true "
                   "worker (the first logged owns it); rate = false merges / all assignments. live = uids "
                   "as output; consolidated = _remap_from(consolidation merges) [paper's +offline]; final = "
                   "_remap_from(live take_merges + consolidation merges) [the records the report ends with].",
    "anonymous_record": "a uid first output with an anonymous 'Worker N' label. Only its ANONYMOUS period "
                        "is attributed to it: the person-frames output under it while the label was "
                        "'Worker N' (frames, owner = majority true worker of those) and the violation "
                        "episodes opened while it was anonymous. Kinds: 'fragment' = a SECOND record of a "
                        "worker whose badge is read in the run (the worker already had a uid when it was "
                        "created, e.g. a re-entry with the badge hidden); 'initial_anonymous' = the worker's "
                        "FIRST record, created because the badge was hidden when the track was committed, "
                        "renamed in place when the badge is read; 'anonymous_identity' = a record of a worker "
                        "whose badge is never read (d, d_noreturn).",
    "at_reacquisition": "renamed in place (same uid now carries a configured name) or merged live "
                        "(take_merges during process) -- correct if into the owner's name; "
                        "'episodes_in_named_record_at_reacquisition' checks the WorkerHistory record holding "
                        "each of the record's anonymous-period episodes right after that frame; "
                        "'episodes_open_at_reacquisition' = those still open then.",
    "at_end_of_session": "after pipe.close() (consolidation + history.close): the label of the record that "
                         "holds the anonymous record (live + consolidation merges followed).",
    "wrong_merge": "kind 'assignment': a worker's person-frames given a uid first logged for another "
                   "worker; kind 'live_merge'/'consolidation_merge': a merge of two uids first logged for "
                   "different workers. misattributed_worker's data (from_uid) went into into_uid, owned by "
                   "into_owner. forward_corrected_frame = a later frame of that worker under a uid it owns "
                   "(the identity layer stops misattributing; NOT an undo). final_record_workers = the true "
                   "workers whose person-frames end in that final record; still_merged_at_end = both the "
                   "misattributed worker and the owner are among them (i.e. not undone).",
    "undo_invariants": "checked every frame: a WorkerHistory record's frames_seen never decreases; an "
                       "IdentityManager worker's frames_seen never decreases; a uid merged away live is never "
                       "output again; an episode changes record only along a merge made in that frame, and "
                       "never returns to a record it left.",
    "binder": "binder_misassigned_reads = person-frames whose track was assigned ANOTHER worker's marker "
              "this frame (to_smaller_box: the receiving box had a strictly smaller area than the badge "
              "owner's box -- workid._assign gives a tag inside two boxes to the smaller one); "
              "binder_wrong_bindings = distinct (worker, ByteTrack id, wrong name) binder labels, i.e. "
              "tracks the binder bound to another worker's name; "
              "own_badge_decoded_but_not_bound_to_own_track = the worker's marker was decoded somewhere in "
              "the frame but not assigned to its track: on a (re-)entering worker's first frame ByteTrack "
              "outputs no id yet, while two boxes contain the tag the binder's 'too close to call' rule "
              "refuses, or the tag went to the other (smaller) box.",
    "record_owner": "a record whose label is a configured name belongs to that name's worker; an anonymous "
                    "record to the majority true worker of the assignments that end in it.",
    "oracle_boxes": "Person boxes are the full (amodal) extent of each rendered figure, as painted (int() "
                    "rounding: 134 or 135 px wide at 320 px, varying per frame); class-wise NMS at IoU 0.7 is "
                    "emulated (the deployed detector's ultralytics default), keeping the less-occluded box.",
}

CAVEATS = {
    "crossings": "Binder outcomes while two boxes overlap depend on the oracle geometry: (1) boxes are "
                 "full-extent (amodal), so the back worker's box can cover the front worker's badge; a real "
                 "detector's box for a half-hidden person may be tighter; (2) workid._assign gives a tag that "
                 "lies inside two boxes to the SMALLER box (refusing only if the areas are within 5 % AND the "
                 "containments within 0.05). In c / c_named both figures are 320 px, so which box is smaller "
                 "is decided by a 1 px int() rounding of the rendered width (random per frame); c_persp draws "
                 "the back worker 6 % smaller, as consistent perspective would, so his box is always smaller. "
                 "Treat binder mis-assignments in the crossings as conditional on amodal boxes.",
    "c_named": "In c_named both tracks get a sticky badge binding within a few frames of entering, long before "
               "the crossing, and ByteTrack keeps both ids through it, so the per-frame hiding cannot change "
               "any identity output there: its result is fixed before the crossing starts.",
    "oracle_tracking": "Oracle boxes move smoothly (sway + constant-speed walk) and are never missed except by "
                       "the NMS emulation, so ByteTrack's Kalman prediction bridges the crossings easily; a "
                       "real detector's jitter and misses could cause id switches the oracle does not. Whether "
                       "any ByteTrack id ever covered both workers is reported per scenario "
                       "(info.bytetrack_ids_covering_two_workers).",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Part F.2: ArUco badges inside the identity pipeline")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--nseeds", type=int, default=NSEEDS)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    ap.add_argument("--drop-occluded", type=float, default=0.0,
                    help="withhold a person box while less than this fraction of it is unoccluded "
                         "(default 0: the oracle reports every box)")
    ap.add_argument("--nms-iou", type=float, default=NMS_IOU,
                    help="emulate the deployed detector's class-wise NMS at this IoU (default 0.7, the "
                         "ultralytics default Detector.detect runs with); 0 = off")
    ap.add_argument("--no-frames", action="store_true", help="do not write per_frame.csv.gz")
    a = ap.parse_args(argv)
    if not sys.flags.dont_write_bytecode:
        print("refusing to run without `python -B` (no .pyc may be written into phase2/src); "
              "relaunch as: .venv\\Scripts\\python.exe -B tools\\revision\\badge_identity_eval.py ...",
              file=sys.stderr)
        return 2
    assert SIMILAR[WORKERS[0]["marker"] % len(SIMILAR)] is SIMILAR[WORKERS[1]["marker"] % len(SIMILAR)]
    os.makedirs(a.out, exist_ok=True)
    seeds = list(range(a.seed0, a.seed0 + a.nseeds))
    started = datetime.datetime.now().isoformat(timespec="seconds")
    t_all = time.time()
    results, settings = [], None
    jsonl = os.path.join(a.out, "runs.jsonl")
    open(jsonl, "w", encoding="utf-8").close()
    frames_fh = (None if a.no_frames else
                 gzip.open(os.path.join(a.out, "per_frame.csv.gz"), "wt", newline="", encoding="utf-8"))
    frames_wr = None
    for scen in a.scenarios:
        for seed in seeds:
            res, pipe = run_one(scen, seed, a.drop_occluded, a.nms_iou)
            if settings is None:
                settings = settings_of(pipe)
            frames = res.pop("frames")                       # streamed, not kept in memory
            if frames_fh is not None and frames:
                if frames_wr is None:
                    frames_wr = csv.DictWriter(frames_fh, fieldnames=list(frames[0].keys()))
                    frames_wr.writeheader()
                frames_wr.writerows(frames)
                frames_fh.flush()
            results.append(res)
            r = res["row"]
            with open(jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(res, default=_json_default) + "\n")
            print(f"[{datetime.datetime.now():%H:%M:%S}] {scen} seed {seed}: {r['frames']} frames "
                  f"{r['run_s']:.0f} s | named A {r.get('A_t_named_s')} B {r.get('B_t_named_s')} s"
                  f" | wrong-name frames {r['wrong_name_frames']} | false merges live/final "
                  f"{r['fm_live_false_merges']}/{r['fm_final_false_merges']} | fragments "
                  f"{r['fragments']} | checks {'ok' if r['checks_passed'] else 'FAIL ' + r['checks_failed']}",
                  flush=True)
    _write_csv(os.path.join(a.out, "per_seed.csv"), [r["row"] for r in results])
    _write_csv(os.path.join(a.out, "fragments.csv"), [f for r in results for f in r["fragments"]])
    _write_csv(os.path.join(a.out, "episodes.csv"), [e for r in results for e in r["episodes"]])
    _write_csv(os.path.join(a.out, "windows.csv"), [w for r in results for w in r["windows"]])
    _write_csv(os.path.join(a.out, "wrong_names.csv"), [w for r in results for w in r["wrong_names"]])
    _write_csv(os.path.join(a.out, "wrong_merges.csv"), [w for r in results for w in r["wrong_merges"]])
    if frames_fh is not None:
        frames_fh.close()
    import supervision
    meta = {
        "command": " ".join(getattr(sys, "orig_argv", None) or [sys.executable] + sys.argv),
        "cwd": os.getcwd(), "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t_all, 1),
        "seconds_per_scenario": {s: round(sum(r["row"]["run_s"] for r in results
                                              if r["row"]["scenario"] == s), 1) for s in a.scenarios},
        "seeds": [seeds[0], seeds[-1]], "scenarios": a.scenarios, "drop_occluded": a.drop_occluded,
        "nms_iou": a.nms_iou, "python_dont_write_bytecode": bool(sys.flags.dont_write_bytecode),
        "scenario_titles": {s: SCEN_TITLE[s] for s in a.scenarios},
        "variant_of": {s: VARIANT_OF[s] for s in a.scenarios if s in VARIANT_OF},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "opencv": cv2.__version__, "supervision": supervision.__version__,
                     "platform": platform.platform()},
        "threads": {"cv2": cv2.getNumThreads(),
                    **{v: os.environ.get(v) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                                      "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}},
        "git": {"head": _git("rev-parse", "HEAD"), "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "phase2_phase5_status": _git("status", "--porcelain", "--", "phase2", "phase5_workid")
                or "clean"},
        "settings": settings, "fps": FPS, "workers": WORKERS,
        "harness": ("real SafetyPipeline (phase2/src/pipeline.py) with src.detector.Detector replaced "
                    "in-process by a stub returning ground-truth Person (3) and No-Helmet (1) boxes at "
                    "confidence 0.9 (full-extent boxes; class-wise NMS at IoU 0.7 emulated); per-frame "
                    "state captured by wrapping instance methods; pipe.close() at the end"),
    }
    summary = {"meta": meta, "definitions": DEFINITIONS, "caveats": CAVEATS,
               "code_evidence": code_evidence(),
               "scenarios": {s: summarise(results, s) for s in a.scenarios}}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_json_default)
    print(f"wrote {a.out} in {meta['seconds'] / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
