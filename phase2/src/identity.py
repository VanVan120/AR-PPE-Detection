"""Persistent worker identity — one layer above `tracker_id`.

`tracker_id` is *motion* continuity: ByteTrack retires it when a worker is occluded past
`lost_track_buffer` or leaves frame, and the same person returns as a different number.
Everything keyed on it — compliance state, violation history — resets with it. That is
what makes a demo look broken ("Person 3" becomes "Person 11" after walking behind a van).

`IdentityManager` maintains a **worker** above that: a durable identity that survives
track churn. It fuses two signals with a strict precedence:

  1. **ArUco marker (authoritative).** If a tag is visible, that IS the worker, with a
     real name. Never overridden by appearance.
  2. **Appearance (bridging).** No tag visible: match the person's appearance descriptor
     against the gallery of known workers to recover who they were.

Three invariants keep the fusion honest:

  * **Stable uid.** A worker's `uid` is assigned once and never changes. Names change
    (an anonymous "Worker 2" becomes "Alice Tan" when a badge is finally read) and the
    lookup index changes with them, but anything holding a uid — above all the violation
    history in `workerlog.py` — stays attached to the right person across that rename.
  * **Mutual exclusion.** One worker cannot be two people in the same frame. A worker
    already claimed by a visible track is excluded from matching any other track, so a
    lookalike can never duplicate an identity.
  * **Promotion, not duplication.** When a marker confirms someone appearance had been
    calling "Worker 2", the record is renamed **in place** — its history and appearance
    exemplars carry over rather than a second record appearing.

Ambiguity is resolved by *refusing to guess*: a match must clear an absolute similarity
threshold AND beat the runner-up by a margin, otherwise the person becomes a new worker.
Inventing a spurious new worker is a cheap, visible error; silently merging two people is
an expensive, invisible one.

Four mechanisms sharpen the tracking (each off by default here, on in `config.yaml`, so
the published baseline numbers reproduce with the defaults):

  * **Probation** (`probation_frames`): a track must be seen that many times before a
    worker is created or matched for it. Its descriptors are pooled meanwhile. A one-frame
    false detection therefore never becomes a permanent "Worker N", and a match is made
    from several looks at the person rather than the first half-visible one.
  * **Spatio-temporal gating** (`gate`): each worker remembers where it was last seen.
    A worker who was visible at the same time as the new track is two people by
    definition and is excluded outright. A recently-seen worker who could not physically
    have reached the new position is vetoed even when the colours agree; and when
    appearance alone is inconclusive but exactly one recently-seen worker could be here,
    that one is accepted (`source="position"`). Camera motion is compensated: the
    per-frame median displacement of continuing tracks is accumulated as a global image
    shift, and a lost worker's predicted position drifts with it.
  * **Merges** (`take_merges()`, `consolidate()`): a badge that names a track holding an
    anonymous record folds that record into the named one instead of stranding it; and an
    end-of-session pass merges anonymous fragments whose presence intervals never overlap
    (two records visible at the same moment are provably two people). Both report
    (src, dst) uid pairs so the violation history can follow.
  * **Coasting** (`coast_frames` / `coast_s`, `ghosts()`): a worker the tracker just lost
    keeps a predicted, shift-compensated box for a short while, so the label does not
    flicker off and on through a brief occlusion. The overlay draws it visibly different.

**Frames versus seconds.** The manager counts frames, but a live loop rarely processes
every camera frame: a laptop reading a 30 fps webcam at 8 fps would make a gate expressed
in frames four times tighter in real time than intended. So the time-based knobs exist in
both units: `gate_speed` / `gate_horizon` / `coast_frames` are per frame and serve the
synthetic harness; `gate_speed_s` / `gate_horizon_s` / `coast_s` are per second and take
over whenever `update()` is given `now_s`, which every real entry point supplies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .reid import cosine_similarity, crop_person

Box = Tuple[float, float, float, float]


@dataclass
class Worker:
    uid: str                                    # stable handle — never changes
    label: str                                  # display name — may be renamed by a badge
    source: str                                 # how the CURRENT label was established
    exemplars: List[np.ndarray] = field(default_factory=list)
    first_seen: int = 0
    last_seen: int = 0
    frames_seen: int = 0
    marker_confirmed: bool = False              # a real tag has vouched for this identity
    # --- position memory (for gating and coasting) ---
    last_box: Optional[Box] = None
    shift_at_last: Tuple[float, float] = (0.0, 0.0)
    first_box: Optional[Box] = None
    shift_at_first: Tuple[float, float] = (0.0, 0.0)
    last_seen_s: Optional[float] = None         # wall-clock / clip time, when supplied
    first_seen_s: Optional[float] = None
    # --- presence intervals [first_frame, last_frame], for the overlap constraint ---
    intervals: List[List[int]] = field(default_factory=list)

    def centroid(self) -> Optional[np.ndarray]:
        if not self.exemplars:
            return None
        v = np.mean(np.stack(self.exemplars), axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v


@dataclass
class IdentityResult:
    label: str
    uid: str
    source: str                                 # 'marker' | 'appearance' | 'position' | 'new'
    similarity: float = 1.0


@dataclass
class _Pending:
    """A track still in probation: evidence gathered, nothing decided."""
    first_frame: int
    first_box: Optional[Box]
    shift_at_first: Tuple[float, float]
    last_seen: int
    first_s: Optional[float] = None
    frames: int = 0
    embs: List[np.ndarray] = field(default_factory=list)
    last_box: Optional[Box] = None


class IdentityManager:
    """Assign a persistent worker identity to every tracked person, every frame."""

    # A spatial veto yields to DECISIVE appearance evidence: the best candidate must be
    # this similar and beat every other known worker by this margin. Identical kit can
    # never be decisive (everyone scores alike), so lookalikes stay vetoed; a distinctly
    # dressed worker who genuinely reappears far away (a camera swing nobody else was
    # visible to measure, a long walk behind a wall) is not thrown away. Co-visibility is
    # NOT a veto but a proof, and nothing overrides it.
    OVERRIDE_SIM = 0.85
    OVERRIDE_MARGIN = 0.20

    def __init__(self, embedder, match_threshold: float = 0.62, margin: float = 0.04,
                 max_exemplars: int = 8, forget_after: int = 9000,
                 min_box_height: int = 48, refresh_every: int = 15,
                 appearance_enabled: bool = True,
                 probation_frames: int = 1, gate: bool = False,
                 gate_slack: float = 0.5, gate_speed: float = 0.03,
                 gate_horizon: int = 90, gate_floor: float = 0.45,
                 coast_frames: int = 0, consolidate_threshold: Optional[float] = None,
                 consolidate_margin: float = 0.10,
                 gate_speed_s: Optional[float] = None,
                 gate_horizon_s: Optional[float] = None,
                 coast_s: Optional[float] = None):
        self.embedder = embedder
        self.match_threshold = float(match_threshold)
        self.margin = float(margin)
        self.max_exemplars = max(1, int(max_exemplars))
        self.forget_after = max(1, int(forget_after))
        self.min_box_height = max(1, int(min_box_height))
        self.refresh_every = max(1, int(refresh_every))
        self.appearance_enabled = bool(appearance_enabled)
        self.probation_frames = max(1, int(probation_frames))
        self.gate = bool(gate)
        self.gate_slack = max(0.0, float(gate_slack))
        self.gate_speed = max(0.0, float(gate_speed))          # body heights per frame
        self.gate_horizon = max(0, int(gate_horizon))           # frames
        self.gate_floor = float(gate_floor)
        self.coast_frames = max(0, int(coast_frames))
        # Second-based twins, used whenever update() is given `now_s`.
        self.gate_speed_s = None if gate_speed_s is None else max(0.0, float(gate_speed_s))
        self.gate_horizon_s = (None if gate_horizon_s is None
                               else max(0.0, float(gate_horizon_s)))
        self.coast_s = None if coast_s is None else max(0.0, float(coast_s))
        self.consolidate_threshold = (float(consolidate_threshold)
                                      if consolidate_threshold is not None
                                      else self.match_threshold)
        # The offline pass writes the final report, so its mistakes are permanent: it
        # demands a wider lead over the runner-up than the live matcher does.
        self.consolidate_margin = max(self.margin, float(consolidate_margin))
        self.reset()

    def reset(self) -> None:
        self.workers: Dict[str, Worker] = {}          # uid -> Worker
        self._by_label: Dict[str, str] = {}           # label -> uid
        self._track_to_worker: Dict[int, str] = {}    # track_id -> uid
        self._pending: Dict[int, _Pending] = {}       # track_id -> probation evidence
        self._frame_idx = 0
        self._now_s: Optional[float] = None
        self._uid_seq = 0
        self._auto_seq = 0
        self._last_refresh: Dict[str, int] = {}
        self._shift: Tuple[float, float] = (0.0, 0.0)  # accumulated global image shift
        self._prev_boxes: Dict[int, Box] = {}
        self._out_uids: set = set()                   # uids visible in the last update
        self._merges: List[Tuple[str, str]] = []
        self.stats = {"appearance_matches": 0, "marker_bindings": 0,
                      "new_workers": 0, "promotions": 0, "reassignments": 0,
                      "position_matches": 0, "merges": 0}

    # --- public API -----------------------------------------------------------
    def update(self, frame: np.ndarray, track_ids: Sequence[int],
               boxes: Sequence[Sequence[float]],
               marker_labels: Optional[Dict[int, str]] = None,
               now_s: Optional[float] = None) -> Dict[int, IdentityResult]:
        """Resolve identities for the tracks visible this frame.

        Tracks still in probation are absent from the result: nothing is known about
        them yet, and callers fall back to the anonymous track id. `now_s` (seconds,
        wall-clock or clip time) switches the time-based knobs to their per-second
        values; without it the manager counts frames.
        """
        self._frame_idx += 1
        self._now_s = None if now_s is None else float(now_s)
        marker_labels = dict(marker_labels or {})
        track_ids = [int(t) for t in track_ids]
        box_by_tid: Dict[int, Box] = {
            int(t): tuple(float(v) for v in b[:4]) for t, b in zip(track_ids, boxes)}
        self._update_shift(box_by_tid)
        out: Dict[int, IdentityResult] = {}
        claimed: set = set()          # uids already used THIS frame

        # -- pass 1: markers are authoritative --------------------------------
        for tid in track_ids:
            label = marker_labels.get(tid)
            if label is None:
                continue
            uid = self._resolve_marker_identity(tid, label)
            claimed.add(uid)
            w = self.workers[uid]
            pend = self._pending.pop(tid, None)
            if pend is not None:
                # Evidence gathered during probation belongs to this worker now.
                for e in pend.embs:
                    self._add_exemplar(w, e)
                self._touch(w, box_by_tid.get(tid), since=pend.first_frame,
                            since_s=pend.first_s)
            else:
                self._touch(w, box_by_tid.get(tid))
            out[tid] = IdentityResult(w.label, uid, "marker", 1.0)
            self.stats["marker_bindings"] += 1

        # -- pass 2: tracks already bound and still valid ----------------------
        pending: List[int] = []
        for tid in track_ids:
            if tid in out:
                continue
            uid = self._track_to_worker.get(tid)
            if uid is not None and uid in self.workers and uid not in claimed:
                claimed.add(uid)
                w = self.workers[uid]
                self._touch(w, box_by_tid.get(tid))
                out[tid] = IdentityResult(w.label, uid, w.source, 1.0)
            else:
                # Either unseen, or its worker was just claimed by a marker on a
                # different track -- this track's identity is now open again.
                if uid is not None and uid in claimed:
                    self._track_to_worker.pop(tid, None)
                pending.append(tid)

        # -- pass 3: probation, then appearance (+ position) matching ----------
        if pending:
            embeds = self._embed_tracks(frame, pending, box_by_tid)
            for tid in pending:
                emb = embeds.get(tid)
                box = box_by_tid.get(tid)
                p = self._pending.get(tid)
                if p is None:
                    p = _Pending(first_frame=self._frame_idx, first_box=box,
                                 shift_at_first=self._shift, last_seen=self._frame_idx,
                                 first_s=self._now_s)
                    self._pending[tid] = p
                p.frames += 1
                p.last_seen = self._frame_idx
                p.last_box = box
                if emb is not None:
                    p.embs.append(np.asarray(emb, dtype=np.float32))
                if p.frames < self.probation_frames:
                    continue                          # still gathering evidence
                self._pending.pop(tid, None)
                pooled = self._pool(p.embs)
                uid, sim, how = None, 0.0, "new"
                if pooled is not None and self.appearance_enabled:
                    uid, sim, how = self._match(pooled, exclude=claimed, box=p.first_box,
                                                at_frame=p.first_frame, at_s=p.first_s,
                                                shift=p.shift_at_first)
                if uid is None:
                    uid = self._new_worker(None)
                    out[tid] = IdentityResult(self.workers[uid].label, uid, "new", 0.0)
                    self.stats["new_workers"] += 1
                else:
                    out[tid] = IdentityResult(self.workers[uid].label, uid, how, float(sim))
                    self.stats["position_matches" if how == "position"
                               else "appearance_matches"] += 1
                claimed.add(uid)
                self._track_to_worker[tid] = uid
                w = self.workers[uid]
                self._touch(w, box, since=p.first_frame, since_s=p.first_s)
                for e in p.embs:
                    self._add_exemplar(w, e)

        # -- refresh exemplars for stable tracks so appearance drifts with the worker
        if self.appearance_enabled:
            self._refresh_exemplars(frame, out, box_by_tid)
        self._out_uids = {r.uid for r in out.values()}
        self._gc()
        return out

    def label_for_track(self, track_id: int) -> Optional[str]:
        uid = self._track_to_worker.get(int(track_id))
        return self.workers[uid].label if uid in self.workers else None

    def roster(self) -> List[Worker]:
        """Known workers, most recently seen first."""
        return sorted(self.workers.values(), key=lambda w: -w.last_seen)

    def take_merges(self) -> List[Tuple[str, str]]:
        """(src_uid, dst_uid) pairs merged since the last call. The caller folds the
        violation history the same way (`WorkerHistory.merge`)."""
        merges, self._merges = self._merges, []
        return merges

    def ghosts(self) -> List[dict]:
        """Workers the tracker has just lost, with a predicted box: `coast_frames` (or
        `coast_s`, when the manager is given time) worth of coasting so a label survives a
        brief occlusion. Never includes a worker that is visible right now. Each entry:
        uid, label, badge, box (shift-compensated), age (frames), age_s."""
        time_mode = self._now_s is not None and self.coast_s is not None
        if not time_mode and self.coast_frames <= 0:
            return []
        if time_mode and self.coast_s <= 0:
            return []
        rows = []
        for w in self.workers.values():
            if w.uid in self._out_uids or w.last_box is None:
                continue
            age = self._frame_idx - w.last_seen
            if age < 1:
                continue
            age_s = None
            if time_mode and w.last_seen_s is not None:
                age_s = self._now_s - w.last_seen_s
                if age_s <= 0 or age_s > self.coast_s:
                    continue
            elif age > self.coast_frames:
                continue
            dx = self._shift[0] - w.shift_at_last[0]
            dy = self._shift[1] - w.shift_at_last[1]
            x1, y1, x2, y2 = w.last_box
            rows.append({"uid": w.uid, "label": w.label, "badge": w.marker_confirmed,
                         "box": [x1 + dx, y1 + dy, x2 + dx, y2 + dy], "age": int(age),
                         "age_s": None if age_s is None else round(float(age_s), 3)})
        rows.sort(key=lambda r: (r["age"], r["label"]))
        return rows

    def consolidate(self) -> List[Tuple[str, str]]:
        """End-of-session pass: fold anonymous fragments into the records they belong to.

        Two records whose presence intervals overlap were visible at the same moment and
        are therefore two people; everything else is a candidate, decided by the SAME
        rule as the live matcher (threshold + margin on appearance, or a single spatially
        plausible neighbour at the fragment boundary). Greedy, best pair first. A
        badge-confirmed record is never merged away; an anonymous one merges only into a
        record that started earlier or is named. Returns the (src, dst) pairs, which are
        also queued for `take_merges()`.
        """
        done: List[Tuple[str, str]] = []
        while True:
            best = None                                  # (score, src_uid, dst_uid)
            anon = [w for w in self.workers.values()
                    if not w.marker_confirmed and w.exemplars]
            for src in anon:
                src_c = src.centroid()
                if src_c is None:
                    continue
                cands = [w for w in self.workers.values()
                         if w.uid != src.uid and w.exemplars
                         and (w.marker_confirmed or w.first_seen < src.first_seen
                              or (w.first_seen == src.first_seen and w.uid < src.uid))
                         and not _overlaps(src.intervals, w.intervals)]
                if not cands:
                    continue
                uids, scores, recent, plausible = [], [], [], set()
                for w in cands:
                    c = w.centroid()
                    if c is None:
                        continue
                    uids.append(w.uid)
                    scores.append(float(cosine_similarity(src_c[None, :], c[None, :])[0, 0]))
                    if self.gate:
                        verdict = self._boundary_plausible(src, w)
                        if verdict is not None:
                            recent.append(w.uid)
                            if verdict:
                                plausible.add(w.uid)
                uid, score, _how = self._choose(uids, scores, recent, plausible,
                                                threshold=self.consolidate_threshold,
                                                margin=self.consolidate_margin)
                if uid is not None and (best is None or score > best[0]):
                    best = (score, src.uid, uid)
            if best is None:
                return done
            _s, src_uid, dst_uid = best
            self._merge_into(src_uid, dst_uid)
            done.append((src_uid, dst_uid))

    # --- marker handling ------------------------------------------------------
    def _resolve_marker_identity(self, tid: int, label: str) -> str:
        """Bind `tid` to the worker named `label`, merging an anonymous identity into it.

        Three cases:
          * this track currently holds an ANONYMOUS identity and the name is new ->
            rename that record in place, so its history and exemplars carry over;
          * the name is new and the track has no anonymous record -> create the worker;
          * the name already exists -> reuse it, taking the track from any other holder;
            an anonymous record the track was holding is MERGED into the named one.
        """
        existing_uid = self._by_label.get(label)
        current_uid = self._track_to_worker.get(tid)
        current = self.workers.get(current_uid) if current_uid else None

        if existing_uid is None and current is not None and not current.marker_confirmed:
            self._by_label.pop(current.label, None)
            current.label = label
            current.source = "marker"
            current.marker_confirmed = True
            self._by_label[label] = current.uid
            self.stats["promotions"] += 1
            uid = current.uid
        elif existing_uid is None:
            uid = self._new_uid()
            self.workers[uid] = Worker(uid=uid, label=label, source="marker",
                                       first_seen=self._frame_idx,
                                       last_seen=self._frame_idx, marker_confirmed=True)
            self._by_label[label] = uid
        else:
            uid = existing_uid
            w = self.workers[uid]
            w.marker_confirmed = True
            w.source = "marker"
            if current is not None and current.uid != uid and not current.marker_confirmed:
                # The person we had been calling "Worker N" IS this named worker: fold
                # the anonymous record in rather than leaving it stranded.
                self._merge_into(current.uid, uid)
            # The tag is ground truth: if another track was wearing this identity, it
            # loses it rather than the system showing two people with one name.
            for t, k in list(self._track_to_worker.items()):
                if k == uid and t != tid:
                    self._track_to_worker.pop(t, None)
                    self.stats["reassignments"] += 1
        self._track_to_worker[tid] = uid
        return uid

    # --- appearance handling --------------------------------------------------
    def _embed_tracks(self, frame, tids: Sequence[int],
                      box_by_tid: Dict[int, Sequence[float]]) -> Dict[int, np.ndarray]:
        crops, keep = [], []
        for tid in tids:
            box = box_by_tid.get(tid)
            if box is None or not self._box_ok(box):
                continue
            c = crop_person(frame, box)
            if c is None:
                continue
            crops.append(c)
            keep.append(tid)
        if not crops:
            return {}
        vecs = self.embedder.embed(crops)
        return {tid: vecs[i] for i, tid in enumerate(keep)}

    def _box_ok(self, box: Sequence[float]) -> bool:
        """Reject boxes too small or too oddly shaped to describe reliably. A distant or
        half-merged box yields a descriptor that pollutes the gallery."""
        x1, y1, x2, y2 = (float(v) for v in box[:4])
        h, w = y2 - y1, x2 - x1
        if h < self.min_box_height or w <= 1:
            return False
        return 0.15 <= (w / h) <= 1.6           # upright-ish person

    @staticmethod
    def _pool(embs: List[np.ndarray]) -> Optional[np.ndarray]:
        """The descriptor a probation period produces: the normalised mean. A single
        embedding is returned untouched so the one-frame path is bit-identical to the
        original behaviour."""
        if not embs:
            return None
        if len(embs) == 1:
            return embs[0]
        v = np.mean(np.stack(embs), axis=0).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def _match(self, emb: np.ndarray, exclude: set, box: Optional[Box] = None,
               at_frame: Optional[int] = None, at_s: Optional[float] = None,
               shift: Tuple[float, float] = (0.0, 0.0)) -> Tuple[Optional[str], float, str]:
        """Best gallery match subject to the threshold AND a runner-up margin, with the
        spatio-temporal gate applied when enabled. Returns (uid, similarity, how)."""
        uids = [u for u, w in self.workers.items() if u not in exclude and w.exemplars]
        if self.gate and at_frame is not None:
            # Co-visibility is proof of two people: a worker seen on or after the frame
            # this track first appeared (while not being this track) is not a candidate
            # at all, and no appearance evidence can bring them back.
            uids = [u for u in uids if self.workers[u].last_seen < at_frame]
        if not uids:
            return None, 0.0, "new"
        scores = []
        for u in uids:
            gallery = np.stack(self.workers[u].exemplars)
            scores.append(float(cosine_similarity(emb[None, :], gallery).max()))
        recent: List[str] = []
        plausible: set = set()
        if self.gate and box is not None and at_frame is not None:
            for u in uids:
                w = self.workers[u]
                if w.last_box is None:
                    continue
                beyond, travel = self._gap(w.last_seen, w.last_seen_s, at_frame, at_s)
                if beyond:
                    continue                          # too old to carry position info
                recent.append(u)
                if self._plausible(w.last_box, w.shift_at_last, box, shift, travel):
                    plausible.add(u)
        return self._choose(uids, scores, recent, plausible, self.match_threshold)

    def _choose(self, uids: List[str], scores: List[float], recent: List[str],
                plausible: set, threshold: float,
                margin: Optional[float] = None) -> Tuple[Optional[str], float, str]:
        """The decision rule shared by live matching and consolidation."""
        if not uids:
            return None, 0.0, "new"
        margin = self.margin if margin is None else float(margin)
        vetoed = {u for u in recent if u not in plausible}
        if vetoed:
            # Decisive appearance beats an impossible-looking position (see OVERRIDE_*).
            uid_all, best_all = self._decide(uids, scores, threshold, margin)
            if uid_all in vetoed and best_all >= self.OVERRIDE_SIM and len(scores) >= 2:
                second = sorted(scores, reverse=True)[1]
                if best_all - second >= self.OVERRIDE_MARGIN:
                    return uid_all, best_all, "appearance"
        pool = [(u, s) for u, s in zip(uids, scores) if u not in vetoed]
        uid, best = self._decide([u for u, _ in pool], [s for _, s in pool], threshold,
                                 margin)
        if uid is not None:
            return uid, best, "appearance"
        if len(plausible) == 1:
            (pu,) = tuple(plausible)
            ps = scores[uids.index(pu)]
            # Position may only settle a choice appearance could not: the lone plausible
            # worker must also LOOK the most like this person of everyone known, and
            # clear the floor. A stranger stepping into a vacated spot fails both.
            if ps >= self.gate_floor and ps >= max(scores):
                return pu, ps, "position"
        return None, (max(scores) if scores else 0.0), "new"

    def _decide(self, uids: List[str], scores: List[float], threshold: float,
                margin: Optional[float] = None) -> Tuple[Optional[str], float]:
        if not uids:
            return None, 0.0
        margin = self.margin if margin is None else float(margin)
        order = np.argsort(scores)[::-1]
        best_i = int(order[0])
        best = scores[best_i]
        if best < threshold:
            return None, best
        if len(scores) > 1:
            second = scores[int(order[1])]
            if (best - second) < margin:
                return None, best              # ambiguous -> refuse to guess
        return uids[best_i], best

    def _gap(self, last_frame: int, last_s: Optional[float], at_frame: int,
             at_s: Optional[float]) -> Tuple[bool, float]:
        """How long a worker has been gone, as (beyond the horizon?, allowed travel in
        body heights). Seconds when the manager has been given time and the per-second
        knobs exist; frames otherwise."""
        if (at_s is not None and last_s is not None
                and (self.gate_speed_s is not None or self.gate_horizon_s is not None)):
            dt_s = max(0.0, float(at_s) - float(last_s))
            horizon_s = (self.gate_horizon_s if self.gate_horizon_s is not None
                         else self.gate_horizon / 30.0)
            speed_s = (self.gate_speed_s if self.gate_speed_s is not None
                       else self.gate_speed * 30.0)
            return dt_s > horizon_s, speed_s * dt_s
        dt = max(0, int(at_frame) - int(last_frame))
        return dt > self.gate_horizon, self.gate_speed * dt

    def _plausible(self, last_box: Box, shift_then: Tuple[float, float], box: Box,
                   shift_now: Tuple[float, float], travel: float) -> bool:
        """Could the worker last seen at `last_box` be at `box` now, having had `travel`
        body heights of movement allowance?

        The prediction drifts with the accumulated camera shift; the tolerance is the slack
        plus the allowance, in units of body height so it is independent of resolution and
        distance. A person cannot halve or double in height in the process either.
        """
        lx1, ly1, lx2, ly2 = last_box
        x1, y1, x2, y2 = box
        lh, nh = ly2 - ly1, y2 - y1
        if lh <= 0 or nh <= 0:
            return False
        if not (0.5 <= nh / lh <= 2.0):
            return False
        dx = shift_now[0] - shift_then[0]
        dy = shift_now[1] - shift_then[1]
        pcx, pcy = (lx1 + lx2) / 2.0 + dx, (ly1 + ly2) / 2.0 + dy
        ncx, ncy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        h = 0.5 * (lh + nh)
        radius = h * (self.gate_slack + max(0.0, travel))
        return math.hypot(ncx - pcx, ncy - pcy) <= radius

    def _boundary_plausible(self, src: Worker, cand: Worker) -> Optional[bool]:
        """Position check across the gap between two non-overlapping records. None when
        the pair carries no position information (interleaved records, or a gap longer
        than the horizon)."""
        if cand.last_seen < src.first_seen:
            a_box, a_shift, a_frame, a_s = (cand.last_box, cand.shift_at_last,
                                            cand.last_seen, cand.last_seen_s)
            b_box, b_shift, b_frame, b_s = (src.first_box, src.shift_at_first,
                                            src.first_seen, src.first_seen_s)
        elif cand.first_seen > src.last_seen:
            a_box, a_shift, a_frame, a_s = (src.last_box, src.shift_at_last,
                                            src.last_seen, src.last_seen_s)
            b_box, b_shift, b_frame, b_s = (cand.first_box, cand.shift_at_first,
                                            cand.first_seen, cand.first_seen_s)
        else:
            return None
        if a_box is None or b_box is None:
            return None
        beyond, travel = self._gap(a_frame, a_s, b_frame, b_s)
        if beyond:
            return None
        return self._plausible(a_box, a_shift, b_box, b_shift, travel)

    def _new_uid(self) -> str:
        self._uid_seq += 1
        return f"w{self._uid_seq}"

    def _new_worker(self, emb: Optional[np.ndarray]) -> str:
        self._auto_seq += 1
        uid = self._new_uid()
        label = f"Worker {self._auto_seq}"
        w = Worker(uid=uid, label=label, source="new",
                   first_seen=self._frame_idx, last_seen=self._frame_idx)
        if emb is not None:
            w.exemplars.append(np.asarray(emb, dtype=np.float32))
        self.workers[uid] = w
        self._by_label[label] = uid
        return uid

    def _add_exemplar(self, w: Worker, emb: np.ndarray) -> None:
        """Keep a small, DIVERSE exemplar set. Storing near-duplicates would fill the
        buffer with one pose and lose the appearance range that makes re-matching work."""
        emb = np.asarray(emb, dtype=np.float32)
        if w.exemplars:
            sims = cosine_similarity(emb[None, :], np.stack(w.exemplars))
            if float(sims.max()) > 0.985:        # already represented
                return
        w.exemplars.append(emb)
        if len(w.exemplars) > self.max_exemplars:
            w.exemplars.pop(0)

    def _refresh_exemplars(self, frame, out: Dict[int, IdentityResult],
                           box_by_tid: Dict[int, Sequence[float]]) -> None:
        due = [tid for tid, res in out.items()
               if self._frame_idx - self._last_refresh.get(res.uid, -10 ** 9)
               >= self.refresh_every]
        if not due:
            return
        embeds = self._embed_tracks(frame, due, box_by_tid)
        for tid, emb in embeds.items():
            w = self.workers.get(out[tid].uid)
            if w is not None:
                self._add_exemplar(w, emb)
                self._last_refresh[w.uid] = self._frame_idx

    # --- position memory ------------------------------------------------------
    def _update_shift(self, box_by_tid: Dict[int, Box]) -> None:
        """Accumulate the global image shift from tracks that continued from the previous
        frame: the median of their centre displacements. With several tracks this is the
        camera's motion; with one it also carries that person's walk, which the gate's
        slack absorbs. No continuing track -> the shift is simply carried forward."""
        if self._prev_boxes:
            dxs, dys = [], []
            for tid, b in box_by_tid.items():
                pb = self._prev_boxes.get(tid)
                if pb is None:
                    continue
                dxs.append((b[0] + b[2]) / 2.0 - (pb[0] + pb[2]) / 2.0)
                dys.append((b[1] + b[3]) / 2.0 - (pb[1] + pb[3]) / 2.0)
            if dxs:
                self._shift = (self._shift[0] + float(np.median(dxs)),
                               self._shift[1] + float(np.median(dys)))
        self._prev_boxes = dict(box_by_tid)

    def _touch(self, w: Worker, box: Optional[Box] = None,
               since: Optional[int] = None, since_s: Optional[float] = None) -> None:
        w.last_seen = self._frame_idx
        w.frames_seen += 1
        if self._now_s is not None:
            w.last_seen_s = self._now_s
            if w.first_seen_s is None:
                w.first_seen_s = self._now_s if since_s is None else float(since_s)
        if box is not None:
            w.last_box = box
            w.shift_at_last = self._shift
            if w.first_box is None:
                w.first_box = box
                w.shift_at_first = self._shift
        start = self._frame_idx if since is None else min(int(since), self._frame_idx)
        if w.intervals and w.intervals[-1][1] >= start - 1:
            w.intervals[-1][1] = max(w.intervals[-1][1], self._frame_idx)
        else:
            w.intervals.append([start, self._frame_idx])

    def _merge_into(self, src_uid: str, dst_uid: str) -> None:
        """Fold worker `src` into `dst`: gallery, presence, position memory, bindings."""
        src = self.workers.pop(src_uid, None)
        dst = self.workers.get(dst_uid)
        if src is None or dst is None:
            if src is not None:
                self.workers[src_uid] = src           # nothing to merge into; undo
            return
        if self._by_label.get(src.label) == src_uid:
            self._by_label.pop(src.label, None)
        for e in src.exemplars:
            self._add_exemplar(dst, e)
        dst.frames_seen += src.frames_seen
        if src.first_seen < dst.first_seen or dst.first_box is None:
            dst.first_seen = min(dst.first_seen, src.first_seen)
            if src.first_box is not None:
                dst.first_box, dst.shift_at_first = src.first_box, src.shift_at_first
            if src.first_seen_s is not None and (dst.first_seen_s is None
                                                 or src.first_seen_s < dst.first_seen_s):
                dst.first_seen_s = src.first_seen_s
        if src.last_seen > dst.last_seen or dst.last_box is None:
            dst.last_seen = max(dst.last_seen, src.last_seen)
            if src.last_box is not None:
                dst.last_box, dst.shift_at_last = src.last_box, src.shift_at_last
            if src.last_seen_s is not None and (dst.last_seen_s is None
                                                or src.last_seen_s > dst.last_seen_s):
                dst.last_seen_s = src.last_seen_s
        dst.intervals = _union(dst.intervals + src.intervals)
        for t, k in list(self._track_to_worker.items()):
            if k == src_uid:
                self._track_to_worker[t] = dst_uid
        self._last_refresh.pop(src_uid, None)
        self._out_uids.discard(src_uid)
        self._merges.append((src_uid, dst_uid))
        self.stats["merges"] += 1

    # --- housekeeping ---------------------------------------------------------
    def _gc(self) -> None:
        """Forget workers unseen for a long time. Marker-confirmed workers are kept:
        a named person who steps away for ten minutes should still be recognised on
        return, and there are only ever a handful of them. Pending tracks that vanished
        during probation are dropped too, so a recycled id cannot inherit their evidence."""
        stale = [u for u, w in self.workers.items()
                 if not w.marker_confirmed
                 and (self._frame_idx - w.last_seen) > self.forget_after]
        for u in stale:
            w = self.workers.pop(u, None)
            if w is not None and self._by_label.get(w.label) == u:
                self._by_label.pop(w.label, None)
            self._last_refresh.pop(u, None)
            for t, wk in list(self._track_to_worker.items()):
                if wk == u:
                    self._track_to_worker.pop(t, None)
        for tid in [t for t, p in self._pending.items()
                    if self._frame_idx - p.last_seen > self.probation_frames]:
            self._pending.pop(tid, None)


def _overlaps(a: List[List[int]], b: List[List[int]]) -> bool:
    """Do two sorted interval lists share any frame?"""
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i][0] <= b[j][1] and b[j][0] <= a[i][1]:
            return True
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return False


def _union(intervals: List[List[int]]) -> List[List[int]]:
    """Sort and coalesce intervals (touching or overlapping ones become one)."""
    out: List[List[int]] = []
    for s, e in sorted((list(iv) for iv in intervals), key=lambda iv: iv[0]):
        if out and s <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
