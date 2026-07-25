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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .reid import cosine_similarity, crop_person


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


@dataclass
class IdentityResult:
    label: str
    uid: str
    source: str                                 # 'marker' | 'appearance' | 'new'
    similarity: float = 1.0


class IdentityManager:
    """Assign a persistent worker identity to every tracked person, every frame."""

    def __init__(self, embedder, match_threshold: float = 0.62, margin: float = 0.04,
                 max_exemplars: int = 8, forget_after: int = 9000,
                 min_box_height: int = 48, refresh_every: int = 15,
                 appearance_enabled: bool = True):
        self.embedder = embedder
        self.match_threshold = float(match_threshold)
        self.margin = float(margin)
        self.max_exemplars = max(1, int(max_exemplars))
        self.forget_after = max(1, int(forget_after))
        self.min_box_height = max(1, int(min_box_height))
        self.refresh_every = max(1, int(refresh_every))
        self.appearance_enabled = bool(appearance_enabled)
        self.reset()

    def reset(self) -> None:
        self.workers: Dict[str, Worker] = {}          # uid -> Worker
        self._by_label: Dict[str, str] = {}           # label -> uid
        self._track_to_worker: Dict[int, str] = {}    # track_id -> uid
        self._frame_idx = 0
        self._uid_seq = 0
        self._auto_seq = 0
        self._last_refresh: Dict[str, int] = {}
        self.stats = {"appearance_matches": 0, "marker_bindings": 0,
                      "new_workers": 0, "promotions": 0, "reassignments": 0}

    # --- public API -----------------------------------------------------------
    def update(self, frame: np.ndarray, track_ids: Sequence[int],
               boxes: Sequence[Sequence[float]],
               marker_labels: Optional[Dict[int, str]] = None) -> Dict[int, IdentityResult]:
        """Resolve identities for the tracks visible this frame."""
        self._frame_idx += 1
        marker_labels = dict(marker_labels or {})
        track_ids = [int(t) for t in track_ids]
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
            self._touch(w)
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
                self._touch(w)
                out[tid] = IdentityResult(w.label, uid, w.source, 1.0)
            else:
                # Either unseen, or its worker was just claimed by a marker on a
                # different track -- this track's identity is now open again.
                if uid is not None and uid in claimed:
                    self._track_to_worker.pop(tid, None)
                pending.append(tid)

        # -- pass 3: appearance matching for the rest -------------------------
        box_by_tid = {int(t): b for t, b in zip(track_ids, boxes)}
        if pending:
            embeds = self._embed_tracks(frame, pending, box_by_tid)
            for tid in pending:
                emb = embeds.get(tid)
                uid, sim = (None, 0.0)
                if emb is not None and self.appearance_enabled:
                    uid, sim = self._match(emb, exclude=claimed)
                if uid is None:
                    uid = self._new_worker(emb)
                    out[tid] = IdentityResult(self.workers[uid].label, uid, "new", 0.0)
                    self.stats["new_workers"] += 1
                else:
                    out[tid] = IdentityResult(self.workers[uid].label, uid,
                                              "appearance", float(sim))
                    self.stats["appearance_matches"] += 1
                claimed.add(uid)
                self._track_to_worker[tid] = uid
                w = self.workers[uid]
                self._touch(w)
                if emb is not None:
                    self._add_exemplar(w, emb)

        # -- refresh exemplars for stable tracks so appearance drifts with the worker
        if self.appearance_enabled:
            self._refresh_exemplars(frame, out, box_by_tid)
        self._gc()
        return out

    def label_for_track(self, track_id: int) -> Optional[str]:
        uid = self._track_to_worker.get(int(track_id))
        return self.workers[uid].label if uid in self.workers else None

    def roster(self) -> List[Worker]:
        """Known workers, most recently seen first."""
        return sorted(self.workers.values(), key=lambda w: -w.last_seen)

    # --- marker handling ------------------------------------------------------
    def _resolve_marker_identity(self, tid: int, label: str) -> str:
        """Bind `tid` to the worker named `label`, merging an anonymous identity into it.

        Three cases:
          * this track currently holds an ANONYMOUS identity and the name is new ->
            rename that record in place, so its history and exemplars carry over;
          * the name is new and the track has no anonymous record -> create the worker;
          * the name already exists -> reuse it, taking the track from any other holder.
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

    def _match(self, emb: np.ndarray, exclude: set) -> Tuple[Optional[str], float]:
        """Best gallery match subject to the threshold AND a runner-up margin."""
        uids = [u for u, w in self.workers.items() if u not in exclude and w.exemplars]
        if not uids:
            return None, 0.0
        scores = []
        for u in uids:
            gallery = np.stack(self.workers[u].exemplars)
            scores.append(float(cosine_similarity(emb[None, :], gallery).max()))
        order = np.argsort(scores)[::-1]
        best_i = int(order[0])
        best = scores[best_i]
        if best < self.match_threshold:
            return None, best
        if len(scores) > 1:
            second = scores[int(order[1])]
            if (best - second) < self.margin:
                return None, best              # ambiguous -> refuse to guess
        return uids[best_i], best

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

    # --- housekeeping ---------------------------------------------------------
    def _touch(self, w: Worker) -> None:
        w.last_seen = self._frame_idx
        w.frames_seen += 1

    def _gc(self) -> None:
        """Forget workers unseen for a long time. Marker-confirmed workers are kept:
        a named person who steps away for ten minutes should still be recognised on
        return, and there are only ever a handful of them."""
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
