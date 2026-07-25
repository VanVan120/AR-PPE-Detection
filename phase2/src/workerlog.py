"""Per-worker violation history — turn a stream of events into a per-person record.

`eventlog.py` writes one JSONL row per violation as it fires: good for an audit trail,
useless for answering the question an inspector actually asks — *"who was unsafe, how
often, and for how long?"*

`WorkerHistory` accumulates that. It keys everything on the identity **uid** from
`identity.py`, never on `tracker_id` and never on the display name:

  * `tracker_id` dies every time the tracker loses someone, which would split one
    worker's record into a dozen fragments;
  * the display name *changes* when a badge promotes "Worker 2" to "Alice Tan", and
    keying on it would strand everything recorded before the badge was read.

A violation is stored as an **episode** (start -> end), not a counter, so the report can
give real durations — "Alice Tan: 2 no-helmet episodes, 47 s non-compliant" — rather than
a bare count that says nothing about severity of exposure. An episode still open when the
session ends is closed at the final timestamp and marked, so its duration is never
silently reported as zero.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Episode:
    class_name: str
    label: str
    severity: str
    start_frame: int
    start_s: float
    end_frame: Optional[int] = None
    end_s: Optional[float] = None
    truncated: bool = False          # still open when the session ended

    @property
    def closed(self) -> bool:
        return self.end_s is not None

    @property
    def duration_s(self) -> float:
        return max(0.0, float(self.end_s) - float(self.start_s)) if self.closed else 0.0

    def duration_upto(self, now_s: Optional[float] = None) -> float:
        """Duration including an episode that is still running.

        Lets a live view show a violation's elapsed time WITHOUT closing it. Closing an
        open episode just to read it would end it prematurely, and the next frame would
        open a fresh one — so merely looking at the report would shatter one long
        violation into a string of short ones.
        """
        if self.closed:
            return self.duration_s
        if now_s is None:
            return 0.0
        return max(0.0, float(now_s) - float(self.start_s))


@dataclass
class WorkerRecord:
    uid: str
    label: str
    marker_confirmed: bool = False
    first_s: float = 0.0
    last_s: float = 0.0
    frames_seen: int = 0
    frames_violating: int = 0
    episodes: List[Episode] = field(default_factory=list)

    @property
    def observed_s(self) -> float:
        return max(0.0, self.last_s - self.first_s)

    @property
    def violation_s(self) -> float:
        return sum(e.duration_s for e in self.episodes)

    def violation_s_at(self, now_s=None) -> float:
        return sum(e.duration_upto(now_s) for e in self.episodes)

    @property
    def compliance_pct(self) -> float:
        """Share of the frames this worker was visible in which they were compliant.

        Frame-based, not time-based: a worker who is only visible intermittently should
        be judged on the frames where we could actually see them, not on wall-clock.
        """
        if self.frames_seen <= 0:
            return 100.0
        return 100.0 * (1.0 - self.frames_violating / self.frames_seen)

    def by_type(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for e in self.episodes:
            out[e.class_name] = out.get(e.class_name, 0) + 1
        return out


class WorkerHistory:
    def __init__(self):
        self.records: Dict[str, WorkerRecord] = {}
        self._open: Dict[tuple, Episode] = {}          # (uid, class_name) -> Episode
        self._started = False

    # --- ingest ---------------------------------------------------------------
    def update(self, frame_no: int, elapsed_s: float, frame_compliance,
               identity_by_track: Dict[int, object]) -> None:
        """Fold one frame of compliance + identity into the per-worker records."""
        active_now: Dict[tuple, object] = {}

        for person in getattr(frame_compliance, "persons", []):
            ident = identity_by_track.get(person.tracker_id)
            if ident is None:
                continue                                # not identified this frame
            uid = getattr(ident, "uid", None)
            if uid is None:
                continue
            rec = self._record(uid, getattr(ident, "label", uid), elapsed_s)
            # The name can change under us (badge promotion); the uid cannot.
            rec.label = getattr(ident, "label", rec.label)
            if getattr(ident, "source", "") == "marker":
                rec.marker_confirmed = True
            rec.frames_seen += 1
            rec.last_s = float(elapsed_s)
            if person.active:
                rec.frames_violating += 1
            for av in person.active:
                active_now[(uid, av.class_name)] = av

        # close episodes that ended
        for key in [k for k in self._open if k not in active_now]:
            ep = self._open.pop(key)
            ep.end_frame, ep.end_s = int(frame_no), float(elapsed_s)

        # open episodes that started
        for key, av in active_now.items():
            if key in self._open:
                continue
            uid, class_name = key
            ep = Episode(class_name=class_name, label=getattr(av, "label", class_name),
                         severity=getattr(av, "severity", "high"),
                         start_frame=int(frame_no), start_s=float(elapsed_s))
            self._open[key] = ep
            rec = self.records.get(uid)
            if rec is not None:
                rec.episodes.append(ep)

    def close(self, elapsed_s: float, frame_no: int = -1) -> None:
        """End the session: close every still-open episode so no duration reads as 0."""
        for key in list(self._open):
            ep = self._open.pop(key)
            ep.end_frame, ep.end_s = int(frame_no), float(elapsed_s)
            ep.truncated = True

    def _record(self, uid: str, label: str, elapsed_s: float) -> WorkerRecord:
        rec = self.records.get(uid)
        if rec is None:
            rec = WorkerRecord(uid=uid, label=label, first_s=float(elapsed_s),
                               last_s=float(elapsed_s))
            self.records[uid] = rec
        return rec

    # --- output ---------------------------------------------------------------
    def report(self, now_s=None) -> dict:
        """Snapshot of the session. Pass `now_s` to include the elapsed time of episodes
        that are still open; this NEVER mutates state, so it is safe to poll live."""
        workers = []
        for rec in sorted(self.records.values(),
                          key=lambda r: (-r.violation_s_at(now_s), -r.frames_violating,
                                         r.label)):
            workers.append({
                "uid": rec.uid,
                "worker": rec.label,
                "identified_by_badge": rec.marker_confirmed,
                "frames_seen": rec.frames_seen,
                "observed_s": round(rec.observed_s, 1),
                "violation_episodes": len(rec.episodes),
                "violation_s": round(rec.violation_s_at(now_s), 1),
                "compliance_pct": round(rec.compliance_pct, 1),
                "by_type": rec.by_type(),
                "episodes": [
                    {"violation": e.class_name, "severity": e.severity,
                     "start_s": round(e.start_s, 1),
                     "end_s": round(e.end_s, 1) if e.closed else None,
                     "duration_s": round(e.duration_upto(now_s), 1),
                     "ongoing": not e.closed,
                     "truncated": e.truncated}
                    for e in rec.episodes
                ],
            })
        total_eps = sum(len(r.episodes) for r in self.records.values())
        return {
            "workers_seen": len(self.records),
            "identified_by_badge": sum(1 for r in self.records.values() if r.marker_confirmed),
            "total_violation_episodes": total_eps,
            "total_violation_s": round(
                sum(r.violation_s_at(now_s) for r in self.records.values()), 1),
            "per_worker": workers,
        }

    def format_report(self, width: int = 68, now_s=None) -> str:
        """A console table for the end-of-session summary."""
        rep = self.report(now_s)
        if not rep["per_worker"]:
            return "No workers were identified this session."
        lines = ["", "=" * width, "PER-WORKER SAFETY REPORT", "=" * width,
                 f"{'worker':<20}{'seen':>7}{'events':>8}{'unsafe':>9}{'compliant':>11}",
                 "-" * width]
        for w in rep["per_worker"]:
            name = w["worker"] + ("" if w["identified_by_badge"] else " *")
            lines.append(f"{name[:20]:<20}{w['observed_s']:>6.0f}s"
                         f"{w['violation_episodes']:>8}"
                         f"{w['violation_s']:>8.0f}s"
                         f"{w['compliance_pct']:>10.0f}%")
            for cls, n in sorted(w["by_type"].items()):
                lines.append(f"    - {cls}: {n}")
        lines.append("-" * width)
        lines.append(f"{rep['workers_seen']} worker(s), "
                     f"{rep['identified_by_badge']} confirmed by badge, "
                     f"{rep['total_violation_episodes']} violation episode(s), "
                     f"{rep['total_violation_s']:.0f}s unsafe in total")
        lines.append("* = identity from appearance only (no badge seen) — treat the name "
                     "as provisional")
        lines.append("=" * width)
        return "\n".join(lines)

    def save_json(self, path: str, now_s=None) -> None:
        if not path:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.report(now_s), fh, indent=2, ensure_ascii=False)
