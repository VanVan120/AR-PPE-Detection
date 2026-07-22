"""Worker-attributed violation event log (JSONL).

One JSON object per line, each tagged with an `event` type:

  * `session_start`   — written when the log opens (delimits append-mode sessions)
  * `violation`       — written when a violation *fires* (deduplicated by the
                        compliance debounce — not per frame); ties a violation to a
                        worker (via Work ID when available, else the anonymous track
                        id), a frame index and a timestamp
  * `workflow_mistake`— written when the activity recogniser's order monitor flags an
                        out-of-order assembly step (once per step, not per frame);
                        carries the step, a kind and a human-readable reason
  * `session_bindings`— the final {track_id: worker} map, written at close, so a
                        consumer can re-key earlier anonymous (`identified:false`)
                        rows to the worker who was identified later in the session
  * `session_end`     — written when the log closes (with the violation count)

Timestamps are timezone-aware (local time with UTC offset). This file is the
structured data source for the future AI daily safety report (Phase 3).
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


def _now_iso() -> str:
    """Timezone-aware local timestamp (with offset) — unambiguous across sessions."""
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class EventLog:
    def __init__(self, path: str):
        self.path = path
        self.count = 0
        self.mistakes = 0
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", encoding="utf-8")
            self._write({"event": "session_start", "ts": _now_iso()})

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def _write(self, rec: dict) -> None:
        if self._fh is None:
            return
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def log_violation(self, frame_idx: int, elapsed_s: float, track_id: int,
                      worker: "str | None", ev) -> None:
        """`ev` is a compliance.ActiveViolation. `worker` is the identity at FIRE time
        (None if not yet identified) — a later identity is reconcilable via the
        `session_bindings` record on the shared `track_id`."""
        if self._fh is None:
            return
        self._write({
            "event": "violation",
            "ts": _now_iso(),
            "frame": int(frame_idx),
            "elapsed_s": round(float(elapsed_s), 2),
            "worker": worker or f"#{int(track_id)}",
            "identified": worker is not None,
            "track_id": int(track_id),
            "violation": ev.class_name,
            "label": ev.label,
            "severity": ev.severity,
        })
        self.count += 1

    def log_mistake(self, frame_idx: int, elapsed_s: float, step: str, detail: str,
                    kind: str = "order_violation", stream: str = "ego") -> None:
        """A workflow-order mistake detected by the activity recogniser's procedure
        monitor (fires once per out-of-order step, not per frame). Distinct from a PPE
        `violation`; the `mistakes` count is reported separately at session end."""
        if self._fh is None:
            return
        self._write({
            "event": "workflow_mistake",
            "ts": _now_iso(),
            "frame": int(frame_idx),
            "elapsed_s": round(float(elapsed_s), 2),
            "stream": stream,
            "step": step,
            "kind": kind,
            "detail": detail,
        })
        self.mistakes += 1

    def log_bindings(self, bindings: dict) -> None:
        """Persist the final track_id -> worker map so a consumer can attribute the
        anonymous fire-time rows to the worker resolved later in the session."""
        if self._fh is None or not bindings:
            return
        self._write({"event": "session_bindings", "ts": _now_iso(),
                     "bindings": {str(k): v for k, v in bindings.items()}})

    def close(self) -> None:
        if self._fh is not None:
            self._write({"event": "session_end", "ts": _now_iso(),
                         "violations": self.count, "mistakes": self.mistakes})
            self._fh.close()
            self._fh = None
