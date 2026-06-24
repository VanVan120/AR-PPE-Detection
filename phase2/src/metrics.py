"""Performance instrumentation — end-to-end FPS and per-stage latency.

Times each pipeline stage (detect / track / compliance / render) with a context
manager and the whole frame with start_frame()/end_frame(). Stage keys are
arbitrary — the HUD and summary iterate over whatever stages were timed. Exposes
smoothed live numbers for the HUD and an aggregate summary for the exit report —
the data that later decides whether the model is fast enough for on-device.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager


class PerfTracker:
    def __init__(self, window: int = 30):
        self._window = window
        self._stage_total: dict[str, float] = defaultdict(float)
        self._stage_count: dict[str, int] = defaultdict(int)
        self._stage_recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._frame_times: list[float] = []
        self._recent_frames: deque = deque(maxlen=window)
        self._frame_start: float | None = None

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self._stage_total[name] += dt
            self._stage_count[name] += 1
            self._stage_recent[name].append(dt)

    def start_frame(self) -> None:
        self._frame_start = time.perf_counter()

    def end_frame(self) -> None:
        if self._frame_start is None:
            return
        dt = time.perf_counter() - self._frame_start
        self._frame_times.append(dt)
        self._recent_frames.append(dt)
        self._frame_start = None

    @property
    def frames(self) -> int:
        return len(self._frame_times)

    @property
    def live_fps(self) -> float:
        if not self._recent_frames:
            return 0.0
        avg = sum(self._recent_frames) / len(self._recent_frames)
        return 1.0 / avg if avg > 0 else 0.0

    def live_stage_ms(self) -> dict[str, float]:
        """Smoothed recent per-stage latency in milliseconds (for the HUD)."""
        out = {}
        for name, dq in self._stage_recent.items():
            if dq:
                out[name] = (sum(dq) / len(dq)) * 1000.0
        return out

    def summary(self) -> dict:
        n = len(self._frame_times)
        if n == 0:
            return {"frames": 0}
        fps = [1.0 / t for t in self._frame_times if t > 0]
        avg_frame = sum(self._frame_times) / n
        stage_ms = {
            name: (self._stage_total[name] / self._stage_count[name]) * 1000.0
            for name in self._stage_total if self._stage_count[name]
        }
        return {
            "frames": n,
            "avg_fps": (sum(fps) / len(fps)) if fps else 0.0,
            "min_fps": min(fps) if fps else 0.0,
            "max_fps": max(fps) if fps else 0.0,
            "avg_frame_ms": avg_frame * 1000.0,
            "stage_latency_ms": stage_ms,
        }

    def print_summary(self, device: str = "") -> None:
        s = self.summary()
        print("\n" + "=" * 60)
        print(" Performance summary" + (f"  (device: {device})" if device else ""))
        print("=" * 60)
        if not s.get("frames"):
            print("  no frames processed")
            return
        print(f"  frames        : {s['frames']}")
        print(f"  FPS  avg/min/max : {s['avg_fps']:.1f} / {s['min_fps']:.1f} / {s['max_fps']:.1f}")
        print(f"  frame latency : {s['avg_frame_ms']:.1f} ms avg")
        for name, ms in s["stage_latency_ms"].items():
            print(f"    - {name:<7}: {ms:.1f} ms")
