"""Simulated AR heads-up overlay — the 'glasses view' drawn on the live feed.

Pure OpenCV (no UI framework). Draws, per tracked person, a box coloured by their
worst active violation (green = compliant) with their ID and warning labels, plus
two HUD panels: a status panel (FPS, per-stage latency, counts) and a live alert
list of active violations colour-coded by severity. Text is ASCII-only because
cv2.putText cannot render non-ASCII glyphs.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .compliance import FrameCompliance
from .config import SEVERITY_RANK

# BGR colours
SEVERITY_COLORS = {
    "high":   (40, 40, 220),    # red
    "medium": (0, 140, 255),    # orange
    "low":    (0, 215, 255),    # yellow
}
OK_COLOR = (70, 180, 75)        # green — compliant
WHITE = (255, 255, 255)
GREY = (180, 180, 180)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def severity_color(severity: Optional[str]) -> tuple[int, int, int]:
    if severity is None:
        return OK_COLOR
    return SEVERITY_COLORS.get(severity, SEVERITY_COLORS["high"])


def annotate(frame: np.ndarray, fc: FrameCompliance, hud: dict) -> np.ndarray:
    """Draw the full AR overlay onto `frame` in place and return it."""
    _draw_people(frame, fc)
    _draw_status_panel(frame, fc, hud)
    _draw_alert_list(frame, fc)
    if hud.get("recording"):
        _draw_rec(frame)
    return frame


# --- people ------------------------------------------------------------------
def _draw_people(frame: np.ndarray, fc: FrameCompliance) -> None:
    # Draw larger boxes first so smaller ones stay legible on top.
    for p in sorted(fc.persons, key=lambda s: _area(s.bbox), reverse=True):
        x1, y1, x2, y2 = (int(round(v)) for v in p.bbox)
        color = severity_color(p.worst_severity)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ID tag at the top-left corner.
        tag = f"#{p.tracker_id}"
        if p.is_compliant:
            tag += " OK"
        _label(frame, tag, x1, y1, color)

        # Stack violation labels just inside the top of the box.
        oy = y1 + 4
        for av in sorted(p.active, key=lambda v: SEVERITY_RANK.get(v.severity, 0), reverse=True):
            oy = _label(frame, f"! {av.label}", x1 + 2, oy + 18,
                        severity_color(av.severity), anchor_top=True) + 2


# --- status panel (top-left) -------------------------------------------------
def _draw_status_panel(frame: np.ndarray, fc: FrameCompliance, hud: dict) -> None:
    counts = {"high": 0, "medium": 0, "low": 0}
    for e in fc.events:
        sev = e.severity if e.severity in counts else "high"  # unknown -> high (contract)
        counts[sev] += 1

    lines = [
        ("AR SAFETY MONITOR", WHITE, 0.6, 2),
        (f"FPS: {hud.get('fps', 0.0):4.1f}    device: {hud.get('device', '?')}", GREY, 0.5, 1),
    ]
    sm = hud.get("stage_ms", {})
    if sm:
        stage_str = "  ".join(f"{k[:3]} {v:.0f}ms" for k, v in sm.items())
        lines.append((stage_str, GREY, 0.5, 1))
    lines.append((f"persons: {len(fc.persons)}    violations: {len(fc.events)}", WHITE, 0.5, 1))
    sev_str = f"HIGH {counts['high']}   MED {counts['medium']}   LOW {counts['low']}"
    sev_color = (40, 40, 220) if counts["high"] else (
        (0, 140, 255) if counts["medium"] else OK_COLOR)
    lines.append((sev_str, sev_color, 0.5, 1))

    _text_panel(frame, 10, 10, lines)


# --- alert list (top-right) --------------------------------------------------
def _draw_alert_list(frame: np.ndarray, fc: FrameCompliance, max_lines: int = 8) -> None:
    if not fc.events:
        return
    events = sorted(fc.events, key=lambda e: (-SEVERITY_RANK.get(e.severity, 0), e.person_id))
    rows = [(f"#{e.person_id}  {e.label}  [{e.severity.upper()}]", severity_color(e.severity))
            for e in events[:max_lines]]
    extra = len(events) - len(rows)
    if extra > 0:
        rows.append((f"+{extra} more", GREY))

    # Measure width to right-align the panel.
    scale, thick, pad = 0.5, 1, 8
    title = "ACTIVE ALERTS"
    widths = [cv2.getTextSize(title, _FONT, 0.55, 1)[0][0]]
    widths += [cv2.getTextSize(t, _FONT, scale, thick)[0][0] for t, _ in rows]
    panel_w = max(widths) + pad * 2
    h, w = frame.shape[:2]
    x1 = max(0, w - panel_w - 10)
    line_h = 22
    panel_h = pad * 2 + line_h * (len(rows) + 1)
    _panel(frame, x1, 10, x1 + panel_w, 10 + panel_h)

    y = 10 + pad + 16
    cv2.putText(frame, title, (x1 + pad, y), _FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    for text, color in rows:
        y += line_h
        cv2.putText(frame, text, (x1 + pad, y), _FONT, scale, color, thick, cv2.LINE_AA)


def _draw_rec(frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    cv2.circle(frame, (w - 24, h - 22), 8, (40, 40, 220), -1)
    cv2.putText(frame, "REC", (w - 70, h - 16), _FONT, 0.6, (40, 40, 220), 2, cv2.LINE_AA)


# --- low-level helpers -------------------------------------------------------
def _text_panel(frame: np.ndarray, x: int, y: int, lines) -> None:
    pad, line_h = 8, 24
    widths = [cv2.getTextSize(t, _FONT, sc, th)[0][0] for t, _, sc, th in lines]
    panel_w = max(widths) + pad * 2
    panel_h = pad * 2 + line_h * len(lines)
    _panel(frame, x, y, x + panel_w, y + panel_h)
    cy = y + pad + 14
    for text, color, scale, thick in lines:
        cv2.putText(frame, text, (x + pad, cy), _FONT, scale, color, thick, cv2.LINE_AA)
        cy += line_h


def _label(frame: np.ndarray, text: str, x: int, y: int,
           color: tuple[int, int, int], anchor_top: bool = False) -> int:
    """Draw a filled label. Returns the baseline y used (for stacking)."""
    scale, thick = 0.5, 1
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)
    if anchor_top:
        y_top = y - th - base
    else:
        y_top = max(0, y - th - base - 2)
    y_top = max(0, y_top)
    cv2.rectangle(frame, (x, y_top), (x + tw + 4, y_top + th + base + 2), color, -1)
    cv2.putText(frame, text, (x + 2, y_top + th + 1), _FONT, scale, WHITE, thick, cv2.LINE_AA)
    return y_top + th + base + 2


def _panel(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
           color: tuple[int, int, int] = (0, 0, 0), alpha: float = 0.45) -> None:
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    sub = frame[y1:y2, x1:x2]
    rect = np.full_like(sub, color)
    cv2.addWeighted(rect, alpha, sub, 1 - alpha, 0, sub)


def _area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
