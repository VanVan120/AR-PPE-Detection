"""Simulated AR heads-up overlay — the 'glasses view' drawn on the live feed.

Pure OpenCV (no UI framework), designed to read like a real AR HUD:

  * a top **header bar** with the brand, an overall-status indicator dot, FPS + device;
  * per-person **corner-bracket reticles** coloured by their worst active violation
    (green = compliant), with a rounded ID/name pill and stacked violation chips;
  * a right-side **ACTIVE ALERTS** card (severity-striped rows);
  * a bottom-left **WORKFLOW** card (Phase 3: current step, a mistake banner, and the
    anticipated next steps) shown only when activity recognition is on;
  * a bottom **status bar** with person / severity chips, a controls hint and REC dot.

Panels are translucent rounded cards. Text is ASCII-only because cv2.putText cannot
render non-ASCII glyphs. Everything is drawn in place, fast enough for the live loop,
and safe to render headless (used by the offline preview harness too).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .compliance import FrameCompliance
from .config import SEVERITY_RANK

# --- palette (BGR) -----------------------------------------------------------
SEVERITY_COLORS = {
    "high":   (48, 48, 226),     # red
    "medium": (0, 145, 255),     # orange
    "low":    (0, 205, 245),     # amber
}
OK_COLOR = (96, 196, 108)        # calm green — compliant
WHITE    = (244, 244, 248)
MUTED    = (172, 178, 188)
FAINT    = (120, 126, 136)
INK      = (24, 22, 20)          # near-black panel ground (slight warm bias)
ACCENT   = (232, 190, 64)        # cyan/teal brand accent
ACCENT_D = (150, 118, 30)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONTD = cv2.FONT_HERSHEY_DUPLEX


def _ascii(text: str) -> str:
    """cv2.putText cannot render non-ASCII glyphs (they show as boxes). Map them to
    '?' the same way the tag generator does, so externally-configured worker names
    (e.g. Malaysian/Chinese names) degrade gracefully instead of rendering as garbage."""
    return "".join(c if ord(c) < 128 else "?" for c in str(text))


def severity_color(severity: Optional[str]) -> tuple[int, int, int]:
    if severity is None:
        return OK_COLOR
    return SEVERITY_COLORS.get(severity, SEVERITY_COLORS["high"])


def _overall_color(fc: FrameCompliance) -> tuple[int, int, int]:
    sevs = {e.severity for e in fc.events}
    if "high" in sevs:
        return SEVERITY_COLORS["high"]
    if "medium" in sevs:
        return SEVERITY_COLORS["medium"]
    if "low" in sevs:
        return SEVERITY_COLORS["low"]
    return OK_COLOR


def annotate(frame: np.ndarray, fc: FrameCompliance, hud: dict,
             worker_of: Optional[dict] = None) -> np.ndarray:
    """Draw the full AR overlay onto `frame` in place and return it."""
    _draw_people(frame, fc, worker_of or {})
    # Info cards stack up from the bottom-left, clear of the upper scene where people's
    # heads and their name pills sit.
    used = _draw_workers(frame, hud)
    _draw_workflow(frame, hud, bottom_offset=used)
    _draw_alert_list(frame, fc)
    _draw_header(frame, fc, hud)          # header + status bar drawn last: sit above boxes
    _draw_status_bar(frame, fc, hud)
    return frame


# --- drawing primitives ------------------------------------------------------
def _fill_round(frame, x1, y1, x2, y2, color, radius) -> None:
    """Solid rounded-rect (no blending)."""
    r = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))
    if r <= 0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
        return
    cv2.rectangle(frame, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(frame, (cx, cy), r, color, -1)


def _panel(frame, x1, y1, x2, y2, color=INK, alpha=0.58, radius=12,
           border=None) -> None:
    """Translucent rounded card, optionally with a coloured border (color, thickness)."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    r = int(max(0, min(radius, rh // 2, rw // 2)))
    mask = np.zeros((rh, rw), np.uint8)
    if r > 0:
        cv2.rectangle(mask, (r, 0), (rw - r, rh), 255, -1)
        cv2.rectangle(mask, (0, r), (rw, rh - r), 255, -1)
        for cx, cy in ((r, r), (rw - r, r), (r, rh - r), (rw - r, rh - r)):
            cv2.circle(mask, (cx, cy), r, 255, -1)
    else:
        mask[:] = 255
    overlay = np.empty_like(roi)
    overlay[:] = color
    blended = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0.0)
    idx = mask > 0
    roi[idx] = blended[idx]
    if border is not None:
        bcol, bth = border
        e = 1
        cv2.line(roi, (r, e), (rw - r, e), bcol, bth, cv2.LINE_AA)
        cv2.line(roi, (r, rh - e), (rw - r, rh - e), bcol, bth, cv2.LINE_AA)
        cv2.line(roi, (e, r), (e, rh - r), bcol, bth, cv2.LINE_AA)
        cv2.line(roi, (rw - e, r), (rw - e, rh - r), bcol, bth, cv2.LINE_AA)
        cv2.ellipse(roi, (r, r), (r - e, r - e), 180, 0, 90, bcol, bth, cv2.LINE_AA)
        cv2.ellipse(roi, (rw - r, r), (r - e, r - e), 270, 0, 90, bcol, bth, cv2.LINE_AA)
        cv2.ellipse(roi, (r, rh - r), (r - e, r - e), 90, 0, 90, bcol, bth, cv2.LINE_AA)
        cv2.ellipse(roi, (rw - r, rh - r), (r - e, r - e), 0, 0, 90, bcol, bth, cv2.LINE_AA)


def _text(frame, text, x, y, color=WHITE, scale=0.5, thick=1, font=_FONT) -> int:
    """Draw AA text with a subtle shadow for legibility on any background. Returns width."""
    text = _ascii(text)
    cv2.putText(frame, text, (x + 1, y + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)
    return cv2.getTextSize(text, font, scale, thick)[0][0]


def _chip(frame, x, y, text, fg=WHITE, bg=INK, scale=0.46, thick=1) -> tuple[int, int]:
    """A rounded pill label anchored top-left at (x, y). Returns (width, height)."""
    text = _ascii(text)
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)
    px, py = 8, 5
    w, h = tw + 2 * px, th + base + 2 * py
    _fill_round(frame, x, y, x + w, y + h, bg, radius=h // 2)
    cv2.putText(frame, text, (x + px, y + py + th), _FONT, scale, fg, thick, cv2.LINE_AA)
    return w, h


def _dot(frame, cx, cy, color, r=6) -> None:
    cv2.circle(frame, (cx, cy), r + 1, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, color, -1, cv2.LINE_AA)


# --- people (AR reticles) ----------------------------------------------------
def _brackets(frame, x1, y1, x2, y2, color, thick=2) -> None:
    L = int(max(12, min(30, (x2 - x1) * 0.22, (y2 - y1) * 0.22)))
    for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + sx * L, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + sy * L), color, thick, cv2.LINE_AA)


def _draw_people(frame: np.ndarray, fc: FrameCompliance, worker_of: dict) -> None:
    h, w = frame.shape[:2]
    # Draw larger boxes first so smaller ones stay legible on top.
    for p in sorted(fc.persons, key=lambda s: _area(s.bbox), reverse=True):
        x1, y1, x2, y2 = (int(round(v)) for v in p.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = severity_color(p.worst_severity)
        if not p.is_compliant:                       # faint full box adds emphasis
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        _brackets(frame, x1, y1, x2, y2, color, thick=2)

        # ID pill (worker Work ID when bound, else anonymous track id) + status dot
        name = worker_of.get(p.tracker_id) or f"ID {p.tracker_id}"
        chip_bg = color if not p.is_compliant else (34, 40, 34)
        chip_fg = WHITE if not p.is_compliant else OK_COLOR
        cy = y1 - 26 if y1 - 26 >= 34 else y1 + 4
        cw, ch = _chip(frame, x1, cy, ("  " + name) if p.is_compliant else name,
                       fg=chip_fg, bg=chip_bg)
        if p.is_compliant:
            _dot(frame, x1 + 10, cy + ch // 2, OK_COLOR, r=4)

        # Violation chips stacked just inside the top-left of the box.
        oy = cy + ch + 4
        for av in sorted(p.active, key=lambda v: SEVERITY_RANK.get(v.severity, 0), reverse=True)[:3]:
            _, vh = _chip(frame, x1, oy, "! " + av.label, fg=WHITE,
                          bg=severity_color(av.severity), scale=0.44)
            oy += vh + 3


# --- header bar (top) --------------------------------------------------------
def _draw_header(frame: np.ndarray, fc: FrameCompliance, hud: dict) -> None:
    h, w = frame.shape[:2]
    bar_h = 40
    _panel(frame, 0, 0, w, bar_h, color=INK, alpha=0.62, radius=0)
    cv2.line(frame, (0, bar_h), (w, bar_h), ACCENT_D, 1, cv2.LINE_AA)

    status = _overall_color(fc)
    _dot(frame, 20, bar_h // 2, status, r=7)
    x = _text(frame, "AR SAFETY MONITOR", 36, 26, WHITE, 0.62, 1, font=_FONTD) + 36
    state = ("CLEAR" if status == OK_COLOR else "ALERT")
    _text(frame, "| " + state, x + 12, 26, status, 0.5, 1)

    # right cluster: FPS + device
    right = []
    fps = hud.get("fps")
    if fps is not None:
        right.append((f"{fps:4.1f} FPS", WHITE))
    right.append((_ascii(str(hud.get("device", "?")).upper()), ACCENT))
    rx = w - 14
    for text, col in reversed(right):
        tw = cv2.getTextSize(_ascii(text), _FONT, 0.5, 1)[0][0]
        rx -= tw
        _text(frame, text, rx, 26, col, 0.5, 1)
        rx -= 16


# --- alert list (top-right, below header) ------------------------------------
def _draw_alert_list(frame: np.ndarray, fc: FrameCompliance, max_lines: int = 7) -> None:
    if not fc.events:
        return
    events = sorted(fc.events, key=lambda e: (-SEVERITY_RANK.get(e.severity, 0), e.person_id))
    rows = [(_ascii(f"ID {e.person_id}   {e.label}"), e.severity) for e in events[:max_lines]]
    extra = len(events) - len(rows)

    scale, pad, stripe = 0.48, 12, 10       # severity is encoded by the left stripe colour
    title = f"ACTIVE ALERTS  ({len(events)})"
    widths = [cv2.getTextSize(title, _FONT, 0.5, 1)[0][0]]
    widths += [stripe + cv2.getTextSize(t, _FONT, scale, 1)[0][0] for t, _ in rows]
    if extra > 0:
        widths.append(stripe + cv2.getTextSize(f"+{extra} more", _FONT, 0.44, 1)[0][0])
    panel_w = min(max(widths) + pad * 2, frame.shape[1] - 24)
    row_h = 25
    panel_h = 34 + row_h * len(rows) + (20 if extra > 0 else 0) + 6
    h, w = frame.shape[:2]
    x1 = w - panel_w - 12
    y1 = 52
    _panel(frame, x1, y1, x1 + panel_w, y1 + panel_h, alpha=0.6,
           border=(SEVERITY_COLORS["high"] if any(s == "high" for _, s in rows) else ACCENT_D, 1))

    _text(frame, title, x1 + pad, y1 + 22, WHITE, 0.5, 1)
    cv2.line(frame, (x1 + pad, y1 + 31), (x1 + panel_w - pad, y1 + 31), (72, 76, 84), 1, cv2.LINE_AA)
    y = y1 + 31
    for text, sev in rows:
        y += row_h
        col = severity_color(sev)
        cv2.rectangle(frame, (x1 + pad, y - 13), (x1 + pad + 3, y + 3), col, -1)  # severity stripe
        _text(frame, text, x1 + pad + stripe, y, WHITE, scale, 1)
    if extra > 0:
        _text(frame, f"+{extra} more", x1 + pad + stripe, y + row_h - 4, MUTED, 0.44, 1)


# --- worker roster (top-left) — Work ID --------------------------------------
def _draw_workers(frame: np.ndarray, hud: dict, max_rows: int = 5) -> int:
    """Live roster of identified workers: who is on site and their current state.

    Anchored bottom-left and returns the vertical space it consumed, so the workflow card
    can stack above it. It deliberately does NOT sit top-left: that is where person name
    pills are drawn, and the panel would bury them.

    Each row carries an explicit provenance tag — `badge` (an ArUco tag was read, the
    name is certain) or `seen` (appearance match only, the name is a best guess). Showing
    that distinction is the point: an inspector must never mistake a provisional
    appearance match for a confirmed identity.
    """
    workers = hud.get("workers")
    if not workers:
        return 0
    rows = list(workers)[:max_rows]
    extra = len(workers) - len(rows)

    pad, scale = 12, 0.46
    title = f"WORKERS  ({len(workers)})"
    widths = [cv2.getTextSize(title, _FONT, 0.46, 1)[0][0]]
    for wk in rows:
        label = _ascii(wk.get("label", "?"))
        widths.append(18 + cv2.getTextSize(label, _FONT, scale, 1)[0][0] + 52)
    panel_w = min(max(widths) + pad * 2, frame.shape[1] // 2)
    row_h = 24
    panel_h = 34 + row_h * len(rows) + (18 if extra > 0 else 0) + 8
    x1 = 12
    y1 = max(52, frame.shape[0] - panel_h - 44)          # sit just above the status bar
    _panel(frame, x1, y1, x1 + panel_w, y1 + panel_h, alpha=0.6, border=(ACCENT_D, 1))

    _text(frame, title, x1 + pad, y1 + 22, ACCENT, 0.46, 1)
    cv2.line(frame, (x1 + pad, y1 + 31), (x1 + panel_w - pad, y1 + 31),
             (72, 76, 84), 1, cv2.LINE_AA)
    y = y1 + 31
    for wk in rows:
        y += row_h
        present = bool(wk.get("present", True))
        violating = bool(wk.get("violating", False))
        col = SEVERITY_COLORS["high"] if violating else (OK_COLOR if present else FAINT)
        _dot(frame, x1 + pad + 4, y - 5, col, r=4)
        name_col = WHITE if present else MUTED
        _text(frame, wk.get("label", "?"), x1 + pad + 16, y, name_col, scale, 1)
        tag = "badge" if wk.get("badge") else "seen"
        tag_col = ACCENT if wk.get("badge") else FAINT
        tw = cv2.getTextSize(tag, _FONT, 0.38, 1)[0][0]
        _text(frame, tag, x1 + panel_w - pad - tw, y, tag_col, 0.38, 1)
    if extra > 0:
        _text(frame, f"+{extra} more", x1 + pad + 16, y + 16, MUTED, 0.4, 1)
    return panel_h + 8


# --- workflow card (bottom-left) — Phase 3 -----------------------------------
def _draw_workflow(frame: np.ndarray, hud: dict, bottom_offset: int = 0) -> None:
    act = hud.get("activity")
    if act is None:
        return
    step = _ascii(getattr(act, "step", "") or "")
    conf = float(getattr(act, "confidence", 0.0) or 0.0)
    mistake = bool(getattr(act, "mistake", False))
    detail = _ascii(getattr(act, "detail", "") or "")
    nexts = [_ascii(s) for s in (getattr(act, "next_steps", None) or [])][:3]

    h, w = frame.shape[:2]
    pad = 12
    lines_w = [cv2.getTextSize("WORKFLOW", _FONT, 0.46, 1)[0][0]]
    lines_w.append(cv2.getTextSize(f"{step}  {conf:.2f}", _FONT, 0.56, 1)[0][0])
    if nexts:
        lines_w.append(cv2.getTextSize("NEXT  " + "  ".join(nexts), _FONT, 0.44, 1)[0][0])
    if mistake and detail:
        lines_w.append(cv2.getTextSize(detail[:52], _FONT, 0.42, 1)[0][0])
    panel_w = min(max(lines_w) + pad * 2 + 12, w - 24)
    panel_h = 30 + 28 + (24 if mistake else 0) + (24 if nexts else 0) + pad
    x1, y1 = 12, max(52, h - panel_h - 44 - int(bottom_offset))
    _panel(frame, x1, y1, x1 + panel_w, y1 + panel_h, alpha=0.6,
           border=(SEVERITY_COLORS["high"] if mistake else ACCENT_D, 1))

    _text(frame, "WORKFLOW", x1 + pad, y1 + 22, ACCENT, 0.46, 1)
    y = y1 + 30
    y += 22
    _text(frame, step, x1 + pad, y, WHITE, 0.56, 1, font=_FONTD)
    cw = cv2.getTextSize(_ascii(step), _FONTD, 0.56, 1)[0][0]
    _text(frame, f"{conf:.2f}", x1 + pad + cw + 12, y, MUTED, 0.46, 1)
    if mistake:
        y += 24
        _dot(frame, x1 + pad + 4, y - 4, SEVERITY_COLORS["high"], r=4)
        _text(frame, ("MISTAKE  " + detail)[:56], x1 + pad + 14, y,
              SEVERITY_COLORS["high"], 0.42, 1)
    if nexts:
        y += 24
        nx = _text(frame, "NEXT", x1 + pad, y, FAINT, 0.44, 1) + x1 + pad + 10
        for s in nexts:
            cw, _ = _chip(frame, nx, y - 13, s, fg=INK, bg=ACCENT, scale=0.4)
            nx += cw + 6


# --- bottom status bar -------------------------------------------------------
def _draw_status_bar(frame: np.ndarray, fc: FrameCompliance, hud: dict) -> None:
    h, w = frame.shape[:2]
    bar_h = 34
    y1 = h - bar_h
    _panel(frame, 0, y1, w, h, color=INK, alpha=0.62, radius=0)
    cv2.line(frame, (0, y1), (w, y1), ACCENT_D, 1, cv2.LINE_AA)

    counts = {"high": 0, "medium": 0, "low": 0}
    for e in fc.events:
        counts[e.severity if e.severity in counts else "high"] += 1

    x = 14
    yc = y1 + 9
    cw, _ = _chip(frame, x, yc, f"PERSONS {len(fc.persons)}", fg=WHITE, bg=(46, 44, 40))
    x += cw + 8
    for key, lab in (("high", "HIGH"), ("medium", "MED"), ("low", "LOW")):
        n = counts[key]
        bg = severity_color(key) if n else (40, 44, 40)
        fg = WHITE if n else FAINT
        cw, _ = _chip(frame, x, yc, f"{lab} {n}", fg=fg, bg=bg)
        x += cw + 8

    # right cluster: controls hint + REC
    rx = w - 14
    if hud.get("recording"):
        _text(frame, "REC", rx - 34, y1 + 22, SEVERITY_COLORS["high"], 0.5, 1)
        _dot(frame, rx - 44, y1 + 17, SEVERITY_COLORS["high"], r=5)
        rx -= 60
    hint = "q quit   s shot   r rec"
    tw = cv2.getTextSize(hint, _FONT, 0.44, 1)[0][0]
    _text(frame, hint, rx - tw, y1 + 22, FAINT, 0.44, 1)


def _area(bbox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
