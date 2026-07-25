"""Glasses-ready rendering — the view an AR headset can actually display.

`overlay.py` composites a HUD **onto the camera image**. That is right for a monitor and
for video-passthrough headsets (Quest 3, Vision Pro), where the camera feed *is* the
display. It is wrong for **optical see-through** glasses (HoloLens, Xreal, Rokid), where
the wearer looks through the lens at the real world and the display only *adds* light:

  * **Black is transparent.** Painting the camera frame would lay a video on top of
    reality — the one thing a see-through display must never do. The renderer therefore
    starts from a black canvas and draws graphics only.
  * **Dark UI disappears.** `overlay.py`'s translucent near-black cards are the standard
    way to make text readable on a bright photo. On an additive display they emit almost
    no light, so they simply vanish. See-through UI has to be **bright strokes and text on
    nothing** — outlines, not filled cards.
  * **The display is smaller than the camera.** Glasses see ~30-50 deg; a webcam sees
    much more. Anything drawn near the frame edge falls outside the lens and is never
    seen, so content is confined to a **safe zone** and workers outside it get an
    **edge indicator** instead of being silently dropped.
  * **Angular resolution is low.** Strokes and text are scaled up; a 0.4-scale label that
    reads fine on a monitor is unreadable through a lens.

`simulate_glasses()` closes the loop: it additively blends the graphics layer over the
real frame and crops to the display FOV, which is a physically reasonable model of what
the wearer actually sees. That makes the design reviewable with no headset in the room.
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .compliance import FrameCompliance
from .config import SEVERITY_RANK
from .overlay import (_area, _ascii, _brackets, _text, OK_COLOR, SEVERITY_COLORS,
                      WHITE, severity_color)

# Bright, saturated palette: an additive display cannot render dark UI.
AR_ACCENT = (255, 232, 120)
AR_MUTED = (190, 190, 200)

# Fraction of the camera frame the glasses display actually covers. A 60-degree camera
# feeding a ~36-degree display gives ~0.6. Overridable from config.
DEFAULT_FOV_RATIO = 0.62


def safe_rect(width: int, height: int, fov_ratio: float = DEFAULT_FOV_RATIO
              ) -> Tuple[int, int, int, int]:
    """The centred region the lens can actually show, as (x1, y1, x2, y2)."""
    r = float(np.clip(fov_ratio, 0.15, 1.0))
    w, h = int(width * r), int(height * r)
    x1, y1 = (width - w) // 2, (height - h) // 2
    return x1, y1, x1 + w, y1 + h


def _clamp_to(rect, x, y) -> Tuple[int, int]:
    x1, y1, x2, y2 = rect
    return int(np.clip(x, x1, x2)), int(np.clip(y, y1, y2))


def _centre(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def render_seethrough(shape, fc: FrameCompliance, hud: dict,
                      worker_of: Optional[dict] = None,
                      fov_ratio: float = DEFAULT_FOV_RATIO,
                      scale: float = 1.0) -> np.ndarray:
    """Build the graphics-only layer for an optical see-through display.

    Returns a BGR image the same size as the camera frame whose black pixels mean
    'emit nothing' (i.e. fully transparent on the lens).
    """
    h, w = int(shape[0]), int(shape[1])
    layer = np.zeros((h, w, 3), np.uint8)
    rect = safe_rect(w, h, fov_ratio)
    worker_of = worker_of or {}

    # The status line owns a band at the top of the safe zone; person labels must not be
    # drawn into it, or a worker standing high in frame collides with the alert count.
    header_bottom = rect[1] + int(42 * scale)
    _draw_people_ar(layer, fc, worker_of, rect, scale, header_bottom)
    _draw_edge_indicators(layer, fc, rect, scale)
    _draw_summary_ar(layer, fc, hud, rect, scale)

    # Hard clip to the lens. A display physically cannot emit outside its own FOV, so
    # enforcing it here makes the invariant true by construction rather than depending on
    # every draw call getting its bounds right — stroke width and anti-aliasing both spill
    # a pixel or two past a clamped coordinate.
    x1, y1, x2, y2 = rect
    mask = np.zeros((h, w), np.uint8)
    mask[y1:y2, x1:x2] = 1
    return layer * mask[:, :, None]


def _draw_people_ar(layer, fc: FrameCompliance, worker_of: dict, rect, scale,
                    header_bottom: int) -> None:
    """Corner reticles + a name and the worst violation, in bright strokes only."""
    x1r, y1r, x2r, y2r = rect
    th = max(2, int(round(3 * scale)))
    for p in sorted(fc.persons, key=lambda s: _area(s.bbox), reverse=True):
        cx, cy = _centre(p.bbox)
        if not (x1r <= cx <= x2r and y1r <= cy <= y2r):
            continue                                   # handled by the edge indicator
        bx1, by1, bx2, by2 = (int(round(v)) for v in p.bbox)
        bx1, by1 = max(x1r, bx1), max(y1r, by1)
        bx2, by2 = min(x2r, bx2), min(y2r, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        colour = severity_color(p.worst_severity)
        _brackets(layer, bx1, by1, bx2, by2, colour, th)

        name = worker_of.get(p.tracker_id) or f"ID {p.tracker_id}"
        line = int(30 * scale)                         # must exceed the glyph height
        ty = by1 - int(14 * scale)
        if p.active:
            ty -= line                                 # make room for the violation line
        if ty < header_bottom:                         # would hit the status band -> go below
            ty = max(by1 + int(28 * scale), header_bottom + line)
        _text(layer, name, bx1, ty, WHITE, 0.62 * scale, max(1, int(round(2 * scale))))
        if p.active:
            worst = max(p.active,
                        key=lambda v: SEVERITY_RANK.get(v.severity, SEVERITY_RANK["high"]))
            _text(layer, worst.label, bx1, ty + line, colour,
                  0.56 * scale, max(1, int(round(2 * scale))))


def _draw_edge_indicators(layer, fc: FrameCompliance, rect, scale) -> None:
    """A worker outside the lens FOV still matters — especially an unsafe one. Draw a
    chevron on the safe-zone boundary pointing at them rather than dropping them."""
    x1r, y1r, x2r, y2r = rect
    for p in fc.persons:
        cx, cy = _centre(p.bbox)
        if x1r <= cx <= x2r and y1r <= cy <= y2r:
            continue
        colour = severity_color(p.worst_severity)
        size = int(14 * scale)
        # Keep the whole chevron INSIDE the safe zone: anything past the boundary is off
        # the lens and would simply not be displayed. The tip touches the edge and the
        # body sits inboard, so it still reads as "they are out that way".
        ex, ey = _clamp_to((x1r + size, y1r + size, x2r - size, y2r - size), cx, cy)
        dx, dy = cx - ex, cy - ey
        n = float(np.hypot(dx, dy)) or 1.0
        ux, uy = dx / n, dy / n
        tip = (int(ex + ux * size), int(ey + uy * size))
        base = (ex - int(ux * size * 0.4), ey - int(uy * size * 0.4))
        left = (int(base[0] - uy * size * 0.6), int(base[1] + ux * size * 0.6))
        right = (int(base[0] + uy * size * 0.6), int(base[1] - ux * size * 0.6))
        cv2.drawContours(layer, [np.array([tip, left, right])], 0, colour, -1, cv2.LINE_AA)


def _draw_summary_ar(layer, fc: FrameCompliance, hud: dict, rect, scale) -> None:
    """One glanceable status line at the top of the safe zone, plus the unsafe workers.

    Deliberately minimal: the monitor HUD's four cards are far too much information for a
    small, always-on display worn on someone's face.
    """
    x1r, y1r, x2r, y2r = rect
    pad = int(10 * scale)
    n_alerts = len(fc.events)
    colour = SEVERITY_COLORS["high"] if n_alerts else OK_COLOR
    status = f"{n_alerts} ALERT" + ("S" if n_alerts != 1 else "") if n_alerts else "ALL CLEAR"
    cv2.circle(layer, (x1r + pad + int(6 * scale), y1r + pad + int(6 * scale)),
               int(6 * scale), colour, -1, cv2.LINE_AA)
    _text(layer, status, x1r + pad + int(20 * scale), y1r + pad + int(12 * scale),
          colour, 0.62 * scale, max(1, int(round(2 * scale))))

    persons = len(fc.persons)
    right = f"{persons} on site"
    tw = cv2.getTextSize(_ascii(right), cv2.FONT_HERSHEY_SIMPLEX, 0.52 * scale, 1)[0][0]
    _text(layer, right, x2r - pad - tw, y1r + pad + int(12 * scale), AR_MUTED,
          0.52 * scale, max(1, int(round(1 * scale))))

    # Unsafe workers the wearer CANNOT see. Anyone inside the lens FOV already carries a
    # label on their body, so listing them again would duplicate information on a display
    # that has very little room for it. This list earns its space by covering only the
    # people who would otherwise go unnoticed — turn your head and they appear.
    offscreen = []
    for p in fc.persons:
        if not p.active:
            continue
        cx, cy = _centre(p.bbox)
        if x1r <= cx <= x2r and y1r <= cy <= y2r:
            continue
        offscreen.append(p)
    if not offscreen:
        return
    y = y2r - pad
    for p in sorted(offscreen, key=lambda s: -_area(s.bbox))[:3]:
        worst = max(p.active,
                    key=lambda v: SEVERITY_RANK.get(v.severity, SEVERITY_RANK["high"]))
        side = "<" if _centre(p.bbox)[0] < x1r else ">"
        _text(layer, f"{side} {worst.label}", x1r + pad, y,
              severity_color(worst.severity), 0.56 * scale, max(1, int(round(2 * scale))))
        y -= int(28 * scale)
    _text(layer, "OFF-VIEW", x1r + pad, y, AR_MUTED, 0.44 * scale, 1)


def simulate_glasses(frame: np.ndarray, layer: np.ndarray,
                     fov_ratio: float = DEFAULT_FOV_RATIO,
                     dim_world: float = 0.75) -> np.ndarray:
    """Approximate what the wearer sees: the real world plus emitted light, cropped to
    the lens FOV.

    `cv2.add` (saturating addition) models an additive combiner, so graphics brighten the
    scene and never occlude it — the reason see-through UI cannot rely on dark panels.
    `dim_world` stands in for the tinted visor most of these headsets use. This is a
    design-review aid, not a photometric simulation: it ignores per-eye offset, lens
    distortion, vergence and the real transmittance curve.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = safe_rect(w, h, fov_ratio)
    world = (frame.astype(np.float32) * float(np.clip(dim_world, 0.0, 1.0))).astype(np.uint8)
    seen = cv2.add(world, layer)
    return seen[y1:y2, x1:x2].copy()


def draw_safe_zone(canvas: np.ndarray, fov_ratio: float = DEFAULT_FOV_RATIO,
                   colour=(90, 90, 90)) -> np.ndarray:
    """Dashed outline of the lens FOV — a design aid for the composite preview, showing
    which parts of the HUD a headset would actually be able to show."""
    h, w = canvas.shape[:2]
    x1, y1, x2, y2 = safe_rect(w, h, fov_ratio)
    dash = 14
    for x in range(x1, x2, dash * 2):
        cv2.line(canvas, (x, y1), (min(x + dash, x2), y1), colour, 1, cv2.LINE_AA)
        cv2.line(canvas, (x, y2 - 1), (min(x + dash, x2), y2 - 1), colour, 1, cv2.LINE_AA)
    for y in range(y1, y2, dash * 2):
        cv2.line(canvas, (x1, y), (x1, min(y + dash, y2)), colour, 1, cv2.LINE_AA)
        cv2.line(canvas, (x2 - 1, y), (x2 - 1, min(y + dash, y2)), colour, 1, cv2.LINE_AA)
    _text(canvas, "glasses FOV", x1 + 6, y1 + 16, colour, 0.42, 1)
    return canvas
