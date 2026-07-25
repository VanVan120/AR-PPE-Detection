"""Generate the home-screen icons. Run once; the PNGs are committed.

Drawn rather than downloaded so the repo stays self-contained and there is no image with
an unclear licence in it. The whole square is painted (no transparent corners) because
Android crops a *maskable* icon to whatever shape the launcher uses — a logo drawn to the
edges would lose its corners, so everything meaningful stays inside the middle ~60%.

    python phase8_phoneapp/make_icons.py
"""
from __future__ import annotations

import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "static")

BG = (14, 16, 18)            # BGR — near-black, matches the app background
HAT = (64, 190, 232)         # BGR of #e8be40, the app accent
BRACKET = (244, 244, 244)


def draw(size: int) -> np.ndarray:
    S = 512
    img = np.full((S, S, 3), BG, np.uint8)

    # Detection brackets: four corners, the visual shorthand for "this is looking at
    # something". Kept inside the maskable safe zone.
    m, ln, th = 132, 60, 12
    for cx, cy, dx, dy in ((m, m, 1, 1), (S - m, m, -1, 1),
                           (m, S - m, 1, -1), (S - m, S - m, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * ln, cy), BRACKET, th, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * ln), BRACKET, th, cv2.LINE_AA)

    # Hard hat: a dome plus a brim.
    cv2.ellipse(img, (S // 2, 300), (112, 104), 0, 180, 360, HAT, -1, cv2.LINE_AA)
    cv2.ellipse(img, (S // 2, 300), (168, 26), 0, 180, 360, HAT, -1, cv2.LINE_AA)
    cv2.rectangle(img, (S // 2 - 168, 296), (S // 2 + 168, 316), HAT, -1, cv2.LINE_AA)
    cv2.ellipse(img, (S // 2, 316), (168, 22), 0, 0, 180, HAT, -1, cv2.LINE_AA)
    # A rib, so it reads as a helmet and not a mound at 48 px.
    cv2.line(img, (S // 2, 200), (S // 2, 296), BG, 14, cv2.LINE_AA)

    if size != S:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    for size in (192, 512):
        path = os.path.join(OUT, f"icon-{size}.png")
        cv2.imwrite(path, draw(size))
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
