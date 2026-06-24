#!/usr/bin/env python3
"""Build a throwaway demo / smoke-test clip from the Phase 1 dataset images.

Selects images the detector actually finds people in, then applies a slow
Ken-Burns pan/zoom to each so there is real intra-segment motion for ByteTrack to
follow. This is a SLIDESHOW WITH SYNTHETIC MOTION, not worn-camera footage — fine
for demoing/smoke-testing the pipeline, but use a real first-person recording for
an honest reality-check.

    python tools/make_sample_clip.py
    python tools/make_sample_clip.py --images <dir> --out data/clips/sample.mp4 --count 5
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE2 = os.path.dirname(HERE)
DEFAULT_IMAGES = os.path.normpath(os.path.join(PHASE2, "..", "data", "dataset", "test", "images"))
DEFAULT_OUT = os.path.join(PHASE2, "data", "clips", "sample_walkthrough.mp4")
DEFAULT_WEIGHTS = os.path.join(PHASE2, "models", "best.pt")


def _device():
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _cover_resize(img, w, h):
    """Resize so the image covers (w,h), then centre-crop to exactly (w,h)."""
    ih, iw = img.shape[:2]
    scale = max(w / iw, h / ih)
    rw, rh = int(np.ceil(iw * scale)), int(np.ceil(ih * scale))
    img = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    x0 = (rw - w) // 2
    y0 = (rh - h) // 2
    return img[y0:y0 + h, x0:x0 + w]


def _ken_burns(base, w, h, frames):
    """Yield `frames` frames panning + zooming across a (w,h) base image."""
    for t in range(frames):
        a = t / max(1, frames - 1)
        zoom = 1.0 + 0.18 * a                      # zoom in over the segment
        cw, ch = int(w / zoom), int(h / zoom)
        # pan the crop centre diagonally
        max_dx, max_dy = w - cw, h - ch
        cx = int(max_dx * a)
        cy = int(max_dy * (1 - a))
        crop = base[cy:cy + ch, cx:cx + cw]
        yield cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Make a demo/smoke-test clip")
    ap.add_argument("--images", default=DEFAULT_IMAGES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--count", type=int, default=5, help="how many source images to use")
    ap.add_argument("--frames", type=int, default=36, help="frames per source image")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scan", type=int, default=60, help="how many images to score for selection")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))
    if not paths:
        print(f"No images in {args.images}")
        return 1

    from ultralytics import YOLO
    model = YOLO(args.weights)
    names = {int(k): str(v) for k, v in model.names.items()}
    dev = _device()

    # Score candidates by how much PPE content they show.
    scored = []
    for p in paths[:args.scan]:
        img = cv2.imread(p)
        if img is None:
            continue
        r = model.predict(img, conf=0.35, device=dev, verbose=False)[0]
        cls = [int(c) for c in (r.boxes.cls.tolist() if r.boxes is not None else [])]
        persons = sum(1 for c in cls if names.get(c) == "Person")
        viols = sum(1 for c in cls if names.get(c) in ("No-Helmet", "No-Vest"))
        if persons >= 1:
            scored.append((persons + 2 * viols, persons, viols, p))
    scored.sort(reverse=True)
    chosen = scored[:args.count]
    if not chosen:
        print("No images with a detected person were found — using the first few images.")
        chosen = [(0, 0, 0, p) for p in paths[:args.count]]
    print(f"Selected {len(chosen)} images:")
    for score, persons, viols, p in chosen:
        print(f"  {os.path.basename(p)}  persons={persons} violations={viols}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.width, args.height))
    n = 0
    for _, _, _, p in chosen:
        img = cv2.imread(p)
        if img is None:
            continue
        base = _cover_resize(img, args.width, args.height)
        for frame in _ken_burns(base, args.width, args.height, args.frames):
            writer.write(frame)
            n += 1
    writer.release()
    print(f"Wrote {args.out}  ({n} frames @ {args.fps}fps, {args.width}x{args.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
