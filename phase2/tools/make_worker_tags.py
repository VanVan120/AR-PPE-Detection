#!/usr/bin/env python3
"""Generate printable ArUco worker tags for Work ID.

Each tag is a marker (worn on helmet/vest) plus the worker's name and id, on a
white card with a quiet zone (ArUco needs the white border to detect reliably).
Marker ids + names come from config.yaml (`workid.markers`) unless overridden.

    python tools/make_worker_tags.py
    python tools/make_worker_tags.py --markers "0=Alice Tan,1=Bob Lim" --size 500
    python tools/make_worker_tags.py --dictionary DICT_5X5_100 --out outputs/tags

Print the PNGs at a constant physical size (e.g. 8-10 cm) for a real site.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PHASE2 = os.path.dirname(HERE)
sys.path.insert(0, PHASE2)


def _markers_from_config():
    try:
        from src.config import load_config
        cfg = load_config(os.path.join(PHASE2, "config.yaml"))
        return cfg.workid_dictionary, dict(cfg.workid_markers)
    except Exception:
        return "DICT_4X4_50", {}


def _parse_markers(s: str) -> dict:
    out = {}
    for part in s.split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        v = v.strip()
        if not v:                    # "3=" — a blank name is a typo, not intent
            print(f"  skipping '{part}' (empty worker name)", file=sys.stderr)
            continue
        try:
            out[int(k.strip())] = v
        except ValueError:
            pass
    return out


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:40] or "worker"


def make_tag(dictionary, marker_id: int, label: str, side: int) -> np.ndarray:
    marker = cv2.aruco.generateImageMarker(dictionary, int(marker_id), side)  # grayscale
    marker = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    quiet = max(8, side // 6)                       # white border (quiet zone)
    label_h = max(48, side // 5)
    W = side + 2 * quiet
    H = side + 2 * quiet + label_h
    card = np.full((H, W, 3), 255, dtype=np.uint8)
    card[quiet:quiet + side, quiet:quiet + side] = marker
    cv2.rectangle(card, (1, 1), (W - 2, H - 2), (0, 0, 0), 2)   # cut border

    font = cv2.FONT_HERSHEY_SIMPLEX
    name = "".join(c if ord(c) < 128 else "?" for c in label)   # cv2 text is ASCII-only
    scale = 0.9
    while scale > 0.3:
        (tw, th), _ = cv2.getTextSize(name, font, scale, 2)
        if tw <= W - 2 * quiet:
            break
        scale -= 0.05
    ty = side + 2 * quiet + (label_h + th) // 2
    cv2.putText(card, name, ((W - tw) // 2, ty), font, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(card, f"ID {marker_id}", (quiet, H - 8), font, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
    return card


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate printable ArUco worker tags")
    ap.add_argument("--markers", default=None, help='"0=Alice,1=Bob" (else read from config.yaml)')
    ap.add_argument("--dictionary", default=None, help="ArUco dictionary (else config.yaml)")
    ap.add_argument("--size", type=int, default=440, help="marker side in pixels")
    ap.add_argument("--out", default=os.path.join(PHASE2, "outputs", "worker_tags"))
    args = ap.parse_args(argv)

    cfg_dict, cfg_markers = _markers_from_config()
    dict_name = args.dictionary or cfg_dict
    markers = _parse_markers(args.markers) if args.markers else cfg_markers
    if not markers:
        print("No markers to generate. Add workid.markers in config.yaml or pass --markers.",
              file=sys.stderr)
        return 1
    if not hasattr(cv2, "aruco"):
        print("cv2.aruco unavailable — install opencv-contrib-python.", file=sys.stderr)
        return 1
    dict_id = getattr(cv2.aruco, dict_name, None)
    if dict_id is None:
        print(f"Unknown ArUco dictionary '{dict_name}'.", file=sys.stderr)
        return 1
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    os.makedirs(args.out, exist_ok=True)
    for mid, name in sorted(markers.items()):
        card = make_tag(dictionary, mid, name, args.size)
        path = os.path.join(args.out, f"tag_{mid:03d}_{_safe(name)}.png")
        ok, buf = cv2.imencode(".png", card)
        if ok:
            buf.tofile(path)                        # unicode-safe write on Windows
            print(f"  wrote {path}  ({name}, marker {mid}, {dict_name})")
    print(f"\n{len(markers)} tag(s) -> {args.out}. Print at a constant physical size (~8-10 cm).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
