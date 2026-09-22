"""Contact sheet of near-duplicate pairs found by `tools/leakage_audit.py`.

The audit reports a contamination *rate*; a rate is only worth quoting once a human has
looked at the pairs behind it and agreed they really are the same photo. This draws a
fixed-seed random sample of rows from `duplicates.csv` as one JPEG: each row is the test
image on the left, its best train/valid match on the right, with the ncc value (and the
transform and hamming distance that produced it) printed on the pair.

It makes no judgement about whether a pair is a true duplicate. That is the author's call
after looking at the sheet.

    python tools/pairs_sample.py --dataset-dir data/ppe_download \
        --audit-dir paper_materials/leakage_audit

Writes <audit-dir>/pairs_sample.jpg.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np

CELL_W, CELL_H = 360, 270          # each image pane
LABEL_H = 26                       # caption strip under each row
PAD = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit(img, w=CELL_W, h=CELL_H):
    """Letterbox `img` into a w x h pane without distorting its aspect ratio."""
    pane = np.full((h, w, 3), 32, np.uint8)
    if img is None:
        cv2.putText(pane, "unreadable", (10, h // 2), FONT, 0.6, (60, 60, 220), 1, cv2.LINE_AA)
        return pane
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * s))), max(1, int(round(ih * s)))
    pane[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw] = \
        cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return pane


def ellipsize(text, max_chars):
    return text if len(text) <= max_chars else "..." + text[-(max_chars - 3):]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Contact sheet of audit duplicate pairs")
    ap.add_argument("--dataset-dir", required=True, help="dataset root the csv paths are relative to")
    ap.add_argument("--audit-dir", default="paper_materials/leakage_audit",
                    help="dir holding duplicates.csv; the sheet is written here")
    ap.add_argument("--rows", type=int, default=24, help="rows to sample (all, if fewer exist)")
    ap.add_argument("--seed", type=int, default=0, help="fixed seed, so the sheet is reproducible")
    ap.add_argument("--out", default=None, help="override output path")
    a = ap.parse_args(argv)

    root = os.path.abspath(a.dataset_dir)
    csv_path = os.path.join(a.audit_dir, "duplicates.csv")
    if not os.path.isfile(csv_path):
        print(f"no duplicates.csv at {csv_path} — run tools/leakage_audit.py first", file=sys.stderr)
        return 1

    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{csv_path} has no rows: no duplicates were found, so there is no sheet to draw.")
        return 0

    n = min(a.rows, len(rows))
    idx = np.random.default_rng(a.seed).permutation(len(rows))[:n]
    idx = sorted(int(i) for i in idx)          # keep csv order within the sample
    sample = [rows[i] for i in idx]
    print(f"{len(rows)} duplicate row(s); drawing {n} at seed {a.seed}"
          f"{' (all of them)' if n == len(rows) else ''}")

    row_h = CELL_H + LABEL_H + PAD
    sheet = np.full((row_h * n + PAD, CELL_W * 2 + PAD * 3, 3), 18, np.uint8)

    for k, r in enumerate(sample):
        left = cv2.imread(os.path.join(root, r["test_image"]), cv2.IMREAD_COLOR)
        right = cv2.imread(os.path.join(root, r["match"]), cv2.IMREAD_COLOR)
        y = PAD + k * row_h
        sheet[y:y + CELL_H, PAD:PAD + CELL_W] = fit(left)
        sheet[y:y + CELL_H, PAD * 2 + CELL_W:PAD * 2 + CELL_W * 2] = fit(right)

        # duplicates.csv line 1 is the header, so data row idx[k] sits on line idx[k]+2.
        cap = (f"ncc {float(r['ncc']):.4f}   transform {r['transform']}   "
               f"hamming {r['hamming']}   [pair {k + 1}/{n}, csv line {idx[k] + 2}]")
        cv2.putText(sheet, cap, (PAD, y + CELL_H + 18), FONT, 0.52, (90, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, ellipsize("test:  " + r["test_image"].replace("\\", "/"), 46),
                    (PAD + 6, y + 20), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(sheet, ellipsize("match: " + r["match"].replace("\\", "/"), 46),
                    (PAD * 2 + CELL_W + 6, y + 20), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(sheet, (0, y + CELL_H + LABEL_H + PAD // 2),
                 (sheet.shape[1], y + CELL_H + LABEL_H + PAD // 2), (60, 60, 60), 1)

    out = a.out or os.path.join(a.audit_dir, "pairs_sample.jpg")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if not cv2.imwrite(out, sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        print(f"failed to write {out}", file=sys.stderr)
        return 1
    print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]}, {n} pair(s))")
    print("Inspect the pairs by eye before quoting the contamination rate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
