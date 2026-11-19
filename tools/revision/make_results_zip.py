"""Revision R1: pack paper_materials/revision_r1/ into paper_materials/revision_r1_results.zip.

Everything in the results folder goes in except what the revision rules keep out of any upload:
weights (*.pt and other model files), images and video (frames), numpy arrays, ultralytics label
caches (*.cache, which hold dataset annotations), and the per-condition split folders that
eval_grouped.py writes (eval_splits/, eval_runs/: image lists and validator output). Every
skipped file is listed in the zip's MANIFEST.txt, so nothing is left out silently.

    .venv\\Scripts\\python.exe tools\\revision\\make_results_zip.py
"""
from __future__ import annotations

import datetime
import hashlib
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "paper_materials", "revision_r1")
OUT = os.path.join(ROOT, "paper_materials", "revision_r1_results.zip")
SKIP_EXT = {".pt", ".pth", ".onnx", ".engine", ".tflite", ".pb", ".safetensors", ".jpg", ".jpeg",
            ".png", ".bmp", ".webp", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".npy", ".npz", ".cache"}
SKIP_DIRS = {"eval_splits", "eval_runs", "__pycache__"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    kept, skipped = [], []
    for dp, dns, fns in os.walk(SRC):
        rel_dir = os.path.relpath(dp, SRC)
        for d in sorted(dns):
            if d in SKIP_DIRS:
                skipped.append((os.path.normpath(os.path.join(rel_dir, d)) + "/", "folder"))
        dns[:] = sorted(d for d in dns if d not in SKIP_DIRS)
        for fn in sorted(fns):
            rel = os.path.normpath(os.path.join(rel_dir, fn))
            if os.path.splitext(fn)[1].lower() in SKIP_EXT:
                skipped.append((rel, "weights/image/video/array/label cache"))
            else:
                kept.append(rel)
    tmp = OUT + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        lines = [f"revision_r1 results, packed {datetime.datetime.now().isoformat(timespec='seconds')}",
                 f"by tools/revision/make_results_zip.py from {SRC}", "",
                 f"{len(kept)} files included (sha256, bytes, path):"]
        for rel in kept:
            p = os.path.join(SRC, rel)
            z.write(p, os.path.join("revision_r1", rel).replace(os.sep, "/"))
            lines.append(f"{sha256(p)}  {os.path.getsize(p):>10}  {rel.replace(os.sep, '/')}")
        lines += ["", f"{len(skipped)} left out (not results, or not allowed in an upload):"]
        lines += [f"  {rel.replace(os.sep, '/')}  ({why})" for rel, why in skipped]
        z.writestr("revision_r1/MANIFEST.txt", "\n".join(lines) + "\n")
    os.replace(tmp, OUT)
    print(f"{OUT}: {len(kept)} files, {len(skipped)} left out, {os.path.getsize(OUT) / 1e6:.1f} MB")
    for rel, why in skipped:
        print(f"  left out: {rel}  ({why})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
