"""Revision R1, Part D: unpack the MOT17 train sequences and lay out TrackEval ground truth.

Only the 7 train sequences of the FRCNN variant are extracted (all three detector variants share
the same images and ground truth). The ground truth is copied UNCHANGED into TrackEval's
MotChallenge2DBox layout, so TrackEval applies its own standard pedestrian / distractor /
zero-marked handling; nothing here filters or edits a ground-truth row.

The ~7 updates-per-second condition keeps every round(fps / 7)-th frame (frame 1, 1+k, 1+2k, ...).
Its ground truth is the same rows restricted to the kept frames, renumbered 1..K so TrackEval can
read it, with seqLength = K and frameRate = fps / k in seqinfo.ini; frame_map.json records the
renumbering. Data stays under data/ (gitignored).

    .venv\\Scripts\\python.exe tools\\revision\\mot17_prepare.py
"""
from __future__ import annotations

import configparser
import csv
import json
import os
import shutil
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MOT = os.path.join(ROOT, "data", "mot17")
ZIP = os.path.join(MOT, "MOT17.zip")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
# Camera motion as described by MOTChallenge for these sequences (and in the task brief).
CAMERA = {"MOT17-02-FRCNN": "static", "MOT17-04-FRCNN": "static", "MOT17-05-FRCNN": "moving",
          "MOT17-09-FRCNN": "static", "MOT17-10-FRCNN": "moving", "MOT17-11-FRCNN": "moving",
          "MOT17-13-FRCNN": "moving"}
TE = os.path.join(MOT, "trackeval")
OUT = os.path.join(ROOT, "paper_materials", "revision_r1", "partD")
RATE = 7.0


def seqinfo(path):
    c = configparser.ConfigParser()
    c.optionxform = str            # keep 'frameRate' etc. as written (default lowercases)
    c.read(path)
    s = c["Sequence"]
    return {k: s[k] for k in s}


def write_seqinfo(path, info):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("[Sequence]\n")
        for k in ("name", "imDir", "frameRate", "seqLength", "imWidth", "imHeight", "imExt"):
            fh.write(f"{k}={info[k]}\n")


def extract():
    want = tuple(f"MOT17/train/{s}/" for s in SEQS)
    with zipfile.ZipFile(ZIP) as z:
        members = [m for m in z.namelist() if m.startswith(want)]
        todo = [m for m in members if not os.path.exists(os.path.join(MOT, m))]
        for m in todo:
            z.extract(m, MOT)
    return len(members), len(todo)


def main() -> int:
    if not os.path.isfile(ZIP):
        print(f"[FAIL] {ZIP} not found", file=sys.stderr)
        return 1
    n_members, n_new = extract()
    print(f"[extract] {n_members} members for {len(SEQS)} sequences ({n_new} newly written)")
    seqs_out = []
    for split, step_rule in (("MOT17-train", None), ("MOT17sub7-train", RATE)):
        gt_root = os.path.join(TE, "gt", "mot_challenge", split)
        os.makedirs(os.path.join(TE, "gt", "mot_challenge", "seqmaps"), exist_ok=True)
        with open(os.path.join(TE, "gt", "mot_challenge", "seqmaps", f"{split}.txt"), "w",
                  encoding="utf-8") as fh:
            fh.write("name\n" + "\n".join(SEQS) + "\n")
        frame_maps = {}
        for s in SEQS:
            src = os.path.join(MOT, "MOT17", "train", s)
            info = seqinfo(os.path.join(src, "seqinfo.ini"))
            n_img = len([f for f in os.listdir(os.path.join(src, "img1")) if f.endswith(".jpg")])
            fps, length = float(info["frameRate"]), int(info["seqLength"])
            if n_img != length:
                raise SystemExit(f"[FAIL] {s}: {n_img} images but seqLength {length}")
            dst = os.path.join(gt_root, s)
            os.makedirs(os.path.join(dst, "gt"), exist_ok=True)
            rows = list(csv.reader(open(os.path.join(src, "gt", "gt.txt"), encoding="utf-8")))
            if step_rule is None:
                shutil.copyfile(os.path.join(src, "gt", "gt.txt"), os.path.join(dst, "gt", "gt.txt"))
                shutil.copyfile(os.path.join(src, "seqinfo.ini"), os.path.join(dst, "seqinfo.ini"))
                cls = {}
                for r in rows:
                    cls[int(r[7])] = cls.get(int(r[7]), 0) + 1
                seqs_out.append({"sequence": s, "camera": CAMERA[s], "fps": fps, "frames": length,
                                 "width": int(info["imWidth"]), "height": int(info["imHeight"]),
                                 "gt_rows": len(rows), "gt_rows_by_class": cls,
                                 "gt_ids_pedestrian_considered": len({int(r[1]) for r in rows
                                                                      if int(r[7]) == 1 and int(r[6]) == 1})})
            else:
                k = max(1, round(fps / step_rule))
                kept = list(range(1, length + 1, k))
                new_of = {f: i + 1 for i, f in enumerate(kept)}
                frame_maps[s] = {"step": k, "kept_frames": kept}
                with open(os.path.join(dst, "gt", "gt.txt"), "w", encoding="utf-8", newline="") as fh:
                    for r in rows:
                        f = int(r[0])
                        if f in new_of:
                            fh.write(",".join([str(new_of[f])] + r[1:]) + "\n")
                sub = dict(info, seqLength=str(len(kept)), frameRate=f"{fps / k:g}")
                write_seqinfo(os.path.join(dst, "seqinfo.ini"), sub)
        if frame_maps:
            with open(os.path.join(TE, "gt", "mot_challenge", f"{split}_frame_map.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(frame_maps, fh)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mot17_sequences.json"), "w", encoding="utf-8") as fh:
        json.dump({"sequences": seqs_out, "zip": ZIP, "zip_bytes": os.path.getsize(ZIP),
                   "subsample_rate_hz": RATE,
                   "subsample_steps": {s["sequence"]: max(1, round(s["fps"] / RATE)) for s in seqs_out}},
                  fh, indent=2)
    for s in seqs_out:
        print(f"{s['sequence']}: {s['camera']:<6} {s['fps']:g} fps, {s['frames']} frames, "
              f"{s['width']}x{s['height']}, {s['gt_ids_pedestrian_considered']} pedestrian ids")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
