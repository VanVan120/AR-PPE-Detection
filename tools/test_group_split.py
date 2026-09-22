"""Self-test for tools/group_split.py on a generated fixture.

The fixture is built to contain exactly the three things that can break a source-grouped
split:

  * **nested re-export names** — `sceneNN_png_jpg.rf.<hash>.jpg` alongside
    `sceneNN_jpg.rf.<hash>.jpg`, which must collapse onto one source `sceneNN`;
  * **a generic stem** — many unrelated photographs all named `image_<k>`, which must NOT
    be welded into one giant group just because the stem repeats;
  * **a cross-stem duplicate pair** — the same photograph exported under two different
    names, which the stem rule alone cannot catch and the pixel matcher must;
  * **a non-generic stem inside the oversized generic group**, whose two files are not
    pixel-similar to each other, so only the stem link holds them together. Decomposing
    that group must dissolve the generic stem's links and keep this one's.

Asserts: every image is in the CSV and no split is empty (an empty test split would make
the leakage audit pass vacuously); no group spans two splits (a format check — the CSV
makes split a function of the group id, so this cannot fail and is not evidence); **no
non-generic source stem spans two splits or is scattered across groups** (the invariant
that can actually fail); the generic stem is decomposed rather than welded; the same input
twice gives the same sha256; `--apply` writes its three files and refuses an incomplete
CSV; and `leakage_audit.py` finds 0 contaminated test images in the materialised output.

    python tools/test_group_split.py        # -> ALL_GROUP_SPLIT True
"""
from __future__ import annotations

import collections
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
rng = np.random.default_rng(7)


def photo(seed=None):
    """A crude distinctive 'photograph' — same generator shape as test_leakage_audit.py."""
    r = np.random.default_rng(seed) if seed is not None else rng
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = r.integers(40, 200, 3)
    for _ in range(14):
        p1 = (int(r.integers(0, 320)), int(r.integers(0, 240)))
        p2 = (int(r.integers(0, 320)), int(r.integers(0, 240)))
        cv2.rectangle(img, p1, p2, [int(c) for c in r.integers(0, 255, 3)], -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def write(d, split, name, img, boxes):
    cv2.imwrite(os.path.join(d, split, "images", name), img,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    lab = os.path.join(d, split, "labels", os.path.splitext(name)[0] + ".txt")
    with open(lab, "w", encoding="utf-8") as fh:
        for ci, x, y, w, h in boxes:
            fh.write(f"{ci} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def boxes_for(k):
    """Two classes, deliberately unbalanced, so the class-share check has work to do."""
    out = [(0, 0.5, 0.5, 0.3, 0.3)]
    if k % 3 == 0:
        out.append((1, 0.25, 0.25, 0.2, 0.2))
    if k % 7 == 0:
        out.append((1, 0.75, 0.75, 0.2, 0.2))
    return out


def build_fixture(d):
    for s in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(d, s, sub), exist_ok=True)
    with open(os.path.join(d, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("names:\n- Alpha\n- Beta\nnc: 2\n"
                 "train: ../train/images\nval: ../valid/images\ntest: ../test/images\n")

    expect_same_source = []

    # 40 ordinary sources, 3 augmented copies each, scattered across the 3 splits so the
    # ORIGINAL split is deliberately leaky (the thing group_split has to repair).
    for k in range(40):
        base = photo(1000 + k)
        names = []
        for c in range(3):
            im = cv2.convertScaleAbs(base, alpha=1.0 + 0.03 * c, beta=2 * c)
            nm = f"scene{k:02d}_jpg.rf.{k:03d}{c}aaaaaaaaaaaaaaaaaaaa.jpg"
            write(d, ("train", "valid", "test")[c % 3], nm, im, boxes_for(k))
            names.append(nm)
        # a NESTED re-export of the same source: must land in the same group
        nm = f"scene{k:02d}_png_jpg.rf.{k:03d}9bbbbbbbbbbbbbbbbbbb.jpg"
        write(d, "test" if k % 2 else "train", nm, base, boxes_for(k))
        names.append(nm)
        expect_same_source.append(names)

    # 30 UNRELATED photographs called image_<k>. Each stem is distinct, so the stem rule
    # already keeps them apart; this guards against over-grouping by prefix.
    for k in range(30):
        write(d, ("train", "valid", "test")[k % 3], f"image_{k}_jpg.rf.g{k:03d}cccccccccccccccc.jpg",
              photo(5000 + k), boxes_for(k))

    # The dangerous case: 24 UNRELATED photographs sharing ONE generic stem, `image`.
    # The stem rule alone would weld them into a single oversized group and then strand
    # 24 different photographs in one split. group_split must notice the group is both
    # oversized and generically named, and decompose it by pixel duplicates instead.
    for k in range(24):
        write(d, ("train", "valid", "test")[k % 3],
              f"image_jpg.rf.h{k:03d}ffffffffffffffffff.jpg", photo(7000 + k), boxes_for(k))

    # A cross-stem duplicate pair: one photograph, two unrelated names, opposite splits.
    twin = photo(9999)
    write(d, "train", "alpha_jpg.rf.dup1dddddddddddddddddd.jpg", twin, boxes_for(1))
    write(d, "test", "zulu_jpg.rf.dup2eeeeeeeeeeeeeeeeeee.jpg",
          cv2.flip(twin, 1), boxes_for(1))
    expect_same_source.append(["alpha_jpg.rf.dup1dddddddddddddddddd.jpg",
                               "zulu_jpg.rf.dup2eeeeeeeeeeeeeeeeeee.jpg"])

    # A NON-generic stem dragged into the oversized generic group by one duplicate link,
    # whose two files are NOT pixel-similar to each other. Only the stem link holds them
    # together, so if decomposition dissolved non-generic stem links they would scatter.
    hook = photo(7000)                       # == the first image_jpg photo, so they link
    write(d, "train", "roofsite_jpg.rf.k001gggggggggggggggggg.jpg",
          cv2.convertScaleAbs(hook, alpha=1.02, beta=2), boxes_for(2))
    write(d, "test", "roofsite_jpg.rf.k002hhhhhhhhhhhhhhhhhh.jpg",
          photo(8123), boxes_for(2))         # a completely different photograph
    expect_same_source.append(["roofsite_jpg.rf.k001gggggggggggggggggg.jpg",
                               "roofsite_jpg.rf.k002hhhhhhhhhhhhhhhhhh.jpg"])
    return expect_same_source


def build_split(d, csv_path, report_dir):
    r = subprocess.run(
        [PY, os.path.join(HERE, "group_split.py"), "--dataset-dir", d,
         "--out-csv", csv_path, "--report-dir", report_dir, "--generic-pct", "5"],
        capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:])
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit("group_split.py --build failed")
    return r.stdout


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    d = tempfile.mkdtemp(prefix="grpsplit_")
    checks = {}
    try:
        expect_same_source = build_fixture(d)
        n_imgs = sum(len(os.listdir(os.path.join(d, s, "images")))
                     for s in ("train", "valid", "test"))
        print(f"fixture: {n_imgs} images in {d}")

        csv1 = os.path.join(d, "split1.csv")
        csv2 = os.path.join(d, "split2.csv")
        rep = os.path.join(d, "rep")
        out1 = build_split(d, csv1, rep)
        report = json.load(open(os.path.join(rep, "group_split_report.json"), encoding="utf-8"))

        rows = list(csv.DictReader(open(csv1, newline="", encoding="utf-8")))
        by_name = {os.path.basename(r["file"]): r for r in rows}
        checks["every image is in the csv"] = (len(rows) == n_imgs)

        # Guard against a vacuous pass later: an empty test split would make the
        # leakage audit report 0 contaminated images no matter how broken the split is.
        per_split = collections.Counter(r["split"] for r in rows)
        print(f"  split sizes: {dict(per_split)}")
        checks["all three splits are non-empty"] = all(
            per_split[s] > 0 for s in ("train", "valid", "test"))

        # 1. no group spans two splits.
        # Note this is true by construction -- the CSV writes split as a function of the
        # group id -- so it is kept as a format check, not evidence. The invariant that
        # CAN fail is checked in 1b: no non-generic SOURCE STEM may span two splits.
        gsplits = {}
        spanning = []
        for r in rows:
            gsplits.setdefault(r["group"], set()).add(r["split"])
        for g, ss in gsplits.items():
            if len(ss) > 1:
                spanning.append((g, sorted(ss)))
        checks["no group spans two splits (format check)"] = not spanning
        if spanning:
            print("  spanning groups:", spanning[:5])

        # 1a. THE load-bearing assertion: a real source photograph must not be in two
        # splits. A generic stem may be, because it names several photographs.
        sys.path.insert(0, HERE)
        import group_split as gsmod
        stem_splits, stem_groups = {}, {}
        for r in rows:
            st = gsmod.source_stem(os.path.basename(r["file"]))
            stem_splits.setdefault(st, set()).add(r["split"])
            stem_groups.setdefault(st, set()).add(r["group"])
        leaked = sorted(s for s, v in stem_splits.items()
                        if len(v) > 1 and not gsmod.is_generic_stem(s))
        shattered = sorted(s for s, v in stem_groups.items()
                           if len(v) > 1 and not gsmod.is_generic_stem(s))
        checks["no non-generic source stem spans two splits"] = not leaked
        checks["no non-generic source stem is split across groups"] = not shattered
        print(f"  source stems: {len(stem_splits)}; non-generic spanning splits "
              f"{len(leaked)}; non-generic split across groups {len(shattered)}")
        if leaked:
            print("  LEAKED stems:", leaked[:5])
        if shattered:
            print("  shattered stems:", shattered[:5])

        # 1b. the things that share a source really do share a group (and so a split)
        bad = []
        for names in expect_same_source:
            gs = {by_name[n]["group"] for n in names if n in by_name}
            if len(gs) != 1:
                bad.append((names[0], sorted(gs)))
        checks["nested re-exports and the cross-stem twin share their source's group"] = not bad
        if bad:
            print("  split across groups:", bad[:5])

        # 1c. the generic stem did NOT weld 30 unrelated photos together
        gen_groups = {by_name[n]["group"] for n in by_name
                      if n.startswith("image_") and not n.startswith("image_jpg.rf.")}
        checks["distinct image_<k> stems stay apart"] = (len(gen_groups) >= 25)
        print(f"  the 30 distinct image_<k> photos occupy {len(gen_groups)} group(s)")

        # 1d. the ONE shared generic stem was decomposed, not left as a 24-photo group
        shared = {by_name[n]["group"] for n in by_name if n.startswith("image_jpg.rf.")}
        handled = report.get("generic_stem_groups", [])
        checks["the oversized generic-stem group was decomposed"] = (len(shared) >= 20)
        checks["the report records the generic stem it decomposed"] = any(
            "image" in (h.get("generic_stems_present") or []) for h in handled)
        print(f"  the 24 photos sharing the stem 'image' occupy {len(shared)} group(s)")
        for h in handled:
            print(f"  generic group handled: size {h['size']} "
                  f"({h['pct_of_dataset']}%), dominant stem {h['dominant_stem']!r} "
                  f"-> {h['action']}")

        # 2. determinism
        build_split(d, csv2, os.path.join(d, "rep2"))
        rows2 = list(csv.DictReader(open(csv2, newline="", encoding="utf-8")))
        sys.path.insert(0, HERE)
        import group_split as gs
        sha1, sha2 = gs.assignment_sha256(rows), gs.assignment_sha256(rows2)
        checks["two runs give the same sha256"] = (sha1 == sha2)
        print(f"  sha256 run1 {sha1[:16]}…  run2 {sha2[:16]}…")

        # 3. apply, then audit the result: test must not be contaminated by train+valid
        applied = os.path.join(d, "grouped")
        r = subprocess.run([PY, os.path.join(HERE, "group_split.py"), "--apply", csv1,
                            "--dataset-dir", d, "--out", applied], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:], file=sys.stderr)
            raise SystemExit("group_split.py --apply failed")
        for f in ("data.yaml", "split_summary.json", "test_one_per_source.txt"):
            checks[f"--apply wrote {f}"] = os.path.isfile(os.path.join(applied, f))

        aud = os.path.join(d, "audit")
        r = subprocess.run([PY, os.path.join(HERE, "leakage_audit.py"), "--dataset-dir", applied,
                            "--against", "train", "valid", "--out", aud],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:]); print(r.stderr[-3000:], file=sys.stderr)
            raise SystemExit("leakage_audit.py on the output failed")
        s = json.load(open(os.path.join(aud, "summary.json"), encoding="utf-8"))
        print(f"  audit of the regrouped output: {s['contaminated']}/{s['test_images']} "
              f"contaminated test images")
        checks["leakage_audit finds 0 contaminated test images"] = (s["contaminated"] == 0)

        # --apply must refuse a CSV that does not match the dataset
        bad_csv = os.path.join(d, "bad.csv")
        with open(bad_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["file", "group", "split"], lineterminator="\n")
            w.writeheader(); w.writerows(rows[:-3])
        r = subprocess.run([PY, os.path.join(HERE, "group_split.py"), "--apply", bad_csv,
                            "--dataset-dir", d, "--out", os.path.join(d, "bad_out")],
                           capture_output=True, text=True)
        checks["--apply fails loudly on an incomplete csv"] = (r.returncode != 0)

    finally:
        shutil.rmtree(d, ignore_errors=True)

    print()
    for k, v in checks.items():
        print(f"  [{'ok ' if v else 'FAIL'}] {k}")
    allok = all(checks.values())
    print(f"\nALL_GROUP_SPLIT {allok}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
