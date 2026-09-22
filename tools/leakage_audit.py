"""Train-test near-duplicate audit for a YOLO-format dataset (Roboflow export).

Why: community datasets are often assembled from overlapping sources, and the same
photo (or a flipped / rotated / re-compressed copy) can sit in both train/ and test/.
That inflates test metrics. This script measures how much of the test split has a
near-duplicate in train (and valid), and writes a de-duplicated test list so the
detector can be re-scored on clean images only.

    python tools/leakage_audit.py --dataset-dir data/ppe_download
    python tools/leakage_audit.py --dataset-dir data/ppe_download --against train valid

Method (two stages, numpy + OpenCV only):
  1. Candidate search: 64-bit difference hash (dHash) of every image. Each TEST image is
     hashed under the 8 flips/rotations of the square (dihedral group), so mirrored or
     rotated copies are still found. Pairs within --hamming bits are candidates.
  2. Confirmation: normalised cross-correlation of 32x32 grey thumbnails (under the same
     transform) must reach --ncc. This removes hash collisions between different photos.

Outputs (in --out, default outputs/leakage_audit/):
  duplicates.csv        one row per contaminated test image and its best train match
  summary.json          counts and the contamination rate
  test_dedup.txt        absolute paths of the CLEAN test images (usable as `test:` in a
                        data.yaml: ultralytics accepts a .txt list of image paths)
  data_dedup.yaml       copy of data.yaml pointing `test:` at test_dedup.txt

Then re-score:   yolo val model=best_refined.pt data=outputs/leakage_audit/data_dedup.yaml split=test
Report BOTH numbers (full test split and de-duplicated) in the paper.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import cv2, numpy as np

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(EXT)]
    return sorted(out)


def dihedral(img):
    """The 8 symmetries of the square: 4 rotations x optional horizontal flip."""
    outs = []
    for flip in (False, True):
        base = cv2.flip(img, 1) if flip else img
        for k in range(4):
            outs.append(np.ascontiguousarray(np.rot90(base, k)))
    return outs


def dhash(gray):
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).astype(np.uint8).ravel()
    return np.packbits(bits)                      # 8 bytes


def thumb(gray):
    t = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 1e-6 else t


def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def main():
    ap = argparse.ArgumentParser(description="Train-test near-duplicate audit")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--test", default="test")
    ap.add_argument("--against", nargs="+", default=["train"], help="splits to search (train [valid])")
    ap.add_argument("--hamming", type=int, default=6, help="max dHash distance for a candidate (0-64)")
    ap.add_argument("--ncc", type=float, default=0.90, help="min thumbnail correlation to confirm")
    ap.add_argument("--out", default="outputs/leakage_audit")
    a = ap.parse_args()

    root = os.path.abspath(a.dataset_dir)
    test_imgs = list_images(os.path.join(root, a.test, "images"))
    ref_imgs = []
    for s in a.against:
        ref_imgs += list_images(os.path.join(root, s, "images"))
    if not test_imgs or not ref_imgs:
        sys.exit(f"no images found under {root} (expected <split>/images/)")
    print(f"test: {len(test_imgs)} images | reference ({'+'.join(a.against)}): {len(ref_imgs)} images")

    ref_hash = np.zeros((len(ref_imgs), 8), np.uint8)
    ref_thumb = np.zeros((len(ref_imgs), 1024), np.float32)
    ok = np.zeros(len(ref_imgs), bool)
    for i, p in enumerate(ref_imgs):
        g = load_gray(p)
        if g is None:
            continue
        ref_hash[i], ref_thumb[i], ok[i] = dhash(g), thumb(g), True
        if (i + 1) % 5000 == 0:
            print(f"  hashed {i + 1}/{len(ref_imgs)} reference images")

    rows, clean = [], []
    for j, p in enumerate(test_imgs):
        g = load_gray(p)
        if g is None:
            clean.append(p); continue
        best = (-1.0, -1, -1, 99)                         # ncc, ref index, transform, hamming
        for t, v in enumerate(dihedral(g)):
            h = dhash(v)
            dist = _POP[np.bitwise_xor(ref_hash, h)].sum(axis=1)
            cand = np.where((dist <= a.hamming) & ok)[0]
            if cand.size == 0:
                continue
            tv = thumb(v)
            ncc = ref_thumb[cand] @ tv
            k = int(np.argmax(ncc))
            if ncc[k] > best[0]:
                best = (float(ncc[k]), int(cand[k]), t, int(dist[cand[k]]))
        if best[0] >= a.ncc:
            rows.append(dict(test_image=os.path.relpath(p, root), match=os.path.relpath(ref_imgs[best[1]], root),
                             transform=best[2], hamming=best[3], ncc=round(best[0], 4)))
        else:
            clean.append(p)
        if (j + 1) % 500 == 0:
            print(f"  checked {j + 1}/{len(test_imgs)} test images, {len(rows)} contaminated so far")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "duplicates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test_image", "match", "transform", "hamming", "ncc"])
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(a.out, "test_dedup.txt"), "w") as f:
        f.write("\n".join(clean) + "\n")
    rate = len(rows) / len(test_imgs)
    summary = dict(test_images=len(test_imgs), reference_images=len(ref_imgs), against=a.against,
                   contaminated=len(rows), clean=len(clean), contamination_rate=round(rate, 4),
                   hamming=a.hamming, ncc=a.ncc)
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    try:
        import yaml
        spec = yaml.safe_load(open(os.path.join(root, "data.yaml")))
        spec["path"] = root
        spec["train"], spec["val"] = "train/images", "valid/images"
        spec["test"] = os.path.abspath(os.path.join(a.out, "test_dedup.txt"))
        yaml.safe_dump(spec, open(os.path.join(a.out, "data_dedup.yaml"), "w"), sort_keys=False)
    except Exception as e:                                # yaml is optional
        print(f"[note] data_dedup.yaml not written ({e}); point `test:` at test_dedup.txt by hand")
    print(json.dumps(summary, indent=2))
    print(f"\n{len(rows)}/{len(test_imgs)} test images ({rate:.2%}) have a near-duplicate in {'+'.join(a.against)}.")
    print("Spot-check duplicates.csv by eye (open 20 pairs) before quoting the rate.")


if __name__ == "__main__":
    main()
