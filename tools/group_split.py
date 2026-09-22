"""Re-split a Roboflow YOLO dataset so no source photo straddles two splits.

Why: `tools/leakage_audit.py` showed that the PPE export's own split leaks badly — 77.95%
of its test images have a near-duplicate in train/valid, and 98.57% share a Roboflow
source stem with train/valid. Its 41,730 images come from ~5,800 source photographs, and
the split was plainly made per *image* after augmentation rather than per *photograph*.
Any accuracy measured on that split is inflated.

This builds a split whose unit is the source photograph, not the file:

  1. **Stem link.** Two images with the same Roboflow source stem are linked. The stem is
     the file name with the `.rf.<hash>` tail removed and then every trailing `_<ext>`
     token removed, repeatedly, so nested re-exports (`foo_png_jpg.rf.…`) collapse onto
     the same source as `foo_jpg.rf.…`.
  2. **Duplicate link.** Two images the matcher of `leakage_audit.py` calls near-duplicates
     are linked — same dHash/NCC test, ncc 0.90, all 8 flips and rotations — searched over
     the WHOLE dataset, not just test against train.
  3. **Group** = a connected component of those links. Groups are assigned whole to
     train / valid / test, so no photograph and no copy of one can span two splits.

The candidate search is multi-index hashing: with a 64-bit hash in 8 one-byte blocks and a
radius of 6 bits, two hashes within the radius must agree exactly on at least two blocks
(pigeonhole), so indexing every block and taking the union of the query's eight buckets
gives a superset of the true neighbours. That turns an all-pairs scan into a few hundred
candidates per query.

    # build the split definition (writes the CSV that IS the split)
    python tools/group_split.py --dataset-dir data/ppe_download \
        --out-csv tools/splits/ppe_grouped_split.csv --report-dir paper_materials/retrain

    # materialise it as a real dataset directory
    python tools/group_split.py --apply tools/splits/ppe_grouped_split.csv \
        --dataset-dir data/ppe_download --out data/ppe_grouped

`--apply` is what Kaggle runs: it re-derives nothing, it just obeys the CSV.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import leakage_audit as la          # reuse the EXACT matcher, not a copy of it  # noqa: E402

SPLITS = ("train", "valid", "test")
RATIO = {"train": 0.80, "valid": 0.10, "test": 0.10}
EXT_TOKENS = ("png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff", "gif")
RF_TAIL = re.compile(r"\.rf\.[A-Za-z0-9]+$")
_EXT_TOKEN_RE = re.compile(r"_(" + "|".join(EXT_TOKENS) + r")$", re.I)

# A stem that names nothing in particular, so that many unrelated photographs can share
# it and the shared name is not evidence of a shared source. Checked only for groups big
# enough to matter.
#
# This is deliberately STRICT: a bare word (`image`, `frame`) or a bare number only.
# A NUMBERED variant such as `image_150` is NOT generic, because on this dataset it names
# one photograph and its augmentations. That was measured, not assumed: `image_150`'s 121
# files have a mean pairwise thumbnail NCC of 0.69-0.87 against a baseline of 0.06 for
# unrelated stems, i.e. they are one scene. They are nonetheless spread over 12 components
# under the ncc-0.90 matcher, because heavy augmentation breaks the chain — so the stem
# link is the ONLY thing holding them together, and dissolving it would put augmented
# copies of one photograph on both sides of the split. The earlier, looser rule (which
# also matched `<word><digits>`) classified 79% of this dataset's stems as generic and did
# exactly that to 11 stems / 469 images.
#
# The asymmetry is deliberate: over-grouping costs some test-set diversity, under-grouping
# causes leakage. Prefer over-grouping.
GENERIC_WORDS = {"image", "images", "img", "frame", "frames", "photo", "photos", "picture",
                 "pictures", "screenshot", "screenshots", "capture", "untitled", "default",
                 "download", "unnamed", "new", "test", "temp", "tmp", "file", "scan",
                 "pic", "pics", "shot", "snap", "output", "result", "data", "sample"}
_GENERIC_RE = re.compile(
    r"^[-_ ]*(" + "|".join(sorted(GENERIC_WORDS)) + r")[-_ ]*$", re.I)


# ---------------------------------------------------------------- source stem

def source_stem(basename: str) -> str:
    """`foo_png_jpg.rf.deadbeef.jpg` -> `foo`. Idempotent, and collapses nested exports."""
    head = os.path.splitext(basename)[0]
    head = RF_TAIL.sub("", head)
    while True:
        m = _EXT_TOKEN_RE.search(head)
        if not m:
            return head
        head = head[:m.start()]


def is_generic_stem(stem: str) -> bool:
    s = stem.strip()
    if not s:
        return True
    if re.fullmatch(r"[-_ ]*\d+[-_ ]*", s):      # a bare number
        return True
    return bool(_GENERIC_RE.fullmatch(s))


# ---------------------------------------------------------------- union-find

class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        p = self.p
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


# ---------------------------------------------------------------- dataset scan

def scan_dataset(root):
    """Every image under <root>/<split>/images, as (rel_posix_path, abs_path)."""
    out = []
    for split in SPLITS:
        d = os.path.join(root, split, "images")
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.lower().endswith(la.EXT):
                out.append((f"{split}/images/{name}", os.path.join(d, name)))
    return out


def label_path_for(abs_img):
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in abs_img:
        return None
    head, _, tail = abs_img.rpartition(sa)
    return os.path.splitext(head + sb + tail)[0] + ".txt"


def class_counts(abs_img):
    """{class_id: n_boxes} for one image."""
    lp = label_path_for(abs_img)
    out = collections.Counter()
    if not lp or not os.path.isfile(lp):
        return out
    with open(lp, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out[int(float(line.split()[0]))] += 1
                except (ValueError, IndexError):
                    pass
    return out


# ---------------------------------------------------------------- hashing

def hash_all(items, progress_every=2500):
    """Identity + all 8 dihedral hashes, and the normalised 32x32 thumbnail, per image.

    Only the identity thumbnail is stored. A thumbnail of a flipped/rotated image is the
    same 32x32 grid with its pixels permuted, and mean-subtraction and normalisation are
    invariant under permutation, so the 8 transformed thumbnails are recovered exactly by
    permuting this one (verified to 1.5e-8 against recomputing them).
    """
    n = len(items)
    H = np.zeros((n, 8, 8), np.uint8)          # [image, transform, byte]
    TH = np.zeros((n, 32, 32), np.float32)
    ok = np.zeros(n, bool)
    t0 = time.time()
    for i, (_rel, ap) in enumerate(items):
        g = la.load_gray(ap)
        if g is not None:
            for t, v in enumerate(la.dihedral(g)):
                H[i, t] = la.dhash(v)
            TH[i] = la.thumb(g).reshape(32, 32)
            ok[i] = True
        if progress_every and (i + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"  hashed {i + 1}/{n}  ({el:.0f}s elapsed, "
                  f"~{el / (i + 1) * (n - i - 1):.0f}s left)", flush=True)
    print(f"  hashed {n}/{n} in {time.time() - t0:.0f}s "
          f"({int(ok.sum())} readable, {int((~ok).sum())} unreadable)", flush=True)
    return H, TH, ok


def build_index(H0, ok):
    """Eight buckets-of-indices, one per hash byte: byte position -> value -> indices."""
    idx = np.flatnonzero(ok)
    buckets = []
    for b in range(8):
        order = np.argsort(H0[idx, b], kind="stable")
        s = idx[order]
        vals = H0[s, b]
        starts = np.searchsorted(vals, np.arange(256), side="left")
        ends = np.searchsorted(vals, np.arange(256), side="right")
        buckets.append((s, starts, ends))
    return buckets


def dup_links(H, TH, ok, buckets, stem_id, hamming, ncc_min, restrict=None,
              allow_same_stem=False, progress_every=5000):
    """Yield (i, j) pairs the leakage_audit matcher confirms as near-duplicates.

    Same-stem pairs are skipped by default: they are already linked by their stem, so
    confirming them cannot change a single connected component. `allow_same_stem=True` is
    used for the second pass that decomposes an oversized generic-stem group, where the
    duplicate relation alone is what matters.
    """
    H0 = np.ascontiguousarray(H[:, 0, :])
    queries = np.flatnonzero(ok) if restrict is None else np.array(
        [i for i in restrict if ok[i]], dtype=np.int64)
    in_scope = None
    if restrict is not None:
        in_scope = np.zeros(len(ok), bool)
        in_scope[queries] = True
    n_pairs = 0
    t0 = time.time()
    for qn, i in enumerate(queries):
        for t in range(8):
            q = H[i, t]
            parts = []
            for b in range(8):
                s, starts, ends = buckets[b]
                lo, hi = starts[q[b]], ends[q[b]]
                if hi > lo:
                    parts.append(s[lo:hi])
            if not parts:
                continue
            cand = np.unique(np.concatenate(parts))
            cand = cand[cand != i]
            if in_scope is not None:
                cand = cand[in_scope[cand]]
            if not allow_same_stem and cand.size:
                cand = cand[stem_id[cand] != stem_id[i]]
            if not cand.size:
                continue
            dist = la._POP[np.bitwise_xor(H0[cand], q)].sum(axis=1)
            cand = cand[dist <= hamming]
            if not cand.size:
                continue
            tq = np.ascontiguousarray(la.dihedral(TH[i])[t]).ravel()
            sim = TH[cand].reshape(cand.size, -1) @ tq
            for j in cand[sim >= ncc_min]:
                n_pairs += 1
                yield i, int(j)
        if progress_every and (qn + 1) % progress_every == 0:
            el = time.time() - t0
            print(f"  matched {qn + 1}/{len(queries)} images, {n_pairs} duplicate link(s) "
                  f"({el:.0f}s elapsed, ~{el / (qn + 1) * (len(queries) - qn - 1):.0f}s left)",
                  flush=True)
    print(f"  matched {len(queries)}/{len(queries)} images, {n_pairs} duplicate link(s) "
          f"in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- assignment

def assign_groups(groups, sizes, seed):
    """Whole groups to splits, aiming at RATIO by image count.

    Shuffle with the seed, then give each group to whichever split is furthest below its
    target. The shuffle is what the seed changes; the greedy step keeps the totals close.
    """
    order = list(range(len(groups)))
    random.Random(seed).shuffle(order)
    total = sum(sizes)
    have = {s: 0 for s in SPLITS}
    out = {}
    for gi in order:
        deficit = {s: RATIO[s] * total - have[s] for s in SPLITS}
        pick = max(SPLITS, key=lambda s: (deficit[s], s == "train", s))
        out[gi] = pick
        have[pick] += sizes[gi]
    return out, have


def instance_shares(group_class_counts, assign, n_groups):
    """Per-split {class_id: n_boxes}, plus the overall totals."""
    per = {s: collections.Counter() for s in SPLITS}
    overall = collections.Counter()
    for gi in range(n_groups):
        cc = group_class_counts[gi]
        per[assign[gi]].update(cc)
        overall.update(cc)
    return per, overall


def share_check(per, overall, split="test", tol_pp=3.0):
    """Every class's share of boxes in `split` vs its share overall, in points."""
    tot_all = sum(overall.values())
    tot_split = sum(per[split].values())
    rows, worst = [], 0.0
    for ci in sorted(overall):
        s_all = 100.0 * overall[ci] / tot_all if tot_all else 0.0
        s_spl = 100.0 * per[split][ci] / tot_split if tot_split else 0.0
        d = s_spl - s_all
        worst = max(worst, abs(d))
        rows.append({"class_id": ci, "overall_pct": s_all, f"{split}_pct": s_spl,
                     "diff_pp": d, "boxes_overall": overall[ci],
                     f"boxes_{split}": per[split][ci]})
    return rows, worst, worst <= tol_pp


# ---------------------------------------------------------------- build

def assignment_sha256(rows):
    """Stable over platforms: forward slashes, sorted, `file,split` per line."""
    h = hashlib.sha256()
    for f, s in sorted((r["file"], r["split"]) for r in rows):
        h.update(f"{f},{s}\n".encode())
    return h.hexdigest()


def cmd_build(a) -> int:
    # Multi-index hashing is only a superset of the true neighbours while the radius is
    # smaller than the number of blocks: 8 one-byte blocks tolerate at most 7 differing
    # bits before a pair can differ in every block and be missed entirely.
    if a.hamming >= 8:
        print(f"--hamming {a.hamming} exceeds what the 8-block index can guarantee "
              f"(needs < 8); it would silently MISS duplicates. Refusing.", file=sys.stderr)
        return 1

    root = os.path.abspath(a.dataset_dir)
    items = scan_dataset(root)
    if not items:
        print(f"no images under {root}/<split>/images", file=sys.stderr)
        return 1
    n = len(items)
    print(f"{n} images under {root}")

    stems = [source_stem(os.path.basename(rel)) for rel, _ in items]
    stem_ids = {s: k for k, s in enumerate(sorted(set(stems)))}
    stem_id = np.array([stem_ids[s] for s in stems], dtype=np.int64)
    print(f"{len(stem_ids)} distinct source stems")

    print("hashing (identity + 8 dihedral) ...")
    H, TH, ok = hash_all(items)
    print("indexing (multi-index hashing on the 8 hash bytes) ...")
    buckets = build_index(np.ascontiguousarray(H[:, 0, :]), ok)

    dsu = DSU(n)
    by_stem = collections.defaultdict(list)                       # --- stem links ---
    for i, s in enumerate(stems):
        by_stem[s].append(i)
    stem_link_count = 0
    for idxs in by_stem.values():
        for j in idxs[1:]:
            if dsu.union(idxs[0], j):
                stem_link_count += 1

    print(f"searching for near-duplicates across the whole dataset "
          f"(ncc {a.ncc}, hamming {a.hamming}, 8 transforms) ...")
    cross = 0
    for i, j in dup_links(H, TH, ok, buckets, stem_id, a.hamming, a.ncc):
        if dsu.union(i, j):
            cross += 1
    print(f"stem links merged {stem_link_count} image(s); duplicate links merged "
          f"{cross} further component(s)")

    # --- groups ---------------------------------------------------------------
    comp = collections.defaultdict(list)
    for i in range(n):
        comp[dsu.find(i)].append(i)
    groups = list(comp.values())

    # --- oversized generic-stem groups get decomposed by duplicates alone -----
    limit = a.generic_pct / 100.0 * n
    generic_handled = []
    final_groups = []
    for g in groups:
        if len(g) <= limit:
            final_groups.append(g)
            continue
        cnt = collections.Counter(stems[i] for i in g)
        dominant, dom_n = cnt.most_common(1)[0]
        gen = [s for s in cnt if is_generic_stem(s)]
        info = {"size": len(g), "pct_of_dataset": round(100.0 * len(g) / n, 3),
                "distinct_stems": len(cnt), "dominant_stem": dominant,
                "dominant_stem_images": dom_n,
                "generic_stems_present": sorted(gen)[:50],
                "n_generic_stems": len(gen)}
        if not gen:
            info["action"] = "kept whole (no generic stem in it)"
            generic_handled.append(info)
            final_groups.append(g)
            continue

        # Decompose by the duplicate relation — but a stem's links are dissolved only on
        # EVIDENCE that the stem really holds unrelated photographs, never on its name
        # alone. The evidence is the matcher's own verdict: a stem among whose images not
        # one single pair is a near-duplicate is a name several different photographs
        # happen to share. A stem with internal duplicate links is one photograph (or one
        # scene) and its link is the only thing holding the heavily-augmented copies
        # together, so it is kept.
        pairs = list(dup_links(H, TH, ok, buckets, stem_id, a.hamming, a.ncc,
                               restrict=g, allow_same_stem=True, progress_every=0))
        internal = collections.Counter()
        for i, j in pairs:
            if stems[i] == stems[j]:
                internal[stems[i]] += 1

        by_s = collections.defaultdict(list)
        for i in g:
            by_s[stems[i]].append(i)
        sub = DSU(n)
        kept_stems, dissolved_stems = [], []
        for s, idxs in by_s.items():
            if is_generic_stem(s) and internal[s] == 0 and len(idxs) > 1:
                dissolved_stems.append(s)
                continue
            kept_stems.append(s)
            for j in idxs[1:]:
                sub.union(idxs[0], j)
        for i, j in pairs:
            sub.union(i, j)
        pieces = collections.defaultdict(list)
        for i in g:
            pieces[sub.find(i)].append(i)
        info["action"] = (
            f"dissolved {len(dissolved_stems)} stem link(s) on evidence, kept "
            f"{len(kept_stems)}; {len(pieces)} resulting component(s) "
            f"(largest {max(len(p) for p in pieces.values())})")
        info["n_components"] = len(pieces)
        info["stems_dissolved"] = sorted(dissolved_stems)[:50]
        info["stems_kept"] = sorted(kept_stems)[:50]
        info["n_stems_dissolved"] = len(dissolved_stems)
        info["n_stems_kept"] = len(kept_stems)
        generic_handled.append(info)
        print(f"[generic] group of {len(g)} images ({info['pct_of_dataset']}% > "
              f"{a.generic_pct}%): {len(gen)}/{len(cnt)} stems generic by name "
              f"(dominant {dominant!r}, generic={is_generic_stem(dominant)}); "
              f"{len(dissolved_stems)} dissolved on evidence, {len(kept_stems)} kept -> "
              f"{len(pieces)} component(s)")
        final_groups.extend(pieces.values())
    groups = final_groups
    sizes = [len(g) for g in groups]
    print(f"{len(groups)} groups over {n} images")

    # --- per-group class counts, once -----------------------------------------
    per_img_cc = [class_counts(ap) for _rel, ap in items]
    gcc = []
    for g in groups:
        c = collections.Counter()
        for i in g:
            c.update(per_img_cc[i])
        gcc.append(c)

    # --- seed search on class shares only -------------------------------------
    tried, seed_used, assign, have, rows, worst = [], None, None, None, None, None
    for seed in range(a.seed, a.seed + a.max_seeds):
        assign_try, have_try = assign_groups(groups, sizes, seed)
        per, overall = instance_shares(gcc, assign_try, len(groups))
        rows_try, worst_try, okc = share_check(per, overall, "test", a.tol_pp)
        tried.append({"seed": seed, "worst_diff_pp": round(worst_try, 3), "passed": okc})
        print(f"  seed {seed}: worst class share difference in test "
              f"{worst_try:.2f} pp -> {'PASS' if okc else 'fail'}")
        if okc:
            seed_used, assign, have, rows, worst = seed, assign_try, have_try, rows_try, worst_try
            break
    if seed_used is None:
        print(f"no seed in {a.seed}..{a.seed + a.max_seeds - 1} kept every class within "
              f"{a.tol_pp} pp; using the best one and reporting it as such.", file=sys.stderr)
        best = min(tried, key=lambda r: r["worst_diff_pp"])
        seed_used = best["seed"]
        assign, have = assign_groups(groups, sizes, seed_used)
        per, overall = instance_shares(gcc, assign, len(groups))
        rows, worst, _ = share_check(per, overall, "test", a.tol_pp)
    per, overall = instance_shares(gcc, assign, len(groups))

    # --- rows + csv -----------------------------------------------------------
    gid_of = {}
    for gi, g in enumerate(groups):
        for i in g:
            gid_of[i] = gi
    out_rows = [{"file": items[i][0], "group": f"g{gid_of[i]:06d}", "split": assign[gid_of[i]]}
                for i in range(n)]
    out_rows.sort(key=lambda r: r["file"])
    os.makedirs(os.path.dirname(os.path.abspath(a.out_csv)) or ".", exist_ok=True)
    with open(a.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "group", "split"], lineterminator="\n")
        w.writeheader()
        w.writerows(out_rows)
    sha = assignment_sha256(out_rows)
    print(f"\nwrote {a.out_csv}  ({n} rows)\nassignment sha256: {sha}")

    # --- the invariant that actually matters, measured on the output ----------
    # "No group spans two splits" is true by construction (split is a function of the
    # group id), so checking it proves nothing. The real question is whether a SOURCE
    # PHOTOGRAPH ended up in two splits. A generic stem legitimately may — it names
    # several different photographs — so that case is reported separately, not as a fault.
    stem_splits = collections.defaultdict(set)
    stem_groups = collections.defaultdict(set)
    for r in out_rows:
        st = source_stem(os.path.basename(r["file"]))
        stem_splits[st].add(r["split"])
        stem_groups[st].add(r["group"])
    bad = sorted(s for s, v in stem_splits.items()
                 if len(v) > 1 and not is_generic_stem(s))
    gen_span = sorted(s for s, v in stem_splits.items()
                      if len(v) > 1 and is_generic_stem(s))
    shattered = sorted(s for s, v in stem_groups.items() if len(v) > 1)
    invariants = {
        "non_generic_stems_spanning_splits": bad,
        "n_non_generic_stems_spanning_splits": len(bad),
        "generic_stems_spanning_splits": gen_span[:50],
        "n_generic_stems_spanning_splits": len(gen_span),
        "stems_spanning_multiple_groups": shattered[:50],
        "n_stems_spanning_multiple_groups": len(shattered),
        "n_stems_classified_generic": sum(1 for s in stem_splits if is_generic_stem(s)),
        "n_stems": len(stem_splits),
    }
    print(f"invariant: non-generic source stems spanning >1 split: {len(bad)} "
          f"(must be 0){'  <-- LEAK' if bad else '  OK'}")
    print(f"           generic stems spanning >1 split: {len(gen_span)} "
          f"(allowed: a generic stem names several photographs)")
    print(f"           stems spanning >1 group: {len(shattered)}")
    if bad:
        print(f"           offending stems: {bad[:10]}", file=sys.stderr)

    # --- report ---------------------------------------------------------------
    sz = np.array(sizes)
    big = sorted(range(len(groups)), key=lambda gi: -sizes[gi])[:10]
    report = {
        "dataset_dir": root, "images": n, "distinct_source_stems": len(stem_ids),
        "groups": len(groups),
        "group_size": {"median": float(np.median(sz)), "mean": float(sz.mean()),
                       "p95": float(np.percentile(sz, 95)), "max": int(sz.max()),
                       "min": int(sz.min()), "singletons": int((sz == 1).sum())},
        "largest_groups": [{
            "group": f"g{gi:06d}", "size": sizes[gi],
            "pct_of_dataset": round(100.0 * sizes[gi] / n, 3),
            "distinct_stems": len({stems[i] for i in groups[gi]}),
            "examples": [os.path.basename(items[i][0]) for i in groups[gi][:3]],
        } for gi in big],
        "generic_stem_groups": generic_handled,
        "matcher": {"ncc": a.ncc, "hamming": a.hamming, "transforms": 8,
                    "scope": "whole dataset", "same_stem_pairs_skipped": True,
                    "note": "same-stem pairs are already linked by their stem, so testing "
                            "them cannot change a component"},
        "stem_rule": "strip .rf.<hash>, then strip trailing _<ext> tokens repeatedly",
        "seed_used": seed_used, "seeds_tried": tried, "tol_pp": a.tol_pp,
        "worst_test_class_share_diff_pp": round(worst, 3),
        "images_per_split": have,
        "groups_per_split": {s: sum(1 for gi in range(len(groups)) if assign[gi] == s)
                             for s in SPLITS},
        "class_shares_test": rows,
        "invariants": invariants,
        "assignment_sha256": sha,
        "csv": os.path.relpath(os.path.abspath(a.out_csv), os.getcwd()).replace("\\", "/"),
    }
    if a.report_dir:
        os.makedirs(a.report_dir, exist_ok=True)
        rp = os.path.join(a.report_dir, "group_split_report.json")
        json.dump(report, open(rp, "w", encoding="utf-8"), indent=2)
        print(f"wrote {rp}")

    print(f"\nimages per split: " + "  ".join(
        f"{s} {have[s]} ({100.0 * have[s] / n:.2f}%)" for s in SPLITS))
    print(f"groups per split: " + "  ".join(
        f"{s} {report['groups_per_split'][s]}" for s in SPLITS))
    print(f"group sizes: median {np.median(sz):.0f}  p95 {np.percentile(sz, 95):.0f}  "
          f"max {sz.max()}  singletons {int((sz == 1).sum())}")
    print(f"seed {seed_used} after {len(tried)} tried; worst test class-share diff "
          f"{worst:.2f} pp (tolerance {a.tol_pp})")
    return 0


# ---------------------------------------------------------------- apply

def cmd_apply(a) -> int:
    root = os.path.abspath(a.dataset_dir)
    out = os.path.abspath(a.out)
    with open(a.apply, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{a.apply} has no rows", file=sys.stderr)
        return 1
    want = {r["file"]: r for r in rows}
    if len(want) != len(rows):
        print(f"{a.apply} has duplicate 'file' values", file=sys.stderr)
        return 1

    have = {rel for rel, _ in scan_dataset(root)}
    missing = sorted(set(want) - have)
    extra = sorted(have - set(want))
    if missing:
        print(f"[FAIL] {len(missing)} file(s) in the CSV are not in {root}; first 10:",
              file=sys.stderr)
        for m in missing[:10]:
            print(f"   {m}", file=sys.stderr)
        return 1
    if extra:
        print(f"[FAIL] {len(extra)} image(s) in {root} are not in the CSV; first 10:",
              file=sys.stderr)
        for m in extra[:10]:
            print(f"   {m}", file=sys.stderr)
        return 1
    print(f"{len(rows)} files in the CSV, all present in {root}, none unaccounted for")

    # Images are placed under <split>/images/ by basename. Two source paths that share a
    # basename would silently overwrite each other and shrink the dataset, so refuse.
    seen = {}
    clash = []
    for rel, r in want.items():
        key = (r["split"], os.path.basename(rel))
        if key in seen:
            clash.append((seen[key], rel, r["split"]))
        else:
            seen[key] = rel
    if clash:
        print(f"[FAIL] {len(clash)} basename collision(s) within a target split — placing "
              f"these would silently lose images; first 10:", file=sys.stderr)
        for aa, bb, s in clash[:10]:
            print(f"   {s}: {aa}  vs  {bb}", file=sys.stderr)
        return 1

    for s in SPLITS:
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(out, s, sub), exist_ok=True)

    linked = copied = labels_linked = labels_missing = 0
    for rel, r in sorted(want.items()):
        src = os.path.join(root, rel.replace("/", os.sep))
        name = os.path.basename(rel)
        dst = os.path.join(out, r["split"], "images", name)
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            shutil.copy2(src, dst)
            copied += 1
        lsrc = label_path_for(src)
        if lsrc and os.path.isfile(lsrc):
            ldst = os.path.join(out, r["split"], "labels", os.path.splitext(name)[0] + ".txt")
            if os.path.exists(ldst):
                os.remove(ldst)
            try:
                os.link(lsrc, ldst)
            except OSError:
                shutil.copy2(lsrc, ldst)
            labels_linked += 1
        else:
            labels_missing += 1
    print(f"images: {linked} hard-linked, {copied} copied | labels: {labels_linked} placed, "
          f"{labels_missing} absent in the source")

    # --- data.yaml (original class names, absolute path) ----------------------
    import yaml
    spec = yaml.safe_load(open(os.path.join(root, "data.yaml"), encoding="utf-8"))
    names = spec.get("names")
    new = {"path": out, "train": "train/images", "val": "valid/images",
           "test": "test/images", "nc": spec.get("nc", len(names) if names else 0),
           "names": names}
    if spec.get("roboflow"):
        new["roboflow"] = spec["roboflow"]
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump(new, fh, sort_keys=False, allow_unicode=True)

    # --- summary --------------------------------------------------------------
    per_split_cc = {s: collections.Counter() for s in SPLITS}
    per_split_imgs = collections.Counter()
    per_split_groups = collections.defaultdict(set)
    for rel, r in want.items():
        s = r["split"]
        per_split_imgs[s] += 1
        per_split_groups[s].add(r["group"])
        per_split_cc[s].update(class_counts(os.path.join(root, rel.replace("/", os.sep))))
    name_of = (lambda ci: names[ci]) if isinstance(names, list) else (lambda ci: str(ci))
    summary = {
        "source_dataset": root, "out_dataset": out,
        "csv": os.path.basename(a.apply),
        "assignment_sha256": assignment_sha256(rows),
        "seed": a.seed_note,
        "nc": new["nc"], "names": names,
        "splits": {s: {
            "images": per_split_imgs[s],
            "source_groups": len(per_split_groups[s]),
            "instances_per_class": {name_of(ci): per_split_cc[s][ci]
                                    for ci in sorted(per_split_cc[s])},
            "instances_total": sum(per_split_cc[s].values()),
        } for s in SPLITS},
    }
    sp = os.path.join(out, "split_summary.json")
    json.dump(summary, open(sp, "w", encoding="utf-8"), indent=2)

    # --- one image per test group, seed 0 -------------------------------------
    by_group = collections.defaultdict(list)
    for rel, r in want.items():
        if r["split"] == "test":
            by_group[r["group"]].append(rel)
    rng = random.Random(0)
    picks = [rng.choice(sorted(v)) for _k, v in sorted(by_group.items())]
    tp = os.path.join(out, "test_one_per_source.txt")
    with open(tp, "w", encoding="utf-8") as fh:
        for rel in picks:
            fh.write(os.path.join(out, "test", "images", os.path.basename(rel)) + "\n")

    print(f"\nwrote {os.path.join(out, 'data.yaml')}")
    print(f"wrote {sp}")
    print(f"wrote {tp}  ({len(picks)} images, one per test group)")
    print(f"assignment sha256: {summary['assignment_sha256']}")
    for s in SPLITS:
        d = summary["splits"][s]
        print(f"  {s:<6} {d['images']:>6} images  {d['source_groups']:>6} groups  "
              f"{d['instances_total']:>7} boxes")
    return 0


# ---------------------------------------------------------------- cli

def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Source-grouped re-split of a YOLO dataset")
    ap.add_argument("--dataset-dir", default="data/ppe_download")
    ap.add_argument("--apply", default=None, metavar="CSV",
                    help="materialise this split definition instead of building one")
    ap.add_argument("--out", default="data/ppe_grouped", help="output dataset dir (--apply)")
    ap.add_argument("--out-csv", default="tools/splits/ppe_grouped_split.csv")
    ap.add_argument("--report-dir", default=None)
    ap.add_argument("--ncc", type=float, default=0.90)
    ap.add_argument("--hamming", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42, help="first seed to try")
    ap.add_argument("--max-seeds", type=int, default=50)
    ap.add_argument("--tol-pp", type=float, default=3.0,
                    help="max allowed class-share difference in test, in points")
    ap.add_argument("--generic-pct", type=float, default=1.0,
                    help="a group larger than this %% of the dataset is checked for a "
                         "generic stem and decomposed if it has one")
    ap.add_argument("--seed-note", default=None,
                    help="seed to record in split_summary.json (--apply)")
    a = ap.parse_args(argv)
    return cmd_apply(a) if a.apply else cmd_build(a)


if __name__ == "__main__":
    raise SystemExit(main())
