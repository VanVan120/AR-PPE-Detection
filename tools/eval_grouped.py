"""Score one weights file on the source-grouped test split.

This is the measurement the corrected paper number comes from. It runs the same weights
four times with identical settings apart from the one thing being varied:

  full_640            the whole grouped test split at imgsz 640   <- the headline number
  one_per_source_640  one image per test source group at imgsz 640
  full_480            the whole grouped test split at imgsz 480
  full_320            the whole grouped test split at imgsz 320

`one_per_source` matters because the grouped test split still holds several augmented
copies of each source photograph. Copies of one photo are not independent samples, so the
full split effectively weights a source by how many copies of it the export happened to
contain. Scoring one image per group removes that weighting; a large gap between the two
means the score depends on which photos were duplicated most.

The 480 and 320 rows are there because the AR/phone path runs at reduced resolution — they
say what the deployed configuration actually achieves, not just the training resolution.

    python tools/eval_grouped.py --weights best_grouped.pt \
        --dataset-dir data/ppe_grouped --out-dir paper_materials/retrain/eval_grouped

Device is cuda when torch reports it usable, else cpu, and is recorded in the output.
Before each run the image and label counts ultralytics reports are checked against the
expected counts; a mismatch is recorded and flagged, never silently accepted.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rescore as rs            # reuse the capture/count-check machinery  # noqa: E402

METRICS = ("precision", "recall", "f1", "map50", "map50_95")
OVERALL_METRICS = ("precision", "recall", "f1", "f1_macro", "map50", "map50_95")


def metrics_from(res, model):
    """Per-class and overall P / R / F1 / mAP@50 / mAP@50-95."""
    box = res.box
    names = res.names if hasattr(res, "names") else model.names
    idx = [int(i) for i in rs._arr(getattr(box, "ap_class_index", []))]
    p, r = rs._arr(getattr(box, "p", [])), rs._arr(getattr(box, "r", []))
    f1 = rs._arr(getattr(box, "f1", []))
    ap50, ap = rs._arr(getattr(box, "ap50", [])), rs._arr(getattr(box, "ap", []))

    per_class = {}
    for j, ci in enumerate(idx):
        name = names.get(ci, str(ci)) if isinstance(names, dict) else names[ci]
        per_class[name] = {
            "precision": p[j] if j < len(p) else None,
            "recall": r[j] if j < len(r) else None,
            "f1": f1[j] if j < len(f1) else None,
            "map50": ap50[j] if j < len(ap50) else None,
            "map50_95": ap[j] if j < len(ap) else None,
        }
    mp, mr = float(getattr(box, "mp", 0.0)), float(getattr(box, "mr", 0.0))
    # Two different "overall F1"s exist and they are not equal. `f1` is the harmonic mean
    # of the MEAN precision and MEAN recall, which is what eval_ppe.py and the published
    # table report, so it is kept as the comparable figure. `f1_macro` is the mean of the
    # per-class F1s, which is the one most readers assume. Both are reported rather than
    # silently picking one — they can differ by a point or more when classes are uneven.
    cls_f1 = [v["f1"] for v in per_class.values() if v["f1"] is not None]
    overall = {
        "precision": mp, "recall": mr,
        "f1": (2 * mp * mr / (mp + mr)) if (mp + mr) else 0.0,
        "f1_macro": (sum(cls_f1) / len(cls_f1)) if cls_f1 else None,
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }
    return overall, per_class


def write_yaml_for(base_yaml, test_value, path):
    spec = dict(base_yaml)
    spec["test"] = test_value
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    return path


def run_one(tag, weights, data_yaml, imgsz, device, expect_images, expect_labels, project,
            independent=None):
    """`independent` is a count derived WITHOUT going through the image list handed to
    ultralytics (the files on disk, or split_summary.json), so the check is not merely
    ultralytics echoing back the list it was given."""
    from ultralytics import YOLO
    print("\n" + "=" * 78)
    print(f" {tag}: imgsz {imgsz}, device {device}")
    print(f" expecting {expect_images} images / {expect_labels} label files"
          + (f"; independently {independent} on disk" if independent is not None else ""))
    print("=" * 78)

    model = YOLO(weights)
    res, printed = rs.val_capturing(model, data=data_yaml, split="test", imgsz=imgsz,
                                    device=device, project=project, name=tag,
                                    exist_ok=True, plots=False)
    counts = rs.reported_counts(printed)
    overall, per_class = metrics_from(res, model)

    checks = {"expected_images": expect_images, "expected_label_files": expect_labels,
              "independent_images": independent,
              **counts,
              "images_match": counts["summary_images"] == expect_images,
              "label_files_match": counts["scan_labels_found"] == expect_labels,
              "no_corrupt": counts["scan_corrupt"] == 0,
              "matches_independent_count": (independent is None
                                            or counts["summary_images"] == independent)}
    gates = ("images_match", "label_files_match", "no_corrupt", "matches_independent_count")
    status = "OK" if all(checks[k] for k in gates) else "MISMATCH"
    print(f"[count check {status}] ultralytics reported {counts['summary_images']} images / "
          f"{counts['summary_instances']} instances; label scan found "
          f"{counts['scan_labels_found']} labels, "
          f"{counts['scan_backgrounds_missing_or_empty']} missing-or-empty, "
          f"{counts['scan_corrupt']} corrupt")
    if status == "MISMATCH":
        print(f"  ^ counts disagree ({', '.join(k for k in gates if not checks[k])}) — "
              f"recorded and flagged, not silently accepted.")
    print(f"  P {overall['precision']*100:.2f}  R {overall['recall']*100:.2f}  "
          f"F1 {overall['f1']*100:.2f}  mAP50 {overall['map50']*100:.2f}  "
          f"mAP50-95 {overall['map50_95']*100:.2f}")

    return {"tag": tag, "imgsz": imgsz, "device": device, "data_yaml": data_yaml,
            "count_check": checks, "count_check_status": status,
            "classes_scored": sorted(per_class),
            "overall": overall, "per_class": per_class}


def write_md(path, payload):
    runs = payload["runs"]
    classes = sorted({c for r in runs for c in r["per_class"]})
    L, A = [], None
    A = L.append
    A(f"# Grouped-split evaluation of `{os.path.basename(payload['weights'])}`")
    A("")
    if payload.get("note"):
        A(f"> **{payload['note']}**")
        A("")
    A(f"- weights: `{payload['weights']}`")
    A(f"- dataset: `{payload['dataset_dir']}` (source-grouped split)")
    A(f"- device: `{payload['device']}` · ultralytics {payload['versions']['ultralytics']} "
      f"· torch {payload['versions']['torch']} · python {payload['versions']['python']}")
    A(f"- `model.val`, `plots=False`, every other setting at its ultralytics default")
    if payload.get("limit"):
        A(f"- **limited to the first {payload['limit']} test images** (same images in "
          f"every row)")
    A("")

    A("## Overall")
    A("")
    A("| run | imgsz | images | precision | recall | F1 | F1 (macro) | mAP@50 | mAP@50-95 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in runs:
        o = r["overall"]
        flag = "" if r["count_check_status"] == "OK" else " ⚠"
        A(f"| {r['tag']}{flag} | {r['imgsz']} | {r['count_check']['summary_images']} | "
          + " | ".join(rs.pct(o.get(m)) for m in OVERALL_METRICS) + " |")
    A("")
    A("**F1** is the harmonic mean of the mean precision and mean recall — the definition "
      "`eval_ppe.py` and the published table use, so it is the comparable one. "
      "**F1 (macro)** is the mean of the per-class F1s. They are not equal when classes "
      "are uneven; both are given so neither is mistaken for the other.")
    A("")

    A("## Per class")
    A("")
    for m in METRICS:
        A(f"### {m}")
        A("")
        A("| class | " + " | ".join(r["tag"] for r in runs) + " |")
        A("|---" * (1 + len(runs)) + "|")
        for c in classes:
            A(f"| {c} | " + " | ".join(rs.pct(r["per_class"].get(c, {}).get(m))
                                       for r in runs) + " |")
        A("")

    A("## Count checks")
    A("")
    A("Each run's counts are checked three ways: against the length of the image list it "
      "was handed, against the label files that list resolves to, and — for the "
      "un-truncated runs — against the independent count recorded in "
      "`split_summary.json` when the split was materialised. The third is the one that "
      "could catch a wrong list, rather than ultralytics echoing back the list it was "
      "given.")
    A("")
    A("| run | expected images | images scored | independent | expected label files | "
      "labels found | missing/empty | corrupt | instances | status |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in runs:
        c = r["count_check"]
        ind = c.get("independent_images")
        A(f"| {r['tag']} | {c['expected_images']} | {c['summary_images']} | "
          f"{'—' if ind is None else ind} | "
          f"{c['expected_label_files']} | {c['scan_labels_found']} | "
          f"{c['scan_backgrounds_missing_or_empty']} | {c['scan_corrupt']} | "
          f"{c['summary_instances']} | {r['count_check_status']} |")
    A("")
    mism = [r["tag"] for r in runs if r["count_check_status"] != "OK"]
    A("**All count checks passed.**" if not mism else
      f"**Count mismatch in: {', '.join(mism)} — treat those rows with suspicion.**")
    A("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Score weights on the source-grouped test split")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--dataset-dir", default="data/ppe_grouped")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="eval_grouped", help="basename for the json/md")
    ap.add_argument("--imgsz", type=int, default=640, help="the primary resolution")
    ap.add_argument("--extra-imgsz", type=int, nargs="*", default=[480, 320])
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N test images (smoke tests)")
    ap.add_argument("--note", default=None,
                    help="a banner to put at the top of the md, e.g. to mark a smoke test")
    a = ap.parse_args(argv)

    root = os.path.abspath(a.dataset_dir)
    dy = os.path.join(root, "data.yaml")
    if not os.path.isfile(dy):
        print(f"no data.yaml in {root} — run group_split.py --apply first", file=sys.stderr)
        return 1
    if not os.path.isfile(a.weights):
        print(f"weights not found: {a.weights}", file=sys.stderr)
        return 1
    base = yaml.safe_load(open(dy, encoding="utf-8"))
    base["path"] = root

    import torch
    device = "cpu"
    if torch.cuda.is_available():
        try:
            torch.zeros(8, device="cuda").sum().item()
            device = "0"
        except Exception as e:
            print(f"[device] CUDA present but unusable ({e}); using CPU.")
    print(f"[device] {device} (torch {torch.__version__}, "
          f"cuda_available={torch.cuda.is_available()})")

    os.makedirs(a.out_dir, exist_ok=True)
    work = os.path.join(a.out_dir, "eval_splits")
    os.makedirs(work, exist_ok=True)

    full_imgs = rs.list_images(os.path.join(root, "test", "images"))
    if not full_imgs:
        print(f"no images under {root}/test/images", file=sys.stderr)
        return 1
    ops_path = os.path.join(root, "test_one_per_source.txt")

    # An independent count, not derived from the list handed to ultralytics: what
    # split_summary.json recorded when the split was materialised. Without this the count
    # check only proves ultralytics echoed back the list it was given.
    indep_full = None
    sp = os.path.join(root, "split_summary.json")
    if os.path.isfile(sp):
        try:
            indep_full = json.load(open(sp, encoding="utf-8"))["splits"]["test"]["images"]
            print(f"[check] split_summary.json records {indep_full} test images")
        except Exception as e:
            print(f"[warn] could not read a test image count from {sp}: {e!r}")
    else:
        print(f"[warn] no {sp}; the full-split count check falls back to the file listing "
              f"only, which is weaker.")

    jobs = []
    if a.limit:
        sel = full_imgs[:a.limit]
        y = rs.write_list_yaml(base, sel, os.path.join(work, "test_limited.txt"),
                               os.path.join(work, "data_limited.yaml"))
        nl = rs.count_labels(sel)
        jobs.append((f"full_{a.imgsz}", y, a.imgsz, len(sel), nl, None))
        for z in a.extra_imgsz:
            jobs.append((f"full_{z}", y, z, len(sel), nl, None))
        print(f"[limit] every run scores the same first {len(sel)} test images; "
              f"one_per_source is skipped, and the independent count check does not "
              f"apply to a deliberately truncated list")
    else:
        y_full = rs.write_list_yaml(base, full_imgs, os.path.join(work, "test_full.txt"),
                                    os.path.join(work, "data_full.yaml"))
        nl = rs.count_labels(full_imgs)
        jobs.append((f"full_{a.imgsz}", y_full, a.imgsz, len(full_imgs), nl, indep_full))
        if os.path.isfile(ops_path):
            ops = [l.strip() for l in open(ops_path, encoding="utf-8") if l.strip()]
            y_ops = write_yaml_for(base, os.path.abspath(ops_path),
                                   os.path.join(work, "data_one_per_source.yaml"))
            indep_ops = None
            if os.path.isfile(sp):
                try:
                    indep_ops = json.load(
                        open(sp, encoding="utf-8"))["splits"]["test"]["source_groups"]
                    print(f"[check] split_summary.json records {indep_ops} test source "
                          f"groups, so the one-per-source list should be that long")
                except Exception:
                    pass
            jobs.append((f"one_per_source_{a.imgsz}", y_ops, a.imgsz, len(ops),
                         rs.count_labels(ops), indep_ops))
        else:
            print(f"[warn] {ops_path} not found; the one-per-source row is skipped.")
        for z in a.extra_imgsz:
            jobs.append((f"full_{z}", y_full, z, len(full_imgs), nl, indep_full))

    project = os.path.abspath(os.path.join(a.out_dir, "eval_runs"))
    runs = [run_one(tag, a.weights, y, z, device, ni, nl, project, independent=ind)
            for tag, y, z, ni, nl, ind in jobs]

    class_sets = {r["tag"]: tuple(r["classes_scored"]) for r in runs}
    payload = {
        "weights": a.weights, "dataset_dir": root, "device": device,
        "note": a.note, "limit": a.limit or None,
        "primary_imgsz": a.imgsz, "extra_imgsz": a.extra_imgsz,
        "versions": {"python": sys.version.split()[0],
                     "torch": __import__("torch").__version__,
                     "ultralytics": __import__("ultralytics").__version__},
        "class_set_consistent": len(set(class_sets.values())) <= 1,
        "classes_per_run": {k: list(v) for k, v in class_sets.items()},
        "runs": runs,
    }
    jp = os.path.join(a.out_dir, f"{a.name}.json")
    mp = os.path.join(a.out_dir, f"{a.name}.md")
    json.dump(payload, open(jp, "w", encoding="utf-8"), indent=2)
    try:
        write_md(mp, payload)
    except Exception as e:
        print(f"[error] {jp} written, but rendering the md failed: {e!r}", file=sys.stderr)
        raise

    print("\n" + "=" * 78)
    for r in runs:
        o = r["overall"]
        print(f" {r['tag']:<22} P {o['precision']*100:6.2f}  R {o['recall']*100:6.2f}  "
              f"F1 {o['f1']*100:6.2f}  mAP50 {o['map50']*100:6.2f}  "
              f"mAP50-95 {o['map50_95']*100:6.2f}")
    print(f"\nWrote {jp}\nWrote {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
