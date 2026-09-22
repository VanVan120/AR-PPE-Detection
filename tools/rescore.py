"""Re-score the detector on the full test split, the de-duplicated split, and 5 controls.

`tools/leakage_audit.py` says how much of the PPE test split has a near-duplicate in
train/valid, and writes a clean test list. Re-scoring on the clean list alone does not
settle anything, because the clean list is also *smaller*, and a smaller test set moves a
score on its own. So this runs three things with ultralytics `model.val` at imgsz 640,
everything else at its defaults:

  full      the whole test split
  dedup     the de-duplicated list from the audit (data_dedup.yaml)
  ctrl0..4  five subsets that drop the SAME NUMBER of test images, chosen at random
            (seeds 0-4), so the dedup result can be read against the spread that
            shrinking the test set produces by itself

If the de-duplicated score sits inside the control range, the duplicates were not
inflating the number. If it sits below it, they were. The controls are the point.

    python tools/rescore.py --dataset-dir data/ppe_download --model best_refined.pt \
        --audit-dir paper_materials/leakage_audit --out-dir paper_materials

Device: GPU (device=0) if CUDA is usable, else CPU for *every* run — never a mix, since
the device can change a score slightly and that would contaminate the comparison.
Writes <out-dir>/rescore.json and <out-dir>/rescore.md, the per-split image lists and
data.yamls under <out-dir>/rescore_splits/, and each run's ultralytics plots under
<out-dir>/rescore_runs/<tag>/ (redirected there so nothing lands in runs/). One write
happens outside <out-dir> and cannot be suppressed: ultralytics caches its label scan as
`<dataset>/test/labels.cache`. Every run uses a different image list, so that cache is
rewritten each time; it is a derived file and safe to delete.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys

import yaml

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
CLASS_ORDER_KEY = "__overall__"


# ---------------------------------------------------------------- dataset helpers

def list_images(d):
    out = []
    for root, _, files in os.walk(d):
        out += [os.path.join(root, f) for f in files if f.lower().endswith(EXT)]
    return sorted(out)


def label_for(img_path):
    """Ultralytics derives a label path by swapping the last os.sep+'images'+os.sep."""
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in img_path:
        return None
    head, _, tail = img_path.rpartition(sa)
    return os.path.splitext(head + sb + tail)[0] + ".txt"


def count_labels(img_paths):
    """How many of these images have a label file on disk (the rest are backgrounds)."""
    n = 0
    for p in img_paths:
        lp = label_for(p)
        if lp and os.path.isfile(lp):
            n += 1
    return n


def write_list_yaml(base_yaml, img_paths, list_path, yaml_path):
    """A data.yaml whose `test:` is a .txt list of absolute image paths."""
    with open(list_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(img_paths) + "\n")
    spec = dict(base_yaml)
    spec["test"] = os.path.abspath(list_path)
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    return yaml_path


# ---------------------------------------------------------------- metric extraction

def _arr(x):
    try:
        return [float(v) for v in x]
    except Exception:
        return []


def metrics_from(res, model):
    box = res.box
    names = res.names if hasattr(res, "names") else model.names
    idx = [int(i) for i in _arr(getattr(box, "ap_class_index", []))]
    p, r = _arr(getattr(box, "p", [])), _arr(getattr(box, "r", []))
    ap50, ap = _arr(getattr(box, "ap50", [])), _arr(getattr(box, "ap", []))

    per_class = {}
    for j, ci in enumerate(idx):
        name = names.get(ci, str(ci)) if isinstance(names, dict) else names[ci]
        per_class[name] = {
            "precision": p[j] if j < len(p) else None,
            "recall": r[j] if j < len(r) else None,
            "map50": ap50[j] if j < len(ap50) else None,
            "map50_95": ap[j] if j < len(ap) else None,
        }
    overall = {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
    }
    return overall, per_class


class _Tee:
    """Pass output through to the real stream while keeping a copy for parsing."""

    def __init__(self, stream, sink):
        self.stream, self.sink = stream, sink

    def write(self, s):
        self.stream.write(s)
        self.sink.append(s)
        return len(s)

    def flush(self):
        self.stream.flush()

    def isatty(self):                       # keeps tqdm on plain, parseable lines
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


class _LogSink(logging.Handler):
    """Ultralytics logs through its own LOGGER, whose handler holds the ORIGINAL
    stderr — so redirecting sys.stderr alone does not capture it."""

    def __init__(self, sink):
        super().__init__()
        self.sink = sink

    def emit(self, record):
        try:
            self.sink.append(record.getMessage() + "\n")
        except Exception:
            pass


def val_capturing(model, **kw):
    """Run model.val(**kw), returning (results, everything ultralytics printed)."""
    sink = []
    handler = _LogSink(sink)
    try:
        from ultralytics.utils import LOGGER
    except Exception:
        LOGGER = None
    if LOGGER is not None:
        LOGGER.addHandler(handler)
    out, err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(out, sink), _Tee(err, sink)
    try:
        res = model.val(**kw)
    finally:
        sys.stdout, sys.stderr = out, err
        if LOGGER is not None:
            LOGGER.removeHandler(handler)
    return res, "".join(sink)


def reported_counts(text):
    """Parse the counts ultralytics itself printed.

    Two independent places state them:
      "... 4190 images, 0 backgrounds, 0 corrupt"   (the label scan / cache line)
      "   all   4190   23467   0.961 ..."           (the summary row: images, instances)

    Careful with the scan line. Ultralytics formats it as
    `f"{nf} images, {nm + ne} backgrounds, {nc} corrupt"`, where `nf` counts images whose
    LABEL FILE WAS FOUND, `nm` those whose label was missing and `ne` those whose label
    existed but was empty. So the first number is the label-found count, not the image
    count, and "backgrounds" sums two different things. The true image count is the
    summary row's first column (the validator's `seen`), which is what the image gate
    uses; the scan number is checked against the number of label files on disk.
    """
    clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text).replace("\r", "\n")
    scan = None
    for m in re.finditer(r"(\d+)\s+images,\s*(\d+)\s+backgrounds,\s*(\d+)\s+corrupt", clean):
        scan = m                                   # the last one is this run's
    rows = re.findall(r"^\s+all\s+(\d+)\s+(\d+)\s", clean, re.M)
    return {
        "scan_labels_found": int(scan.group(1)) if scan else None,
        "scan_backgrounds_missing_or_empty": int(scan.group(2)) if scan else None,
        "scan_corrupt": int(scan.group(3)) if scan else None,
        "summary_images": int(rows[-1][0]) if rows else None,
        "summary_instances": int(rows[-1][1]) if rows else None,
    }


# ---------------------------------------------------------------- one run

def run_one(tag, data_yaml, expect_images, expect_labels, model_path, imgsz, device, project,
            plots=True, focus=None):
    from ultralytics import YOLO

    print("\n" + "=" * 78)
    print(f" {tag}: {data_yaml}")
    print(f" expecting {expect_images} images / {expect_labels} label files, device={device}")
    print("=" * 78)

    model = YOLO(model_path)
    res, printed = val_capturing(model, data=data_yaml, split="test", imgsz=imgsz,
                                 device=device, project=project, name=tag, exist_ok=True,
                                 plots=plots)
    counts = reported_counts(printed)
    overall, per_class = metrics_from(res, model)

    checks = {
        "expected_images": expect_images,
        "expected_label_files": expect_labels,
        **counts,
        "images_match": counts["summary_images"] == expect_images,
        "label_files_match": counts["scan_labels_found"] == expect_labels,
        "no_corrupt": counts["scan_corrupt"] == 0,
    }
    gates = ("images_match", "label_files_match", "no_corrupt")
    status = "OK" if all(checks[k] for k in gates) else "MISMATCH"
    print(f"[count check {status}] ultralytics reported: {counts['summary_images']} images "
          f"/ {counts['summary_instances']} instances scored; label scan found "
          f"{counts['scan_labels_found']} labels, {counts['scan_backgrounds_missing_or_empty']} "
          f"missing-or-empty, {counts['scan_corrupt']} corrupt | expected {expect_images} "
          f"images, {expect_labels} label files")
    if status == "MISMATCH":
        failed = [k for k in gates if not checks[k]]
        print(f"  ^ counts disagree ({', '.join(failed)}) — recorded in rescore.json and "
              f"flagged in rescore.md, NOT silently accepted.")

    run = {"tag": tag, "data_yaml": data_yaml, "device": device, "imgsz": imgsz,
           "count_check": checks, "count_check_status": status,
           "classes_scored": sorted(per_class),
           "overall": overall, "per_class": per_class}

    if focus:
        present = [c for c in focus if c in per_class]
        missing = [c for c in focus if c not in per_class]
        run["focus"] = {
            "classes_requested": list(focus),
            "classes_present": present,
            "classes_missing": missing,
            "mean": {k: mean([per_class[c][k] for c in present]) for k in
                     ("precision", "recall", "map50", "map50_95")} if present else {},
        }
        if missing:
            print(f"  [focus] not scored in this split (no ground-truth instances): "
                  f"{', '.join(missing)}")
        if present:
            f = run["focus"]["mean"]
            print(f"  [focus] mean over {len(present)} class(es): P {f['precision']*100:.2f} "
                  f"R {f['recall']*100:.2f} mAP50 {f['map50']*100:.2f} "
                  f"mAP50-95 {f['map50_95']*100:.2f}")
    return run


# ---------------------------------------------------------------- report

def pct(x):
    return "  -  " if x is None else f"{x * 100:.2f}"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def write_md(path, payload):
    runs = payload["runs"]
    ctrl = [r for r in runs if r["tag"].startswith("ctrl")]
    classes = sorted({c for r in runs for c in r["per_class"]})
    L = []
    A = L.append

    A("# Re-score of `%s` on the PPE test split" % payload["model"])
    A("")
    A(f"- device: `{payload['device']}` (same for every run) · imgsz {payload['imgsz']} · "
      f"`model.val`, all other settings at ultralytics defaults")
    A(f"- ultralytics {payload['versions']['ultralytics']} · torch {payload['versions']['torch']} · "
      f"python {payload['versions']['python']}")
    A(f"- full test split: {payload['n_full']} images · duplicates removed: "
      f"{payload['n_removed']} · remaining: {payload['n_dedup']}")
    A(f"- controls: {len(ctrl)} subsets, each dropping {payload['n_removed']} randomly chosen "
      f"test images (seeds {payload['control_seeds']})")
    A("")

    A("## Overall")
    A("")
    A("| run | images | precision | recall | mAP@50 | mAP@50-95 |")
    A("|---|---:|---:|---:|---:|---:|")
    for r in runs:
        o = r["overall"]
        flag = "" if r["count_check_status"] == "OK" else " ⚠"
        A(f"| {r['tag']}{flag} | {r['count_check']['summary_images']} | {pct(o['precision'])} | "
          f"{pct(o['recall'])} | {pct(o['map50'])} | {pct(o['map50_95'])} |")
    if ctrl:
        A("")
        A("| controls | precision | recall | mAP@50 | mAP@50-95 |")
        A("|---|---:|---:|---:|---:|")
        for stat, fn in (("mean", mean), ("min", min), ("max", max)):
            vals = {k: fn([c["overall"][k] for c in ctrl]) for k in
                    ("precision", "recall", "map50", "map50_95")}
            A(f"| {stat} | {pct(vals['precision'])} | {pct(vals['recall'])} | "
              f"{pct(vals['map50'])} | {pct(vals['map50_95'])} |")
    A("")

    if payload.get("focus_classes"):
        foc = [r for r in runs if r.get("focus", {}).get("mean")]
        A("## Focus classes")
        A("")
        A(f"Mean over only: {', '.join(payload['focus_classes'])}. This is a separate "
          f"average; it does not alter any per-class or overall figure above.")
        A("")
        if foc:
            A("| run | classes averaged | precision | recall | mAP@50 | mAP@50-95 |")
            A("|---|---:|---:|---:|---:|---:|")
            for r in foc:
                m = r["focus"]["mean"]
                A(f"| {r['tag']} | {len(r['focus']['classes_present'])} | "
                  f"{pct(m.get('precision'))} | {pct(m.get('recall'))} | "
                  f"{pct(m.get('map50'))} | {pct(m.get('map50_95'))} |")
            fc = [r for r in foc if r["tag"].startswith("ctrl")]
            if fc:
                A("")
                A("| controls (focus) | precision | recall | mAP@50 | mAP@50-95 |")
                A("|---|---:|---:|---:|---:|")
                for stat, fn in (("mean", mean), ("min", min), ("max", max)):
                    vals = {k: fn([c["focus"]["mean"][k] for c in fc
                                   if c["focus"]["mean"].get(k) is not None])
                            for k in ("precision", "recall", "map50", "map50_95")}
                    A(f"| {stat} | {pct(vals['precision'])} | {pct(vals['recall'])} | "
                      f"{pct(vals['map50'])} | {pct(vals['map50_95'])} |")
            miss = sorted({c for r in foc for c in r["focus"]["classes_missing"]})
            if miss:
                A("")
                A(f"Requested but not scored in at least one split (no ground-truth "
                  f"instances there): {', '.join(miss)}.")
        else:
            A("None of the requested classes had ground-truth instances in any run.")
        A("")

    A("## Per class")
    A("")
    for metric in ("precision", "recall", "map50", "map50_95"):
        A(f"### {metric}")
        A("")
        A("| class | " + " | ".join(r["tag"] for r in runs) +
          (" | ctrl mean | ctrl min | ctrl max |" if ctrl else " |"))
        A("|---" * (1 + len(runs) + (3 if ctrl else 0)) + "|")
        for c in classes:
            cells = [pct(r["per_class"].get(c, {}).get(metric)) for r in runs]
            if ctrl:
                cv = [x["per_class"].get(c, {}).get(metric) for x in ctrl]
                cells += [pct(mean(cv)),
                          pct(min([v for v in cv if v is not None], default=None)),
                          pct(max([v for v in cv if v is not None], default=None))]
            A(f"| {c} | " + " | ".join(cells) + " |")
        A("")

    if payload.get("verdict"):
        A("## Does de-duplication move the score beyond what shrinking it does anyway?")
        A("")
        for line in payload["verdict"]:
            A(f"- {line}")
        A("")

    if payload.get("benchmark_compare"):
        A("## Full split vs `phase2/benchmark.json`")
        A("")
        A("| metric | benchmark.json | this run (full) | difference |")
        A("|---|---:|---:|---:|")
        for k, v in payload["benchmark_compare"]["overall"].items():
            A(f"| {k} | {pct(v['benchmark'])} | {pct(v['rescore'])} | {v['delta_pp']:+.2f} pp |")
        A("")
        A("| class | metric | benchmark.json | this run (full) | difference |")
        A("|---|---|---:|---:|---:|")
        for c, ms in payload["benchmark_compare"]["per_class"].items():
            for k, v in ms.items():
                A(f"| {c} | {k} | {pct(v['benchmark'])} | {pct(v['rescore'])} | {v['delta_pp']:+.2f} pp |")
        A("")

    mism = [r["tag"] for r in runs if r["count_check_status"] != "OK"]
    A("## Count checks")
    A("")
    A("Before each run's metrics were recorded, three things ultralytics itself reported were "
      "compared with the expected counts: the number of images it scored, the number of label "
      "files its scan found, and that nothing was corrupt. A run that fails any of them is "
      "marked ⚠ in the tables above and listed below; the numbers are still reported, so "
      "nothing is hidden.")
    A("")
    A("| run | expected images | images scored | expected label files | labels found | "
      "missing/empty | corrupt | instances | status |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in runs:
        c = r["count_check"]
        A(f"| {r['tag']} | {c['expected_images']} | {c['summary_images']} | "
          f"{c['expected_label_files']} | {c['scan_labels_found']} | "
          f"{c['scan_backgrounds_missing_or_empty']} | {c['scan_corrupt']} | "
          f"{c['summary_instances']} | {r['count_check_status']} |")
    A("")
    A("**All count checks passed.**" if not mism else
      f"**Count mismatch in: {', '.join(mism)} — treat those rows with suspicion.**")
    A("")
    if payload.get("class_set_consistent") is False:
        A("**Warning: the runs did not all score the same set of classes**, so their "
          "overall means are averages over different denominators and are not directly "
          "comparable. Per-run class sets:")
        A("")
        for r in runs:
            A(f"- `{r['tag']}`: {', '.join(r['classes_scored']) or '(none)'}")
        A("")
    else:
        A(f"All runs scored the same {len(classes)} classes "
          f"({', '.join(classes)}), so the overall means share a denominator.")
        A("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))


# ---------------------------------------------------------------- main

def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Re-score on full / de-duplicated / control test splits")
    ap.add_argument("--dataset-dir", default="data/ppe_download")
    ap.add_argument("--model", default="best_refined.pt")
    ap.add_argument("--audit-dir", default="paper_materials/leakage_audit")
    ap.add_argument("--out-dir", default="paper_materials")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--controls", type=int, default=5)
    ap.add_argument("--benchmark", default="phase2/benchmark.json",
                    help="json to compare the full-split result against; pass '' to skip")
    # --- additive options; the defaults reproduce the original behaviour exactly ---
    ap.add_argument("--focus-classes", nargs="+", default=None, metavar="NAME",
                    help="also report a mean over just these classes (e.g. for a 25-class "
                         "dataset where only 5 classes matter). Does not change any per-class "
                         "or overall number; adds a separate 'focus' block.")
    ap.add_argument("--no-plots", dest="plots", action="store_false",
                    help="pass plots=False to model.val (no val_batch*.jpg written)")
    ap.set_defaults(plots=True)
    a = ap.parse_args(argv)

    root = os.path.abspath(a.dataset_dir)
    if not os.path.isfile(a.model):
        print(f"model not found: {a.model}", file=sys.stderr)
        return 1

    all_test = list_images(os.path.join(root, "test", "images"))
    if not all_test:
        print(f"no test images under {root}", file=sys.stderr)
        return 1
    n_full, n_full_labels = len(all_test), count_labels(all_test)

    base = yaml.safe_load(open(os.path.join(root, "data.yaml")))
    base["path"] = root
    base["train"], base["val"] = "train/images", "valid/images"

    # --- device: decided ONCE, used for every run -----------------------------
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        try:                                   # prove it actually works before committing
            torch.zeros(8, device="cuda").sum().item()
        except Exception as e:
            print(f"[device] CUDA present but unusable ({e}); every run will use CPU.")
            device = "cpu"
    print(f"[device] every run uses device={device} "
          f"(torch {torch.__version__}, cuda_available={torch.cuda.is_available()})")

    work = os.path.join(a.out_dir, "rescore_splits")
    os.makedirs(work, exist_ok=True)

    # --- the jobs -------------------------------------------------------------
    jobs = []

    full_yaml = write_list_yaml(base, all_test, os.path.join(work, "test_full.txt"),
                                os.path.join(work, "data_full.yaml"))
    jobs.append(("full", full_yaml, n_full, n_full_labels))

    summary_path = os.path.join(a.audit_dir, "summary.json")
    dedup_yaml_path = os.path.join(a.audit_dir, "data_dedup.yaml")
    dedup_list = os.path.join(a.audit_dir, "test_dedup.txt")
    n_removed = 0
    n_dedup = n_full
    if os.path.isfile(summary_path) and os.path.isfile(dedup_yaml_path):
        summary = json.load(open(summary_path))
        n_removed = int(summary.get("contaminated", 0))
        kept = [l.strip() for l in open(dedup_list, encoding="utf-8") if l.strip()]
        n_dedup = len(kept)
        jobs.append(("dedup", os.path.abspath(dedup_yaml_path), n_dedup, count_labels(kept)))
        if n_removed != n_full - n_dedup:
            print(f"[warn] summary.json says {n_removed} contaminated but the clean list is "
                  f"{n_full - n_dedup} shorter; using {n_full - n_dedup} for the controls.")
            n_removed = n_full - n_dedup
    else:
        print(f"[warn] no audit output at {a.audit_dir}; running the full split only.")

    seeds = []
    if n_removed > 0:
        import numpy as np
        for s in range(a.controls):
            seeds.append(s)
            keep_idx = sorted(np.random.default_rng(s)
                              .permutation(n_full)[: n_full - n_removed].tolist())
            keep = [all_test[i] for i in keep_idx]
            y = write_list_yaml(base, keep, os.path.join(work, f"test_ctrl{s}.txt"),
                                os.path.join(work, f"data_ctrl{s}.yaml"))
            jobs.append((f"ctrl{s}", y, len(keep), count_labels(keep)))
    elif os.path.isfile(summary_path):
        print("[note] no duplicates found, so no control subsets are needed (the spec asks for "
              "controls only if duplicates were found).")
        seeds = []

    # --- run them -------------------------------------------------------------
    val_project = os.path.abspath(os.path.join(a.out_dir, "rescore_runs"))
    runs = []
    for tag, y, ni, nl in jobs:
        try:
            runs.append(run_one(tag, y, ni, nl, a.model, a.imgsz, device, val_project,
                                plots=a.plots, focus=a.focus_classes))
        except Exception as e:
            if device != "cpu":
                print(f"[device] run '{tag}' failed on GPU ({e}); restarting EVERY run on CPU "
                      f"so the comparison stays on one device.", file=sys.stderr)
                device, runs = "cpu", []
                for t2, y2, ni2, nl2 in jobs:
                    runs.append(run_one(t2, y2, ni2, nl2, a.model, a.imgsz, device, val_project,
                                        plots=a.plots, focus=a.focus_classes))
                break
            raise

    # --- verdict: dedup vs the control spread ---------------------------------
    by_tag = {r["tag"]: r for r in runs}
    verdict = []
    ctrl = [r for r in runs if r["tag"].startswith("ctrl")]
    if "dedup" in by_tag and ctrl:
        for k in ("precision", "recall", "map50", "map50_95"):
            d = by_tag["dedup"]["overall"][k]
            cv = [c["overall"][k] for c in ctrl]
            lo, hi = min(cv), max(cv)
            inside = lo <= d <= hi
            verdict.append(
                f"**{k}**: de-duplicated {d * 100:.2f} vs control range "
                f"[{lo * 100:.2f}, {hi * 100:.2f}] (mean {mean(cv) * 100:.2f}) — "
                f"{'INSIDE' if inside else 'OUTSIDE'} the control range"
                f"{'' if inside else (' (below)' if d < lo else ' (above)')}.")

    # The same inside/outside question, asked of the focus-class mean.
    if a.focus_classes and "dedup" in by_tag and ctrl \
            and by_tag["dedup"].get("focus", {}).get("mean"):
        for k in ("precision", "recall", "map50", "map50_95"):
            d = by_tag["dedup"]["focus"]["mean"].get(k)
            cv = [c["focus"]["mean"].get(k) for c in ctrl
                  if c.get("focus", {}).get("mean", {}).get(k) is not None]
            if d is None or not cv:
                continue
            lo, hi = min(cv), max(cv)
            inside = lo <= d <= hi
            verdict.append(
                f"**{k} (focus classes)**: de-duplicated {d * 100:.2f} vs control range "
                f"[{lo * 100:.2f}, {hi * 100:.2f}] (mean {mean(cv) * 100:.2f}) — "
                f"{'INSIDE' if inside else 'OUTSIDE'} the control range"
                f"{'' if inside else (' (below)' if d < lo else ' (above)')}.")

    # --- compare the full split with phase2/benchmark.json ---------------------
    bench_cmp = None
    try:
        b = json.load(open(a.benchmark, encoding="utf-8")) \
            if (os.path.isfile(a.benchmark) and "full" in by_tag) else None
        f = by_tag.get("full")
        if b is not None and f is not None:
            keys = ("precision", "recall", "map50", "map50_95")
            bench_cmp = {"benchmark_file": a.benchmark,
                         "benchmark_num_images": b.get("num_images"),
                         "rescore_num_images": f["count_check"]["summary_images"],
                         "overall": {}, "per_class": {}}
            for k in keys:
                bv, rv = b.get("overall", {}).get(k), f["overall"].get(k)
                if bv is not None and rv is not None:
                    bench_cmp["overall"][k] = {"benchmark": bv, "rescore": rv,
                                               "delta_pp": (rv - bv) * 100}
            for c, bm in b.get("per_class", {}).items():
                rm = f["per_class"].get(c, {})
                row = {}
                for k in keys:
                    bv, rv = bm.get(k), rm.get(k)
                    if bv is not None and rv is not None:
                        row[k] = {"benchmark": bv, "rescore": rv, "delta_pp": (rv - bv) * 100}
                if row:
                    bench_cmp["per_class"][c] = row
    except Exception as e:
        print(f"[warn] could not compare with {a.benchmark}: {e!r}; the run's own metrics "
              f"are unaffected and are still reported.", file=sys.stderr)
        bench_cmp = None

    # Overall precision/recall/mAP are means over the classes PRESENT in a split, so a
    # class missing from one run would silently change that run's denominator.
    class_sets = {r["tag"]: tuple(r["classes_scored"]) for r in runs}
    consistent = len(set(class_sets.values())) <= 1
    if not consistent:
        print("\n[warn] the runs did not all score the same classes; overall means are not "
              "directly comparable. Per-run class sets:")
        for t, cs in class_sets.items():
            print(f"    {t}: {', '.join(cs) or '(none)'}")

    payload = {
        "model": a.model, "dataset_dir": root, "imgsz": a.imgsz, "device": device,
        "n_full": n_full, "n_removed": n_removed, "n_dedup": n_dedup,
        "control_seeds": seeds,
        "focus_classes": a.focus_classes,
        "plots": a.plots,
        "class_set_consistent": consistent,
        "classes_per_run": {k: list(v) for k, v in class_sets.items()},
        "versions": {"python": sys.version.split()[0],
                     "torch": __import__("torch").__version__,
                     "ultralytics": __import__("ultralytics").__version__},
        "runs": runs, "verdict": verdict, "benchmark_compare": bench_cmp,
    }

    os.makedirs(a.out_dir, exist_ok=True)
    jpath = os.path.join(a.out_dir, "rescore.json")
    mpath = os.path.join(a.out_dir, "rescore.md")
    # The metrics cost hours of CPU; get them on disk before anything that could raise.
    json.dump(payload, open(jpath, "w", encoding="utf-8"), indent=2)
    try:
        write_md(mpath, payload)
    except Exception as e:
        print(f"[error] rescore.json was written, but rendering rescore.md failed: {e!r}",
              file=sys.stderr)
        raise

    print("\n" + "=" * 78)
    for line in verdict:
        print(" " + line.replace("**", ""))
    if bench_cmp:
        print(f"\n vs {a.benchmark} (full split):")
        for k, v in bench_cmp["overall"].items():
            print(f"   {k:<10} benchmark {v['benchmark'] * 100:6.2f}  "
                  f"now {v['rescore'] * 100:6.2f}  ({v['delta_pp']:+.2f} pp)")
    print(f"\nWrote {jpath}\nWrote {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
