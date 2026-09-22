"""Validate `kaggle_ppe_grouped.ipynb` locally, before it costs 12 h of Kaggle GPU.

Three things can be checked on this laptop, and they are the three that would waste the
whole Kaggle run if wrong:

  1. **The embedded argument sets are real Python and are accepted by ultralytics.**
     They were lifted verbatim out of the original checkpoints, so they contain ~108 keys
     including several that are not training arguments at all. This runs 1 epoch on a
     handful of images at imgsz 320 on CPU with each stage's arguments — everything as
     recorded except `epochs`, `imgsz`, `device`, `plots`, `data`, `project` and `name` —
     and reports whether ultralytics took them. The temporary run is deleted afterwards.
  2. **The split assertion in cell 6 matches reality** — the sha256 hard-coded in the
     notebook against the one in the locally applied `split_summary.json`.
  3. **The repo references in cell 5 exist** — the branch name the notebook clones and the
     CSV path it expects inside the clone.

    python tools/validate_grouped_notebook.py --weights best_refined.pt

Anything it cannot check locally (the Roboflow download, the GPU, the 12 h budget) is
listed at the end so the gap is explicit rather than assumed away.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

SPLIT_KEYS = ("epochs", "imgsz", "batch", "optimizer", "lr0", "patience", "seed", "device")


def cell_sources(nb_path):
    nb = json.load(open(nb_path, encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"]], nb


def extract_dict(sources, name):
    for src in sources:
        m = re.search(rf"^{name} = (\{{.*?^\}})", src, re.S | re.M)
        if m:
            ns = {}
            exec(f"{name} = {m.group(1)}", ns)       # noqa: S102 - our own generated cell
            return ns[name]
    return None


def find_literal(sources, name):
    for src in sources:
        m = re.search(rf"^{name} = '([^']*)'", src, re.M)
        if m:
            return m.group(1)
    return None


def tiny_dataset(grouped, n_train, n_val, dest):
    """A few images from the grouped split, hard-linked, with a 5-class data.yaml."""
    import yaml
    spec = yaml.safe_load(open(os.path.join(grouped, "data.yaml"), encoding="utf-8"))
    for s in ("train", "valid"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(dest, s, sub), exist_ok=True)
    took = {}
    for src_split, dst_split, k in (("train", "train", n_train), ("valid", "valid", n_val)):
        d = os.path.join(grouped, src_split, "images")
        names = sorted(os.listdir(d))[:k]
        for nm in names:
            for sub, ext in (("images", None), ("labels", ".txt")):
                s_name = nm if ext is None else os.path.splitext(nm)[0] + ext
                s = os.path.join(grouped, src_split, sub, s_name)
                t = os.path.join(dest, dst_split, sub, s_name)
                if not os.path.isfile(s):
                    continue
                try:
                    os.link(s, t)
                except OSError:
                    shutil.copy2(s, t)
        took[dst_split] = len(names)
    out = {"path": os.path.abspath(dest), "train": "train/images", "val": "valid/images",
           "test": "valid/images", "nc": spec.get("nc"), "names": spec.get("names")}
    yp = os.path.join(dest, "data.yaml")
    with open(yp, "w", encoding="utf-8") as fh:
        yaml.safe_dump(out, fh, sort_keys=False, allow_unicode=True)
    return yp, took


def try_train(stage, args, weights, data_yaml, project, imgsz, batch):
    """1 epoch on CPU with the recorded arguments. Returns (ok, message)."""
    from ultralytics import YOLO
    a = dict(args)
    a.update(data=data_yaml, project=project, name=f"validate_{stage}",
             epochs=1, imgsz=imgsz, device="cpu", plots=False, batch=batch,
             exist_ok=True, val=True, workers=0)
    print(f"\n  {stage}: {len(args)} recorded args; overriding epochs=1 imgsz={imgsz} "
          f"device=cpu plots=False batch={batch} workers=0")
    print(f"  as recorded: " + "  ".join(f"{k}={args.get(k)!r}" for k in SPLIT_KEYS))
    try:
        YOLO(weights).train(**a)
        return True, "accepted"
    except Exception as e:                            # noqa: BLE001 - we report whatever it is
        return False, f"{type(e).__name__}: {e}"


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    ap = argparse.ArgumentParser(description="Validate the grouped-retrain notebook locally")
    ap.add_argument("--notebook", default="kaggle_ppe_grouped.ipynb")
    ap.add_argument("--grouped-dir", default="data/ppe_grouped")
    ap.add_argument("--weights", default="best_refined.pt",
                    help="starting weights for the 1-epoch acceptance run (class count "
                         "must match the dataset; the weights themselves are irrelevant)")
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--val-images", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--branch", default="paper-prep")
    ap.add_argument("--skip-train", action="store_true",
                    help="check everything except the two 1-epoch runs")
    a = ap.parse_args(argv)

    checks, notes = {}, []
    sources, nb = cell_sources(a.notebook)
    print(f"{a.notebook}: {len(nb['cells'])} cells")

    # --- 1. the embedded argument sets ---------------------------------------
    s1 = extract_dict(sources, "STAGE1_ARGS")
    s2 = extract_dict(sources, "STAGE2_ARGS")
    checks["stage 1 args are a real Python dict"] = isinstance(s1, dict) and bool(s1)
    checks["stage 2 args are a real Python dict"] = isinstance(s2, dict) and bool(s2)
    if s1 and s2:
        print(f"  stage 1: {len(s1)} args | " +
              "  ".join(f"{k}={s1.get(k)!r}" for k in SPLIT_KEYS))
        print(f"  stage 2: {len(s2)} args | " +
              "  ".join(f"{k}={s2.get(k)!r}" for k in SPLIT_KEYS))
        checks["model/save_dir are not passed as train args"] = not (
            {"model", "save_dir"} & (set(s1) | set(s2)))
        # the recipe must not have drifted from what the checkpoints recorded
        checks["stage 1 recipe matches best.pt"] = (
            s1.get("epochs") == 50 and s1.get("batch") == 96 and
            s1.get("optimizer") == "auto" and s1.get("lr0") == 0.01 and
            s1.get("patience") == 12 and s1.get("seed") == 0)
        checks["stage 2 recipe matches best_refined.pt"] = (
            s2.get("epochs") == 20 and s2.get("batch") == 48 and
            s2.get("optimizer") == "SGD" and s2.get("lr0") == 0.001 and
            s2.get("patience") == 10 and s2.get("seed") == 0)

    # --- 2. the sha256 assertion in cell 6 -----------------------------------
    want = find_literal(sources, "EXPECTED_SHA256")
    sp = os.path.join(a.grouped_dir, "split_summary.json")
    checks["notebook hard-codes a sha256"] = bool(want)
    if want and os.path.isfile(sp):
        got = json.load(open(sp, encoding="utf-8")).get("assignment_sha256")
        checks["notebook sha256 == the locally applied split"] = (want == got)
        print(f"  sha256 in notebook : {want}")
        print(f"  sha256 applied here: {got}")
    else:
        notes.append(f"{sp} not present, so the sha256 assertion was not cross-checked")

    # --- 3. cell 5's repo references -----------------------------------------
    joined = "\n".join(sources)
    checks[f"notebook clones branch {a.branch}"] = f"--branch {a.branch}" in joined
    csv_rel = os.path.join("tools", "splits", "ppe_grouped_split.csv")
    checks["the split csv it expects exists in this repo"] = os.path.isfile(csv_rel)
    checks["notebook reads the Roboflow key from Kaggle Secrets"] = (
        "UserSecretsClient" in joined and "ROBOFLOW_API_KEY" in joined)
    checks["notebook never prints the Roboflow key"] = not re.search(
        r"print\([^)]*ROBOFLOW_API_KEY(?!\s*\))", joined)
    checks["ultralytics is pinned without -U"] = (
        'pip install -q "ultralytics==' in joined and " -U " not in joined)
    m = re.search(r'ultralytics==([0-9.]+)"', joined)
    if m:
        print(f"  ultralytics pinned to {m.group(1)}")

    # --- 4. the 1-epoch acceptance runs --------------------------------------
    if a.skip_train:
        notes.append("the two 1-epoch acceptance runs were skipped (--skip-train)")
    elif not (s1 and s2):
        notes.append("argument sets could not be extracted, so no acceptance run was tried")
    elif not os.path.isfile(a.weights):
        notes.append(f"{a.weights} not found, so no acceptance run was tried")
    else:
        tmp = tempfile.mkdtemp(prefix="nbvalidate_")
        try:
            data_yaml, took = tiny_dataset(a.grouped_dir, a.images, a.val_images,
                                           os.path.join(tmp, "tiny"))
            print(f"\ntiny dataset: {took['train']} train / {took['valid']} val images "
                  f"from {a.grouped_dir}")
            project = os.path.join(tmp, "runs")
            for stage, args in (("stage1", s1), ("stage2", s2)):
                ok, msg = try_train(stage, args, a.weights, data_yaml, project,
                                    a.imgsz, a.batch)
                checks[f"ultralytics accepts the {stage} argument set"] = ok
                print(f"  -> {stage}: {msg}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            print(f"\ndeleted the temporary run and dataset ({tmp})")

    print()
    for k, v in checks.items():
        print(f"  [{'ok ' if v else 'FAIL'}] {k}")
    if notes:
        print("\n  not checked locally:")
        for nt in notes:
            print(f"   - {nt}")
    print("\n  cannot be checked off Kaggle: the Roboflow download (needs the key and "
          "~2.8 GB), the T4 x2 GPU, multi-GPU device='0,1', and the 12 h budget.")
    allok = all(checks.values())
    print(f"\nALL_NOTEBOOK_VALIDATE {allok}")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
