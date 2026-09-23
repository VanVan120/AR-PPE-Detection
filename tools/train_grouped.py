"""Train the PPE detector on the source-grouped split, on this laptop's GPU.

This is the local alternative to `kaggle_ppe_grouped.ipynb`. Same split, same recipe, same
two stages; the only difference is that 6 GB of laptop VRAM cannot hold the original batch,
so the batch is smaller and gradient accumulation puts the *effective* batch back where it
was. See `local_recipe()` for exactly how, and REPORT_local_train.md for the table.

Nothing about the recipe is transcribed by hand. Both argument sets are read straight out of
the original checkpoints (`best.pt` and `best_refined.pt` carry `train_args`, which is the
same mapping ultralytics writes to `args.yaml`), so there is no third copy to drift.

    .venv-gpu\\Scripts\\python.exe tools\\train_grouped.py
    .venv-gpu\\Scripts\\python.exe tools\\train_grouped.py --resume
    .venv-gpu\\Scripts\\python.exe tools\\train_grouped.py --smoke

Stage 1 trains from `yolov8s.pt`; stage 2 continues from stage 1's best checkpoint. Each
stage is evaluated with `tools/eval_grouped.py` as soon as it finishes, so a crash in stage 2
cannot cost the stage-1 result.
"""
from __future__ import annotations

# Must be set before ultralytics is imported anywhere. ultralytics will otherwise pip-install
# packages into this environment behind our back (phase4_deploy/README.md, "Traps", #1).
import os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import argparse
import csv
import datetime
import json
import math
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import group_split as gs  # noqa: E402  (after sys.path)

# The split this script is allowed to train on. Defined by tools/splits/ppe_grouped_split.csv
# and recorded in paper_materials/retrain/REPORT_retrain.md §3.6.
EXPECTED_SHA256 = "60d0437ee36fa6f62a2f235fd2159becbfa812309bed394cdca1b6f2b9293596"

DEFAULT_CSV = os.path.join(HERE, "splits", "ppe_grouped_split.csv")
DEFAULT_DATA = os.path.join(ROOT, "data", "ppe_grouped")
DEFAULT_OUT = os.path.join(ROOT, "paper_materials", "retrain", "local_out")
DEFAULT_PROJECT = os.path.join(ROOT, "paper_materials", "retrain", "local_runs")

# Dropped from the recorded arguments: `model` is supplied explicitly (stage 1 from
# yolov8s.pt, stage 2 from stage 1's best), and `save_dir` would override project/name.
DROP = ("model", "save_dir")

STAGES = (
    {"key": "stage1", "ckpt": "best.pt", "name": "grouped_s1"},
    {"key": "stage2", "ckpt": "best_refined.pt", "name": "grouped_s2"},
)


# ----------------------------------------------------------------- the recorded recipe

def args_from_ckpt(path):
    """The `args.yaml` ultralytics embedded in a checkpoint, as a dict.

    Same source `tools/ckpt_train_record.py` reads. Reading it here rather than copying the
    values means the local run cannot silently disagree with the original recipe.
    """
    import torch

    if not os.path.isfile(path):
        raise SystemExit(f"[FAIL] checkpoint not found: {path}\n"
                         f"       stage arguments are read from it; it is not optional.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("train_args")
    if not isinstance(a, dict):
        raise SystemExit(f"[FAIL] {path} carries no train_args mapping")
    return {k: v for k, v in a.items() if k not in DROP}


def local_recipe(orig, batch):
    """Re-fit one stage's batch to a smaller GPU without changing what the optimiser sees.

    ultralytics 8.x (engine/trainer.py) does, at the top of `_setup_train`:

        accumulate   = max(round(nbs / batch), 1)
        weight_decay = weight_decay * batch * accumulate / nbs
        iterations   = ceil(len(dataset) / max(batch, nbs)) * epochs

    so two quantities decide what the run actually optimises: the effective batch
    `batch * accumulate`, and the scaled weight decay. Setting

        nbs          = the original effective batch
        weight_decay = the original *scaled* weight decay

    reproduces both exactly, because then `batch * accumulate / nbs` is 1 and the scaling is
    the identity. It also leaves `max(batch, nbs)` unchanged, which matters for stage 1:
    `optimizer='auto'` picks the optimiser and learning rate from `iterations`, and that
    count would otherwise move.

    Returns the replacement values plus everything needed to show the two side by side.
    """
    o_batch, o_nbs, o_wd = orig["batch"], orig["nbs"], orig["weight_decay"]
    o_accum = max(round(o_nbs / o_batch), 1)
    eff = o_batch * o_accum
    o_scaled_wd = o_wd * o_batch * o_accum / o_nbs

    nbs = eff                       # so that accumulate lands exactly on eff / batch
    accum = max(round(nbs / batch), 1)
    wd = o_scaled_wd                # scaling is now the identity
    scaled_wd = wd * batch * accum / nbs

    # These are the properties being preserved. If a batch size is ever chosen that does not
    # divide the effective batch, this is where it stops -- silently training at a different
    # effective batch is exactly the kind of difference this task forbids.
    if batch * accum != eff:
        raise SystemExit(
            f"[FAIL] batch {batch} cannot reproduce the effective batch {eff}: "
            f"accumulate would be {accum}, giving {batch * accum}. Choose a batch that "
            f"divides {eff}.")
    if abs(scaled_wd - o_scaled_wd) > 1e-12:
        raise SystemExit(f"[FAIL] scaled weight decay {scaled_wd} != original {o_scaled_wd}")
    # `iterations` (trainer.py:283) feeds exactly one decision: the optimiser that
    # `optimizer='auto'` picks (trainer.py:1027). It is not used anywhere else, so it only has
    # to be preserved for a stage that actually asks for 'auto'. Stage 2 names SGD outright,
    # and there max(batch, nbs) legitimately moves from 64 to 48.
    auto = str(orig.get("optimizer", "")).lower() == "auto"
    if auto and max(batch, nbs) != max(o_batch, o_nbs):
        raise SystemExit(
            f"[FAIL] optimizer='auto' and max(batch, nbs) changed from "
            f"{max(o_batch, o_nbs)} to {max(batch, nbs)}; that moves the iteration count "
            f"the automatic optimiser choice is made from.")

    return {
        "batch": batch, "nbs": nbs, "weight_decay": wd,
        "_orig": {"batch": o_batch, "nbs": o_nbs, "weight_decay": o_wd,
                  "accumulate": o_accum, "effective_batch": eff,
                  "scaled_weight_decay": o_scaled_wd},
        "_local": {"batch": batch, "nbs": nbs, "weight_decay": wd,
                   "accumulate": accum, "effective_batch": batch * accum,
                   "scaled_weight_decay": scaled_wd},
    }


def auto_optimizer_choice(n_train, batch, nbs, epochs, nc):
    """What `optimizer='auto'` will select, by ultralytics' own rule.

    Recorded so the report can state it rather than assume it. The name comes from the pinned
    ultralytics source at run time, not from memory -- see `check_trainer_formulas`.
    """
    iterations = math.ceil(n_train / max(batch, nbs)) * epochs
    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    return {"iterations": iterations, "over_10000": iterations > 10000,
            "lr_fit_if_under": lr_fit}


def check_trainer_formulas():
    """Confirm the three formulas above are what the *installed* ultralytics actually runs.

    The pin exists because these change between versions -- 8.4.105 selects MuSGD where
    8.4.75 selects SGD, for instance. Reading the source beats trusting a memory of it.
    """
    import ultralytics
    from ultralytics.engine import trainer as tr

    src = open(tr.__file__, encoding="utf-8").read()
    want = {
        "accumulate": "self.accumulate = max(round(self.args.nbs / self.batch_size), 1)",
        "weight_decay": ("weight_decay = self.args.weight_decay * self.batch_size * "
                         "self.accumulate / self.args.nbs"),
        "iterations": ("iterations = math.ceil(len(self.train_loader.dataset) / "
                       "max(self.batch_size, self.args.nbs)) * self.epochs"),
    }
    found = {k: (v in src) for k, v in want.items()}

    # Which optimiser 'auto' picks above 10k iterations, read out of the source line.
    auto_name = None
    for line in src.splitlines():
        if "if iterations > 10000 else" in line and "name, lr, momentum" in line:
            auto_name = line.strip()
            break
    return {"ultralytics": ultralytics.__version__, "formulas_present": found,
            "auto_selection_line": auto_name, "all_present": all(found.values())}


# ----------------------------------------------------------------- the split check

def verify_split(data_dir, csv_path):
    """Refuse to train unless `data_dir` is the split the CSV defines.

    Two separate things are checked, because either alone can pass on the wrong data:
      1. the CSV still hashes to EXPECTED_SHA256 -- it is the file that was audited;
      2. the images actually on disk are exactly the CSV's assignment, split by split.
    """
    if not os.path.isfile(csv_path):
        raise SystemExit(f"[FAIL] split definition not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    got = gs.assignment_sha256(rows)
    if got != EXPECTED_SHA256:
        raise SystemExit(
            f"[FAIL] {csv_path} does not hash to the audited split.\n"
            f"       expected {EXPECTED_SHA256}\n       got      {got}\n"
            f"       This is not the split REPORT_retrain.md describes. Do not train.")

    want = {}
    for r in rows:
        want.setdefault(r["split"], set()).add(os.path.basename(r["file"]))
    have = {}
    for rel, _ in gs.scan_dataset(os.path.abspath(data_dir)):
        split, _, name = rel.split("/", 2)
        have.setdefault(split, set()).add(name)

    problems = []
    for split in sorted(set(want) | set(have)):
        w, h = want.get(split, set()), have.get(split, set())
        if w != h:
            problems.append(f"  {split}: {len(w)} in the CSV, {len(h)} on disk, "
                            f"{len(w - h)} missing, {len(h - w)} unexpected")
    if problems:
        raise SystemExit("[FAIL] %s does not match the split definition:\n%s\n"
                         "       Rebuild it with group_split.py --apply." %
                         (data_dir, "\n".join(problems)))

    summary_path = os.path.join(data_dir, "split_summary.json")
    recorded = None
    if os.path.isfile(summary_path):
        recorded = json.load(open(summary_path, encoding="utf-8")).get("assignment_sha256")
        if recorded != EXPECTED_SHA256:
            raise SystemExit(f"[FAIL] {summary_path} records sha256 {recorded}, "
                             f"expected {EXPECTED_SHA256}")

    counts = {s: len(v) for s, v in sorted(have.items())}
    print(f"[split ok] {csv_path}")
    print(f"           sha256 {got}")
    print(f"           on disk: " + "  ".join(f"{s} {n}" for s, n in counts.items()))
    return {"sha256": got, "counts": counts, "summary_sha256": recorded}


# ----------------------------------------------------------------- smoke fixtures

def write_smoke_yaml(data_dir, work, n_train, n_val):
    """A data.yaml with tiny train/val lists, for the 1-epoch smoke test."""
    import yaml

    base = yaml.safe_load(open(os.path.join(data_dir, "data.yaml"), encoding="utf-8"))
    os.makedirs(work, exist_ok=True)
    spec = dict(base)
    spec["path"] = os.path.abspath(data_dir)
    for key, split, n in (("train", "train", n_train), ("val", "valid", n_val)):
        imgs = [p for _, p in gs.scan_dataset(os.path.abspath(data_dir))
                if os.sep + split + os.sep in p][:n]
        if len(imgs) < n:
            raise SystemExit(f"[FAIL] only {len(imgs)} images in {split}, wanted {n}")
        lst = os.path.join(work, f"{key}_smoke.txt")
        with open(lst, "w", encoding="utf-8") as fh:
            fh.write("\n".join(imgs) + "\n")
        spec[key] = os.path.abspath(lst)
    spec.pop("test", None)
    out = os.path.join(work, "data_smoke.yaml")
    with open(out, "w", encoding="utf-8") as fh:
        yaml.safe_dump(spec, fh, sort_keys=False, allow_unicode=True)
    return out


# ----------------------------------------------------------------- running a stage

def stage_done(stage_dir):
    return os.path.isfile(os.path.join(stage_dir, ".stage_complete"))


def mark_done(stage_dir):
    with open(os.path.join(stage_dir, ".stage_complete"), "w", encoding="utf-8") as fh:
        fh.write(datetime.datetime.now().isoformat(timespec="seconds") + "\n")


def run_stage(spec, init_weights, data_yaml, project, batch, epochs, workers, resume,
              device, n_train, nc):
    """Train one stage. Returns (best_ckpt, stage_dir, recipe, seconds)."""
    from ultralytics import YOLO

    orig = args_from_ckpt(os.path.join(ROOT, spec["ckpt"]))
    recipe = local_recipe(orig, batch)
    stage_dir = os.path.join(project, spec["name"])
    last = os.path.join(stage_dir, "weights", "last.pt")

    args = dict(orig)
    args.update(data=data_yaml, project=project, name=spec["name"], device=device,
                batch=recipe["batch"], nbs=recipe["nbs"],
                weight_decay=recipe["weight_decay"])
    if epochs is not None:
        args["epochs"] = epochs
    if workers is not None:
        args["workers"] = workers
    # Training plots are kept on purpose: the curves go in the paper.
    args["plots"] = True
    args["exist_ok"] = True
    args.pop("resume", None)

    auto = None
    if str(orig.get("optimizer", "")).lower() == "auto":
        auto = auto_optimizer_choice(n_train, recipe["batch"], recipe["nbs"],
                                     args["epochs"], nc)
        auto["original"] = auto_optimizer_choice(
            n_train, orig["batch"], orig["nbs"], args["epochs"], nc)

    print("\n" + "=" * 78)
    print(f" {spec['key']}: batch {recipe['batch']} x accumulate "
          f"{recipe['_local']['accumulate']} = effective {recipe['_local']['effective_batch']} "
          f"(original {recipe['_orig']['effective_batch']})")
    print(f" weight_decay {recipe['weight_decay']:.6g} -> scaled "
          f"{recipe['_local']['scaled_weight_decay']:.6g} "
          f"(original scaled {recipe['_orig']['scaled_weight_decay']:.6g})")
    if auto:
        print(f" optimizer='auto': {auto['iterations']} iterations "
              f"(original {auto['original']['iterations']}) -> "
              f"{'>' if auto['over_10000'] else '<='} 10000")
    print("=" * 78)

    t0 = time.time()
    if resume and os.path.isfile(last):
        print(f"[resume] continuing {spec['key']} from {last}")
        YOLO(last).train(resume=True)
    else:
        YOLO(init_weights).train(**args)
    secs = time.time() - t0

    best = os.path.join(stage_dir, "weights", "best.pt")
    if not os.path.isfile(best):
        raise SystemExit(f"[FAIL] {spec['key']} finished but {best} is missing")
    mark_done(stage_dir)
    return best, stage_dir, recipe, secs, auto, args["epochs"]


def run_eval(weights, data_dir, out_dir, name, limit, note):
    """Evaluate in a SEPARATE process.

    Two reasons, both of them things that have actually bitten this project: ultralytics'
    `select_device` sets CUDA_VISIBLE_DEVICES process-wide and never restores it
    (phase4_deploy/README.md, "Traps", #2), and a training process holds its CUDA cache, which
    on 6 GB is enough to push the evaluation into OOM.
    """
    cmd = [sys.executable, os.path.join(HERE, "eval_grouped.py"),
           "--weights", weights, "--dataset-dir", data_dir,
           "--out-dir", out_dir, "--name", name]
    if limit:
        cmd += ["--limit", str(limit)]
    if note:
        cmd += ["--note", note]
    env = dict(os.environ, YOLO_AUTOINSTALL="false")
    print("\n[eval] " + " ".join(cmd[1:]))
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[FAIL] eval_grouped.py exited {r.returncode} for {name}")
    return os.path.join(out_dir, f"{name}.json")


# ----------------------------------------------------------------- main

def collect_versions():
    import cv2
    import numpy
    import torch
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "ultralytics": ultralytics.__version__,
        "numpy": numpy.__version__,
        "opencv": cv2.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mb": (round(torch.cuda.get_device_properties(0).total_memory / 2**20)
                         if torch.cuda.is_available() else None),
    }


def git_commit():
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        dirty = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30)
        return {"commit": out.stdout.strip() or None,
                "dirty": bool(dirty.stdout.strip())}
    except Exception as exc:                                     # pragma: no cover
        return {"commit": None, "error": str(exc)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--batch", type=int, default=16,
                    help="local batch size; must divide each stage's effective batch")
    ap.add_argument("--workers", type=int, default=None,
                    help="override the recorded workers (Windows RAM)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--resume", action="store_true",
                    help="continue: skip finished stages, resume an interrupted one")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch per stage on 200 train images, eval on 50 test images")
    ap.add_argument("--smoke-train", type=int, default=200)
    ap.add_argument("--smoke-val", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=None, help="override epochs (testing)")
    ap.add_argument("--keep", action="store_true",
                    help="with --smoke, keep the run and output dirs instead of deleting "
                         "them (used by the resume test)")
    a = ap.parse_args(argv)

    t_start = time.time()
    started = datetime.datetime.now().isoformat(timespec="seconds")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("[FAIL] torch reports no CUDA device. This script is the GPU route; "
                         "use kaggle_ppe_grouped.ipynb or fix the environment.")

    versions = collect_versions()
    formulas = check_trainer_formulas()
    print(f"[env] {versions['gpu']} ({versions['gpu_total_mb']} MB) | torch "
          f"{versions['torch']} cu{versions['torch_cuda']} | ultralytics "
          f"{versions['ultralytics']}")
    if not formulas["all_present"]:
        raise SystemExit(
            "[FAIL] the installed ultralytics does not contain the accumulate / weight-decay "
            "formulas this script compensates for:\n  " + json.dumps(formulas, indent=2) +
            "\n  Refusing to train, because the effective batch would be unknown.")
    print(f"[env] trainer formulas confirmed in ultralytics {formulas['ultralytics']}")
    print(f"[env] optimizer='auto' line: {formulas['auto_selection_line']}")

    split = verify_split(a.data_dir, a.csv)
    n_train = split["counts"].get("train", 0)

    import yaml
    base = yaml.safe_load(open(os.path.join(a.data_dir, "data.yaml"), encoding="utf-8"))
    nc = base.get("nc") or len(base.get("names", []))

    data_yaml = os.path.abspath(os.path.join(a.data_dir, "data.yaml"))
    project, out = os.path.abspath(a.project), os.path.abspath(a.out)
    work = None
    epochs = a.epochs
    if a.smoke:
        project = os.path.join(project + "_smoke")
        out = os.path.join(out + "_smoke")
        work = os.path.join(project, "_fixture")
        data_yaml = write_smoke_yaml(a.data_dir, work, a.smoke_train, a.smoke_val)
        n_train = a.smoke_train
        epochs = a.epochs if a.epochs is not None else 1   # --epochs wins, for the resume test
        print(f"[smoke] {a.smoke_train} train / {a.smoke_val} val images, "
              f"{epochs} epoch(s) per stage")
    os.makedirs(out, exist_ok=True)

    results, init = {}, os.path.join(ROOT, "yolov8s.pt")
    banked = {}

    def write_run_info(complete):
        """Rewritten after every stage, not just at the end.

        This run is hours long and unattended. If stage 2 dies, the stage-1 record -- which
        versions, which split, which effective batch -- still has to survive on disk.
        """
        info = {
            "started": started,
            "ended": datetime.datetime.now().isoformat(timespec="seconds"),
            "hours": round((time.time() - t_start) / 3600, 4),
            "complete": complete,
            # best_grouped.pt is refreshed after each stage, so if stage 2 never finished it
            # holds stage 1's weights. Say so rather than let the file name imply otherwise.
            "best_grouped_from": banked.get("stage"),
            "smoke": a.smoke,
            "git": git_commit(),
            "versions": versions,
            "trainer_formulas": formulas,
            "split": split,
            "batch": a.batch,
            "device": a.device,
            "workers_override": a.workers,
            "stages": results,
        }
        with open(os.path.join(out, "run_info.json"), "w", encoding="utf-8") as fh:
            json.dump(info, fh, indent=2, default=str)

    for spec in STAGES:
        stage_dir = os.path.join(project, spec["name"])
        best = os.path.join(stage_dir, "weights", "best.pt")
        if a.resume and stage_done(stage_dir) and os.path.isfile(best):
            print(f"[resume] {spec['key']} already complete, skipping")
            # Record the recipe even for a skipped stage. run_info.json is required to carry
            # the Part B values, and a --resume that skips both stages would otherwise
            # overwrite a complete record with a thinner one.
            orig = args_from_ckpt(os.path.join(ROOT, spec["ckpt"]))
            recipe = local_recipe(orig, a.batch)
            auto = None
            if str(orig.get("optimizer", "")).lower() == "auto":
                ep = epochs if epochs is not None else orig["epochs"]
                auto = auto_optimizer_choice(n_train, recipe["batch"], recipe["nbs"], ep, nc)
                auto["original"] = auto_optimizer_choice(
                    n_train, orig["batch"], orig["nbs"], ep, nc)
            results[spec["key"]] = {"skipped": True, "best": best, "seconds": 0.0,
                                    "stage_dir": stage_dir, "recipe": recipe,
                                    "auto_optimizer": auto}
            init = best
            banked["stage"] = spec["key"]
            continue

        best, stage_dir, recipe, secs, auto, ran_epochs = run_stage(
            spec, init, data_yaml, project, a.batch, epochs, a.workers,
            a.resume, a.device, n_train, nc)

        name = f"eval_{spec['key']}"
        note = ("SMOKE TEST -- 1 epoch on a handful of images; these numbers measure nothing."
                if a.smoke else None)
        run_eval(best, a.data_dir, os.path.join(out, name), name,
                 a.smoke_val if a.smoke else 0, note)

        for f in ("results.csv", "args.yaml"):
            p = os.path.join(stage_dir, f)
            if os.path.isfile(p):
                os.makedirs(os.path.join(out, spec["key"]), exist_ok=True)
                shutil.copy(p, os.path.join(out, spec["key"], f))
        results[spec["key"]] = {
            "best": best, "seconds": secs, "stage_dir": stage_dir,
            "recipe": recipe, "auto_optimizer": auto, "epochs": ran_epochs,
        }
        init = best
        # Bank the stage before starting the next one.
        shutil.copy(best, os.path.join(out, "best_grouped.pt"))
        banked["stage"] = spec["key"]
        src_summary = os.path.join(a.data_dir, "split_summary.json")
        if os.path.isfile(src_summary):
            shutil.copy(src_summary, out)
        write_run_info(complete=False)

    write_run_info(complete=True)

    print("\n" + "=" * 78)
    for k, v in results.items():
        print(f" {k:<8} {v.get('seconds', 0) / 60:8.1f} min   {v['best']}")
    print(f" wrote {out}")
    print("=" * 78)

    if a.smoke and not a.keep:
        # The task asks for these runs to be deleted. Their weights and metrics are
        # meaningless -- 1 epoch on 200 images -- and leaving a 22 MB best_grouped.pt around
        # that looks like a result is worse than having no file at all.
        for d in (project, out):
            shutil.rmtree(d, ignore_errors=True)
            print(f"[smoke] deleted {d}")
    return 0


if __name__ == "__main__":
    # Windows spawns dataloader workers as fresh processes that re-import this module; without
    # this guard each one would start the training again.
    raise SystemExit(main())
