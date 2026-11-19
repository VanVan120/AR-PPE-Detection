"""Revision R1, Part A: two more training seeds of the grouped-split recipe, back to back.

Runs `tools/train_grouped.py` once per seed, sequentially, with everything but the seed exactly
as in the published run (batch 16, workers 2, the recipe read from the original checkpoints);
the only other difference is `--no-plots`, which the revision rules require and which changes
no training arithmetic (it only skips writing images).

    .venv-gpu\\Scripts\\python.exe tools\\revision\\run_partA.py            # seeds 1 then 2

What this wrapper adds, and nothing else:
  * keeps Windows from idle-sleeping while it runs (SetThreadExecutionState), because an
    interrupted run cannot early-stop as designed (ultralytics does not persist early-stopping
    state; see results/grouped_retrain/READ_FIRST.md section 3);
  * one log file per seed, plus a JSON record of the exact command, start/end times, wall
    time and exit code of every session;
  * if it is relaunched after an interruption, a seed whose run is not complete is continued
    with `--resume`, and the interruption is recorded rather than hidden.

Outputs: training runs (weights) in paper_materials/retrain/local_runs_seed<N>/; evaluation
JSON/MD, run_info.json, results.csv and args.yaml in paper_materials/revision_r1/partA/seed<N>/.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PART = os.path.join(ROOT, "paper_materials", "revision_r1", "partA")

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def keep_awake(on):
    """Ask Windows not to idle-sleep while this process runs. Lid close still sleeps."""
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:                                            # not Windows
        return False


def complete(out_dir):
    p = os.path.join(out_dir, "run_info.json")
    try:
        return bool(json.load(open(p, encoding="utf-8")).get("complete"))
    except Exception:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    os.makedirs(os.path.join(PART, "logs"), exist_ok=True)
    awake = keep_awake(True)
    print(f"[{now()}] keep-awake requested: {awake}", flush=True)
    rc = 0
    try:
        for seed in a.seeds:
            project = os.path.join(ROOT, "paper_materials", "retrain", f"local_runs_seed{seed}")
            out = os.path.join(PART, f"seed{seed}")
            rec_path = os.path.join(PART, f"launcher_seed{seed}.json")
            rec = (json.load(open(rec_path, encoding="utf-8")) if os.path.isfile(rec_path)
                   else {"seed": seed, "project": project, "out": out, "sessions": []})
            if complete(out):
                print(f"[{now()}] seed {seed}: already complete, skipping", flush=True)
                continue
            # Started before but not complete: this launch is a resume after an interruption.
            resume = bool(rec["sessions"]) or os.path.isdir(project)
            cmd = [a.python, os.path.join(ROOT, "tools", "train_grouped.py"),
                   "--seed", str(seed), "--batch", str(a.batch), "--workers", str(a.workers),
                   "--no-plots", "--project", project, "--out", out]
            if resume:
                cmd.append("--resume")
            sess = {"start": now(), "command": subprocess.list2cmdline(cmd), "resume": resume,
                    "keep_awake": awake}
            if resume:
                prev = rec["sessions"][-1] if rec["sessions"] else {}
                ended = prev.get("end") or ("without a recorded end (the process was killed: "
                                            "reboot, crash or power loss)")
                sess["interruption"] = (f"previous session started {prev.get('start')}, ended "
                                        f"{ended}, exit code {prev.get('exit_code')}")
            rec["sessions"].append(sess)
            json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=2)

            log = os.path.join(PART, "logs", f"seed{seed}.log")
            print(f"[{now()}] seed {seed}: {sess['command']}\n    log: {log}", flush=True)
            t0 = time.time()
            env = dict(os.environ, YOLO_AUTOINSTALL="false", PYTHONUNBUFFERED="1",
                       PYTHONIOENCODING="utf-8")
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(f"\n===== session start {sess['start']} =====\n{sess['command']}\n")
                fh.flush()
                p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            sess.update(end=now(), exit_code=p.returncode,
                        wall_seconds=round(time.time() - t0, 1))
            json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=2)
            print(f"[{now()}] seed {seed}: exit {p.returncode} after "
                  f"{sess['wall_seconds'] / 3600:.2f} h", flush=True)
            if p.returncode != 0:
                rc = p.returncode
                print(f"[{now()}] stopping: seed {seed} failed; later seeds not started",
                      flush=True)
                break
    finally:
        keep_awake(False)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
