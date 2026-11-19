"""Revision R1, Part A: two more training seeds of the grouped-split recipe, back to back.

Runs `tools/train_grouped.py` once per seed, sequentially, with everything but the seed exactly
as in the published run (batch 16, workers 2, the recipe read from the original checkpoints);
the only other difference is `--no-plots`, which the revision rules require and which changes
no training arithmetic (it only skips writing images).

    .venv-gpu\\Scripts\\python.exe tools\\revision\\run_partA.py            # seeds 1 then 2

What this wrapper adds, and nothing else:
  * keeps Windows from idle-sleeping while it runs (SetThreadExecutionState);
  * one log file per seed, plus a JSON record of the exact command, start/end times, wall
    time, exit code and, for every interrupted session, what interrupted it;
  * if a session fails, it is continued with `--resume --faithful-resume` (train_grouped.py:
    the resumed stage gets back its early-stopping state and, after a late resume, a reset
    loader, so it stops and augments as an uninterrupted stage would);
  * a watchdog: after a CUDA error the training process has been seen to print its traceback
    and then hang with the GPU memory still held. A session whose log shows a new traceback and
    which has not exited 90 s later, or whose log has not grown for 20 minutes, is killed and
    recorded as such;
  * between attempts: a 3-minute pause, then tools/revision/gpu_check.py; if the SM clock under
    load is not well above idle (median < 1,000 MHz) it waits 10 minutes and checks again.
    At most MAX_ATTEMPTS sessions per seed.
"""
from __future__ import annotations

import argparse
import ctypes
import datetime
import glob
import json
import os
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PART = os.path.join(ROOT, "paper_materials", "revision_r1", "partA")
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
MAX_ATTEMPTS = 12
COOLDOWN_S = 180
HANG_AFTER_TRACEBACK_S = 90
STALL_S = 20 * 60


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def keep_awake(on):
    try:
        flags = ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0)
        return bool(ctypes.windll.kernel32.SetThreadExecutionState(flags))
    except Exception:
        return False


def complete(out_dir):
    try:
        return bool(json.load(open(os.path.join(out_dir, "run_info.json"), encoding="utf-8")).get("complete"))
    except Exception:
        return False


def kill_tree(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def watchdog(proc, log, start_size, verdict):
    """Kill a session that hangs after a traceback or stops writing to its log."""
    last_size, last_growth, tb_seen = start_size, time.time(), None
    while proc.poll() is None:
        time.sleep(20)
        try:
            size = os.path.getsize(log)
        except OSError:
            continue
        if size != last_size:
            last_size, last_growth = size, time.time()
        if tb_seen is None:
            with open(log, "rb") as fh:
                fh.seek(start_size)
                if b"Traceback (most recent call last)" in fh.read():
                    tb_seen = time.time()
        if tb_seen is not None and time.time() - tb_seen > HANG_AFTER_TRACEBACK_S and proc.poll() is None:
            verdict["killed"] = "traceback printed, process still alive after 90 s (hung); killed"
            kill_tree(proc.pid)
            return
        if time.time() - last_growth > STALL_S and proc.poll() is None:
            verdict["killed"] = "log did not grow for 20 min (hung); killed"
            kill_tree(proc.pid)
            return


def gpu_ok(python):
    out = os.path.join(PART, "gpu_check")
    before = set(glob.glob(os.path.join(out, "gpu_check_*.json")))
    p = subprocess.run([python, os.path.join(HERE, "gpu_check.py"), "--out", out],
                       capture_output=True, text=True, cwd=ROOT)
    new = sorted(set(glob.glob(os.path.join(out, "gpu_check_*.json"))) - before)
    if p.returncode != 0 or not new:
        return False, f"gpu_check exit {p.returncode}"
    r = json.load(open(new[-1], encoding="utf-8"))
    med = (r.get("load", {}).get("sm") or {}).get("median", 0)
    return med >= 1000, f"{os.path.basename(new[-1])}: median SM {med} MHz, {r.get('achieved_tflops')} TFLOPS"


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
            log = os.path.join(PART, "logs", f"seed{seed}.log")
            rec = (json.load(open(rec_path, encoding="utf-8")) if os.path.isfile(rec_path)
                   else {"seed": seed, "project": project, "out": out, "sessions": []})
            while not complete(out):
                if len(rec["sessions"]) >= MAX_ATTEMPTS:
                    print(f"[{now()}] seed {seed}: {MAX_ATTEMPTS} sessions used; giving up", flush=True)
                    rc = 2
                    break
                resume = bool(rec["sessions"]) or os.path.isdir(project)
                if resume:
                    time.sleep(COOLDOWN_S)
                    for _ in range(6):
                        ok, msg = gpu_ok(a.python)
                        print(f"[{now()}] GPU check before resuming: {'OK' if ok else 'NOT OK'} ({msg})", flush=True)
                        if ok:
                            break
                        time.sleep(600)
                    if not ok:
                        print(f"[{now()}] seed {seed}: GPU still not healthy; stopping", flush=True)
                        rc = 3
                        break
                cmd = [a.python, os.path.join(ROOT, "tools", "train_grouped.py"),
                       "--seed", str(seed), "--batch", str(a.batch), "--workers", str(a.workers),
                       "--no-plots", "--project", project, "--out", out]
                if resume:
                    cmd += ["--resume", "--faithful-resume"]
                sess = {"start": now(), "command": subprocess.list2cmdline(cmd), "resume": resume,
                        "keep_awake": awake}
                if resume and rec["sessions"]:
                    prev = rec["sessions"][-1]
                    ended = prev.get("end") or "without a recorded end (process killed from outside)"
                    sess["interruption"] = (f"previous session started {prev.get('start')}, ended {ended}, "
                                            f"exit code {prev.get('exit_code')}"
                                            + (f"; {prev['killed']}" if prev.get("killed") else ""))
                rec["sessions"].append(sess)
                json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=2)
                print(f"[{now()}] seed {seed}: {sess['command']}\n    log: {log}", flush=True)
                t0 = time.time()
                env = dict(os.environ, YOLO_AUTOINSTALL="false", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(f"\n===== session start {sess['start']} =====\n{sess['command']}\n")
                    fh.flush()
                    start_size = fh.tell()
                    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
                    verdict = {}
                    w = threading.Thread(target=watchdog, args=(p, log, start_size, verdict), daemon=True)
                    w.start()
                    p.wait()
                sess.update(end=now(), exit_code=p.returncode, wall_seconds=round(time.time() - t0, 1),
                            **({"killed": verdict["killed"]} if verdict.get("killed") else {}))
                json.dump(rec, open(rec_path, "w", encoding="utf-8"), indent=2)
                print(f"[{now()}] seed {seed}: exit {p.returncode} after {sess['wall_seconds'] / 3600:.2f} h"
                      + (f" ({verdict['killed']})" if verdict.get("killed") else ""), flush=True)
            if rc:
                break
            print(f"[{now()}] seed {seed}: complete", flush=True)
    finally:
        keep_awake(False)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
