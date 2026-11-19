"""Revision R1, Part A: per-seed table and spread over seeds 0, 1 and 2.

Seed 0 is the published run (results/grouped_retrain/); seeds 1 and 2 are the new runs written by
tools/revision/run_partA.py (paper_materials/revision_r1/partA/seed<N>/). Every number is read
from the evaluation JSON that tools/eval_grouped.py wrote, and from each stage's results.csv.
No metric is recomputed here except the mean, SD and range across seeds.

    .venv\\Scripts\\python.exe tools\\revision\\partA_summary.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
PART = os.path.join(ROOT, "paper_materials", "revision_r1", "partA")
PUB = os.path.join(ROOT, "results", "grouped_retrain")
METRICS = ("precision", "recall", "map50", "map50_95")
CONDITIONS = ("full_640", "one_per_source_640", "full_480", "full_320")


def seed_dirs():
    d = {0: {"eval": {s: os.path.join(PUB, f"eval_{s}", f"eval_{s}.json") for s in ("stage1", "stage2")},
             "csv": {s: os.path.join(PUB, s, "results.csv") for s in ("stage1", "stage2")},
             "log": None, "launcher": None, "run_info": os.path.join(PUB, "run_info.json")}}
    for n in (1, 2):
        base = os.path.join(PART, f"seed{n}")
        d[n] = {"eval": {s: os.path.join(base, f"eval_{s}", f"eval_{s}.json") for s in ("stage1", "stage2")},
                "csv": {s: os.path.join(base, s, "results.csv") for s in ("stage1", "stage2")},
                "log": os.path.join(PART, "logs", f"seed{n}.log"),
                "launcher": os.path.join(PART, f"launcher_seed{n}.json"),
                "run_info": os.path.join(base, "run_info.json")}
    return d


def training_record(csv_path):
    """Epochs run, best epoch by ultralytics' 8.4.75 fitness (mAP50-95; first maximum, because an
    epoch only counts as better if fitness > best_fitness), and summed epoch time."""
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    rows = [{k.strip(): v for k, v in r.items()} for r in rows]
    m = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    best = max(range(len(m)), key=lambda i: (m[i], -i))
    t = [float(r["time"]) for r in rows]
    # `time` is cumulative within a session and restarts at a resume; sum the per-session maxima.
    sessions, cur = [], 0.0
    for x in t:
        if x < cur:
            sessions.append(cur)
        cur = x
    sessions.append(cur)
    return {"epochs_run": len(rows), "best_epoch": int(rows[best]["epoch"]),
            "best_val_map50_95": m[best], "epoch_time_h": sum(sessions) / 3600,
            "sessions_in_csv": len(sessions)}


def early_stop_line(log_path):
    if not log_path or not os.path.isfile(log_path):
        return []
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    return re.findall(r"Training stopped early[^\n]*Best results observed at epoch \d+", txt)


def spread(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"n": len(v), "mean": statistics.mean(v), "sd": statistics.stdev(v) if len(v) > 1 else None,
            "min": min(v), "max": max(v), "range": max(v) - min(v)}


def main() -> int:
    seeds = seed_dirs()
    have = {n: all(os.path.isfile(p) for p in s["eval"].values()) for n, s in seeds.items()}
    out = {"seeds_available": [n for n, ok in have.items() if ok], "per_seed": {}, "spread": {}}
    for n, s in seeds.items():
        if not have[n]:
            continue
        rec = {}
        for st in ("stage1", "stage2"):
            ev = json.load(open(s["eval"][st], encoding="utf-8"))
            rec[st] = {"weights": ev["weights"], "runs": {r["tag"]: {
                "overall": {m: r["overall"][m] for m in METRICS},
                "per_class": {c: {m: v[m] for m in METRICS} for c, v in r["per_class"].items()},
                "count_check": r["count_check_status"]} for r in ev["runs"]},
                "training": training_record(s["csv"][st]) if os.path.isfile(s["csv"][st]) else None}
        rec["early_stopping_messages"] = early_stop_line(s["log"])
        if s["launcher"] and os.path.isfile(s["launcher"]):
            rec["launcher"] = json.load(open(s["launcher"], encoding="utf-8"))
        if os.path.isfile(s["run_info"]):
            ri = json.load(open(s["run_info"], encoding="utf-8"))
            rec["run_info"] = {k: ri.get(k) for k in ("started", "ended", "hours", "complete",
                                                       "seed", "plots", "batch", "workers_override")}
            rec["run_info"]["stage_seconds"] = {k: v.get("seconds") for k, v in ri.get("stages", {}).items()}
        out["per_seed"][n] = rec

    avail = out["seeds_available"]
    for st in ("stage1", "stage2"):
        for cond in CONDITIONS:
            key = f"{st} | {cond}"
            ent = {"overall": {}, "per_class": {}}
            for m in METRICS:
                ent["overall"][m] = spread([out["per_seed"][n][st]["runs"][cond]["overall"][m] for n in avail])
            classes = sorted(out["per_seed"][avail[0]][st]["runs"][cond]["per_class"]) if avail else []
            for c in classes:
                ent["per_class"][c] = {m: spread([out["per_seed"][n][st]["runs"][cond]["per_class"][c][m]
                                                  for n in avail]) for m in METRICS}
            out["spread"][key] = ent

    with open(os.path.join(PART, "partA_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    with open(os.path.join(PART, "partA_per_seed.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["seed", "stage", "condition", "class", *METRICS])
        for n in avail:
            for st in ("stage1", "stage2"):
                for cond in CONDITIONS:
                    r = out["per_seed"][n][st]["runs"][cond]
                    wr.writerow([n, st, cond, "all", *[r["overall"][m] for m in METRICS]])
                    for c, v in r["per_class"].items():
                        wr.writerow([n, st, cond, c, *[v[m] for m in METRICS]])
    print(f"seeds available: {avail}; wrote partA_summary.json and partA_per_seed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
