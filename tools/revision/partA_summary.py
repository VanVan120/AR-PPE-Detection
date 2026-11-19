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
    runs = os.path.join(ROOT, "paper_materials", "retrain")
    d = {0: {"eval": {s: os.path.join(PUB, f"eval_{s}", f"eval_{s}.json") for s in ("stage1", "stage2")},
             "csv": {s: os.path.join(PUB, s, "results.csv") for s in ("stage1", "stage2")},
             "args": {s: os.path.join(runs, "local_runs", f"grouped_s{s[-1]}") for s in ("stage1", "stage2")},
             "log": None, "launcher": None, "run_info": os.path.join(PUB, "run_info.json")}}
    for n in (1, 2):
        base = os.path.join(PART, f"seed{n}")
        d[n] = {"eval": {s: os.path.join(base, f"eval_{s}", f"eval_{s}.json") for s in ("stage1", "stage2")},
                "csv": {s: os.path.join(base, s, "results.csv") for s in ("stage1", "stage2")},
                "args": {s: os.path.join(runs, f"local_runs_seed{n}", f"grouped_s{s[-1]}")
                         for s in ("stage1", "stage2")},
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


def designed_stop(csv_path, patience, epochs):
    """Where ultralytics' EarlyStopping (8.4.75: stop once epoch - best_epoch >= patience, with
    best = first maximum of fitness = mAP50-95, epochs counted from 1) ends an *uninterrupted* run
    whose validation curve is this results.csv. Used for seed 0, whose resumes reset the counter."""
    rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(open(csv_path, encoding="utf-8"))]
    best_f, best_e = -1.0, 0
    for r in rows:
        e, f = int(float(r["epoch"])), float(r["metrics/mAP50-95(B)"])
        if f > best_f:
            best_f, best_e = f, e
        if e - best_e >= patience or e == epochs:
            return {"stop_epoch": e, "best_epoch": best_e, "best_val_map50_95": best_f,
                    "early": e - best_e >= patience and e < epochs}
    return {"stop_epoch": None, "best_epoch": best_e, "best_val_map50_95": best_f, "early": False}


def stage_args(project_stage_dir):
    """epochs / patience / seed / plots from the args.yaml ultralytics wrote for the stage."""
    p = os.path.join(project_stage_dir, "args.yaml")
    if not os.path.isfile(p):
        return {}
    txt = open(p, encoding="utf-8").read()
    get = lambda k: (re.search(rf"^{k}: (.*)$", txt, re.M) or [None, None])[1]
    return {"epochs": int(get("epochs")), "patience": int(get("patience")), "seed": int(get("seed")),
            "plots": get("plots"), "deterministic": get("deterministic")}


def early_stop_line(log_path):
    if not log_path or not os.path.isfile(log_path):
        return []
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    return re.findall(r"Training stopped early[^\n]*Best results observed at epoch \d+", txt)


def session_errors(log_path):
    """Last exception line of each launcher session in the seed's log (None if it ended cleanly)."""
    if not log_path or not os.path.isfile(log_path):
        return []
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    parts = re.split(r"^===== session start (\S+) =====$", txt, flags=re.M)[1:]
    pairs = list(zip(parts[0::2], parts[1::2]))
    res = []
    for start, body in pairs:
        errs = re.findall(r"^([A-Za-z_.]*(?:Error|Exception)): (.*)$", body, flags=re.M)
        res.append({"start": start, "error": f"{errs[-1][0]}: {errs[-1][1].strip()}" if errs else None})
    return res


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
            rec[st]["args"] = stage_args(s["args"][st])
            if rec[st]["training"] and rec[st]["args"]:
                rec[st]["training"]["designed_stop"] = designed_stop(
                    s["csv"][st], rec[st]["args"]["patience"], rec[st]["args"]["epochs"])
        rec["early_stopping_messages"] = early_stop_line(s["log"])
        rec["session_errors"] = session_errors(s["log"])
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
    with open(os.path.join(PART, "partA_tables.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(out))
    print(f"seeds available: {avail}; wrote partA_summary.json, partA_per_seed.csv and partA_tables.md")
    return 0


STAGE_NAME = {"stage1": "stage 1 (`best_grouped_stage1.pt` equivalent)",
              "stage2": "stage 2 (`best_grouped.pt` equivalent, the deployed recipe)"}
CLASSES = ("Helmet", "No-Helmet", "No-Vest", "Person", "Vest")


def _p(x):
    return "" if x is None else f"{100 * x:.1f}"


def _msd(sp):
    if not sp:
        return ""
    return f"{100 * sp['mean']:.1f} ± {'–' if sp['sd'] is None else f'{100 * sp['sd']:.1f}'}"


def _rng(sp):
    return "" if not sp else f"{100 * sp['min']:.1f}–{100 * sp['max']:.1f}"


def markdown(out):
    """The Part A tables for REPORT.md, all numbers in % (P, R, mAP), read from `out`."""
    avail, ps, L = out["seeds_available"], out["per_seed"], []
    L.append("**Training record.** Epochs are 1-based, as in `results.csv`. Training time is the sum of the "
             "epoch times `results.csv` records; it excludes start-up, evaluation and time lost to crashes.\n")
    L.append("| seed | stage | max epochs / patience | epochs run | best epoch (val mAP@50-95) | ended by | "
             "uninterrupted early stopping would end at (best) | training time (h) | sessions in `results.csv` |")
    L.append("|---|---|---|---:|---|---|---|---:|---:|")
    for n in avail:
        for st in ("stage1", "stage2"):
            tr, ar = ps[n][st].get("training"), ps[n][st].get("args") or {}
            if not tr:
                continue
            ds = tr.get("designed_stop") or {}
            ended = "epoch limit" if tr["epochs_run"] == ar.get("epochs") else "early stopping"
            L.append(f"| {n} | {st[-1]} | {ar.get('epochs')} / {ar.get('patience')} | {tr['epochs_run']} | "
                     f"{tr['best_epoch']} ({tr['best_val_map50_95']:.4f}) | {ended} | "
                     f"{ds.get('stop_epoch')} ({ds.get('best_epoch')}) | {tr['epoch_time_h']:.2f} | "
                     f"{tr['sessions_in_csv']} |")
    for st in ("stage2", "stage1"):
        key = f"{st} | full_640"
        L.append(f"\n**Grouped test set, full_640, {STAGE_NAME[st]}.** P and R are ultralytics' values at its "
                 "F1-optimal confidence; mAP@50 and mAP@50-95 as in Table 3.\n")
        L.append("| seed | P | R | mAP@50 | mAP@50-95 | No-Helmet P | No-Helmet R | No-Helmet mAP@50 | "
                 "No-Helmet mAP@50-95 |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n in avail:
            r = ps[n][st]["runs"]["full_640"]
            nh = r["per_class"]["No-Helmet"]
            L.append(f"| {n}{' (published)' if n == 0 else ''} | "
                     + " | ".join(_p(r["overall"][m]) for m in METRICS) + " | "
                     + " | ".join(_p(nh[m]) for m in METRICS) + " |")
        sp = out["spread"][key]
        L.append("| **mean ± SD** | " + " | ".join(_msd(sp["overall"][m]) for m in METRICS) + " | "
                 + " | ".join(_msd(sp["per_class"]["No-Helmet"][m]) for m in METRICS) + " |")
        L.append("| range | " + " | ".join(_rng(sp["overall"][m]) for m in METRICS) + " | "
                 + " | ".join(_rng(sp["per_class"]["No-Helmet"][m]) for m in METRICS) + " |")
    for st in ("stage2", "stage1"):
        sp = out["spread"][f"{st} | full_640"]
        L.append(f"\n**Per class, full_640, {STAGE_NAME[st]}: mean ± SD (range) over seeds "
                 f"{', '.join(map(str, avail))}.**\n")
        L.append("| class | P | R | mAP@50 | mAP@50-95 |")
        L.append("|---|---|---|---|---|")
        rows = [("all", sp["overall"])] + [(c, sp["per_class"][c]) for c in CLASSES if c in sp["per_class"]]
        for c, v in rows:
            L.append(f"| {c} | " + " | ".join(f"{_msd(v[m])} ({_rng(v[m])})" for m in METRICS) + " |")
    L.append("\n**Other test conditions (Table S20): mAP@50 per seed, and mean ± SD.**\n")
    L.append("| stage | condition | all: " + " / ".join(f"seed {n}" for n in avail) + " | all: mean ± SD | "
             "all mAP@50-95: mean ± SD | No-Helmet: " + " / ".join(f"seed {n}" for n in avail)
             + " | No-Helmet: mean ± SD |")
    L.append("|---|---|---|---|---|---|---|")
    for st in ("stage2", "stage1"):
        for cond in CONDITIONS:
            sp = out["spread"][f"{st} | {cond}"]
            al = " / ".join(_p(ps[n][st]["runs"][cond]["overall"]["map50"]) for n in avail)
            nh = " / ".join(_p(ps[n][st]["runs"][cond]["per_class"]["No-Helmet"]["map50"]) for n in avail)
            L.append(f"| {st[-1]} | {cond} | {al} | {_msd(sp['overall']['map50'])} | "
                     f"{_msd(sp['overall']['map50_95'])} | {nh} | {_msd(sp['per_class']['No-Helmet']['map50'])} |")
    sess = [(n, s) for n in avail for s in (ps[n].get("launcher") or {}).get("sessions", [])]
    if sess:
        L.append("\n**Sessions of the new runs** (from `launcher_seed<N>.json`; the error is the last exception "
                 "line the session printed).\n")
        L.append("| seed | session | start | end | exit code | wall time (h) | resumed | error |")
        L.append("|---|---:|---|---|---:|---:|---|---|")
        for n in avail:
            errs = {e["start"]: e["error"] for e in ps[n].get("session_errors", [])}
            for i, s in enumerate((ps[n].get("launcher") or {}).get("sessions", []), 1):
                w = s.get("wall_seconds")
                L.append(f"| {n} | {i} | {s['start'].replace('T', ' ')} | {(s.get('end') or '').replace('T', ' ')} | "
                         f"{s.get('exit_code', '')} | {'' if w is None else f'{w / 3600:.2f}'} | "
                         f"{'yes' if s.get('resume') else 'no'} | {errs.get(s['start']) or ''} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
