"""Revision R1, Part E: timing of the compliance logic at the deployed settings (synthetic).

Replaces the CMOT violation-episode analysis (CMOT frames could not be obtained). Sequences come
from the Dataset D generator (phase5_workid.reid_eval.make_sequence) and are used with their
ground-truth person boxes and ground-truth track ids. No-Helmet evidence is injected as a box
inside the helmet band of a chosen worker during a chosen period; outside that period no
evidence is injected for anyone. The boxes and evidence go through phase2's ComplianceMonitor,
built exactly as SafetyPipeline builds it (phase2/src/pipeline.py:55-58, 87-89) from
phase2/config.yaml: debounce 5, clear 15, containment 0.30, lost_track_buffer 30.
Nothing in phase2/src is changed.

Design, per seed (20 seeds, 100-119):
  * one 900-frame (30 s at the generator's nominal 30 fps) sequence, 4 workers, 'similar' kit;
  * worker 0 violates for 1 s, worker 1 for 3 s, worker 2 for 10 s, worker 3 never (control);
    each episode starts at a random frame in [360, 420], after every worker's scripted
    occlusion and re-entry (frames 315-342), so one continuous track id carries the episode;
  * the evidence for each episode frame is dropped independently with probability
    p = 0, 0.2 or 0.4 (one draw per frame, shared by both rates, so the rates are paired);
  * the monitor runs at 30 fps (every frame) and at ~7 updates/s (every 4th frame, 7.5 Hz).

Measured per episode: alert raised (yes/no), alert delay = time of the frame whose update first
reports the alert minus the episode start; clear delay = time of the first update after the
episode end at which the violation is no longer active, minus the episode end (the first
compliant frame); number of alerts for that track between episode start and its clear (two or
more = split); false alerts = alerts on the control worker or outside any episode window.

    .venv\\Scripts\\python.exe -B tools\\revision\\compliance_timing.py
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "phase2"))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "paper_materials", "revision_r1", "partE")

FPS = 30.0
FRAMES = 900
EPISODES = {0: 1.0, 1: 3.0, 2: 10.0}          # worker -> duration (s); worker 3 is the control
START_RANGE = (360, 420)
MISS = (0.0, 0.2, 0.4)
RATES = {"30fps": 1, "7.5Hz": 4}               # process every k-th frame
SEEDS = list(range(100, 120))
NAMES = ["Helmet", "No-Helmet", "No-Vest", "Person", "Vest"]   # the detectors' class order


def head_box(b):
    x1, y1, x2, y2 = b
    w, h = x2 - x1, y2 - y1
    return [x1 + 0.2 * w, y1, x2 - 0.2 * w, y1 + 0.22 * h]     # inside the helmet band


def build_monitor():
    from src.config import load_config
    from src.compliance import ComplianceMonitor
    cfg = load_config(os.path.join(ROOT, "phase2", "config.yaml"))
    rules_by_id = {}
    for name, rule in cfg.violation_rules.items():          # as pipeline.py:55-58
        for cid in [i for i, n in enumerate(NAMES) if n == name]:
            rules_by_id[cid] = rule
    person = NAMES.index(cfg.person_class)
    mon = ComplianceMonitor(rules_by_id, cfg.debounce_frames, cfg.clear_frames,
                            cfg.association_containment, cfg.lost_track_buffer)
    return mon, person, {"debounce_frames": cfg.debounce_frames, "clear_frames": cfg.clear_frames,
                         "association_containment": cfg.association_containment,
                         "lost_track_buffer": cfg.lost_track_buffer,
                         "rules_by_id": {k: v.class_name for k, v in rules_by_id.items()},
                         "person_class_id": person}


def run_one(seq, seed, p, step):
    import supervision as sv
    mon, PERSON, settings = build_monitor()
    rng = np.random.default_rng(seed + 30_000)
    starts = {w: int(rng.integers(START_RANGE[0], START_RANGE[1] + 1)) for w in EPISODES}
    # one evidence draw per frame per worker, independent of the rate (paired conditions)
    erng = np.random.default_rng(seed + 40_000 + int(round(p * 1000)))
    keep = {w: erng.random(FRAMES) >= p for w in EPISODES}
    ep = {w: (starts[w], starts[w] + int(round(d * FPS))) for w, d in EPISODES.items()}  # [s, e)
    alerts, active_of = [], {}           # alerts: (frame, tracker_id, worker)
    tid_worker = {}
    for f in range(0, FRAMES, step):
        P = [t for t in seq.tracks[f] if t.worker >= 0]
        for t in P:
            tid_worker[t.track_id] = t.worker
        persons = sv.Detections(xyxy=np.array([t.box for t in P], np.float32).reshape(-1, 4),
                                confidence=np.full(len(P), 0.9, np.float32),
                                class_id=np.full(len(P), PERSON, int),
                                tracker_id=np.array([t.track_id for t in P], int))
        V = [head_box(t.box) for t in P if t.worker in ep and ep[t.worker][0] <= f < ep[t.worker][1]
             and keep[t.worker][f]]
        viol = sv.Detections(xyxy=np.array(V, np.float32).reshape(-1, 4),
                             confidence=np.full(len(V), 0.9, np.float32),
                             class_id=np.full(len(V), NAMES.index("No-Helmet"), int))
        fc = mon.update(persons, viol)
        for ev in fc.new_events:
            if ev.class_name == "No-Helmet":
                alerts.append((f, int(ev.person_id), tid_worker.get(int(ev.person_id), -1)))
        for ps in fc.persons:
            active_of.setdefault(int(ps.tracker_id), {})[f] = any(a.class_name == "No-Helmet" for a in ps.active)
    rows = []
    used = set()
    for w, (s, e) in ep.items():
        tids = {t.track_id for t in seq.tracks[s] if t.worker == w}
        assert len(tids) == 1, (w, s, tids)
        tid = tids.pop()
        # first processed frame at or after the episode end where the track is no longer active
        clear_f = next((f for f in sorted(active_of.get(tid, {})) if f >= e and not active_of[tid][f]), None)
        window_end = clear_f if clear_f is not None else FRAMES
        ep_alerts = [a for a in alerts if a[1] == tid and s <= a[0] < window_end]
        used.update(ep_alerts)
        rows.append({"seed": seed, "miss_p": p, "rate": "30fps" if step == 1 else "7.5Hz", "worker": w,
                     "duration_s": EPISODES[w], "start_frame": s, "end_frame": e,
                     "evidence_frames_seen": int(sum(keep[w][f] for f in range(s, e) if (f % step) == 0)),
                     "updates_in_episode": len(range(s + (-s) % step, e, step)),
                     "alerted": bool(ep_alerts),
                     "alert_delay_s": (ep_alerts[0][0] - s) / FPS if ep_alerts else None,
                     "n_alerts": len(ep_alerts), "split": len(ep_alerts) >= 2,
                     # a clear only exists after an alert; un-alerted episodes have none
                     "clear_delay_s": (clear_f - e) / FPS if (clear_f is not None and ep_alerts) else None})
    false_alerts = [a for a in alerts if a not in used]
    return rows, false_alerts, settings


def summarise(v):
    v = [x for x in v if x is not None]
    if not v:
        return None
    return {"n": len(v), "median": statistics.median(v), "min": min(v), "max": max(v),
            "mean": statistics.mean(v)}


def main() -> int:
    import supervision as sv
    from phase5_workid.reid_eval import make_sequence
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    rows, false_alerts, settings = [], [], None
    for seed in SEEDS:
        seq = make_sequence(n_workers=4, frames=FRAMES, scenario="similar", gap=12, seed=seed)
        for p in MISS:
            for rate, step in RATES.items():
                r, fa, settings = run_one(seq, seed, p, step)
                rows += r
                false_alerts += [{"seed": seed, "miss_p": p, "rate": rate, "frame": a[0],
                                  "tracker_id": a[1], "worker": a[2]} for a in fa]
    with open(os.path.join(OUT, "episodes.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    summary = []
    for rate in RATES:
        for p in MISS:
            for d in sorted(set(EPISODES.values())):
                sel = [r for r in rows if r["rate"] == rate and r["miss_p"] == p and r["duration_s"] == d]
                summary.append({
                    "rate": rate, "miss_p": p, "duration_s": d, "episodes": len(sel),
                    "alerted_share": sum(r["alerted"] for r in sel) / len(sel),
                    "alert_delay_s": summarise([r["alert_delay_s"] for r in sel]),
                    "clear_delay_s": summarise([r["clear_delay_s"] for r in sel]),
                    "split_episodes": sum(r["split"] for r in sel),
                    "never_cleared": sum(r["clear_delay_s"] is None for r in sel),
                    "false_alerts": sum(1 for a in false_alerts if a["rate"] == rate and a["miss_p"] == p)})
    meta = {"command": " ".join([sys.executable] + sys.argv), "seconds": round(time.time() - t0, 1),
            "settings": settings, "frames": FRAMES, "fps_nominal": FPS, "episodes_s": EPISODES,
            "start_range": START_RANGE, "miss_p": MISS, "rates": RATES, "seeds": [SEEDS[0], SEEDS[-1]],
            "scenario": "similar", "workers": 4, "gap": 12,
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                         "supervision": sv.__version__, "platform": platform.platform()},
            "worker_minutes_control": len(SEEDS) * FRAMES / FPS / 60,
            "false_alerts": false_alerts,
            "finished": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": summary}, fh, indent=2)
    for s in summary:
        ad, cd = s["alert_delay_s"], s["clear_delay_s"]
        print(f"{s['rate']:<6} p={s['miss_p']:.1f} {s['duration_s']:>4.0f}s alerted {s['alerted_share']:5.0%} "
              f"alert {ad['median'] if ad else float('nan'):.3f} [{ad['min'] if ad else float('nan'):.3f}-{ad['max'] if ad else float('nan'):.3f}] "
              f"clear {cd['median'] if cd else float('nan'):.3f} [{cd['min'] if cd else float('nan'):.3f}-{cd['max'] if cd else float('nan'):.3f}] "
              f"split {s['split_episodes']} false {s['false_alerts']}")
    print(f"done in {meta['seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
