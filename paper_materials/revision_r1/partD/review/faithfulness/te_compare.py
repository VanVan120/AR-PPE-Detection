"""Score stored Part D Work ID runs vs the same runs with analyze.py's clip-time origin (f/fps).
GT is read from the repo (read-only); trackers and outputs live in this scratch folder."""
import os
import shutil

import numpy as np

for _a, _b in (("float", float), ("int", int), ("bool", bool)):
    if not hasattr(np, _a):
        setattr(np, _a, _b)
import trackeval  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
GT_TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
STORED = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "trackers", "mot_challenge", "MOT17-train")
TT = os.path.join(HERE, "te_trackers", "MOT17-train")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
RUNS = ["coco__workid_phase2", "coco__workid_phase2_offline", "coco__workid_phase2__feedall",
        "coco__workid_phase2__feedall_offline", "grouped__workid_phase2__feedall",
        "grouped__workid_phase2__feedall_offline"]


def uidonly(src, dst):
    os.makedirs(dst, exist_ok=True)
    for f in os.listdir(src):
        with open(os.path.join(src, f), encoding="utf-8") as fi, open(os.path.join(dst, f), "w", encoding="utf-8") as fo:
            fo.writelines(l for l in fi if int(l.split(",", 2)[1]) < 1_000_000)


names = []
for run in RUNS:
    nowf = os.path.join(TT, run + "__nowf", "data")
    for s in SEQS:                                   # sequences not re-run are identical under both origins
        if not os.path.isfile(os.path.join(nowf, f"{s}.txt")):
            shutil.copyfile(os.path.join(STORED, run, "data", f"{s}.txt"), os.path.join(nowf, f"{s}.txt"))
    st = os.path.join(TT, run + "__stored", "data")
    os.makedirs(st, exist_ok=True)
    for s in SEQS:
        shutil.copyfile(os.path.join(STORED, run, "data", f"{s}.txt"), os.path.join(st, f"{s}.txt"))
    for v in ("__stored", "__nowf"):
        uidonly(os.path.join(TT, run + v, "data"), os.path.join(TT, run + "_uidonly" + v, "data"))
        names += [run + v, run + "_uidonly" + v]

ec = trackeval.Evaluator.get_default_eval_config()
ec.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "TIME_PROGRESS": False,
           "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "BREAK_ON_ERROR": True,
           "DISPLAY_LESS_PROGRESS": True})
dc = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
dc.update({"GT_FOLDER": GT_TE, "TRACKERS_FOLDER": os.path.join(HERE, "te_trackers"),
           "OUTPUT_FOLDER": os.path.join(HERE, "te_out"), "TRACKERS_TO_EVAL": names,
           "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train", "DO_PREPROC": True,
           "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "", "PRINT_CONFIG": False})
q = {"PRINT_CONFIG": False}
res, _ = trackeval.Evaluator(ec).evaluate([trackeval.datasets.MotChallenge2DBox(dc)],
                                          [trackeval.metrics.HOTA(q), trackeval.metrics.CLEAR({**q, "THRESHOLD": 0.5}),
                                           trackeval.metrics.Identity({**q, "THRESHOLD": 0.5})])
R = res["MotChallenge2DBox"]


def row(r):
    p = r["pedestrian"]
    return (100 * float(np.mean(p["HOTA"]["HOTA"])), 100 * float(np.mean(p["HOTA"]["AssA"])),
            100 * float(p["Identity"]["IDF1"]), int(p["CLEAR"]["IDSW"]), 100 * float(p["CLEAR"]["MOTA"]),
            int(p["Count"]["IDs"]))


print(f"{'run':48s} {'seq':15s} | {'stored: HOTA AssA IDF1 IDSW MOTA IDs':40s} | analyze.py origin (f/fps)")
for run in RUNS:
    for r in (run, run.replace("_offline", "") + "_uidonly" + ("_offline" if "_offline" in run else "")):
        pass
    for base in (run, run + "_uidonly"):
        a, b = R[base + "__stored"], R[base + "__nowf"]
        for s in ("COMBINED_SEQ", "MOT17-05-FRCNN", "MOT17-11-FRCNN"):
            ra, rb = row(a[s]), row(b[s])
            if s != "COMBINED_SEQ" and ra == rb:
                continue
            fa = " ".join(f"{v:6.2f}" if isinstance(v, float) else f"{v:5d}" for v in ra)
            fb = " ".join(f"{v:6.2f}" if isinstance(v, float) else f"{v:5d}" for v in rb)
            print(f"{base:48s} {s:15s} | {fa} | {fb}")
