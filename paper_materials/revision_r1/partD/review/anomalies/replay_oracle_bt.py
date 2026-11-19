"""Re-run oracle bytetrack_phase2 native with the current mot17_track.py logic (imported, run in
scratch via --smoke-dir semantics) and compare with the stored 18:48 output."""
import os, sys, filecmp, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
spec = importlib.util.spec_from_file_location("mot17_track", os.path.join(ROOT, "tools", "revision", "mot17_track.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
sys.path.insert(0, ROOT)
m.TRACKERS = os.path.join(HERE, "replay", "trackers"); m.RUNS = os.path.join(HERE, "replay", "runs")
for rate in ("native", "7hz"):
    m.run("bytetrack_phase2", "oracle", rate, "pipeline", 1)
    split = "MOT17-train" if rate == "native" else "MOT17sub7-train"
    for s in m.SEQS:
        a = os.path.join(m.TRACKERS, split, "oracle__bytetrack_phase2", "data", f"{s}.txt")
        b = os.path.join(ROOT, "data", "revision_r1_cache", "partD", "trackers", "mot_challenge", split, "oracle__bytetrack_phase2", "data", f"{s}.txt")
        print(rate, s, "identical" if filecmp.cmp(a, b, shallow=False) else "DIFFERENT")
