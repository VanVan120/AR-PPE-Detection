import configparser, hashlib, json, os, csv
ROOT = r"C:\Users\user\Desktop\SummerInternProject"
SRC = os.path.join(ROOT, "data", "mot17", "MOT17", "train")
TE = os.path.join(ROOT, "data", "mot17", "trackeval", "gt", "mot_challenge")
SEQS = ["MOT17-02-FRCNN", "MOT17-04-FRCNN", "MOT17-05-FRCNN", "MOT17-09-FRCNN",
        "MOT17-10-FRCNN", "MOT17-11-FRCNN", "MOT17-13-FRCNN"]
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
def ini(p):
    c = configparser.ConfigParser(); c.optionxform = str; c.read(p); return dict(c["Sequence"])
fm = json.load(open(os.path.join(TE, "MOT17sub7-train_frame_map.json")))
tot_frames = 0; tot_ids = 0; tot_frames7 = 0
for s in SEQS:
    src_gt = os.path.join(SRC, s, "gt", "gt.txt")
    n_gt = os.path.join(TE, "MOT17-train", s, "gt", "gt.txt")
    same_gt = sha(src_gt) == sha(n_gt)
    same_ini = sha(os.path.join(SRC, s, "seqinfo.ini")) == sha(os.path.join(TE, "MOT17-train", s, "seqinfo.ini"))
    info = ini(os.path.join(SRC, s, "seqinfo.ini"))
    fps, L = float(info["frameRate"]), int(info["seqLength"])
    nimg = len([f for f in os.listdir(os.path.join(SRC, s, "img1")) if f.endswith(".jpg")])
    k = max(1, round(fps / 7.0))
    kept = list(range(1, L + 1, k))
    sub = ini(os.path.join(TE, "MOT17sub7-train", s, "seqinfo.ini"))
    rows = [r for r in csv.reader(open(src_gt))]
    # expected sub gt
    new_of = {f: i + 1 for i, f in enumerate(kept)}
    exp = [[str(new_of[int(r[0])])] + r[1:] for r in rows if int(r[0]) in new_of]
    got = [r for r in csv.reader(open(os.path.join(TE, "MOT17sub7-train", s, "gt", "gt.txt")))]
    # frames in source gt
    gt_frames = sorted({int(r[0]) for r in rows})
    max_new = max(int(r[0]) for r in got)
    ped_ids = {int(r[1]) for r in rows if int(r[7]) == 1 and int(r[6]) == 1}
    ped_ids7 = {int(r[1]) for r in rows if int(r[7]) == 1 and int(r[6]) == 1 and int(r[0]) in new_of}
    cls1_flag0 = sum(1 for r in rows if int(r[7]) == 1 and int(r[6]) == 0)
    other_flag1 = sum(1 for r in rows if int(r[7]) != 1 and int(r[6]) == 1)
    tot_frames += L; tot_ids += len(ped_ids); tot_frames7 += len(kept)
    print(s, "native gt same:", same_gt, "ini same:", same_ini, "imgs", nimg, "L", L, "fps", fps,
          "| step", k, "fm step", fm[s]["step"], "fm kept ok", fm[s]["kept_frames"] == kept,
          "| sub seqLength", sub["seqLength"], "exp", len(kept), "frameRate", sub["frameRate"], "exp", f"{fps/k:g}",
          "| sub gt rows", len(got), "exp", len(exp), "equal", got == exp, "max frame", max_new,
          "| gt frame range", gt_frames[0], gt_frames[-1], "n", len(gt_frames),
          "| ped ids", len(ped_ids), "ped ids @7Hz", len(ped_ids7), "| cls1 flag0 rows", cls1_flag0, "non-cls1 flag1 rows", other_flag1,
          "| other ini keys same:", {kk: (sub[kk] == info[kk]) for kk in ("name","imDir","imWidth","imHeight","imExt")})
print("total frames", tot_frames, "total ped ids", tot_ids, "frames@7Hz", tot_frames7)
