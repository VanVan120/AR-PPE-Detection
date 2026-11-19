import os, json, numpy as np, cv2
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
HERE = os.path.dirname(os.path.abspath(__file__))
ev = json.load(open(os.path.join(HERE, "oracle_relinks.json")))
wrong = [e for e in ev if e["relink"] and not e["correct"] and e["vis"] >= 0.6 and not e["clipped"]]
print(len(wrong), "wrong relinks with vis>=0.6 and unclipped at decision")
gts = {}
def best_crop(seq, gid):
    if seq not in gts:
        g = np.loadtxt(os.path.join(MOT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2); gts[seq] = g[(g[:, 7] == 1) & (g[:, 6] == 1)]
    g = gts[seq]; r = g[g[:, 1] == gid]
    r = r[np.argmax(r[:, 8] * 1000 + r[:, 5])]   # most visible, then tallest
    img = cv2.imread(os.path.join(MOT, seq, "img1", f"{int(r[0]):06d}.jpg"))
    x, y, w, h = r[2:6]
    x1, y1 = int(max(0, x)), int(max(0, y)); x2, y2 = int(min(img.shape[1], x + w)), int(min(img.shape[0], y + h))
    c = img[y1:y2, x1:x2]
    c = cv2.resize(c, (max(1, int(200 * c.shape[1] / max(1, c.shape[0]))), 200))
    return c, r
sel = wrong[::max(1, len(wrong) // 10)][:10]
tiles = []
for e in sel:
    a, ra = best_crop(e["seq"], e["prev_gt"]); b, rb = best_crop(e["seq"], e["gt"])
    canvas = np.full((230, 300, 3), 255, np.uint8)
    strip = np.hstack([a, np.full((200, 8, 3), 255, np.uint8), b])[:, :300]
    canvas[28:228, :strip.shape[1]] = strip
    cv2.putText(canvas, f"{e['seq'][6:8]} {e['source'][:3]} s{e['sim']:.2f} GT{e['prev_gt']}|GT{e['gt']}", (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    tiles.append(canvas)
    print(e["seq"], e["source"], e["sim"], "GT", e["prev_gt"], "(f%d v%.2f h%.0f)" % (ra[0], ra[8], ra[5]), "-> GT", e["gt"], "(f%d v%.2f h%.0f)" % (rb[0], rb[8], rb[5]))
rows = [np.hstack(tiles[i:i + 2]) if len(tiles[i:i + 2]) == 2 else np.hstack([tiles[i], np.full_like(tiles[i], 255)]) for i in range(0, len(tiles), 2)]
out = np.vstack(rows)
cv2.imwrite(os.path.join(HERE, "montage_wrong_relinks.jpg"), out); print(out.shape)
