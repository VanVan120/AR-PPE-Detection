import os, sys, numpy as np, cv2
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "common.py")).read())
OUTD = os.path.dirname(os.path.abspath(__file__))
seq = sys.argv[1]
pairs = [tuple(int(x) for x in p.split(":")) for p in sys.argv[2:]]   # uid:gtA:frameA:gtB:frameB
gt = np.loadtxt(os.path.join(MOT, seq, "gt", "gt.txt"), delimiter=",", ndmin=2)
def crop(f, gid):
    r = gt[(gt[:, 0] == f) & (gt[:, 1] == gid)][0]
    img = cv2.imread(os.path.join(MOT, seq, "img1", f"{f:06d}.jpg"))
    x, y, w, h = r[2:6]
    x1, y1, x2, y2 = int(max(0, x)), int(max(0, y)), int(min(img.shape[1], x + w)), int(min(img.shape[0], y + h))
    c = img[y1:y2, x1:x2]
    c = cv2.resize(c, (int(220 * c.shape[1] / max(1, c.shape[0])) or 1, 220))
    return c, r[8]
rows = []
for uid, ga, fa, gb, fb in pairs:
    a, va = crop(fa, ga); b, vb = crop(fb, gb)
    pad = np.full((220, 10, 3), 255, np.uint8)
    canvas = np.full((250, 480, 3), 255, np.uint8)
    strip = np.hstack([a, pad, b])[:, :480]
    canvas[30:30 + strip.shape[0], :strip.shape[1]] = strip
    cv2.putText(canvas, f"uid{uid}: GT{ga}@{fa} v{va:.2f} | GT{gb}@{fb} v{vb:.2f}", (2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    rows.append(canvas)
out = np.vstack(rows)
p = os.path.join(OUTD, f"montage_{seq}.jpg")
cv2.imwrite(p, out)
print(p, out.shape)
