"""Self-test on a generated fixture: 40 train photos, 20 test photos of which 8 are
planted copies (flipped / rotated / re-compressed / brightened). Expect exactly 8 found."""
import os, json, shutil, subprocess, sys, tempfile
import cv2, numpy as np
rng = np.random.default_rng(0)
def photo():
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = rng.integers(40, 200, 3)
    for _ in range(12):
        p1 = (int(rng.integers(0, 320)), int(rng.integers(0, 240))); p2 = (int(rng.integers(0, 320)), int(rng.integers(0, 240)))
        cv2.rectangle(img, p1, p2, [int(c) for c in rng.integers(0, 255, 3)], -1)
    return cv2.GaussianBlur(img, (5, 5), 0)
d = tempfile.mkdtemp()
for s in ("train", "test"):
    os.makedirs(os.path.join(d, s, "images"))
train = [photo() for _ in range(40)]
for i, im in enumerate(train):
    cv2.imwrite(os.path.join(d, "train/images", f"t{i:03d}.jpg"), im)
planted = 0
for i in range(20):
    if i < 8:
        im = train[i * 3].copy()
        if i % 4 == 0: im = cv2.flip(im, 1)
        if i % 4 == 1: im = np.ascontiguousarray(np.rot90(im, 1))
        if i % 4 == 2: im = cv2.convertScaleAbs(im, alpha=1.08, beta=6)
        if i % 4 == 3: im = cv2.resize(im, (256, 192))
        cv2.imwrite(os.path.join(d, "test/images", f"x{i:03d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 70]); planted += 1
    else:
        cv2.imwrite(os.path.join(d, "test/images", f"x{i:03d}.jpg"), photo())
out = os.path.join(d, "out")
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "leakage_audit.py"), "--dataset-dir", d, "--out", out], check=True, capture_output=True)
s = json.load(open(os.path.join(out, "summary.json")))
print("planted", planted, "found", s["contaminated"], "clean", s["clean"])
print("ALL_LEAKAGE", s["contaminated"] == planted and s["clean"] == 12)
shutil.rmtree(d)
