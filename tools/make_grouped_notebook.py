"""Generate `kaggle_ppe_grouped.ipynb` — the retraining notebook the author runs on Kaggle.

The notebook is generated rather than hand-written for one reason: it has to carry the
*exact* training arguments recorded in the two original checkpoints, and transcribing ~108
arguments twice by hand is a good way to introduce a silent difference that makes the
retrain not comparable. This reads them out of `tools/ckpt_train_record.py`'s output and
embeds them literally.

    python tools/make_grouped_notebook.py --sha256 <assignment sha256> \
        --out kaggle_ppe_grouped.ipynb

`--sha256` is the value `group_split.py` printed when the CSV was built; the notebook
asserts the split it materialises on Kaggle hashes to the same thing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

# Recorded but not a training argument: `model` is set by the YOLO(...) constructor, and
# `save_dir` is a computed absolute path from the ORIGINAL run that would silently
# override the project/name we pass.
DROP = ("model", "save_dir")


def load_args(path):
    d = yaml.safe_load(open(path, encoding="utf-8"))
    for k in DROP:
        d.pop(k, None)
    return d


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def py_dict(d):
    """A PYTHON dict literal, not JSON.

    json.dumps would emit `false` / `true` / `null`, which parse as Python identifiers and
    then raise NameError when the cell runs — a bug that survives a syntax check.
    """
    body = ",\n".join(f"    {k!r}: {d[k]!r}" for k in sorted(d))
    return "{\n" + body + ",\n}"


def build(stage1, stage2, sha256, ultra_pin, repo, branch, seed_note):
    s1 = py_dict(stage1)
    s2 = py_dict(stage2)
    cells = []

    cells.append(md(f"""# PPE Detection — retrain on a **source-grouped** split

The published PPE numbers were measured on the Roboflow export's own split, and that split
leaks: **77.95%** of its test images have a near-duplicate in train/valid (ncc 0.90), and
**98.57%** of test images share a Roboflow source stem with train/valid. Its 41,730 images
come from ~5,800 source photographs — the split was made per *image* after augmentation,
not per *photograph*. On the pixel-deduplicated subset the detector scored below all five
size-matched control subsets, so the published numbers are inflated.

This notebook retrains **the same model with the same recipe** on a split whose unit is
the source photograph, so the test split shares no photograph — and no augmented or
re-exported copy of one — with training. The result is the honest number for the paper.

Nothing here re-derives the split. `tools/splits/ppe_grouped_split.csv` in the repo *is*
the split; this notebook applies it and asserts its sha256.

### Settings to choose before running
1. **Accelerator → GPU T4 x2**  (stage 1 was trained on two GPUs: `device='0,1'`)
2. **Internet → On**  (needed for `pip install` and the Roboflow download)
3. **Add-ons → Secrets** → add `ROBOFLOW_API_KEY` and attach it to this notebook
4. Then **Save Version → Save & Run All (Commit)** and close the browser — Kaggle runs it
   offline for up to 12 h and keeps the output.

### What you get
`/kaggle/working/out.zip`, containing `best_grouped.pt`, each stage's `results.csv` and
`args.yaml`, every evaluation JSON/MD, and `split_summary.json`.

> **Two stages, exactly as the original.** Stage 1 is the from-scratch run recorded in
> `best.pt` (50 epochs requested, early stopping fired at 31, batch 96, `optimizer='auto'`,
> `lr0=0.01`). Stage 2 is the refinement recorded in `best_refined.pt` (20 epochs, batch 48,
> `optimizer='SGD'`, `lr0=0.001`). Both argument sets are embedded below verbatim; only
> `data`, `project` and `name` are changed, plus stage 2's starting weights, which by
> definition point at stage 1's output rather than the original Kaggle dataset path.
"""))

    cells.append(md("## 1. GPU + CPU, and the clock\n\nKaggle stops a committed run at 12 h. "
                    "`T_START` is used later to skip stage 2 if stage 1 ran long.\n"))
    cells.append(code("""import os, time, datetime, torch

T_START = time.time()
print('start (UTC):', datetime.datetime.utcfromtimestamp(T_START).isoformat(timespec='seconds'))
print('CPUs:', os.cpu_count(), '| CUDA:', torch.cuda.is_available(),
      '| GPUs:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(' ', torch.cuda.get_device_name(i))
if not torch.cuda.is_available():
    print('NO GPU -- Settings > Accelerator > GPU T4 x2, then restart.')
"""))

    cells.append(md(f"""## 2. Install, with ultralytics pinned

Pinned to **{ultra_pin}**, the version recorded inside `best_refined.pt`. A different
ultralytics version changes augmentation defaults and the loss, which would make the
retrained number incomparable with the published one. No `-U`: upgrading torch can pull a
build that drops support for Kaggle's GPU.
"""))
    cells.append(code(f"""!pip install -q "ultralytics=={ultra_pin}" roboflow
import torch, ultralytics, roboflow
print('torch', torch.__version__, '| ultralytics', ultralytics.__version__,
      '| roboflow', roboflow.__version__)
assert ultralytics.__version__ == '{ultra_pin}', (
    f'expected ultralytics {ultra_pin}, got ' + ultralytics.__version__ +
    ' -- the retrain would not be comparable with the published run')
"""))

    cells.append(md("## 3. Roboflow API key\n\nFrom **Add-ons → Secrets** "
                    "(`ROBOFLOW_API_KEY`), same as the existing notebooks. The key is never "
                    "printed.\n"))
    cells.append(code("""import os
ROBOFLOW_API_KEY = ''
try:
    from kaggle_secrets import UserSecretsClient
    ROBOFLOW_API_KEY = UserSecretsClient().get_secret('ROBOFLOW_API_KEY')
except Exception:
    pass
if not ROBOFLOW_API_KEY:
    ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '')
assert ROBOFLOW_API_KEY, (
    'No Roboflow API key found. Add a Kaggle Secret named ROBOFLOW_API_KEY '
    '(Add-ons -> Secrets, then attach it to this notebook), or set the '
    'ROBOFLOW_API_KEY environment variable. Get your key at '
    'https://app.roboflow.com (Settings -> API).'
)
print('Roboflow key loaded:', bool(ROBOFLOW_API_KEY))
"""))

    cells.append(md("## 4. Download the dataset (~2.8 GB)\n\nThe same export the paper used: "
                    "`segp-fcn6m/ppe-yezzu-fwbjo` version 1, `yolov8` format. The split "
                    "inside it is the leaky one and is about to be replaced.\n"))
    cells.append(code("""import shutil, glob
DEST = '/kaggle/working/ppe'
shutil.rmtree(DEST, ignore_errors=True)   # roboflow skips the download if the dir exists

from roboflow import Roboflow
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace('segp-fcn6m').project('ppe-yezzu-fwbjo')
dataset = project.version(1).download('yolov8', location=DEST)

ORIG = os.path.dirname(glob.glob('/kaggle/working/ppe/**/data.yaml', recursive=True)[0])
print('dataset at:', ORIG, '|', sorted(os.listdir(ORIG)))
n = sum(len(os.listdir(os.path.join(ORIG, s, 'images'))) for s in ('train', 'valid', 'test'))
print('images:', n)
assert n == 41730, f'expected 41730 images, found {n} -- the export has changed'
"""))

    cells.append(md(f"""## 5. Clone the repo (branch `{branch}`)

The split definition and the tooling live in the repo, so the run is reproducible from a
commit rather than from cells pasted into Kaggle.
"""))
    cells.append(code(f"""!rm -rf /kaggle/working/repo
!git clone -q --branch {branch} {repo} /kaggle/working/repo
REPO = '/kaggle/working/repo'
import subprocess
COMMIT = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'],
                        capture_output=True, text=True).stdout.strip()
print('branch  : {branch}')
print('commit  :', COMMIT)
print(subprocess.run(['git', '-C', REPO, 'log', '-1', '--oneline'],
                     capture_output=True, text=True).stdout.strip())
CSV = os.path.join(REPO, 'tools', 'splits', 'ppe_grouped_split.csv')
assert os.path.isfile(CSV), CSV
print('split csv:', CSV, '|', sum(1 for _ in open(CSV)) - 1, 'rows')
"""))

    cells.append(md(f"""## 6. Apply the grouped split

`group_split.py --apply` re-derives nothing: it reads the CSV, hard-links every image and
label into `train/valid/test`, and writes `data.yaml`, `split_summary.json` and
`test_one_per_source.txt`. It fails loudly if any file in the CSV is missing from the
download, or if any downloaded image is not in the CSV.

The sha256 below was computed when the CSV was built locally. If it does not match, the
CSV and the dataset are not the pair this notebook was written for — stop rather than
train on a split nobody has audited.
"""))
    cells.append(code(f"""EXPECTED_SHA256 = '{sha256}'
SEED_NOTE = '{seed_note}'

GROUPED = '/kaggle/working/ppe_grouped'
!rm -rf {{GROUPED}}
!python {{REPO}}/tools/group_split.py --apply {{CSV}} --dataset-dir {{ORIG}} --out {{GROUPED}} --seed-note "{seed_note}"

import json
summary = json.load(open(os.path.join(GROUPED, 'split_summary.json')))
print(json.dumps({{k: v['images'] for k, v in summary['splits'].items()}}, indent=2))
print('sha256 :', summary['assignment_sha256'])
assert summary['assignment_sha256'] == EXPECTED_SHA256, (
    'split sha256 mismatch!\\n  expected ' + EXPECTED_SHA256 +
    '\\n  got      ' + summary['assignment_sha256'] +
    '\\nThe CSV does not describe this dataset. Do not train.')
print('\\nsplit verified against the locally built definition.')
GROUPED_YAML = os.path.join(GROUPED, 'data.yaml')
for s in ('train', 'valid', 'test'):
    d = summary['splits'][s]
    print(f"  {{s:<6}} {{d['images']:>6}} images  {{d['source_groups']:>6}} source groups  "
          f"{{d['instances_total']:>7}} boxes")
"""))

    cells.append(md("""## 7. Stage 1 — the from-scratch run

Exactly the arguments recorded in the original `best.pt`, with only `data`, `project` and
`name` changed. Note `epochs=50` with `patience=12`: the original stopped early at epoch
31, and this run is free to stop wherever it stops. `optimizer='auto'` and `lr0=0.01` are
as recorded.
"""))
    cells.append(code(f"""from ultralytics import YOLO

STAGE1_ARGS = {s1}

PROJECT = '/kaggle/working/runs'
STAGE1_ARGS.update(data=GROUPED_YAML, project=PROJECT, name='grouped_s1')

print('stage 1:', {{k: STAGE1_ARGS[k] for k in
      ('epochs', 'imgsz', 'batch', 'optimizer', 'lr0', 'patience', 'seed', 'device')}})
m1 = YOLO('yolov8s.pt')
m1.train(**STAGE1_ARGS)

S1_DIR = os.path.join(PROJECT, 'grouped_s1')
S1_BEST = os.path.join(S1_DIR, 'weights', 'best.pt')
assert os.path.isfile(S1_BEST), S1_BEST
print('stage 1 best:', S1_BEST)
print('elapsed so far: %.2f h' % ((time.time() - T_START) / 3600))
"""))

    cells.append(md("""## 8. Evaluate stage 1, and bank everything

Copied to `/kaggle/working/out/` now, so that if stage 2 runs out of time the stage-1
result still survives in the notebook output.
"""))
    cells.append(code("""OUT = '/kaggle/working/out'
os.makedirs(OUT, exist_ok=True)

!python {REPO}/tools/eval_grouped.py --weights {S1_BEST} --dataset-dir {GROUPED} --out-dir {OUT}/eval_stage1 --name eval_stage1

import shutil
shutil.copy(os.path.join(GROUPED, 'split_summary.json'), OUT)
os.makedirs(os.path.join(OUT, 'stage1'), exist_ok=True)
for f in ('results.csv', 'args.yaml'):
    p = os.path.join(S1_DIR, f)
    if os.path.isfile(p):
        shutil.copy(p, os.path.join(OUT, 'stage1', f))
shutil.copy(S1_BEST, os.path.join(OUT, 'stage1', 'best.pt'))
print(sorted(os.listdir(OUT)))
print(open(os.path.join(OUT, 'eval_stage1', 'eval_stage1.md')).read()[:1500])
"""))

    cells.append(md("""## 9. Stage 2 — the refinement

Exactly the arguments recorded in `best_refined.pt`, with `data`, `project` and `name`
changed, and starting from **stage 1's** best checkpoint instead of the original Kaggle
dataset path — which is what "continue from stage 1" means.

Skipped if more than 8.5 h have already gone, leaving comfortable room inside Kaggle's
12 h limit for stage 2's own evaluation and the zip.
"""))
    cells.append(code(f"""STAGE2_ARGS = {s2}

STAGE2_ARGS.update(data=GROUPED_YAML, project=PROJECT, name='grouped_s2')

BUDGET_H = 8.5
elapsed_h = (time.time() - T_START) / 3600
S2_BEST = None
if elapsed_h > BUDGET_H:
    print(f'SKIPPING stage 2: {{elapsed_h:.2f}} h already elapsed, over the {{BUDGET_H}} h budget.')
    print('Kaggle stops a committed run at 12 h, and stage 2 plus its evaluation would not')
    print('finish. The stage-1 weights and evaluation are already in /kaggle/working/out.')
    print('To get stage 2: re-run this notebook with stage 1 replaced by the banked')
    print('stage1/best.pt attached as a Kaggle dataset input.')
else:
    print(f'{{elapsed_h:.2f}} h elapsed, under the {{BUDGET_H}} h budget -- running stage 2.')
    print('stage 2:', {{k: STAGE2_ARGS[k] for k in
          ('epochs', 'imgsz', 'batch', 'optimizer', 'lr0', 'patience', 'seed', 'device')}})
    m2 = YOLO(S1_BEST)
    m2.train(**STAGE2_ARGS)
    S2_DIR = os.path.join(PROJECT, 'grouped_s2')
    S2_BEST = os.path.join(S2_DIR, 'weights', 'best.pt')
    assert os.path.isfile(S2_BEST), S2_BEST
    print('stage 2 best:', S2_BEST)
print('elapsed: %.2f h' % ((time.time() - T_START) / 3600))
"""))

    cells.append(md("## 10. Evaluate stage 2\n\nSame four runs as stage 1, so the two stages "
                    "are directly comparable.\n"))
    cells.append(code("""if S2_BEST:
    !python {REPO}/tools/eval_grouped.py --weights {S2_BEST} --dataset-dir {GROUPED} --out-dir {OUT}/eval_stage2 --name eval_stage2
    print(open(os.path.join(OUT, 'eval_stage2', 'eval_stage2.md')).read()[:1500])
else:
    print('stage 2 was skipped, so there is nothing to evaluate here.')
"""))

    cells.append(md("""## 11. Collect the output and zip it

`/kaggle/working/out.zip` is what to download from the **Output** tab. `best_grouped.pt` is
stage 2's weights if stage 2 ran, otherwise stage 1's — the file name always means "the
model trained on the grouped split", and `PROVENANCE.txt` says which stage it came from.
"""))
    cells.append(code("""import shutil, json, subprocess

FINAL_DIR, FINAL_STAGE = (None, None)
if S2_BEST:
    FINAL_DIR, FINAL_STAGE = os.path.join(PROJECT, 'grouped_s2'), 'stage2'
else:
    FINAL_DIR, FINAL_STAGE = S1_DIR, 'stage1'

if FINAL_STAGE == 'stage2':
    os.makedirs(os.path.join(OUT, 'stage2'), exist_ok=True)
    for f in ('results.csv', 'args.yaml'):
        p = os.path.join(FINAL_DIR, f)
        if os.path.isfile(p):
            shutil.copy(p, os.path.join(OUT, 'stage2', f))

shutil.copy(os.path.join(FINAL_DIR, 'weights', 'best.pt'),
            os.path.join(OUT, 'best_grouped.pt'))

with open(os.path.join(OUT, 'PROVENANCE.txt'), 'w') as fh:
    fh.write('best_grouped.pt is the %s checkpoint\\n' % FINAL_STAGE)
    fh.write('repo commit     : %s\\n' % COMMIT)
    fh.write('split sha256    : %s\\n' % EXPECTED_SHA256)
    fh.write('ultralytics     : %s\\n' % ultralytics.__version__)
    fh.write('torch           : %s\\n' % torch.__version__)
    fh.write('total hours     : %.2f\\n' % ((time.time() - T_START) / 3600))

shutil.make_archive('/kaggle/working/out', 'zip', OUT)
print('wrote /kaggle/working/out.zip')
for r, _d, fs in os.walk(OUT):
    for f in sorted(fs):
        p = os.path.join(r, f)
        print(f'  {os.path.relpath(p, OUT):<48} {os.path.getsize(p)/1e6:8.2f} MB')
print()
print(open(os.path.join(OUT, 'PROVENANCE.txt')).read())
"""))

    return {
        "cells": cells,
        "metadata": {
            "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [],
                       "isInternetEnabled": True, "language": "python",
                       "sourceType": "notebook"},
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the grouped-retrain Kaggle notebook")
    ap.add_argument("--stage1-args",
                    default="paper_materials/final_detector_train/prior_run_best_pt/"
                            "args_from_checkpoint.yaml")
    ap.add_argument("--stage2-args",
                    default="paper_materials/final_detector_train/args_from_checkpoint.yaml")
    ap.add_argument("--sha256", required=True)
    ap.add_argument("--ultralytics", default=None,
                    help="version to pin; default = the one recorded in best_refined.pt")
    ap.add_argument("--stage2-summary",
                    default="paper_materials/final_detector_train/summary.json")
    ap.add_argument("--repo", default="https://github.com/VanVan120/AR-PPE-Detection")
    ap.add_argument("--branch", default="paper-prep")
    ap.add_argument("--seed-note", default="unknown",
                    help="the seed group_split.py actually used, recorded into "
                         "split_summary.json on Kaggle")
    ap.add_argument("--out", default="kaggle_ppe_grouped.ipynb")
    a = ap.parse_args(argv)

    pin = a.ultralytics
    if not pin:
        s = json.load(open(a.stage2_summary, encoding="utf-8"))
        pin = s.get("ultralytics_version")
    if not pin:
        import ultralytics
        pin = ultralytics.__version__
        print(f"[note] no version recorded in the checkpoint; pinning the local "
              f"ultralytics {pin}")
    else:
        print(f"pinning ultralytics {pin} (recorded in best_refined.pt)")

    nb = build(load_args(a.stage1_args), load_args(a.stage2_args), a.sha256, pin,
               a.repo, a.branch, a.seed_note)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"wrote {a.out}  ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
