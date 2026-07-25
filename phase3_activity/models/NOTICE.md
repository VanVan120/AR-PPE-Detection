# Phase 3 learned workflow models — what is committed here and why

Two small JSON models are committed so that **mistake detection and next-step
anticipation can be demonstrated with no downloads at all**:

| File | Size | What it is |
|---|---|---|
| `procedure_model.json` | ~52 KB | The learned assembly-**order** model: 16 precedence constraints (`a` must precede `b`), the seen step transitions, start steps, and the step-id → name map. |
| `anticipation_model.json` | ~63 KB | The next-step model: unigram / bigram transition counts + start distribution over the coarse steps. |

Run them with:

```bash
python -m phase3_activity.tas.demo --source sample --inject-fault
```

## Deliberately NOT committed

- **`mstcn_best.pt`** (the trained TAS checkpoint, ~3.6 MB) — it is useless without the
  ~34 GB TSM feature LMDB, so shipping it would only add weight. Retrain with
  `tas/train.py`, or rebuild the two JSONs above from the coarse annotations with
  `tas/procedure.py` and `tas/anticipation.py`.
- **Everything under `phase3_activity/data/`** — the gated Assembly101 download.

## Attribution (required)

These are **derived model parameters** (integer counts and step-name strings) learned
from the Assembly101 coarse annotations. They contain no video, no images and no
extracted features.

> Assembly101: Sener et al., *"Assembly101: A Large-Scale Multi-View Video Dataset for
> Understanding Procedural Activities"*, CVPR 2022.

The dataset is licensed **CC BY-NC 4.0** — attribution required, **non-commercial use
only**. That condition carries over to these derived models and to any use of this
repository's Phase 3 outputs.
