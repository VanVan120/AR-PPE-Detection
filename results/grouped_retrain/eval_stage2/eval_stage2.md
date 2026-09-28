# Grouped-split evaluation of `best.pt`

- weights: `C:\Users\user\Desktop\SummerInternProject\paper_materials\retrain\local_runs\grouped_s2\weights\best.pt`
- dataset: `C:\Users\user\Desktop\SummerInternProject\data\ppe_grouped` (source-grouped split)
- device: `0` · ultralytics 8.4.75 · torch 2.13.0+cu130 · python 3.13.3
- `model.val`, `plots=False`, every other setting at its ultralytics default

## Overall

| run | imgsz | images | precision | recall | F1 | F1 (macro) | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| full_640 | 640 | 4174 | 70.78 | 71.34 | 71.06 | 70.80 | 71.11 | 46.24 |
| one_per_source_640 | 640 | 584 | 62.82 | 70.80 | 66.57 | 66.46 | 66.20 | 44.32 |
| full_480 | 480 | 4174 | 71.37 | 72.24 | 71.80 | 71.54 | 71.96 | 47.04 |
| full_320 | 320 | 4174 | 71.23 | 70.52 | 70.87 | 70.74 | 70.99 | 44.65 |

**F1** is the harmonic mean of the mean precision and mean recall — the definition `eval_ppe.py` and the published table use, so it is the comparable one. **F1 (macro)** is the mean of the per-class F1s. They are not equal when classes are uneven; both are given so neither is mistaken for the other.

## Per class

### precision

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 78.65 | 76.47 | 78.93 | 79.36 |
| No-Helmet | 42.93 | 41.99 | 43.76 | 44.13 |
| No-Vest | 69.12 | 56.07 | 70.02 | 69.71 |
| Person | 82.41 | 67.06 | 82.91 | 81.83 |
| Vest | 80.77 | 72.50 | 81.23 | 81.11 |

### recall

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 89.53 | 86.84 | 89.98 | 85.76 |
| No-Helmet | 50.70 | 52.99 | 52.79 | 48.80 |
| No-Vest | 56.71 | 53.88 | 58.52 | 59.51 |
| Person | 80.92 | 80.07 | 80.07 | 78.51 |
| Vest | 78.87 | 80.23 | 79.83 | 80.03 |

### f1

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 83.74 | 81.33 | 84.09 | 82.44 |
| No-Helmet | 46.49 | 46.85 | 47.85 | 46.35 |
| No-Vest | 62.30 | 54.95 | 63.75 | 64.21 |
| Person | 81.66 | 72.99 | 81.47 | 80.14 |
| Vest | 79.81 | 76.17 | 80.53 | 80.57 |

### map50

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 88.81 | 85.68 | 89.01 | 85.86 |
| No-Helmet | 37.89 | 38.53 | 40.74 | 39.76 |
| No-Vest | 61.12 | 52.34 | 61.59 | 61.53 |
| Person | 84.87 | 75.79 | 84.82 | 84.36 |
| Vest | 82.85 | 78.66 | 83.66 | 83.42 |

### map50_95

| class | full_640 | one_per_source_640 | full_480 | full_320 |
|---|---|---|---|---|
| Helmet | 56.60 | 51.90 | 57.19 | 53.63 |
| No-Helmet | 20.77 | 23.55 | 21.73 | 20.06 |
| No-Vest | 37.27 | 33.11 | 37.26 | 35.00 |
| Person | 63.67 | 58.21 | 64.70 | 62.86 |
| Vest | 52.89 | 54.83 | 54.32 | 51.69 |

## Count checks

Each run's counts are checked three ways: against the length of the image list it was handed, against the label files that list resolves to, and — for the un-truncated runs — against the independent count recorded in `split_summary.json` when the split was materialised. The third is the one that could catch a wrong list, rather than ultralytics echoing back the list it was given.

| run | expected images | images scored | independent | expected label files | labels found | missing/empty | corrupt | instances | status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_640 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| one_per_source_640 | 584 | 584 | 584 | 584 | 584 | 0 | 0 | 2957 | OK |
| full_480 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |
| full_320 | 4174 | 4174 | 4174 | 4174 | 4174 | 0 | 0 | 25232 | OK |

**All count checks passed.**
