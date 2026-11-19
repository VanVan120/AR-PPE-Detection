# Part A, attempt 0: stopped because the GPU was clock-capped (19 Nov 2026)

The first launch of seed 1 (17:28:36) was stopped by hand at 17:36:30, after 28 of 2,088 iterations
of stage-1 epoch 1. Nothing from it is used; the run directory was deleted and seed 1 was started
again from scratch after a restart. This folder keeps its log and records.

**Symptom.** 15.3–15.8 s per iteration from the start, against 0.286 s/it measured for the same
recipe and `--workers 2` before the published run (paper_materials/retrain/REPORT_local_train.md §2.4)
and 661 s for the published run's first epoch (0.32 s/it). At that speed one epoch would take ~9 h.

**Cause: the GPU was held at its minimum clock by a false power reading** (`nvidia-smi`, driver 592.82):

| when | utilisation | memory used | SM clock | reported power | temp |
|---|---:|---:|---:|---:|---:|
| 17:2x, before training, nothing running | 0 % | 110 MiB | — | 86.39 W | 44 °C |
| 17:35, training | 100 % | 3,925 MiB | **210 MHz** | 94.01 W | 48 °C |
| 17:37, training killed, nothing running | 14 % | 104 MiB | **210 MHz** | 88.70 W | 48 °C |

- Power limit 75.84 W (default 80, max 120); "Clocks Event Reasons: SW Power Cap: Active".
- Max SM clock 3,105 MHz. A reading of 86–94 W from an idle GPU at 44–48 °C is not physical, so the
  driver was capping on a false reading.
- The laptop had not been restarted since 24 Sep 2026 17:19, the BSOD during the published run.
- Windows also had a restart pending (Component Based Servicing `RebootPending`, plus
  `PendingFileRenameOperations`).

**Memory was tight but was not the cause.** At the time there were 0.5–1.6 GB available and the
system was paging (≈2,100–3,000 pages input/s). But the published run's notes record 0.288 s/it
with only 212 MB free (workers 4), and here the GPU was the bottleneck at 100 % "utilisation" on
210 MHz.

**Fix.** Restart the laptop (clears the driver state and applies the pending update), check that the
SM clock rises under load, then relaunch with the identical command.
