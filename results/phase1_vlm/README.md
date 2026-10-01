# Phase 1 VLM answers: the raw responses behind the VLM-only numbers
Qwen2.5-VL 7B via Ollama (tag `qwen2.5vl:7b`), run 21 June 2026 on the 82 test images of Dataset A (Construction Site Safety); `vlm/` holds one response per image.
`metrics.json` is that run's scored output: VLM-only, image level, macro over NO-Hardhat and NO-Safety Vest, precision 24.2 / recall 26.7 / F1 25.4 (it also holds the YOLO-World zero-shot track).
3 of 82 requests returned an error from the Ollama server (HTTP 400 Bad Request: files 0034, 0051, 0055) and are scored as no observations.
To rescore, copy `vlm/` to `outputs/vlm/` and run `python run.py --reuse-vlm` (needs Dataset A at `data/dataset`; it rewrites `outputs/`).
