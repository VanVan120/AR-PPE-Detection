"""Assemble per-image records that feed results.json / results.html.

Keeps the qualitative side-by-side data (annotated image path, YOLO detections,
VLM observations, the report line) in one place so `render.py` only formats it.
"""
from __future__ import annotations

from typing import Optional

from .detector_yolo import Detection
from .detector_vlm import VlmResult
from .loader import Sample


def assemble_record(
    sample: Sample,
    display_image_rel: str,
    display_dets: Optional[list[Detection]],
    vres: Optional[VlmResult],
    report: str,
) -> dict:
    """Build one image's combined record. `None` means that pipeline was skipped."""
    rec: dict = {
        "image": sample.name,
        "width": sample.width,
        "height": sample.height,
        "display_image": display_image_rel,
        "report": report,
    }
    if display_dets is not None:
        rec["yolo_world"] = {
            "ran": True,
            "num_detections": len(display_dets),
            "detections": [d.to_dict() for d in display_dets],
        }
    else:
        rec["yolo_world"] = {"ran": False}

    if vres is not None:
        rec["vlm"] = {
            "ran": True,
            "observations": [o.to_dict() for o in vres.observations],
            "error": vres.error,
        }
    else:
        rec["vlm"] = {"ran": False}

    return rec


def summarize(records: list[dict]) -> dict:
    """High-level counts for the results-page header."""
    n = len(records)
    total_det = sum(r["yolo_world"].get("num_detections", 0)
                    for r in records if r["yolo_world"].get("ran"))
    total_obs = sum(len(r["vlm"].get("observations", []))
                    for r in records if r["vlm"].get("ran"))
    vlm_errors = sum(1 for r in records
                     if r["vlm"].get("ran") and r["vlm"].get("error"))
    return {
        "num_images": n,
        "total_detections_shown": total_det,
        "total_vlm_observations": total_obs,
        "vlm_errors": vlm_errors,
    }
