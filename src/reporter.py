"""Turn detections + VLM observations into one concise safety report line.

Template-based and deterministic so it works even when one pipeline is skipped
(Ollama down, or YOLO disabled). Both pipelines' findings are combined; a risk
level (low/medium/high) is derived from the violations and the VLM severities.
"""
from __future__ import annotations

from typing import Optional

from .config import Config
from .detector_yolo import Detection
from .detector_vlm import VlmResult

_HAZARD_HINTS = ("edge", "rebar", "fall", "scaffold", "machinery", "trip", "open hole")


def _dedupe(items: list[str]) -> list[str]:
    seen: list[str] = []
    for it in items:
        if it and it not in seen:
            seen.append(it)
    return seen


def _yolo_counts(detections: list[Detection], cfg: Config):
    # Use the same per-class operating point as the gallery/record so the report
    # text can't disagree with them (cfg.threshold_for falls back to the global
    # confidence_threshold when no per-class override applies / backend is gated).
    shown = [d for d in detections if d.confidence >= cfg.threshold_for(d.class_name)]

    def count(name: str) -> int:
        return sum(1 for d in shown if d.class_name == name)

    person = count("Person")
    hat, no_hat = count("Hardhat"), count("NO-Hardhat")
    vest, no_vest = count("Safety Vest"), count("NO-Safety Vest")
    people = max(person, hat + no_hat, vest + no_vest)
    hazards = _dedupe([d.label for d in shown if d.class_name is None])
    return people, no_hat, no_vest, hazards


def build_report(
    detections: Optional[list[Detection]],
    vlm: Optional[VlmResult],
    cfg: Config,
    threshold: Optional[float] = None,   # kept for backwards-compat; counts use cfg.threshold_for
) -> str:
    """Build a one-line natural-language safety report.

    `detections is None` / `vlm is None` mean that pipeline was not run (vs an
    empty list/result, which means it ran and found nothing).
    """
    has_yolo = detections is not None
    has_vlm = vlm is not None
    if not has_yolo and not has_vlm:
        return "No detection pipelines were run."

    people = no_hat = no_vest = 0
    hazards: list[str] = []
    if has_yolo:
        people, no_hat, no_vest, hazards = _yolo_counts(detections, cfg)

    vlm_obs = vlm.observations if has_vlm else []
    vlm_high = any(o.severity == "high" for o in vlm_obs)
    vlm_med = any(o.severity == "medium" for o in vlm_obs)
    vlm_hazard = any(
        any(h in (o.type + " " + o.description).lower() for h in _HAZARD_HINTS)
        for o in vlm_obs
    )

    # --- risk level ----------------------------------------------------------
    risk = "low"
    if no_hat or no_vest or vlm_med or vlm_obs:
        risk = "medium"
    if (no_hat and no_vest) or vlm_high or hazards or vlm_hazard:
        risk = "high"

    # --- clauses -------------------------------------------------------------
    clauses: list[str] = []
    if has_yolo:
        if people:
            clauses.append(f"{people} worker{'s' if people != 1 else ''} detected")
        violations: list[str] = []
        if no_hat:
            violations.append(f"{no_hat} without a hard hat")
        if no_vest:
            violations.append(f"{no_vest} without a safety vest")
        if violations:
            clauses.append(" and ".join(violations))
        elif people and not hazards:
            clauses.append("required PPE present")
        if hazards:
            clauses.append("possible hazard: " + ", ".join(hazards))
        if not people and not violations and not hazards:
            clauses.append("no workers or PPE violations detected")

    if has_vlm and vlm_obs:
        types = _dedupe([o.type for o in vlm_obs if o.type != "person_detected"])
        detail = f" ({', '.join(types[:3])})" if types else ""
        n = len(vlm_obs)
        clauses.append(f"VLM flagged {n} observation{'s' if n != 1 else ''}{detail}")

    line = "; ".join(clauses) if clauses else "no notable safety observations"
    line = line[0].upper() + line[1:]
    return f"{line} — {risk} risk."
