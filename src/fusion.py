"""Track C — fuse the fine-tuned detector with the VLM into one richer assessment.

Grounded in the measured results: after Track A the fine-tuned detector is the
strong, precise source for the scored PPE / violation classes (box-level, high
precision — NO-Hardhat / NO-Safety Vest F1 ~0.63-0.65). The VLM's distinct value
is no longer "better violation recall" (the detector beats it there now) but:

  1. hazards the detector has NO class for — unprotected edge, exposed rebar,
     proximity / scene context;
  2. a second opinion that can catch a detector miss;
  3. natural-language narration.

`fuse()` combines them with the detector AUTHORITATIVE on the scored violation
classes:
  * both flag a violation         -> CONFIRMED (grounded + corroborated)
  * detector alone                -> DETECTOR  (grounded, high precision)
  * VLM alone                     -> REVIEW    (possible detector miss — surface it)
  * neither                       -> clear
VLM observations that map to no scored class become surfaced CONTEXT HAZARDS, and
a concise fused report + overall risk level are produced.

The point isn't to change box-level mAP (only the detector has boxes) — it's to
improve *image-level* violation detection (recall via OR, precision via AND) and to
produce a report richer than either pipeline alone. fusion_eval.py measures that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .detector_yolo import Detection
from .detector_vlm import VlmResult, Observation
from .evaluator import _vlm_flags_class, _negative_pattern, _normalize


# Statuses for a single violation class after reconciliation.
CONFIRMED = "confirmed"   # detector + VLM agree
DETECTOR = "detector"     # detector only (grounded)
REVIEW = "review"         # VLM only — possible detector miss, flag for review
CLEAR = "clear"           # neither


@dataclass
class ViolationStatus:
    cls: str
    detector_flag: bool
    detector_count: int
    detector_conf: float
    vlm_flag: bool

    @property
    def status(self) -> str:
        if self.detector_flag and self.vlm_flag:
            return CONFIRMED
        if self.detector_flag:
            return DETECTOR
        if self.vlm_flag:
            return REVIEW
        return CLEAR

    def to_dict(self) -> dict:
        return {
            "class": self.cls, "status": self.status,
            "detector_flag": self.detector_flag, "detector_count": self.detector_count,
            "detector_conf": round(self.detector_conf, 4), "vlm_flag": self.vlm_flag,
        }


@dataclass
class FusionResult:
    violations: list[ViolationStatus] = field(default_factory=list)
    people: int = 0
    context_hazards: list[Observation] = field(default_factory=list)
    report: str = ""
    risk: str = "low"

    def to_dict(self) -> dict:
        return {
            "people": self.people,
            "violations": [v.to_dict() for v in self.violations],
            "context_hazards": [o.to_dict() for o in self.context_hazards],
            "report": self.report,
            "risk": self.risk,
        }


def detector_flags_class(dets: list[Detection], cls: str, cfg: Config) -> tuple[bool, int, float]:
    """Image-level: does the detector predict class `cls` at its operating point?"""
    hits = [d for d in dets if d.class_name == cls and d.confidence >= cfg.threshold_for(cls)]
    if not hits:
        return False, 0, 0.0
    return True, len(hits), max(d.confidence for d in hits)


def _count_person(dets: list[Detection], cfg: Config) -> int:
    return sum(1 for d in dets if d.class_name == "Person" and d.confidence >= cfg.threshold_for("Person"))


def _context_hazards(vres: Optional[VlmResult], cfg: Config) -> list[Observation]:
    """VLM observations that don't restate a scored violation class — i.e. the
    hazards/context the scored detector pipeline does not report (unprotected edge,
    exposed rebar, no-mask, etc.). These are the VLM's unique value-add."""
    if vres is None or not vres.observations:
        return []
    # keyword sets for the scored violation classes; an obs matching one of these
    # is already represented by the detector reconciliation, so exclude it here.
    scored_kw: list[str] = []
    for cls in cfg.vlm_eval_classes:
        scored_kw += [_normalize(k) for k in cfg.vlm_violation_keywords.get(cls, [])]
    out: list[Observation] = []
    for obs in vres.observations:
        text = _normalize(f"{obs.type} {obs.description}")
        if obs.type in ("person_detected", "observation"):
            continue
        if any(kw and kw in text for kw in scored_kw):
            continue
        out.append(obs)
    return out


def fuse(dets: Optional[list[Detection]], vres: Optional[VlmResult], cfg: Config) -> FusionResult:
    dets = dets or []
    neg = _negative_pattern(cfg.vlm_negative_keywords)
    violations: list[ViolationStatus] = []
    for cls in cfg.vlm_eval_classes:
        dflag, dn, dconf = detector_flags_class(dets, cls, cfg)
        vflag = _vlm_flags_class(vres, cls, cfg, neg) if vres is not None else False
        violations.append(ViolationStatus(cls, dflag, dn, dconf, vflag))

    people = _count_person(dets, cfg)
    hazards = _context_hazards(vres, cfg)
    risk = _risk(violations, hazards, vres)
    report = build_fused_report(violations, people, hazards, risk)
    return FusionResult(violations=violations, people=people,
                        context_hazards=hazards, report=report, risk=risk)


def _risk(violations: list[ViolationStatus], hazards: list[Observation],
          vres: Optional[VlmResult]) -> str:
    confirmed = [v for v in violations if v.status == CONFIRMED]
    grounded = [v for v in violations if v.status in (CONFIRMED, DETECTOR)]
    review = [v for v in violations if v.status == REVIEW]
    high_haz = any(o.severity == "high" for o in hazards)

    risk = "low"
    if grounded or review or hazards:
        risk = "medium"
    # High: corroborated violation, multiple grounded violations, or a serious
    # context hazard (edge/rebar/fall).
    if confirmed or len(grounded) >= 2 or high_haz:
        risk = "high"
    return risk


def _label(v: ViolationStatus) -> str:
    base = "hard hat" if "hardhat" in v.cls.lower().replace("-", "") else \
           ("safety vest" if "vest" in v.cls.lower() else v.cls)
    return base


def build_fused_report(violations, people, hazards, risk) -> str:
    """A concise but richer report than either pipeline alone."""
    clauses: list[str] = []
    if people:
        clauses.append(f"{people} worker{'s' if people != 1 else ''} detected")

    grounded = []
    reviews = []
    for v in violations:
        if v.status == CONFIRMED:
            grounded.append(f"{v.detector_count} without a {_label(v)} (confirmed)")
        elif v.status == DETECTOR:
            grounded.append(f"{v.detector_count} without a {_label(v)}")
        elif v.status == REVIEW:
            reviews.append(_label(v))
    if grounded:
        clauses.append("; ".join(grounded))
    elif people:
        clauses.append("required PPE present")
    if reviews:
        clauses.append("VLM-only (review): possible missing " + ", ".join(reviews))

    if hazards:
        htypes = []
        for o in hazards:
            t = o.type.replace("_", " ")
            if t not in htypes:
                htypes.append(t)
        clauses.append("VLM also flags: " + ", ".join(htypes[:3]))

    if not clauses:
        clauses.append("no workers or PPE violations detected")
    line = "; ".join(clauses)
    line = line[0].upper() + line[1:]
    return f"{line} — {risk} risk."
