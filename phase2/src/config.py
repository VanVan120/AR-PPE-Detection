"""Load + validate config.yaml for the real-time pipeline.

A single `Config` object carries every knob plus a resolved compute device. Paths
(weights, benchmark) are resolved relative to the config file so `python run.py`
works from the phase2 directory regardless of the process CWD.

Class names are NEVER hardcoded for scoring: the violation-rule names in
config.yaml are validated against the *model's own* class names at `--check`
time (see `validate_against_model`), so a typo or a name that the trained model
does not actually predict is reported clearly instead of silently never firing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import yaml

# Severity ranking — used to pick the *worst* active violation on a person (for
# the box colour) and to order the HUD. Higher = more severe.
SEVERITY_RANK: dict[str, int] = {"high": 3, "medium": 2, "low": 1}

DEFAULTS: dict[str, Any] = {
    "weights": "models/best.pt",
    "source": 0,
    "confidence_threshold": 0.35,
    "imgsz": 640,
    "device": "auto",
    "person_class": "Person",
    "violation_rules": {
        "No-Helmet": {"severity": "high", "label": "No hard hat"},
        "No-Vest": {"severity": "medium", "label": "No safety vest"},
    },
    "association_containment": 0.30,
    "debounce_frames": 5,
    "clear_frames": 15,
    "lost_track_buffer": 30,
    "target_fps": 15,
    "save_output_video": False,
    "benchmark_file": "benchmark.json",
}


@dataclass
class Issue:
    """A single validation result for `run.py --check`."""
    level: str   # "ok" | "warn" | "error"
    message: str

    def __str__(self) -> str:
        mark = {"ok": "[ ok ]", "warn": "[warn]", "error": "[FAIL]"}.get(self.level, "[?]")
        return f"{mark} {self.message}"


@dataclass
class ViolationRule:
    class_name: str
    severity: str
    label: str


@dataclass
class Config:
    config_path: str
    config_dir: str
    weights: str
    source: Union[int, str]
    confidence_threshold: float
    imgsz: int
    device_pref: str
    person_class: str
    violation_rules: dict[str, ViolationRule]
    association_containment: float
    debounce_frames: int
    clear_frames: int
    lost_track_buffer: int
    target_fps: int
    save_output_video: bool
    benchmark_file: str

    # --- resolved paths ------------------------------------------------------
    def _resolve(self, path: str) -> str:
        """Resolve a config-relative path against the config file's directory."""
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.config_dir, path))

    @property
    def weights_path(self) -> str:
        return self._resolve(self.weights)

    @property
    def benchmark_path(self) -> str:
        return self._resolve(self.benchmark_file)

    @property
    def device(self) -> str:
        return resolve_device(self.device_pref)

    @property
    def source_is_path(self) -> bool:
        """True when `source` names a file (not a webcam index)."""
        return isinstance(self.source, str) and not str(self.source).isdigit()

    @property
    def source_display(self) -> str:
        return f"file:{self.source}" if self.source_is_path else f"webcam:{self.source}"

    def resolved_source(self) -> Union[int, str]:
        """The value to hand cv2.VideoCapture: an int index, or a resolved path."""
        if self.source_is_path:
            return self._resolve(str(self.source))
        # "0" or 0 -> webcam index 0
        return int(self.source)


def resolve_device(pref: str) -> str:
    """Resolve "auto" to the best available device, else honour the preference."""
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _read_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _parse_rules(raw: Any) -> dict[str, ViolationRule]:
    rules: dict[str, ViolationRule] = {}
    if not isinstance(raw, dict):
        return rules
    for name, spec in raw.items():
        spec = spec or {}
        severity = str(spec.get("severity", "high")).lower()
        label = str(spec.get("label", name))
        rules[str(name)] = ViolationRule(class_name=str(name), severity=severity, label=label)
    return rules


def load_config(config_path: str = "config.yaml") -> Config:
    """Load config.yaml merged over DEFAULTS."""
    raw: dict[str, Any] = dict(DEFAULTS)
    config_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.exists(config_path):
        file_cfg = _read_yaml(config_path)
        raw.update({k: v for k, v in file_cfg.items() if v is not None})

    return Config(
        config_path=config_path,
        config_dir=config_dir,
        weights=str(raw["weights"]),
        source=raw["source"],
        confidence_threshold=float(raw["confidence_threshold"]),
        imgsz=int(raw["imgsz"]),
        device_pref=str(raw["device"]),
        person_class=str(raw["person_class"]),
        violation_rules=_parse_rules(raw["violation_rules"]),
        association_containment=float(raw["association_containment"]),
        debounce_frames=int(raw["debounce_frames"]),
        clear_frames=int(raw["clear_frames"]),
        lost_track_buffer=int(raw["lost_track_buffer"]),
        target_fps=int(raw["target_fps"]),
        save_output_video=bool(raw["save_output_video"]),
        benchmark_file=str(raw["benchmark_file"]),
    )


def validate_config(cfg: Config) -> list[Issue]:
    """Structural validation that does NOT need to load the model (no torch)."""
    issues: list[Issue] = []

    # weights
    if not os.path.isfile(cfg.weights_path):
        issues.append(Issue("error", f"weights not found: {cfg.weights_path}"))
    else:
        issues.append(Issue("ok", f"weights present: {cfg.weights_path}"))

    # thresholds
    if not (0.0 <= cfg.confidence_threshold <= 1.0):
        issues.append(Issue("error", f"confidence_threshold must be in [0,1]: {cfg.confidence_threshold}"))
    if not (0.0 <= cfg.association_containment <= 1.0):
        issues.append(Issue("error", f"association_containment must be in [0,1]: {cfg.association_containment}"))
    if cfg.imgsz <= 0 or cfg.imgsz % 32 != 0:
        issues.append(Issue("warn", f"imgsz {cfg.imgsz} should be a positive multiple of 32 (YOLO requirement)"))

    # debounce / clear
    if cfg.debounce_frames < 1:
        issues.append(Issue("error", f"debounce_frames must be >= 1: {cfg.debounce_frames}"))
    if cfg.clear_frames < 1:
        issues.append(Issue("error", f"clear_frames must be >= 1: {cfg.clear_frames}"))
    if cfg.target_fps < 1:
        issues.append(Issue("error", f"target_fps must be >= 1: {cfg.target_fps} "
                                     "(0/negative silently disables recording + breaks tracking)"))
    if cfg.lost_track_buffer < 1:
        issues.append(Issue("error", f"lost_track_buffer must be >= 1: {cfg.lost_track_buffer}"))

    # rules
    if not cfg.violation_rules:
        issues.append(Issue("error", "violation_rules is empty — nothing would ever be flagged"))
    for name, rule in cfg.violation_rules.items():
        if rule.severity not in SEVERITY_RANK:
            issues.append(Issue("warn", f"rule '{name}' has unknown severity '{rule.severity}' "
                                        f"(expected {sorted(SEVERITY_RANK)}) — treated as 'high'"))

    # source
    if cfg.source_is_path:
        src = cfg.resolved_source()
        if not os.path.isfile(str(src)):
            issues.append(Issue("warn", f"source file not found: {src} "
                                        "(pass --source or record a clip into data/clips/)"))
        else:
            issues.append(Issue("ok", f"source file present: {src}"))
    else:
        issues.append(Issue("ok", f"source is webcam index {cfg.resolved_source()} "
                                  "(opened at run time)"))

    return issues


def validate_against_model(cfg: Config, model_names: dict[int, str]) -> list[Issue]:
    """Validate the rule + person class names against the model's ACTUAL classes.

    This is where a name that the trained model does not predict (a typo, or the
    proposal's placeholder names) is caught — such a rule would otherwise sit
    silently dead, never firing.
    """
    issues: list[Issue] = []
    available = set(model_names.values())

    if cfg.person_class in available:
        issues.append(Issue("ok", f"person_class '{cfg.person_class}' found in model"))
    else:
        issues.append(Issue("error", f"person_class '{cfg.person_class}' is NOT a model class. "
                                     f"Model classes: {sorted(available)}"))

    for name in cfg.violation_rules:
        if name in available:
            issues.append(Issue("ok", f"violation rule '{name}' found in model"))
        else:
            issues.append(Issue("error", f"violation rule '{name}' is NOT a model class — it can "
                                         f"never fire. Model classes: {sorted(available)}"))
    return issues
