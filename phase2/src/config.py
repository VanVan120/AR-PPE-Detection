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
    # --- Work ID: bind each tracked person to a persistent worker identity via a
    #     visible ArUco marker worn on the helmet/vest (needs cv2.aruco / contrib).
    "workid": {
        "enabled": False,
        "dictionary": "DICT_4X4_50",
        "containment": 0.5,         # min fraction of a marker inside a person box to bind it
        "markers": {},              # marker_id (int) -> worker label (str)
    },
    # --- Worker-attributed event log (JSONL). Opt-in: set a path to enable; "" / null
    #     keeps the default pipeline side-effect-free (no file created). --------------
    "event_log": "",
    # --- Activity / workflow recognition: SCAFFOLD only. Off until the dataset is
    #     confirmed (Assembly101 / Ego4D). backend: placeholder | kinetics ---------
    "activity": {
        "enabled": False,
        "backend": "placeholder",
        "clip_len": 16,
        "stride": 2,
    },
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
    # Work ID
    workid_enabled: bool
    workid_dictionary: str
    workid_containment: float
    workid_markers: dict[int, str]
    # event log + activity scaffold
    event_log: str
    activity_enabled: bool
    activity_backend: str
    activity_clip_len: int
    activity_stride: int

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
    def event_log_path(self) -> str:
        return self._resolve(self.event_log) if self.event_log else ""

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
    file_cfg: dict[str, Any] = {}
    if os.path.exists(config_path):
        file_cfg = _read_yaml(config_path)
        raw.update({k: v for k, v in file_cfg.items() if v is not None})

    # `event_log: null` (and "") is the documented "disable" signal, but the
    # None-filter above drops null and would leave the default in place. Honour an
    # explicit empty/null value as "disabled".
    if "event_log" in file_cfg and not file_cfg["event_log"]:
        raw["event_log"] = ""

    # Merge nested sections per-key so a partial override keeps the other defaults.
    workid = dict(DEFAULTS["workid"])
    if isinstance(raw.get("workid"), dict):
        workid.update(raw["workid"])
    activity = dict(DEFAULTS["activity"])
    if isinstance(raw.get("activity"), dict):
        activity.update(raw["activity"])
    markers = {}
    for k, v in dict(workid.get("markers") or {}).items():
        try:
            markers[int(k)] = str(v)
        except (TypeError, ValueError):
            pass

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
        workid_enabled=bool(workid.get("enabled", False)),
        workid_dictionary=str(workid.get("dictionary", "DICT_4X4_50")),
        workid_containment=float(workid.get("containment", 0.5)),
        workid_markers=markers,
        event_log=str(raw["event_log"]) if raw.get("event_log") else "",
        activity_enabled=bool(activity.get("enabled", False)),
        activity_backend=str(activity.get("backend", "placeholder")),
        activity_clip_len=int(activity.get("clip_len", 16)),
        activity_stride=int(activity.get("stride", 2)),
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

    # Work ID
    if cfg.workid_enabled:
        if not (0.0 <= cfg.workid_containment <= 1.0):
            issues.append(Issue("error", f"workid.containment must be in [0,1]: {cfg.workid_containment}"))
        if not cfg.workid_markers:
            issues.append(Issue("warn", "workid.enabled but no markers mapped — detected markers "
                                        "will get auto labels 'W-<id>'. Add markers in config.yaml."))

    # Activity scaffold
    if cfg.activity_enabled:
        if cfg.activity_clip_len < 1:
            issues.append(Issue("error", f"activity.clip_len must be >= 1: {cfg.activity_clip_len}"))
        if cfg.activity_stride < 1:
            issues.append(Issue("warn", f"activity.stride must be >= 1 (clamped to 1): {cfg.activity_stride}"))
        if cfg.activity_backend not in ("placeholder", "kinetics"):
            issues.append(Issue("warn", f"activity.backend '{cfg.activity_backend}' unknown "
                                        "(expected: placeholder | kinetics)"))

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
