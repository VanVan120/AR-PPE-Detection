"""Load + validate config.yaml and read the dataset's own data.yaml.

A single `Config` object carries every knob plus values derived from the dataset
(class names, split paths) and the resolved compute device. Class names are NEVER
hardcoded — they are read from the dataset's data.yaml at load time, so the
prompt->class mapping can be validated against the real classes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# ----------------------------------------------------------------------------
# Defaults — anything omitted from config.yaml falls back to these.
# ----------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "dataset_dir": "data/dataset",
    "eval_split": "test",
    "sample_for_review": 20,
    "confidence_threshold": 0.25,
    "yolo_conf_floor": 0.01,
    "iou_threshold": 0.5,
    "pipelines": ["yolo_world", "vlm"],
    "yolo_model": "yolov8s-worldv2.pt",
    # Detection backend for the main run: "yolo_world" (zero-shot, default) or
    # "finetuned" (a model trained by train.py). Track A produces the finetuned one.
    "detector_backend": "yolo_world",
    "finetuned_model": "models/finetuned.pt",
    # Test-time augmentation: run inference on augmented views and merge. ~2-3x
    # slower but lifts mAP (+~2.5 on the fine-tuned model here) with no training —
    # the most reliable accuracy lever in a data-limited regime.
    "tta": False,
    # train.py knobs (Track A fine-tuning)
    "finetune_base_model": "yolov8s.pt",
    "finetune_epochs": 80,
    "finetune_imgsz": 640,
    "finetune_batch": 8,
    "finetune_patience": 20,
    "device": "auto",
    "ollama_host": "http://localhost:11434",
    "ollama_model": "qwen2.5vl:7b",
    "ollama_timeout": 180,
    "ollama_num_predict": 640,
    "api_provider": "anthropic",
    "api_model": "claude-opus-4-8",
    "api_max_tokens": 1024,
    "output_dir": "outputs",
    "safety_prompts": [
        "person wearing a hard hat",
        "person without a hard hat",
        "person wearing a high-visibility safety vest",
        "person without a safety vest",
        "person",
    ],
    "prompt_to_class": {},
    "per_class_thresholds": {},            # tuned operating point for the zero-shot backend
    "finetuned_per_class_thresholds": {},  # tuned operating point for the fine-tuned backend
    "vlm_eval_classes": ["NO-Hardhat", "NO-Safety Vest"],
    "vlm_violation_keywords": {
        "NO-Hardhat": ["hard hat", "hardhat", "helmet", "head protection"],
        "NO-Safety Vest": ["safety vest", "hi-vis", "high visibility", "reflective vest", "vest"],
    },
    "vlm_negative_keywords": [
        "no", "without", "missing", "not wearing", "lacks", "lacking",
        "absent", "none", "unprotected", "fails to wear", "no-",
    ],
}

VALID_PIPELINES = {"yolo_world", "vlm"}


@dataclass
class Issue:
    """A single validation result for `run.py --check`."""
    level: str   # "ok" | "warn" | "error"
    message: str

    def __str__(self) -> str:
        mark = {"ok": "[ ok ]", "warn": "[warn]", "error": "[FAIL]"}.get(self.level, "[?]")
        return f"{mark} {self.message}"


@dataclass
class Config:
    config_path: str
    # raw knobs (resolved against DEFAULTS)
    dataset_dir: str
    eval_split: str
    sample_for_review: int
    confidence_threshold: float
    yolo_conf_floor: float
    iou_threshold: float
    pipelines: list[str]
    yolo_model: str
    detector_backend: str
    finetuned_model: str
    tta: bool
    finetune_base_model: str
    finetune_epochs: int
    finetune_imgsz: int
    finetune_batch: int
    finetune_patience: int
    device_pref: str
    ollama_host: str
    ollama_model: str
    ollama_timeout: int
    ollama_num_predict: int
    api_provider: str
    api_model: str
    api_max_tokens: int
    output_dir: str
    safety_prompts: list[str]
    prompt_to_class: dict[str, Optional[str]]
    per_class_thresholds: dict[str, float]
    finetuned_per_class_thresholds: dict[str, float]
    vlm_eval_classes: list[str]
    vlm_violation_keywords: dict[str, list[str]]
    vlm_negative_keywords: list[str]
    # derived from the dataset
    dataset_classes: list[str] = field(default_factory=list)

    # --- derived paths -------------------------------------------------------
    @property
    def data_yaml_path(self) -> str:
        return os.path.join(self.dataset_dir, "data.yaml")

    @property
    def images_dir(self) -> str:
        return os.path.join(self.dataset_dir, self.eval_split, "images")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.dataset_dir, self.eval_split, "labels")

    # --- output sub-dirs -----------------------------------------------------
    @property
    def yolo_out_dir(self) -> str:
        return os.path.join(self.output_dir, "yolo_world")

    @property
    def vlm_out_dir(self) -> str:
        return os.path.join(self.output_dir, "vlm")

    @property
    def images_out_dir(self) -> str:
        return os.path.join(self.output_dir, "images")

    # --- evaluation helpers --------------------------------------------------
    @property
    def scored_classes(self) -> list[str]:
        """Ordered, unique dataset classes that at least one prompt maps to and
        that actually exist in the dataset. This is the class set the YOLO-World
        detector is scored against."""
        seen: list[str] = []
        for prompt in self.safety_prompts:
            cls = self.prompt_to_class.get(prompt)
            if cls and cls in self.dataset_classes and cls not in seen:
                seen.append(cls)
        return seen

    def class_for_prompt(self, prompt: str) -> Optional[str]:
        cls = self.prompt_to_class.get(prompt)
        if cls and cls in self.dataset_classes:
            return cls
        return None

    def _active_threshold_map(self) -> dict[str, float]:
        """The per-class threshold map for the active backend. Each backend is
        tuned separately (different confidence distributions), so the zero-shot and
        fine-tuned models use distinct maps; an empty map means 'global threshold'."""
        if (self.detector_backend or "yolo_world").lower() == "finetuned":
            return self.finetuned_per_class_thresholds
        return self.per_class_thresholds

    def threshold_for(self, class_name: Optional[str]) -> float:
        """Operating-point confidence threshold for a class: the per-class override
        for the active backend (produced by `tune.py`) if present, else the global
        `confidence_threshold`."""
        m = self._active_threshold_map()
        if m and class_name and class_name in m:
            return float(m[class_name])
        return self.confidence_threshold

    @property
    def uses_per_class_thresholds(self) -> bool:
        return bool(self._active_threshold_map())

    # --- device --------------------------------------------------------------
    @property
    def device(self) -> str:
        return resolve_device(self.device_pref)

    def runs_pipeline(self, name: str) -> bool:
        return name in self.pipelines


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


def _read_dataset_classes(data_yaml_path: str) -> list[str]:
    """Read class names from a YOLO data.yaml. Supports both list and dict forms."""
    data = _read_yaml(data_yaml_path)
    names = data.get("names")
    if names is None:
        return []
    if isinstance(names, dict):
        # {0: 'a', 1: 'b'} -> ['a', 'b'] ordered by key
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load config.yaml merged over DEFAULTS, then read dataset class names.

    Reading data.yaml is best-effort: if the dataset is missing, `dataset_classes`
    is left empty and validation (run.py --check) reports it clearly rather than
    crashing here.
    """
    raw: dict[str, Any] = dict(DEFAULTS)
    if os.path.exists(config_path):
        file_cfg = _read_yaml(config_path)
        raw.update({k: v for k, v in file_cfg.items() if v is not None})

    cfg = Config(
        config_path=config_path,
        dataset_dir=raw["dataset_dir"],
        eval_split=raw["eval_split"],
        sample_for_review=int(raw["sample_for_review"]),
        confidence_threshold=float(raw["confidence_threshold"]),
        yolo_conf_floor=float(raw["yolo_conf_floor"]),
        iou_threshold=float(raw["iou_threshold"]),
        pipelines=list(raw["pipelines"]),
        yolo_model=raw["yolo_model"],
        detector_backend=str(raw["detector_backend"]),
        finetuned_model=str(raw["finetuned_model"]),
        tta=bool(raw["tta"]),
        finetune_base_model=str(raw["finetune_base_model"]),
        finetune_epochs=int(raw["finetune_epochs"]),
        finetune_imgsz=int(raw["finetune_imgsz"]),
        finetune_batch=int(raw["finetune_batch"]),
        finetune_patience=int(raw["finetune_patience"]),
        device_pref=raw["device"],
        ollama_host=raw["ollama_host"].rstrip("/"),
        ollama_model=raw["ollama_model"],
        ollama_timeout=int(raw["ollama_timeout"]),
        ollama_num_predict=int(raw["ollama_num_predict"]),
        api_provider=raw["api_provider"],
        api_model=raw["api_model"],
        api_max_tokens=int(raw["api_max_tokens"]),
        output_dir=raw["output_dir"],
        safety_prompts=list(raw["safety_prompts"]),
        prompt_to_class=dict(raw["prompt_to_class"]),
        per_class_thresholds={k: float(v) for k, v in dict(raw["per_class_thresholds"]).items()},
        finetuned_per_class_thresholds={k: float(v) for k, v in dict(raw["finetuned_per_class_thresholds"]).items()},
        vlm_eval_classes=list(raw["vlm_eval_classes"]),
        vlm_violation_keywords=dict(raw["vlm_violation_keywords"]),
        vlm_negative_keywords=list(raw["vlm_negative_keywords"]),
    )

    if os.path.exists(cfg.data_yaml_path):
        try:
            cfg.dataset_classes = _read_dataset_classes(cfg.data_yaml_path)
        except Exception:
            cfg.dataset_classes = []
    return cfg


def validate_config(cfg: Config) -> list[Issue]:
    """Structural validation of config + dataset. Used by run.py --check and as a
    pre-flight warning before a normal run."""
    issues: list[Issue] = []

    # dataset
    if not os.path.isdir(cfg.dataset_dir):
        issues.append(Issue("error", f"dataset_dir not found: {cfg.dataset_dir}"))
    if not os.path.isfile(cfg.data_yaml_path):
        issues.append(Issue("error", f"data.yaml not found: {cfg.data_yaml_path}"))
    elif not cfg.dataset_classes:
        issues.append(Issue("error", f"could not read class names from {cfg.data_yaml_path}"))
    else:
        issues.append(Issue("ok", f"dataset: {len(cfg.dataset_classes)} classes in {cfg.data_yaml_path}"))

    # eval split
    n_imgs = _count_files(cfg.images_dir)
    n_lbls = _count_files(cfg.labels_dir)
    if n_imgs == 0:
        issues.append(Issue("error", f"no images in eval split: {cfg.images_dir}"))
    else:
        issues.append(Issue("ok", f"eval split '{cfg.eval_split}': {n_imgs} images, {n_lbls} label files"))
    if n_imgs and n_lbls == 0:
        issues.append(Issue(
            "error",
            f"no label files in {cfg.labels_dir} — the headline metrics need "
            "ground-truth labels. Use `--images <folder>` for label-free ad-hoc runs.",
        ))

    # pipelines
    bad = [p for p in cfg.pipelines if p not in VALID_PIPELINES]
    if bad:
        issues.append(Issue("error", f"unknown pipeline(s) {bad}; valid: {sorted(VALID_PIPELINES)}"))
    if not cfg.pipelines:
        issues.append(Issue("error", "no pipelines selected"))

    # prompts
    if not cfg.safety_prompts:
        issues.append(Issue("error", "safety_prompts is empty"))

    # prompt->class mapping
    if cfg.dataset_classes:
        invalid = {
            p: c for p, c in cfg.prompt_to_class.items()
            if c is not None and c not in cfg.dataset_classes
        }
        for p, c in invalid.items():
            issues.append(Issue("warn", f"prompt '{p}' maps to unknown class '{c}' — it will not be scored"))
        scored = cfg.scored_classes
        if scored:
            issues.append(Issue("ok", f"YOLO-World scored against classes: {scored}"))
        else:
            issues.append(Issue("warn", "no prompts map to a valid dataset class — YOLO-World metrics will be empty"))

        # VLM eval classes
        bad_vlm = [c for c in cfg.vlm_eval_classes if c not in cfg.dataset_classes]
        if bad_vlm:
            issues.append(Issue("warn", f"vlm_eval_classes not in dataset: {bad_vlm}"))

    return issues


def _count_files(directory: str) -> int:
    if not os.path.isdir(directory):
        return 0
    return sum(1 for e in os.scandir(directory) if e.is_file())
