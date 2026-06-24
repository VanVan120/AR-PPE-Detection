"""VLM pipeline: image -> structured safety observations.

Two interchangeable backends behind one contract (`BaseVLM.observe`):
  * `OllamaVLM`   — local vision model via Ollama's HTTP API (default).
  * `AnthropicVLM`— Claude vision via the Anthropic API (behind `--api`).

Both return a `VlmResult` holding a list of `{type, description, severity}`
observations plus the raw model text. Parsing is defensive (code fences / stray
prose are stripped) because local models don't always honour the format.
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import requests

from .config import Config

SEVERITIES = ("low", "medium", "high")

PROMPT = (
    "You are a construction-site safety inspector reviewing a single still image.\n"
    "Identify safety observations, focusing on PPE — hard hats, high-visibility "
    "safety vests, and masks — plus obvious hazards (a person near an unprotected "
    "edge, exposed rebar).\n"
    "Respond with ONLY a JSON object, no prose, of this exact form:\n"
    '{"observations": [{"type": "<short_snake_case_category>", '
    '"description": "<one short sentence>", "severity": "low|medium|high"}]}\n'
    "Rules:\n"
    "- Use clear types such as: no_hardhat, no_safety_vest, no_mask, "
    "person_detected, unprotected_edge, exposed_rebar.\n"
    "- Phrase a missing-PPE finding explicitly, e.g. type 'no_hardhat', "
    "description 'A worker is not wearing a hard hat.'\n"
    "- severity must be exactly one of: low, medium, high.\n"
    '- If there are no notable safety issues, return {"observations": []}.'
)

# JSON schema used to constrain the Anthropic API response.
_API_SCHEMA = {
    "type": "object",
    "properties": {
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "description": {"type": "string"},
                    "severity": {"type": "string", "enum": list(SEVERITIES)},
                },
                "required": ["type", "description", "severity"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["observations"],
    "additionalProperties": False,
}


@dataclass
class Observation:
    type: str
    description: str
    severity: str   # low | medium | high

    def to_dict(self) -> dict:
        return {"type": self.type, "description": self.description, "severity": self.severity}


@dataclass
class VlmResult:
    observations: list[Observation] = field(default_factory=list)
    raw: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "observations": [o.to_dict() for o in self.observations],
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------
def encode_jpeg_b64(image_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("failed to JPEG-encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Defensive parsing
# ---------------------------------------------------------------------------
def _has_word(text: str, words: tuple[str, ...]) -> bool:
    # whole-word match so "med" doesn't fire on "immediate", "info" on "uninformative"
    return any(re.search(rf"(?<![a-z]){w}(?![a-z])", text) for w in words)


def normalize_severity(value: object) -> str:
    s = str(value or "").strip().lower()
    if _has_word(s, ("high", "critical", "severe", "danger", "dangerous")):
        return "high"
    if _has_word(s, ("medium", "moderate", "med")):
        return "medium"
    if _has_word(s, ("low", "minor", "info", "informational", "negligible")):
        return "low"
    return "medium"


def _coerce_observation(item: object) -> Optional[Observation]:
    if not isinstance(item, dict):
        return None
    typ = str(item.get("type") or item.get("category") or item.get("label") or "observation").strip()
    desc = str(item.get("description") or item.get("desc") or item.get("text") or "").strip()
    sev = normalize_severity(item.get("severity") or item.get("risk") or item.get("level"))
    if not desc and typ == "observation":
        return None
    return Observation(type=typ or "observation", description=desc, severity=sev)


def parse_observations(text: str) -> tuple[list[Observation], Optional[str]]:
    """Parse model text into observations. Returns (observations, error)."""
    if not text or not text.strip():
        return [], "empty response"
    cleaned = _strip_fences(text)
    payload = _load_json_loose(cleaned)
    if payload is None:
        return [], "could not parse JSON from response"

    items: list = []
    if isinstance(payload, dict):
        if isinstance(payload.get("observations"), list):
            items = payload["observations"]
        elif "type" in payload or "description" in payload:
            items = [payload]   # a single observation object
        else:
            # dict of named observations -> take its list-ish values
            for v in payload.values():
                if isinstance(v, list):
                    items.extend(v)
    elif isinstance(payload, list):
        items = payload

    observations = [o for o in (_coerce_observation(it) for it in items) if o]
    return observations, None


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _load_json_loose(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the first *balanced* JSON value embedded in surrounding prose
    # (brace/bracket counting that respects strings), not the outermost span —
    # so a valid object followed by prose containing a stray '}' still parses.
    span = _first_balanced_json(text)
    if span is not None:
        try:
            return json.loads(span)
        except Exception:
            pass
    return None


def _first_balanced_json(text: str) -> Optional[str]:
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        closer = "}" if ch == "{" else "]"
        depth = 0
        in_str = False
        esc = False
        for j in range(i, len(text)):
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == ch:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[i:j + 1]
        return None  # opener never balanced
    return None


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class BaseVLM:
    name = "vlm"

    def observe(self, image_bgr: np.ndarray) -> VlmResult:  # pragma: no cover - interface
        raise NotImplementedError

    def health_check(self) -> tuple[bool, str]:  # pragma: no cover - interface
        raise NotImplementedError


class OllamaVLM(BaseVLM):
    name = "ollama"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.host = os.environ.get("OLLAMA_HOST", cfg.ollama_host).rstrip("/")
        self.model = cfg.ollama_model

    def observe(self, image_bgr: np.ndarray) -> VlmResult:
        try:
            b64 = encode_jpeg_b64(image_bgr)
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": PROMPT,
                    "images": [b64],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": self.cfg.ollama_num_predict},
                },
                timeout=self.cfg.ollama_timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        except requests.RequestException as e:
            return VlmResult(raw="", error=f"ollama request failed: {e}")
        except Exception as e:
            return VlmResult(raw="", error=f"ollama error: {e}")
        obs, err = parse_observations(raw)
        return VlmResult(observations=obs, raw=raw, error=err)

    def health_check(self) -> tuple[bool, str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
            names = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as e:
            return False, f"Ollama not reachable at {self.host} ({e})"
        if not names:
            return False, f"Ollama reachable at {self.host} but no models pulled"
        if self.model not in names and not any(n.split(":")[0] == self.model.split(":")[0] for n in names):
            return False, f"vision model '{self.model}' not found in Ollama (have: {names})"
        return True, f"Ollama OK at {self.host}; using '{self.model}'"


class AnthropicVLM(BaseVLM):
    name = "anthropic"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        import anthropic  # imported lazily; only needed with --api
        self._client = anthropic.Anthropic()

    def observe(self, image_bgr: np.ndarray) -> VlmResult:
        try:
            b64 = encode_jpeg_b64(image_bgr)
            resp = self._client.messages.create(
                model=self.cfg.api_model,
                max_tokens=self.cfg.api_max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64",
                                                     "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": PROMPT},
                    ],
                }],
                output_config={"format": {"type": "json_schema", "schema": _API_SCHEMA}},
            )
            raw = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), "")
        except Exception as e:
            return VlmResult(raw="", error=f"anthropic error: {e}")
        obs, err = parse_observations(raw)
        return VlmResult(observations=obs, raw=raw, error=err)

    def health_check(self) -> tuple[bool, str]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY is not set (required for --api)"
        try:
            import anthropic  # noqa: F401
        except Exception:
            return False, "anthropic package not installed (pip install anthropic)"
        return True, f"Anthropic API path ready; model '{self.cfg.api_model}'"


def build_vlm(cfg: Config, use_api: bool) -> BaseVLM:
    """Construct the VLM backend. `--api` selects Anthropic; otherwise Ollama."""
    if use_api:
        if cfg.api_provider != "anthropic":
            raise ValueError(f"unsupported api_provider '{cfg.api_provider}' (only 'anthropic' implemented)")
        return AnthropicVLM(cfg)
    return OllamaVLM(cfg)


def save_vlm_output(path: str, result: VlmResult) -> None:
    """Persist raw + parsed VLM output for one image."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "observations": [o.to_dict() for o in result.observations],
            "error": result.error,
            "raw": result.raw,
        }, fh, indent=2)


def load_vlm_output(path: str) -> VlmResult:
    """Reconstruct a VlmResult from a previously saved VLM output file.

    Lets a re-run reuse cached VLM responses (`--reuse-vlm`) — useful when only the
    detection side changed (e.g. tuned thresholds, a fine-tuned model), so the
    expensive VLM inference need not be repeated.
    """
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    obs = [
        Observation(
            type=str(o.get("type", "observation")),
            description=str(o.get("description", "")),
            severity=normalize_severity(o.get("severity")),
        )
        for o in data.get("observations", []) if isinstance(o, dict)
    ]
    return VlmResult(observations=obs, raw=data.get("raw", ""), error=data.get("error"))
