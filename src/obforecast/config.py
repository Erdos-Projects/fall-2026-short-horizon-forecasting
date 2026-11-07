"""Run configuration: YAML files with optional ``extends`` inheritance and derived timing."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config. ``extends: other.yaml`` is resolved relative to the file."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _deep_merge(load_config(path.parent / parent), cfg)
    cfg["_source"] = str(path)
    return cfg


def config_hash(cfg: dict) -> str:
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, default=str).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Timing:
    """Window timing. ``cut`` is the first second that is NOT a feature input."""

    feature_last_second: int = 299
    decision_second: int = 299
    entry_second: int = 300
    exit_second: int = 599
    entry_delays: tuple = (0, 2, 5, 10)
    stale_quote_seconds: int = 30
    grid: int = 600

    def __post_init__(self):
        if not (0 < self.feature_last_second < self.entry_second <= self.exit_second < self.grid):
            raise ValueError("timing must satisfy 0 < feature_last < entry <= exit < 600")
        if self.decision_second != self.feature_last_second:
            raise ValueError("the decision is taken right after the last feature second")
        if self.entry_second + max(self.entry_delays) > self.exit_second:
            raise ValueError("entry delays must leave the entry before the exit")

    @property
    def cut(self) -> int:
        return self.feature_last_second + 1

    @classmethod
    def from_config(cls, cfg: dict) -> "Timing":
        t = cfg.get("timing", {})
        return cls(
            feature_last_second=int(t.get("feature_last_second", 299)),
            decision_second=int(t.get("decision_second", 299)),
            entry_second=int(t.get("entry_second", 300)),
            exit_second=int(t.get("exit_second", 599)),
            entry_delays=tuple(int(d) for d in t.get("entry_delays", (0, 2, 5, 10))),
            stale_quote_seconds=int(t.get("stale_quote_seconds", 30)),
        )


def taus(cfg: dict) -> list[float]:
    """Spread-filter thresholds tau = log(ratio); ``null`` means no filter (inf)."""
    return [float("inf") if r is None else float(math.log(r)) for r in cfg["trading"]["tau_ratios"]]
