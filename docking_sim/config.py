from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "runs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return data


def env_kwargs_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    env_cfg = dict(cfg.get("env") or {})
    weights = env_cfg.pop("weights", None)
    reward_name = env_cfg.pop("reward", "safe_docking")
    kwargs = dict(env_cfg)
    kwargs["reward_name"] = reward_name
    if weights is not None:
        kwargs["reward_weights"] = weights
    return kwargs
