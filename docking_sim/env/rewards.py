from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class RewardContext:
    distance: float
    prev_distance: float
    speed: float
    heading_error: float
    fuel_used: float
    success: bool
    crash: bool
    timeout: bool


def _empty_components() -> dict[str, float]:
    return {
        "distance": 0.0,
        "velocity": 0.0,
        "rotation": 0.0,
        "fuel": 0.0,
        "time": 0.0,
        "terminal": 0.0,
    }


def safe_docking(ctx: RewardContext, weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    close_range = max(float(weights.get("close_range", 4.0)), 1e-6)
    proximity = float(np.clip(1.0 - ctx.distance / close_range, 0.0, 1.0))
    # Mostly a gate penalty. A small floor still asks the ship to point at the port while it is far away.
    near = 0.05 + 0.95 * proximity

    parts = _empty_components()
    # Closing in is rewarded. Sitting far away is not taxed just for the range.
    parts["distance"] = float(weights.get("distance", 1.0)) * (ctx.prev_distance - ctx.distance)
    parts["distance"] += float(weights.get("gate", 2.0)) * proximity
    parts["velocity"] = -float(weights.get("velocity", 0.4)) * ctx.speed * near
    parts["rotation"] = -float(weights.get("rotation", 0.35)) * ctx.heading_error * near
    parts["fuel"] = -float(weights.get("fuel", 0.02)) * ctx.fuel_used
    parts["time"] = -float(weights.get("time", 0.01))
    if ctx.success:
        parts["terminal"] = float(weights.get("success", 120.0))
    elif ctx.crash:
        parts["terminal"] = -(
            float(weights.get("crash", 80.0)) + float(weights.get("crash_speed", 15.0)) * ctx.speed
        )

    total = float(sum(parts.values()))
    return total, parts


def distance_only(ctx: RewardContext, weights: dict[str, float]) -> tuple[float, dict[str, float]]:
    parts = _empty_components()
    parts["distance"] = -float(weights.get("distance", 1.0)) * ctx.distance
    if ctx.success:
        parts["terminal"] = float(weights.get("success", 0.0))
    elif ctx.crash:
        parts["terminal"] = -float(weights.get("crash", 0.0))
    return float(sum(parts.values())), parts


REWARD_FNS: dict[str, Callable[[RewardContext, dict[str, float]], tuple[float, dict[str, float]]]] = {
    "safe_docking": safe_docking,
    "distance_only": distance_only,
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "distance": 1.0,
    "velocity": 0.4,
    "rotation": 0.35,
    "fuel": 0.02,
    "time": 0.01,
    "success": 120.0,
    "crash": 80.0,
    "crash_speed": 15.0,
    "gate": 2.0,
    "close_range": 4.0,
}


def make_reward(name: str, weights: dict[str, float] | None = None):
    if name not in REWARD_FNS:
        known = ", ".join(sorted(REWARD_FNS))
        raise ValueError(f"Unknown reward '{name}'. Expected one of: {known}")
    merged = {**DEFAULT_WEIGHTS, **(weights or {})}
    fn = REWARD_FNS[name]

    def _compute(ctx: RewardContext) -> tuple[float, dict[str, float]]:
        return fn(ctx, merged)

    return _compute
