from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from docking_sim.config import ARTIFACTS, env_kwargs_from_config, load_yaml
from docking_sim.env.docking_env import DockingEnv
from docking_sim.replay.rollout import (
    _vecnormalize_for_checkpoint,
    load_policy,
    load_run_meta,
    rollout,
)

EVAL_SEEDS = list(range(1000, 1020))


def eval_row(result: dict[str, Any], checkpoint: str, timesteps: int) -> dict[str, Any]:
    first = result["frames"][0]
    last = result["frames"][-1]
    heading_deg = float(last["heading_error"]) * 180.0 / float(np.pi)
    return {
        "checkpoint": checkpoint,
        "timesteps": int(timesteps),
        "seed": int(result["seed"]),
        "success": int(bool(result["success"])),
        "crash": int(bool(result["crash"])),
        "timeout": int(bool(result["timeout"])),
        "steps": int(result["steps"]),
        "fuel_used": float(first["fuel"]) - float(last["fuel"]),
        "final_speed": float(last["speed"]),
        "final_heading_deg": heading_deg,
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["timesteps"]), []).append(row)
    points = []
    for timesteps in sorted(grouped):
        group = grouped[timesteps]
        successes = [row for row in group if int(row["success"])]
        points.append(
            {
                "timesteps": timesteps,
                "checkpoint": group[-1]["checkpoint"],
                "success_rate": sum(int(row["success"]) for row in group) / len(group),
                "crash_rate": sum(int(row["crash"]) for row in group) / len(group),
                "median_steps": _median([float(row["steps"]) for row in successes]),
                "n": len(group),
            }
        )
    return points


def read_eval_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_eval_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def evaluate_checkpoint(
    run_id: str,
    checkpoint: str,
    *,
    timesteps: int,
    seeds: list[int] | None = None,
) -> list[dict[str, Any]]:
    run_dir = ARTIFACTS / run_id
    cfg = load_yaml(run_dir / "config.yaml")
    meta = load_run_meta(run_dir)
    algo_name = str((cfg.get("algo") or {}).get("name") or meta.get("algorithm") or "ppo")
    zip_path = run_dir / "checkpoints" / f"{checkpoint}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {zip_path}")
    model = load_policy(zip_path, algo_name)
    env_kwargs = env_kwargs_from_config(cfg)
    env_kwargs["randomize_start"] = True
    vecnorm: VecNormalize | None = None
    vn_path = _vecnormalize_for_checkpoint(run_dir, checkpoint)
    if vn_path is not None:
        dummy = DummyVecEnv([lambda: DockingEnv(**env_kwargs)])
        vecnorm = VecNormalize.load(str(vn_path), dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
    chosen = list(EVAL_SEEDS if seeds is None else seeds)
    rows = []
    try:
        for seed in chosen:
            result = rollout(
                run_id=run_id,
                checkpoint=checkpoint,
                seed=int(seed),
                randomize_start=True,
                model=model,
                vecnorm=vecnorm,
            )
            rows.append(eval_row(result, checkpoint, timesteps))
    finally:
        if vecnorm is not None:
            vecnorm.close()
    return rows
