from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from docking_sim.config import ARTIFACTS, env_kwargs_from_config, load_yaml
from docking_sim.env.docking_env import DockingEnv
from docking_sim.replay.benchmarks import (
    BENCHMARK_PROTOCOL_VERSION,
    compatibility,
    get_benchmark,
    scene_fingerprint,
)
from docking_sim.replay.rollout import (
    _vecnormalize_for_checkpoint,
    load_policy,
    load_run_meta,
    rollout,
)

EVAL_SEEDS = list(range(1000, 1020))
DEFAULT_DT = 0.05


def _terminal_reason(result: dict[str, Any], last: dict[str, Any]) -> str:
    explicit = str(last.get("terminal_reason") or "")
    if explicit:
        return explicit
    if bool(result.get("success")):
        return "docked"
    if bool(last.get("hit_asteroid")):
        return "asteroid"
    if bool(last.get("hit_hull")):
        return "hull"
    if bool(last.get("out_of_bounds")):
        return "bounds"
    if bool(last.get("out_of_fuel")):
        return "fuel"
    if bool(result.get("timeout")):
        return "timeout"
    return "unknown"


def eval_row(
    result: dict[str, Any],
    checkpoint: str,
    timesteps: int,
    *,
    benchmark_id: str | None = None,
    scenario: Any | None = None,
    scene_fingerprint_value: str | None = None,
    interface_fingerprint_value: str | None = None,
    training_seed: int | None = None,
) -> dict[str, Any]:
    first = result["frames"][0]
    last = result["frames"][-1]
    heading_deg = float(last["heading_error"]) * 180.0 / float(np.pi)
    row = {
        "checkpoint": checkpoint,
        "timesteps": int(timesteps),
        "seed": int(result["seed"]),
        "success": int(bool(result["success"])),
        "crash": int(bool(result["crash"])),
        "timeout": int(bool(result["timeout"])),
        "steps": int(result["steps"]),
        "fuel_used": float(first["fuel"]) - float(last["fuel"]),
        "time": float(result["steps"]) * float(result.get("dt", DEFAULT_DT)),
        "final_speed": float(last["speed"]),
        "final_heading_deg": heading_deg,
        "final_distance": float(last.get("distance", 0.0)),
        "terminal_reason": _terminal_reason(result, last),
    }
    if benchmark_id is not None:
        row.update(
            {
                "protocol_version": BENCHMARK_PROTOCOL_VERSION,
                "benchmark_id": benchmark_id,
                "scenario_id": getattr(scenario, "id", result.get("scenario_id")),
                "scenario_hash": getattr(scenario, "scenario_hash", result.get("scenario_hash")),
                "stratum": getattr(scenario, "stratum", None),
                "scene_fingerprint": scene_fingerprint_value,
                "interface_fingerprint": interface_fingerprint_value,
                "training_seed": training_seed,
            }
        )
    return row


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return float(0.5 * (ordered[mid - 1] + ordered[mid]))


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    spread = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denominator
    return float(max(0.0, center - spread)), float(min(1.0, center + spread))


def _row_time(row: dict[str, Any]) -> float | None:
    if row.get("time") is not None:
        return float(row["time"])
    if row.get("steps") is None:
        return None
    return float(row["steps"]) * DEFAULT_DT


def _stratum_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("stratum") or "unstratified")
        grouped.setdefault(key, []).append(row)
    return [
        {
            "stratum": stratum,
            "n": len(items),
            "success_rate": sum(int(item["success"]) for item in items) / len(items),
            "crash_rate": sum(int(item["crash"]) for item in items) / len(items),
        }
        for stratum, items in sorted(grouped.items())
    ]


def _failure_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("terminal_reason") or "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def summarize_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in group if int(row["success"])]
    fuel = [float(row["fuel_used"]) for row in group if row.get("fuel_used") is not None]
    times = [value for value in (_row_time(row) for row in group) if value is not None]
    success_fuel = [
        float(row["fuel_used"])
        for row in successes
        if row.get("fuel_used") is not None
    ]
    success_time = [value for value in (_row_time(row) for row in successes) if value is not None]
    successes_count = len(successes)
    ci_low, ci_high = _wilson_interval(successes_count, len(group))
    return {
        "timesteps": int(group[-1]["timesteps"]),
        "checkpoint": group[-1]["checkpoint"],
        "benchmark_id": group[-1].get("benchmark_id"),
        "protocol_version": group[-1].get("protocol_version"),
        "success_rate": successes_count / len(group),
        "crash_rate": sum(int(row["crash"]) for row in group) / len(group),
        "median_steps": _median([float(row["steps"]) for row in successes]),
        "mean_fuel": _mean(fuel),
        "mean_time": _mean(times),
        "mean_fuel_success": _mean(success_fuel),
        "mean_time_success": _mean(success_time),
        "success_ci_low": ci_low,
        "success_ci_high": ci_high,
        "failure_reasons": _failure_breakdown(group),
        "strata": _stratum_summary(group),
        "n": len(group),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["timesteps"]), []).append(row)
    return [summarize_group(grouped[timesteps]) for timesteps in sorted(grouped)]


def latest_outcomes(
    rows: list[dict[str, Any]],
    *,
    benchmark_id: str | None = None,
    checkpoint: str | None = None,
) -> list[dict[str, Any]]:
    if benchmark_id is not None:
        rows = [row for row in rows if row.get("benchmark_id") == benchmark_id]
    if checkpoint is not None:
        rows = [row for row in rows if row.get("checkpoint") == checkpoint]
    if not rows:
        return []
    latest = max(int(row["timesteps"]) for row in rows)
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if int(row["timesteps"]) != latest:
            continue
        key = str(row.get("scenario_id") or f"legacy-seed-{row['seed']}")
        chosen[key] = row
    outcomes = []
    for scenario_key in sorted(chosen):
        row = chosen[scenario_key]
        outcomes.append(
            {
                "seed": int(row.get("seed", -1)),
                "scenario_id": row.get("scenario_id"),
                "scenario_hash": row.get("scenario_hash"),
                "stratum": row.get("stratum"),
                "success": bool(int(row["success"])),
                "crash": bool(int(row["crash"])),
                "fuel_used": None if row.get("fuel_used") is None else float(row["fuel_used"]),
                "time": _row_time(row),
                "terminal_reason": str(row.get("terminal_reason") or "unknown"),
            }
        )
    return outcomes


def paired_summary(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    benchmark_id: str,
    left_checkpoint: str | None = None,
    right_checkpoint: str | None = None,
) -> dict[str, Any]:
    """Summarize only scenario IDs scored by both candidates."""

    def indexed(rows: list[dict[str, Any]], checkpoint: str | None) -> dict[str, dict[str, Any]]:
        filtered = [row for row in rows if row.get("benchmark_id") == benchmark_id]
        if checkpoint is not None:
            filtered = [row for row in filtered if row.get("checkpoint") == checkpoint]
        if not filtered:
            return {}
        latest = max(int(row["timesteps"]) for row in filtered)
        return {
            str(row["scenario_id"]): row
            for row in filtered
            if int(row["timesteps"]) == latest and row.get("scenario_id")
        }

    left = indexed(left_rows, left_checkpoint)
    right = indexed(right_rows, right_checkpoint)
    common_ids = sorted(set(left).intersection(right))
    outcome_counts = {"both_success": 0, "left_only": 0, "right_only": 0, "neither": 0}
    fuel_deltas: list[float] = []
    time_deltas: list[float] = []
    paired = []
    for scenario_id in common_ids:
        left_row = left[scenario_id]
        right_row = right[scenario_id]
        left_success = bool(int(left_row["success"]))
        right_success = bool(int(right_row["success"]))
        if left_success and right_success:
            outcome_counts["both_success"] += 1
        elif left_success:
            outcome_counts["left_only"] += 1
        elif right_success:
            outcome_counts["right_only"] += 1
        else:
            outcome_counts["neither"] += 1
        if left_row.get("fuel_used") is not None and right_row.get("fuel_used") is not None:
            fuel_deltas.append(float(left_row["fuel_used"]) - float(right_row["fuel_used"]))
        left_time = _row_time(left_row)
        right_time = _row_time(right_row)
        if left_time is not None and right_time is not None:
            time_deltas.append(left_time - right_time)
        paired.append(
            {
                "scenario_id": scenario_id,
                "stratum": left_row.get("stratum") or right_row.get("stratum"),
                "left_success": left_success,
                "right_success": right_success,
                "left_terminal_reason": left_row.get("terminal_reason"),
                "right_terminal_reason": right_row.get("terminal_reason"),
            }
        )
    return {
        "benchmark_id": benchmark_id,
        "n_common": len(common_ids),
        "outcomes": outcome_counts,
        "mean_fuel_delta_left_minus_right": _mean(fuel_deltas),
        "mean_time_delta_left_minus_right": _mean(time_deltas),
        "scenarios": paired,
    }


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
    benchmark_id: str = "quick-20",
) -> list[dict[str, Any]]:
    run_dir = ARTIFACTS / run_id
    cfg = load_yaml(run_dir / "config.yaml")
    meta = load_run_meta(run_dir)
    algo_name = str((cfg.get("algo") or {}).get("name") or meta.get("algorithm") or "ppo")
    suite = get_benchmark(benchmark_id)
    candidate_env_kwargs = env_kwargs_from_config(cfg)
    contract = compatibility(candidate_env_kwargs, benchmark_id)
    if not contract["compatible"]:
        raise ValueError(
            f"Run '{run_id}' is incompatible with {benchmark_id}: {contract['reason']}"
        )
    zip_path = run_dir / "checkpoints" / f"{checkpoint}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {zip_path}")
    model = load_policy(zip_path, algo_name)
    env_kwargs = dict(suite.env_kwargs)
    vecnorm: VecNormalize | None = None
    vn_path = _vecnormalize_for_checkpoint(run_dir, checkpoint)
    if vn_path is not None:
        dummy = DummyVecEnv([lambda: DockingEnv(**env_kwargs)])
        vecnorm = VecNormalize.load(str(vn_path), dummy)
        vecnorm.training = False
        vecnorm.norm_reward = False
    if seeds is None:
        scenarios = list(suite.scenarios)
    else:
        wanted = {int(seed) for seed in seeds}
        scenarios = [scenario for scenario in suite.scenarios if scenario.seed in wanted]
    rows = []
    try:
        for scenario in scenarios:
            result = rollout(
                run_id=run_id,
                checkpoint=checkpoint,
                seed=int(scenario.seed),
                scenario=scenario,
                env_kwargs_override=env_kwargs,
                model=model,
                vecnorm=vecnorm,
            )
            rows.append(
                eval_row(
                    result,
                    checkpoint,
                    timesteps,
                    benchmark_id=benchmark_id,
                    scenario=scenario,
                    scene_fingerprint_value=contract["benchmark_scene_fingerprint"],
                    interface_fingerprint_value=contract["benchmark_interface_fingerprint"],
                    training_seed=int(meta.get("seed", -1)),
                )
            )
    finally:
        if vecnorm is not None:
            vecnorm.close()
    return rows
