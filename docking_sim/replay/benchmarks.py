"""Versioned, deterministic mission suites for fair policy comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any

import numpy as np

from docking_sim.env.physics import ShipState, StationPose, WorldConfig


BENCHMARK_PROTOCOL_VERSION = "1.0"

# This is deliberately a scene contract, not a training configuration. Reward
# weights are intentionally excluded so different objectives can be evaluated
# on the same physical mission.
STATIC_BENCHMARK_ENV: dict[str, Any] = {
    "max_steps": 800,
    "dt": 0.05,
    "dock_speed_max": 0.35,
    "dock_angle_max_deg": 12.0,
    "fuel_capacity": 100.0,
    "randomize_start": False,
    "randomize_velocity": False,
    "obs_mode": "state",
    "extended_obs": False,
    "fail_on_empty_fuel": False,
}


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def _float(value: float) -> float:
    """Round generated scenarios so their serialized form is stable and readable."""
    return float(round(float(value), 8))


@dataclass(frozen=True)
class BenchmarkScenario:
    id: str
    benchmark_id: str
    protocol_version: str
    seed: int
    stratum: str
    initial_state: dict[str, float]
    station_pose: dict[str, float]
    asteroids: tuple[tuple[float, float, float], ...] = ()
    disturbance_trace: tuple[tuple[float, float], ...] = ()

    @property
    def scenario_hash(self) -> str:
        mission = self.to_dict(include_hash=False)
        # A quick suite is a strict subset of the canonical suite. Its shared
        # missions must retain the same identity even though suite membership
        # differs.
        mission.pop("benchmark_id", None)
        return stable_hash(mission)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "benchmark_id": self.benchmark_id,
            "protocol_version": self.protocol_version,
            "seed": self.seed,
            "stratum": self.stratum,
            "initial_state": dict(self.initial_state),
            "station_pose": dict(self.station_pose),
            "asteroids": [
                {"x": x, "y": y, "r": radius}
                for x, y, radius in self.asteroids
            ],
            "disturbance_trace": [
                {"x": x, "y": y}
                for x, y in self.disturbance_trace
            ],
        }
        if include_hash:
            data["scenario_hash"] = self.scenario_hash
        return data

    def reset_options(self) -> dict[str, Any]:
        return {
            "state": ShipState(**self.initial_state),
            "station_pose": StationPose(**self.station_pose),
            "asteroids": list(self.asteroids),
            "disturbance_trace": list(self.disturbance_trace),
            "scenario_id": self.id,
            "scenario_hash": self.scenario_hash,
        }


@dataclass(frozen=True)
class BenchmarkSuite:
    id: str
    label: str
    description: str
    protocol_version: str
    scenarios: tuple[BenchmarkScenario, ...]
    env_kwargs: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return scene_fingerprint(self.env_kwargs)

    def to_dict(self, *, include_scenarios: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "protocol_version": self.protocol_version,
            "n_scenarios": len(self.scenarios),
            "scene_fingerprint": self.fingerprint,
            "interface_fingerprint": interface_fingerprint(self.env_kwargs),
            "strata": sorted({scenario.stratum for scenario in self.scenarios}),
        }
        if include_scenarios:
            data["scenarios"] = [scenario.to_dict() for scenario in self.scenarios]
        return data


def _benchmark_state(index: int) -> tuple[str, dict[str, float]]:
    """Return a deterministic, stratum-labelled mission state.

    All scenarios use the same static physical world and observation/action
    contract. The strata make the 100-mission suite useful for reporting where
    a policy succeeds instead of only reporting one pooled percentage.
    """

    rng = np.random.default_rng(1000 + index)
    kind = index % 4
    if kind == 0:
        stratum = "straight_in"
        x = rng.uniform(-1.75, 1.75)
        y = rng.uniform(-6.5, -4.25)
        theta = np.pi / 2.0 + rng.uniform(-0.45, 0.45)
        vx = 0.0
        vy = 0.0
        omega = 0.0
    elif kind == 1:
        stratum = "lateral_offset"
        x = rng.choice((-1.0, 1.0)) * rng.uniform(3.5, 6.0)
        y = rng.uniform(-6.5, -2.5)
        theta = rng.uniform(-np.pi, np.pi)
        vx = 0.0
        vy = 0.0
        omega = 0.0
    elif kind == 2:
        stratum = "drift"
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(-6.5, -2.0)
        theta = rng.uniform(-np.pi, np.pi)
        vx = rng.uniform(-0.45, 0.45)
        vy = rng.uniform(-0.45, 0.45)
        omega = rng.uniform(-0.25, 0.25)
    else:
        stratum = "precision"
        x = rng.uniform(-2.25, 2.25)
        y = rng.uniform(0.0, 2.5)
        theta = np.pi / 2.0 + rng.uniform(-1.0, 1.0)
        vx = rng.uniform(-0.2, 0.2)
        vy = rng.uniform(-0.2, 0.2)
        omega = rng.uniform(-0.15, 0.15)
    return stratum, {
        "x": _float(x),
        "y": _float(y),
        "vx": _float(vx),
        "vy": _float(vy),
        "theta": _float(theta),
        "omega": _float(omega),
        "fuel": 100.0,
    }


def _scenario(index: int) -> BenchmarkScenario:
    stratum, state = _benchmark_state(index)
    return BenchmarkScenario(
        id=f"static-v1-{index:03d}",
        benchmark_id="canonical-100",
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        seed=1000 + index,
        stratum=stratum,
        initial_state=state,
        station_pose=asdict(StationPose()),
    )


_CANONICAL_SCENARIOS = tuple(_scenario(index) for index in range(100))
_QUICK_SCENARIOS = tuple(
    BenchmarkScenario(
        id=scenario.id,
        benchmark_id="quick-20",
        protocol_version=scenario.protocol_version,
        seed=scenario.seed,
        stratum=scenario.stratum,
        initial_state=scenario.initial_state,
        station_pose=scenario.station_pose,
        asteroids=scenario.asteroids,
        disturbance_trace=scenario.disturbance_trace,
    )
    for scenario in _CANONICAL_SCENARIOS[:20]
)

BENCHMARKS: dict[str, BenchmarkSuite] = {
    "quick-20": BenchmarkSuite(
        id="quick-20",
        label="Quick 20",
        description="Fast, deterministic static-station comparison set for iteration.",
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        scenarios=_QUICK_SCENARIOS,
        env_kwargs=dict(STATIC_BENCHMARK_ENV),
    ),
    "canonical-100": BenchmarkSuite(
        id="canonical-100",
        label="Canonical 100",
        description="Portfolio comparison suite: 100 shared missions across four approach strata.",
        protocol_version=BENCHMARK_PROTOCOL_VERSION,
        scenarios=_CANONICAL_SCENARIOS,
        env_kwargs=dict(STATIC_BENCHMARK_ENV),
    ),
}


SHOWCASE_PRESETS: tuple[dict[str, Any], ...] = (
    {
        "id": "straight-in",
        "label": "Straight-in",
        "description": "Nearly aligned approach from below the port.",
        "benchmark_id": "quick-20",
        "scenario_id": "static-v1-000",
    },
    {
        "id": "lateral-offset",
        "label": "Lateral offset",
        "description": "Wide sideways start that forces a correction before capture.",
        "benchmark_id": "quick-20",
        "scenario_id": "static-v1-001",
    },
    {
        "id": "drifting",
        "label": "Drifting arrival",
        "description": "Initial linear and angular rate that must be cancelled.",
        "benchmark_id": "quick-20",
        "scenario_id": "static-v1-002",
    },
    {
        "id": "precision",
        "label": "Close-range precision",
        "description": "Already near the hull; heading and relative speed decide the outcome.",
        "benchmark_id": "quick-20",
        "scenario_id": "static-v1-003",
    },
)


def list_presets() -> list[dict[str, Any]]:
    presets = []
    for preset in SHOWCASE_PRESETS:
        scenario = get_scenario(preset["benchmark_id"], preset["scenario_id"])
        presets.append({**preset, "stratum": scenario.stratum, "scenario_hash": scenario.scenario_hash})
    return presets


def get_preset(preset_id: str) -> dict[str, Any]:
    for preset in list_presets():
        if preset["id"] == preset_id:
            return preset
    known = ", ".join(item["id"] for item in SHOWCASE_PRESETS)
    raise KeyError(f"Unknown preset '{preset_id}'. Expected one of: {known}")


def list_benchmarks() -> list[dict[str, Any]]:
    return [suite.to_dict() for suite in BENCHMARKS.values()]


def get_benchmark(benchmark_id: str) -> BenchmarkSuite:
    try:
        return BENCHMARKS[benchmark_id]
    except KeyError as exc:
        known = ", ".join(BENCHMARKS)
        raise KeyError(f"Unknown benchmark '{benchmark_id}'. Expected one of: {known}") from exc


def get_scenario(benchmark_id: str, scenario_id: str) -> BenchmarkScenario:
    suite = get_benchmark(benchmark_id)
    for scenario in suite.scenarios:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"Scenario '{scenario_id}' is not in benchmark '{benchmark_id}'.")


def _world_fields(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    defaults = WorldConfig()
    fields = {
        name: env_kwargs.get(name, getattr(defaults, name))
        for name in WorldConfig.__dataclass_fields__
    }
    fields["dt"] = float(env_kwargs.get("dt", defaults.dt))
    fields["fuel_capacity"] = float(env_kwargs.get("fuel_capacity", defaults.fuel_capacity))
    fields["dock_speed_max"] = float(env_kwargs.get("dock_speed_max", defaults.dock_speed_max))
    dock_angle_max_deg = float(env_kwargs.get("dock_angle_max_deg", 12.0))
    fields["dock_angle_max"] = float(np.deg2rad(dock_angle_max_deg))
    fields["dock_angle_max_deg"] = dock_angle_max_deg
    return fields


def scene_contract(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "max_steps": int(env_kwargs.get("max_steps", 800)),
        "fail_on_empty_fuel": bool(env_kwargs.get("fail_on_empty_fuel", False)),
        "world": _world_fields(env_kwargs),
    }


def interface_contract(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    world = _world_fields(env_kwargs)
    extended = env_kwargs.get("extended_obs")
    if extended is None:
        extended = bool(
            world["station_omega"]
            or world["station_vx"]
            or world["station_vy"]
            or world["disturbance_std"]
            or world["n_asteroids"]
        )
    obs_mode = str(env_kwargs.get("obs_mode", "state"))
    return {
        "obs_mode": obs_mode,
        "state_dim": 17 if obs_mode == "state" and extended else (9 if obs_mode == "state" else None),
        "image_shape": [128, 160, 3] if obs_mode == "pixels" else None,
        "action_shape": [3],
    }


def scene_fingerprint(env_kwargs: dict[str, Any]) -> str:
    return stable_hash(scene_contract(env_kwargs))


def interface_fingerprint(env_kwargs: dict[str, Any]) -> str:
    return stable_hash(interface_contract(env_kwargs))


def compatibility(
    candidate_env_kwargs: dict[str, Any],
    benchmark_id: str,
) -> dict[str, Any]:
    suite = get_benchmark(benchmark_id)
    candidate_scene = scene_fingerprint(candidate_env_kwargs)
    candidate_interface = interface_fingerprint(candidate_env_kwargs)
    scene_matches = candidate_scene == suite.fingerprint
    interface_matches = candidate_interface == interface_fingerprint(suite.env_kwargs)
    compatible = scene_matches and interface_matches
    reasons = []
    if not scene_matches:
        reasons.append("physical scene contract differs")
    if not interface_matches:
        reasons.append("observation or action contract differs")
    return {
        "benchmark_id": benchmark_id,
        "compatible": compatible,
        "reason": None if compatible else "; ".join(reasons),
        "candidate_scene_fingerprint": candidate_scene,
        "benchmark_scene_fingerprint": suite.fingerprint,
        "candidate_interface_fingerprint": candidate_interface,
        "benchmark_interface_fingerprint": interface_fingerprint(suite.env_kwargs),
    }
