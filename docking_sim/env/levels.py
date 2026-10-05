"""Playable curriculum levels. These configs are not started automatically."""

from __future__ import annotations

from typing import Any

from docking_sim.config import ROOT, env_kwargs_from_config, load_yaml


LEVEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": 1,
        "label": "Close static",
        "summary": "Short approach to a fixed port. Stop, face the approach, and kill the spin.",
        "config": "configs/ppo_64.yaml",
    },
    {
        "id": 2,
        "label": "Random spawn",
        "summary": "The ship starts somewhere in the lower field.",
        "config": "configs/ppo_level2_spawn.yaml",
    },
    {
        "id": 3,
        "label": "Drift",
        "summary": "The ship already has linear and angular rate.",
        "config": "configs/ppo_level3_velocity.yaml",
    },
    {
        "id": 4,
        "label": "Rotating station",
        "summary": "The port turns, so the approach heading moves.",
        "config": "configs/ppo_level4_rotate.yaml",
    },
    {
        "id": 5,
        "label": "Wind",
        "summary": "A persistent disturbance pushes the ship off the line.",
        "config": "configs/ppo_level5_wind.yaml",
    },
    {
        "id": 6,
        "label": "Tight fuel",
        "summary": "The tanks are small, and running dry ends the mission.",
        "config": "configs/ppo_level6_fuel.yaml",
    },
    {
        "id": 7,
        "label": "Asteroid field",
        "summary": "Rocks are scattered through the field.",
        "config": "configs/ppo_level7_asteroids.yaml",
    },
    {
        "id": 8,
        "label": "Moving port",
        "summary": "The station translates, so capture speed is relative to the port.",
        "config": "configs/ppo_level8_moving.yaml",
    },
    {
        "id": 9,
        "label": "Long approach",
        "summary": "The ship starts in the far corner and has a longer flight.",
        "config": "configs/ppo_level9_long.yaml",
    },
    {
        "id": 10,
        "label": "Corridor obstacles",
        "summary": "Rocks sit beside the line to the port, with a gap to weave through.",
        "config": "configs/ppo_level10_corridor.yaml",
    },
    {
        "id": 11,
        "label": "Two-port transfer",
        "summary": "Hold a stop at the lower port, then depart and dock at the upper port.",
        "config": "configs/ppo_level11_transfer.yaml",
    },
)


def list_levels() -> list[dict[str, Any]]:
    return [
        {"id": spec["id"], "label": spec["label"], "summary": spec["summary"]}
        for spec in LEVEL_SPECS
    ]


def level_spec(level_id: int) -> dict[str, Any]:
    for spec in LEVEL_SPECS:
        if int(spec["id"]) == int(level_id):
            return spec
    known = ", ".join(str(spec["id"]) for spec in LEVEL_SPECS)
    raise KeyError(f"Unknown level '{level_id}'. Expected one of: {known}")


def level_env_kwargs(level_id: int) -> dict[str, Any]:
    spec = level_spec(level_id)
    path = ROOT / str(spec["config"])
    return env_kwargs_from_config(load_yaml(path))
