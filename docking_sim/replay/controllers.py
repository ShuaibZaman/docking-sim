"""Small deterministic policy baselines for the docking benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from docking_sim.env.physics import angle_diff, approach_angle, port_velocity, port_world_center


BASELINES: dict[str, dict[str, str]] = {
    "random": {
        "label": "Random policy",
        "description": "Uniform random continuous control; a floor, not a competent controller.",
    },
    "pd": {
        "label": "PD rendezvous",
        "description": "Hand-built position, velocity, and heading feedback controller.",
    },
}


def list_baselines() -> list[dict[str, str]]:
    return [{"id": key, **value} for key, value in BASELINES.items()]


def _body_commands(env, ax: float, ay: float, yaw: float) -> np.ndarray:
    theta = env._state.theta
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    axial = (ax * cos_t + ay * sin_t) / max(env.cfg.thrust_max, 1e-6)
    lateral = (-ax * sin_t + ay * cos_t) / max(env.cfg.lateral_thrust_max, 1e-6)
    return np.clip(np.asarray([axial, lateral, yaw], dtype=np.float32), -1.0, 1.0)


def pd_controller_action(env) -> np.ndarray:
    """A deterministic feedback policy usable as a lower-bound baseline.

    It is intentionally compact rather than optimal: it closes distance with
    a bounded desired relative velocity, damps target-frame drift, then turns
    toward the port approach heading close to capture.
    """

    state = env._state
    port_x, port_y = port_world_center(env.cfg, env._pose)
    port_vx, port_vy = port_velocity(env.cfg, env._pose)
    dx = port_x - state.x
    dy = port_y - state.y
    distance = float(np.hypot(dx, dy))
    direction_x, direction_y = (0.0, 0.0) if distance < 1e-6 else (dx / distance, dy / distance)
    desired_speed = min(1.4, 0.18 + 0.32 * distance)
    if distance < 2.0:
        desired_speed = min(desired_speed, 0.32)
    desired_vx = port_vx + direction_x * desired_speed
    desired_vy = port_vy + direction_y * desired_speed
    ax = 0.75 * dx + 1.35 * (desired_vx - state.vx)
    ay = 0.75 * dy + 1.35 * (desired_vy - state.vy)

    if distance < 2.8:
        target_heading = approach_angle(env.cfg, env._pose)
    else:
        target_heading = float(np.arctan2(dy, dx))
    yaw = 1.8 * angle_diff(target_heading, state.theta) - 0.25 * state.omega
    return _body_commands(env, ax, ay, yaw)


def controller_action(name: str, env) -> np.ndarray:
    key = str(name).lower()
    if key == "random":
        return env.np_random.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
    if key == "pd":
        return pd_controller_action(env)
    known = ", ".join(sorted(BASELINES))
    raise ValueError(f"Unknown controller baseline '{name}'. Expected one of: {known}")
