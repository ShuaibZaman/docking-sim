from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from docking_sim.env.physics import (
    ShipState,
    WorldConfig,
    angle_diff,
    docking_success,
    hits_hull,
    out_of_bounds,
    step_ship,
)
from docking_sim.env.renderer import render_rgb
from docking_sim.env.rewards import RewardContext, make_reward


OBS_DIM = 9
OBS_LOW = np.full((OBS_DIM,), -10.0, dtype=np.float32)
OBS_HIGH = np.full((OBS_DIM,), 10.0, dtype=np.float32)


class DockingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        max_steps: int = 800,
        dt: float = 0.05,
        dock_speed_max: float = 0.35,
        dock_angle_max_deg: float = 12.0,
        fuel_capacity: float = 100.0,
        randomize_start: bool = False,
        randomize_velocity: bool = False,
        reward_name: str = "safe_docking",
        reward_weights: dict[str, float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.randomize_start = bool(randomize_start)
        self.randomize_velocity = bool(randomize_velocity)

        cfg_fields = {f.name for f in WorldConfig.__dataclass_fields__.values()}
        physics_kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}
        self.cfg = WorldConfig(
            dt=float(dt),
            fuel_capacity=float(fuel_capacity),
            dock_speed_max=float(dock_speed_max),
            dock_angle_max=float(np.deg2rad(dock_angle_max_deg)),
            **physics_kwargs,
        )
        self._reward_fn = make_reward(reward_name, reward_weights)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)

        self._state = self._default_state()
        self._step_count = 0
        self._last_components: dict[str, float] = {}

    def _default_state(self) -> ShipState:
        return ShipState(
            x=0.0,
            y=-5.0,
            vx=0.0,
            vy=0.0,
            theta=float(np.pi / 2.0),
            omega=0.0,
            fuel=self.cfg.fuel_capacity,
        )

    def _spawn_state(self) -> ShipState:
        state = self._default_state()
        if self.randomize_start:
            state.x = float(self.np_random.uniform(-6.0, 6.0))
            state.y = float(self.np_random.uniform(-6.5, 1.5))
            state.theta = float(self.np_random.uniform(-np.pi, np.pi))
        if self.randomize_velocity:
            state.vx = float(self.np_random.uniform(-0.6, 0.6))
            state.vy = float(self.np_random.uniform(-0.6, 0.6))
            state.omega = float(self.np_random.uniform(-0.4, 0.4))
        return state

    def _observe(self, state: ShipState) -> np.ndarray:
        rel_x = state.x - self.cfg.port_cx
        rel_y = state.y - self.cfg.port_cy
        heading_error = angle_diff(state.theta, self.cfg.port_approach_angle)
        obs = np.array(
            [
                rel_x / 10.0,
                rel_y / 8.0,
                state.vx / 5.0,
                state.vy / 5.0,
                np.cos(state.theta),
                np.sin(state.theta),
                state.omega / 4.0,
                heading_error / np.pi,
                state.fuel / max(self.cfg.fuel_capacity, 1e-6),
            ],
            dtype=np.float32,
        )
        return np.clip(obs, OBS_LOW, OBS_HIGH)

    def _metrics(self, state: ShipState) -> dict[str, float]:
        rel_x = state.x - self.cfg.port_cx
        rel_y = state.y - self.cfg.port_cy
        distance = float(np.hypot(rel_x, rel_y))
        speed = float(np.hypot(state.vx, state.vy))
        heading_error = abs(angle_diff(state.theta, self.cfg.port_approach_angle))
        return {
            "distance": distance,
            "speed": speed,
            "heading_error": heading_error,
            "fuel": float(state.fuel),
        }

    def _info(
        self,
        state: ShipState,
        *,
        success: bool = False,
        crash: bool = False,
        timeout: bool = False,
        fuel_used: float = 0.0,
        thrust: float = 0.0,
        torque: float = 0.0,
    ) -> dict[str, Any]:
        metrics = self._metrics(state)
        return {
            **metrics,
            "success": bool(success),
            "crash": bool(crash),
            "timeout": bool(timeout),
            "fuel_used": float(fuel_used),
            "thrust": float(thrust),
            "torque": float(torque),
            "x": float(state.x),
            "y": float(state.y),
            "vx": float(state.vx),
            "vy": float(state.vy),
            "theta": float(state.theta),
            "omega": float(state.omega),
            "reward_components": dict(self._last_components),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options and isinstance(options.get("state"), ShipState):
            self._state = options["state"]
        else:
            self._state = self._spawn_state()
        self._step_count = 0
        self._last_components = {}
        obs = self._observe(self._state)
        return obs, self._info(self._state)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        thrust = float(np.clip(action[0], -1.0, 1.0) + 1.0) * 0.5
        torque = float(np.clip(action[1], -1.0, 1.0))

        self._state, fuel_used = step_ship(self._state, thrust, torque, self.cfg)
        self._step_count += 1

        success = docking_success(self._state, self.cfg)
        crash = hits_hull(self._state, self.cfg) or out_of_bounds(self._state, self.cfg)
        timeout = self._step_count >= self.max_steps and not success and not crash

        metrics = self._metrics(self._state)
        ctx = RewardContext(
            distance=metrics["distance"],
            speed=metrics["speed"],
            heading_error=metrics["heading_error"],
            fuel_used=fuel_used,
            success=success,
            crash=crash,
            timeout=timeout,
        )
        reward, components = self._reward_fn(ctx)
        self._last_components = components

        terminated = bool(success or crash)
        truncated = bool(timeout)
        info = self._info(
            self._state,
            success=success,
            crash=crash,
            timeout=timeout,
            fuel_used=fuel_used,
            thrust=thrust,
            torque=torque,
        )
        return self._observe(self._state), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return render_rgb(self._state, self.cfg)
        return None
