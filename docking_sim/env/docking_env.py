from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from docking_sim.env.physics import (
    ShipState,
    StationPose,
    WorldConfig,
    approach_angle,
    docking_success,
    hits_asteroid,
    hits_hull,
    out_of_bounds,
    port_velocity,
    port_world_center,
    station_center,
    step_ship,
    step_station,
    swept_hits_asteroids,
    swept_hits_hull,
    target_relative_speed,
    target_relative_velocity,
    angle_diff,
)
from docking_sim.env.renderer import IMAGE_HEIGHT, IMAGE_WIDTH, render_rgb
from docking_sim.env.rewards import RewardContext, make_reward


BASE_OBS_DIM = 9
EXTENDED_OBS_DIM = 17
OBS_CLIP = 10.0


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
        obs_mode: str = "state",
        extended_obs: bool | None = None,
        fail_on_empty_fuel: bool = False,
        level: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = int(max_steps)
        self.randomize_start = bool(randomize_start)
        self.randomize_velocity = bool(randomize_velocity)
        self.obs_mode = str(obs_mode)
        if self.obs_mode not in {"state", "pixels"}:
            raise ValueError(f"Unknown obs_mode '{self.obs_mode}'. Expected state or pixels.")
        self.fail_on_empty_fuel = bool(fail_on_empty_fuel)
        self.level = None if level is None else int(level)

        cfg_fields = {f.name for f in WorldConfig.__dataclass_fields__.values()}
        physics_kwargs = {k: v for k, v in kwargs.items() if k in cfg_fields}
        self.cfg = WorldConfig(
            dt=float(dt),
            fuel_capacity=float(fuel_capacity),
            dock_speed_max=float(dock_speed_max),
            dock_angle_max=float(np.deg2rad(dock_angle_max_deg)),
            **physics_kwargs,
        )
        if extended_obs is None:
            self.extended_obs = bool(
                self.cfg.station_omega
                or self.cfg.station_vx
                or self.cfg.station_vy
                or self.cfg.disturbance_std
                or self.cfg.n_asteroids
            )
        else:
            self.extended_obs = bool(extended_obs)
        self._reward_fn = make_reward(reward_name, reward_weights)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        if self.obs_mode == "pixels":
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                dtype=np.uint8,
            )
        else:
            dim = EXTENDED_OBS_DIM if self.extended_obs else BASE_OBS_DIM
            self.observation_space = spaces.Box(
                low=-OBS_CLIP,
                high=OBS_CLIP,
                shape=(dim,),
                dtype=np.float32,
            )

        self._state = self._default_state()
        self._pose = StationPose()
        self._asteroids: list[tuple[float, float, float]] = []
        self._wind = (0.0, 0.0)
        self._wind_bias = (0.0, 0.0)
        self._wind_trace: list[tuple[float, float]] | None = None
        self._actuator_scale = np.ones(3, dtype=np.float32)
        self._applied_action = np.zeros(3, dtype=np.float32)
        self._command_action = np.zeros(3, dtype=np.float32)
        self._obs_history: list[np.ndarray] = []
        self._scenario_id: str | None = None
        self._scenario_hash: str | None = None
        self._step_count = 0
        self._prev_distance = 0.0
        self._fuel_used_total = 0.0
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

    def _fresh_pose(self) -> StationPose:
        return StationPose(vx=float(self.cfg.station_vx), vy=float(self.cfg.station_vy))

    def _sample_asteroids(self) -> list[tuple[float, float, float]]:
        count = int(self.cfg.n_asteroids)
        if count <= 0:
            return []
        radius = float(self.cfg.asteroid_radius)
        ship = self._state
        placed: list[tuple[float, float, float]] = []
        for _ in range(count * 40):
            if len(placed) >= count:
                break
            x = float(self.np_random.uniform(self.cfg.x_min + radius, self.cfg.x_max - radius))
            y = float(self.np_random.uniform(self.cfg.y_min + radius, self.cfg.y_max - radius))
            if self._asteroid_blocked(x, y, radius, ship, placed):
                continue
            placed.append((x, y, radius))
        return placed

    def _asteroid_blocked(
        self,
        x: float,
        y: float,
        radius: float,
        ship: ShipState,
        placed: list[tuple[float, float, float]],
    ) -> bool:
        probe = ShipState(
            x=x,
            y=y,
            vx=0.0,
            vy=0.0,
            theta=0.0,
            omega=0.0,
            fuel=1.0,
        )
        keepout = radius + self.cfg.ship_radius
        port_x, port_y = port_world_center(self.cfg, self._pose)
        near_port = (x - port_x) ** 2 + (y - port_y) ** 2 <= (keepout + 0.8) ** 2
        if hits_hull(probe, self.cfg, self._pose) or near_port:
            return True
        if (x - ship.x) ** 2 + (y - ship.y) ** 2 <= (keepout + 0.6) ** 2:
            return True
        for other_x, other_y, other_r in placed:
            if (x - other_x) ** 2 + (y - other_y) ** 2 <= (radius + other_r + 0.3) ** 2:
                return True
        return False

    def _sample_wind(self) -> tuple[float, float]:
        if self._wind_trace is not None and self._step_count < len(self._wind_trace):
            return self._wind_trace[self._step_count]
        std = float(self.cfg.disturbance_std)
        persistence = float(np.clip(self.cfg.disturbance_persistence, 0.0, 0.999))
        innovation = float(np.sqrt(1.0 - persistence * persistence))
        noise_x = float(self.np_random.normal(0.0, std)) if std > 0.0 else 0.0
        noise_y = float(self.np_random.normal(0.0, std)) if std > 0.0 else 0.0
        return (
            float(
                self._wind_bias[0]
                + persistence * (self._wind[0] - self._wind_bias[0])
                + innovation * noise_x
            ),
            float(
                self._wind_bias[1]
                + persistence * (self._wind[1] - self._wind_bias[1])
                + innovation * noise_y
            ),
        )

    def _raw_observation(self, state: ShipState) -> np.ndarray:
        if self.obs_mode == "pixels":
            return render_rgb(state, self.cfg, self._pose, self._asteroids)
        port_x, port_y = port_world_center(self.cfg, self._pose)
        position_std = float(self.cfg.sensor_position_std)
        velocity_std = float(self.cfg.sensor_velocity_std)
        heading_std = float(self.cfg.sensor_heading_std)
        sensed_x = state.x + (float(self.np_random.normal(0.0, position_std)) if position_std > 0.0 else 0.0)
        sensed_y = state.y + (float(self.np_random.normal(0.0, position_std)) if position_std > 0.0 else 0.0)
        sensed_vx = state.vx + (float(self.np_random.normal(0.0, velocity_std)) if velocity_std > 0.0 else 0.0)
        sensed_vy = state.vy + (float(self.np_random.normal(0.0, velocity_std)) if velocity_std > 0.0 else 0.0)
        sensed_theta = state.theta + (float(self.np_random.normal(0.0, heading_std)) if heading_std > 0.0 else 0.0)
        heading_error = angle_diff(sensed_theta, approach_angle(self.cfg, self._pose))
        obs = [
            (sensed_x - port_x) / 10.0,
            (sensed_y - port_y) / 8.0,
            sensed_vx / 5.0,
            sensed_vy / 5.0,
            float(np.cos(sensed_theta)),
            float(np.sin(sensed_theta)),
            state.omega / 4.0,
            heading_error / np.pi,
            state.fuel / max(self.cfg.fuel_capacity, 1e-6),
        ]
        if self.extended_obs:
            port_vx, port_vy = port_velocity(self.cfg, self._pose)
            rock_x, rock_y, present = self._nearest_asteroid(state)
            obs.extend(
                [
                    port_vx / 5.0,
                    port_vy / 5.0,
                    self.cfg.station_omega / 2.0,
                    self._wind[0] / 2.0,
                    self._wind[1] / 2.0,
                    rock_x,
                    rock_y,
                    present,
                ]
            )
        return np.clip(np.asarray(obs, dtype=np.float32), -OBS_CLIP, OBS_CLIP)

    def _observe(self, state: ShipState) -> np.ndarray:
        raw = self._raw_observation(state)
        delay = max(0, int(self.cfg.sensor_delay_steps))
        if delay == 0:
            return raw
        if not self._obs_history:
            self._obs_history = [raw.copy() for _ in range(delay)]
            return raw
        self._obs_history.append(raw)
        return self._obs_history.pop(0)

    def _nearest_asteroid(self, state: ShipState) -> tuple[float, float, float]:
        if not self._asteroids:
            return 0.0, 0.0, 0.0
        rock_x, rock_y, _radius = min(
            self._asteroids,
            key=lambda rock: (state.x - rock[0]) ** 2 + (state.y - rock[1]) ** 2,
        )
        return (rock_x - state.x) / 10.0, (rock_y - state.y) / 8.0, 1.0

    def _metrics(self, state: ShipState) -> dict[str, float]:
        port_x, port_y = port_world_center(self.cfg, self._pose)
        distance = float(np.hypot(state.x - port_x, state.y - port_y))
        relative_vx, relative_vy = target_relative_velocity(state, self.cfg, self._pose)
        absolute_speed = float(np.hypot(state.vx, state.vy))
        speed = target_relative_speed(state, self.cfg, self._pose)
        heading_error = abs(angle_diff(state.theta, approach_angle(self.cfg, self._pose)))
        return {
            "distance": distance,
            "speed": speed,
            "absolute_speed": absolute_speed,
            "relative_vx": relative_vx,
            "relative_vy": relative_vy,
            "heading_error": heading_error,
            "fuel": float(state.fuel),
        }

    def _scene(self) -> dict[str, Any]:
        hull_x, hull_y = station_center(self.cfg, self._pose)
        port_x, port_y = port_world_center(self.cfg, self._pose)
        return {
            "station": {
                "cx": float(hull_x),
                "cy": float(hull_y),
                "theta": float(self._pose.theta),
                "w": float(self.cfg.hull_w),
                "h": float(self.cfg.hull_h),
            },
            "port_pose": {
                "cx": float(port_x),
                "cy": float(port_y),
                "theta": float(self._pose.theta),
                "w": float(self.cfg.port_w),
                "h": float(self.cfg.port_h),
            },
            "asteroids": [
                {"x": float(x), "y": float(y), "r": float(radius)} for x, y, radius in self._asteroids
            ],
            "scenario_id": self._scenario_id,
            "scenario_hash": self._scenario_hash,
        }

    def _info(
        self,
        state: ShipState,
        *,
        success: bool = False,
        crash: bool = False,
        timeout: bool = False,
        fuel_used: float = 0.0,
        axial: float = 0.0,
        lateral: float = 0.0,
        yaw: float = 0.0,
        hit_hull: bool = False,
        hit_asteroid: bool = False,
        out_of_bounds: bool = False,
        out_of_fuel: bool = False,
        terminal_reason: str = "",
    ) -> dict[str, Any]:
        metrics = self._metrics(state)
        return {
            **metrics,
            **self._scene(),
            "success": bool(success),
            "crash": bool(crash),
            "timeout": bool(timeout),
            "hit_hull": bool(hit_hull),
            "hit_asteroid": bool(hit_asteroid),
            "out_of_bounds": bool(out_of_bounds),
            "out_of_fuel": bool(out_of_fuel),
            "terminal_reason": terminal_reason,
            "fuel_used": float(fuel_used),
            "fuel_used_total": float(self._fuel_used_total),
            "axial": float(axial),
            "lateral": float(lateral),
            "yaw": float(yaw),
            "thrust": float(axial),
            "torque": float(yaw),
            "wind_x": float(self._wind[0]),
            "wind_y": float(self._wind[1]),
            "command_axial": float(self._command_action[0]),
            "command_lateral": float(self._command_action[1]),
            "command_yaw": float(self._command_action[2]),
            "actuator_scale": [float(value) for value in self._actuator_scale],
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
        if options and isinstance(options.get("station_pose"), StationPose):
            self._pose = options["station_pose"]
        else:
            self._pose = self._fresh_pose()
        scenario_asteroids = options.get("asteroids") if options else None
        if scenario_asteroids is None:
            self._asteroids = self._sample_asteroids()
        else:
            self._asteroids = [
                (float(rock[0]), float(rock[1]), float(rock[2]))
                for rock in scenario_asteroids
            ]
        scenario_trace = options.get("disturbance_trace") if options else None
        self._wind_trace = (
            [(float(wind[0]), float(wind[1])) for wind in scenario_trace]
            if scenario_trace is not None
            else None
        )
        self._scenario_id = str(options["scenario_id"]) if options and options.get("scenario_id") else None
        self._scenario_hash = str(options["scenario_hash"]) if options and options.get("scenario_hash") else None
        actuator_scale = options.get("actuator_scale") if options else None
        if actuator_scale is None:
            scale_std = float(self.cfg.actuator_scale_std)
            self._actuator_scale = (
                np.clip(
                    self.np_random.normal(1.0, scale_std, size=3),
                    0.25,
                    2.0,
                ).astype(np.float32)
                if scale_std > 0.0
                else np.ones(3, dtype=np.float32)
            )
        else:
            self._actuator_scale = np.asarray(actuator_scale, dtype=np.float32).reshape(3)
        wind_bias = options.get("wind_bias") if options else None
        if wind_bias is None:
            bias_std = float(self.cfg.disturbance_bias_std)
            self._wind_bias = (
                (
                    float(self.np_random.normal(0.0, bias_std)),
                    float(self.np_random.normal(0.0, bias_std)),
                )
                if bias_std > 0.0
                else (0.0, 0.0)
            )
        else:
            self._wind_bias = (float(wind_bias[0]), float(wind_bias[1]))
        self._applied_action = np.zeros(3, dtype=np.float32)
        self._command_action = np.zeros(3, dtype=np.float32)
        self._wind = (0.0, 0.0)
        self._step_count = 0
        self._fuel_used_total = 0.0
        self._last_components = {}
        self._obs_history = []
        self._prev_distance = self._metrics(self._state)["distance"]
        obs = self._observe(self._state)
        return obs, self._info(self._state)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        command = np.clip(action[:3], -1.0, 1.0).astype(np.float32)
        self._command_action = command
        lag = max(0.0, float(self.cfg.actuator_lag_s))
        if lag > 0.0:
            alpha = float(np.clip(self.cfg.dt / (lag + self.cfg.dt), 0.0, 1.0))
            self._applied_action += alpha * (command - self._applied_action)
        else:
            self._applied_action = command.copy()
        applied = np.clip(self._applied_action * self._actuator_scale, -1.0, 1.0)
        axial = float(applied[0])
        lateral = float(applied[1])
        yaw = float(applied[2])

        previous_state = self._state
        previous_pose = self._pose
        self._wind = self._sample_wind()
        self._state, fuel_used = step_ship(
            self._state,
            axial,
            lateral,
            yaw,
            self.cfg,
            disturbance=self._wind,
        )
        self._pose = step_station(self._pose, self.cfg)
        self._step_count += 1

        hit_hull = swept_hits_hull(
            previous_state,
            self._state,
            self.cfg,
            previous_pose,
            self._pose,
        )
        hit_rock = swept_hits_asteroids(
            previous_state,
            self._state,
            self._asteroids,
            self.cfg.ship_radius,
        )
        left_map = out_of_bounds(self._state, self.cfg)
        success = docking_success(self._state, self.cfg, self._pose) and not hit_hull and not hit_rock and not left_map
        out_of_fuel = self.fail_on_empty_fuel and self._state.fuel <= 1e-8 and not success
        crash = bool(hit_hull or hit_rock or left_map or out_of_fuel)
        timeout = self._step_count >= self.max_steps and not success and not crash
        terminal_reason = (
            "docked"
            if success
            else "hull"
            if hit_hull
            else "asteroid"
            if hit_rock
            else "bounds"
            if left_map
            else "fuel"
            if out_of_fuel
            else "timeout"
            if timeout
            else ""
        )

        metrics = self._metrics(self._state)
        self._fuel_used_total += fuel_used
        ctx = RewardContext(
            distance=metrics["distance"],
            prev_distance=self._prev_distance,
            speed=metrics["speed"],
            heading_error=metrics["heading_error"],
            fuel_used=fuel_used,
            success=success,
            crash=crash,
            timeout=timeout,
        )
        self._prev_distance = metrics["distance"]
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
            axial=axial,
            lateral=lateral,
            yaw=yaw,
            hit_hull=hit_hull,
            hit_asteroid=hit_rock,
            out_of_bounds=left_map,
            out_of_fuel=out_of_fuel,
            terminal_reason=terminal_reason,
        )
        return self._observe(self._state), float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array" or self.obs_mode == "pixels":
            return render_rgb(self._state, self.cfg, self._pose, self._asteroids)
        return None
