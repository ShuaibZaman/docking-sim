from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_angle(theta: float) -> float:
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


def angle_diff(a: float, b: float) -> float:
    return wrap_angle(a - b)


@dataclass
class WorldConfig:
    x_min: float = -10.0
    x_max: float = 10.0
    y_min: float = -8.0
    y_max: float = 8.0
    dt: float = 0.05
    mass: float = 1.0
    inertia: float = 0.15
    thrust_max: float = 4.0
    lateral_thrust_max: float = 2.0
    torque_max: float = 2.0
    ship_radius: float = 0.28
    fuel_capacity: float = 100.0
    fuel_thrust_rate: float = 8.0
    fuel_torque_rate: float = 2.0
    hull_cx: float = 0.0
    hull_cy: float = 6.6
    hull_w: float = 4.0
    hull_h: float = 1.4
    port_cx: float = 0.0
    port_cy: float = 5.15
    port_w: float = 1.0
    port_h: float = 0.8
    port_approach_angle: float = float(np.pi / 2.0)
    dock_speed_max: float = 0.35
    dock_angle_max: float = 0.21
    linear_damping: float = 0.0
    angular_damping: float = 0.04


@dataclass
class ShipState:
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float


def hull_aabb(cfg: WorldConfig) -> tuple[float, float, float, float]:
    hw, hh = cfg.hull_w / 2.0, cfg.hull_h / 2.0
    return (
        cfg.hull_cx - hw,
        cfg.hull_cy - hh,
        cfg.hull_cx + hw,
        cfg.hull_cy + hh,
    )


def port_aabb(cfg: WorldConfig) -> tuple[float, float, float, float]:
    pw, ph = cfg.port_w / 2.0, cfg.port_h / 2.0
    return (
        cfg.port_cx - pw,
        cfg.port_cy - ph,
        cfg.port_cx + pw,
        cfg.port_cy + ph,
    )


def circle_aabb_overlap(
    cx: float,
    cy: float,
    radius: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    closest_x = float(np.clip(cx, xmin, xmax))
    closest_y = float(np.clip(cy, ymin, ymax))
    dx = cx - closest_x
    dy = cy - closest_y
    return dx * dx + dy * dy <= radius * radius


def point_in_aabb(
    x: float,
    y: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    return xmin <= x <= xmax and ymin <= y <= ymax


def out_of_bounds(state: ShipState, cfg: WorldConfig) -> bool:
    return not (
        cfg.x_min <= state.x <= cfg.x_max and cfg.y_min <= state.y <= cfg.y_max
    )


def hits_hull(state: ShipState, cfg: WorldConfig) -> bool:
    xmin, ymin, xmax, ymax = hull_aabb(cfg)
    return circle_aabb_overlap(state.x, state.y, cfg.ship_radius, xmin, ymin, xmax, ymax)


def in_port_zone(state: ShipState, cfg: WorldConfig) -> bool:
    xmin, ymin, xmax, ymax = port_aabb(cfg)
    return point_in_aabb(state.x, state.y, xmin, ymin, xmax, ymax)


def docking_success(state: ShipState, cfg: WorldConfig) -> bool:
    if not in_port_zone(state, cfg):
        return False
    speed = float(np.hypot(state.vx, state.vy))
    heading_err = abs(angle_diff(state.theta, cfg.port_approach_angle))
    return speed <= cfg.dock_speed_max and heading_err <= cfg.dock_angle_max


def step_ship(
    state: ShipState,
    axial_cmd: float,
    lateral_cmd: float,
    yaw_cmd: float,
    cfg: WorldConfig,
) -> tuple[ShipState, float]:
    """Semi-implicit Euler.

    axial_cmd in [-1, 1]: +1 nose thrust, -1 brake along the heading.
    lateral_cmd in [-1, 1]: strafe along (-sin theta, cos theta).
    yaw_cmd in [-1, 1].
    """
    axial_cmd = float(np.clip(axial_cmd, -1.0, 1.0))
    lateral_cmd = float(np.clip(lateral_cmd, -1.0, 1.0))
    yaw_cmd = float(np.clip(yaw_cmd, -1.0, 1.0))

    if state.fuel <= 1e-8:
        axial_cmd = 0.0
        lateral_cmd = 0.0
        yaw_cmd = 0.0

    fuel_used = (
        abs(axial_cmd) * cfg.fuel_thrust_rate
        + abs(lateral_cmd) * cfg.fuel_thrust_rate
        + abs(yaw_cmd) * cfg.fuel_torque_rate
    ) * cfg.dt
    fuel_used = float(min(fuel_used, max(state.fuel, 0.0)))

    axial = axial_cmd * cfg.thrust_max
    lateral = lateral_cmd * cfg.lateral_thrust_max
    torque = yaw_cmd * cfg.torque_max
    cos_t = float(np.cos(state.theta))
    sin_t = float(np.sin(state.theta))
    ax = (axial * cos_t - lateral * sin_t) / cfg.mass
    ay = (axial * sin_t + lateral * cos_t) / cfg.mass
    alpha = torque / cfg.inertia

    vx = state.vx + ax * cfg.dt
    vy = state.vy + ay * cfg.dt
    omega = state.omega + alpha * cfg.dt

    if cfg.linear_damping:
        damp = max(0.0, 1.0 - cfg.linear_damping * cfg.dt)
        vx *= damp
        vy *= damp
    if cfg.angular_damping:
        omega *= max(0.0, 1.0 - cfg.angular_damping * cfg.dt)

    x = state.x + vx * cfg.dt
    y = state.y + vy * cfg.dt
    theta = wrap_angle(state.theta + omega * cfg.dt)
    fuel = max(0.0, state.fuel - fuel_used)

    nxt = ShipState(x=x, y=y, vx=vx, vy=vy, theta=theta, omega=omega, fuel=fuel)
    return nxt, fuel_used
