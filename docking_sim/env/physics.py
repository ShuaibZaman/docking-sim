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
    fuel_lateral_rate: float = 4.0
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
    dock_speed_max: float = 0.08
    dock_angle_max: float = 0.21
    dock_omega_max: float = 0.08
    linear_damping: float = 0.0
    angular_damping: float = 0.04
    station_omega: float = 0.0
    station_vx: float = 0.0
    station_vy: float = 0.0
    station_margin: float = 2.5
    disturbance_std: float = 0.0
    disturbance_persistence: float = 0.0
    disturbance_bias_std: float = 0.0
    n_asteroids: int = 0
    asteroid_radius: float = 0.45
    actuator_lag_s: float = 0.0
    actuator_scale_std: float = 0.0
    sensor_position_std: float = 0.0
    sensor_velocity_std: float = 0.0
    sensor_heading_std: float = 0.0
    sensor_delay_steps: int = 0
    dynamics_mode: str = "inertial"
    orbital_mean_motion: float = 0.0
    world_scale_m: float = 1.0
    second_port: bool = False
    port2_cx: float = 0.0
    port2_cy: float = 8.05
    port2_w: float = 1.0
    port2_h: float = 0.8
    port2_approach_angle: float = float(-np.pi / 2.0)
    hold_steps: int = 20


# A dock is a numerical stop. Looser values in older configs are tightened to these.
CAPTURE_SPEED_MAX = 0.08
CAPTURE_OMEGA_MAX = 0.08


def clamp_capture_speed(value: float) -> float:
    return min(float(value), CAPTURE_SPEED_MAX)


def clamp_capture_omega(value: float) -> float:
    return min(float(value), CAPTURE_OMEGA_MAX)


@dataclass(frozen=True)
class PortSpec:
    cx: float
    cy: float
    w: float
    h: float
    approach: float


def primary_port(cfg: WorldConfig) -> PortSpec:
    return PortSpec(cfg.port_cx, cfg.port_cy, cfg.port_w, cfg.port_h, cfg.port_approach_angle)


def port_list(cfg: WorldConfig) -> tuple[PortSpec, ...]:
    primary = primary_port(cfg)
    if not cfg.second_port:
        return (primary,)
    return (
        primary,
        PortSpec(cfg.port2_cx, cfg.port2_cy, cfg.port2_w, cfg.port2_h, cfg.port2_approach_angle),
    )


@dataclass
class ShipState:
    x: float
    y: float
    vx: float
    vy: float
    theta: float
    omega: float
    fuel: float


@dataclass
class StationPose:
    """Offset and spin of the station relative to its home pose.

    x and y are added to the home hull center. theta rotates the hull and the
    port around that center. vx and vy are the translation rates.
    """

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0


def idle_pose(pose: StationPose | None) -> bool:
    if pose is None:
        return True
    return (
        pose.x == 0.0
        and pose.y == 0.0
        and pose.theta == 0.0
        and pose.vx == 0.0
        and pose.vy == 0.0
    )


def station_center(cfg: WorldConfig, pose: StationPose | None = None) -> tuple[float, float]:
    pose = pose or StationPose()
    return cfg.hull_cx + pose.x, cfg.hull_cy + pose.y


def port_world_center(
    cfg: WorldConfig,
    pose: StationPose | None = None,
    port: PortSpec | None = None,
) -> tuple[float, float]:
    pose = pose or StationPose()
    spec = port or primary_port(cfg)
    if port is None and idle_pose(pose):
        return cfg.port_cx, cfg.port_cy
    local_x = spec.cx - cfg.hull_cx
    local_y = spec.cy - cfg.hull_cy
    cos_t = float(np.cos(pose.theta))
    sin_t = float(np.sin(pose.theta))
    rotated_x = cos_t * local_x - sin_t * local_y
    rotated_y = sin_t * local_x + cos_t * local_y
    center_x, center_y = station_center(cfg, pose)
    return center_x + rotated_x, center_y + rotated_y


def port_velocity(
    cfg: WorldConfig,
    pose: StationPose | None = None,
    port: PortSpec | None = None,
) -> tuple[float, float]:
    pose = pose or StationPose()
    spec = port or primary_port(cfg)
    local_x = spec.cx - cfg.hull_cx
    local_y = spec.cy - cfg.hull_cy
    cos_t = float(np.cos(pose.theta))
    sin_t = float(np.sin(pose.theta))
    rotated_y = sin_t * local_x + cos_t * local_y
    spin = cfg.station_omega
    return pose.vx - spin * rotated_y, pose.vy + spin * (cos_t * local_x - sin_t * local_y)


def approach_angle(
    cfg: WorldConfig,
    pose: StationPose | None = None,
    port: PortSpec | None = None,
) -> float:
    pose = pose or StationPose()
    spec = port or primary_port(cfg)
    return wrap_angle(spec.approach + pose.theta)


def to_station_local(
    x: float,
    y: float,
    cfg: WorldConfig,
    pose: StationPose | None = None,
) -> tuple[float, float]:
    pose = pose or StationPose()
    center_x, center_y = station_center(cfg, pose)
    dx = x - center_x
    dy = y - center_y
    cos_t = float(np.cos(pose.theta))
    sin_t = float(np.sin(pose.theta))
    return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy


def step_station(pose: StationPose, cfg: WorldConfig) -> StationPose:
    theta = wrap_angle(pose.theta + cfg.station_omega * cfg.dt)
    vx, vy = pose.vx, pose.vy
    x = pose.x + vx * cfg.dt
    y = pose.y + vy * cfg.dt
    center_x = cfg.hull_cx + x
    center_y = cfg.hull_cy + y
    lo_x = cfg.x_min + cfg.station_margin
    hi_x = cfg.x_max - cfg.station_margin
    lo_y = cfg.y_min + cfg.station_margin
    hi_y = cfg.y_max - cfg.station_margin
    if center_x < lo_x or center_x > hi_x:
        vx = -vx
        x = pose.x + vx * cfg.dt
    if center_y < lo_y or center_y > hi_y:
        vy = -vy
        y = pose.y + vy * cfg.dt
    return StationPose(x=x, y=y, theta=theta, vx=vx, vy=vy)


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


def hits_hull(state: ShipState, cfg: WorldConfig, pose: StationPose | None = None) -> bool:
    if idle_pose(pose):
        xmin, ymin, xmax, ymax = hull_aabb(cfg)
        return circle_aabb_overlap(state.x, state.y, cfg.ship_radius, xmin, ymin, xmax, ymax)
    local_x, local_y = to_station_local(state.x, state.y, cfg, pose)
    half_w = cfg.hull_w / 2.0
    half_h = cfg.hull_h / 2.0
    return circle_aabb_overlap(local_x, local_y, cfg.ship_radius, -half_w, -half_h, half_w, half_h)


def in_port_zone(
    state: ShipState,
    cfg: WorldConfig,
    pose: StationPose | None = None,
    port: PortSpec | None = None,
) -> bool:
    spec = port or primary_port(cfg)
    if port is None and idle_pose(pose):
        xmin, ymin, xmax, ymax = port_aabb(cfg)
        return point_in_aabb(state.x, state.y, xmin, ymin, xmax, ymax)
    local_x, local_y = to_station_local(state.x, state.y, cfg, pose)
    center_x = spec.cx - cfg.hull_cx
    center_y = spec.cy - cfg.hull_cy
    half_w = spec.w / 2.0
    half_h = spec.h / 2.0
    return point_in_aabb(
        local_x,
        local_y,
        center_x - half_w,
        center_y - half_h,
        center_x + half_w,
        center_y + half_h,
    )


def docking_success(
    state: ShipState,
    cfg: WorldConfig,
    pose: StationPose | None = None,
    port: PortSpec | None = None,
) -> bool:
    """In the port, nearly stopped in translation and spin, and facing the approach."""
    if not in_port_zone(state, cfg, pose, port):
        return False
    speed = target_relative_speed(state, cfg, pose, port)
    heading_err = abs(angle_diff(state.theta, approach_angle(cfg, pose, port)))
    return (
        speed <= cfg.dock_speed_max
        and abs(state.omega) <= cfg.dock_omega_max
        and heading_err <= cfg.dock_angle_max
    )


def target_relative_velocity(
    state: ShipState,
    cfg: WorldConfig,
    pose: StationPose | None = None,
    port: PortSpec | None = None,
) -> tuple[float, float]:
    target_vx, target_vy = port_velocity(cfg, pose, port)
    return state.vx - target_vx, state.vy - target_vy


def target_relative_speed(
    state: ShipState,
    cfg: WorldConfig,
    pose: StationPose | None = None,
    port: PortSpec | None = None,
) -> float:
    rel_vx, rel_vy = target_relative_velocity(state, cfg, pose, port)
    return float(np.hypot(rel_vx, rel_vy))


def hits_asteroid(
    state: ShipState,
    asteroids: list[tuple[float, float, float]],
    ship_radius: float,
) -> bool:
    for rock_x, rock_y, radius in asteroids:
        dx = state.x - rock_x
        dy = state.y - rock_y
        limit = ship_radius + radius
        if dx * dx + dy * dy <= limit * limit:
            return True
    return False


def _segment_circle_overlap(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    circle_x: float,
    circle_y: float,
    radius: float,
) -> bool:
    dx = end_x - start_x
    dy = end_y - start_y
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return (start_x - circle_x) ** 2 + (start_y - circle_y) ** 2 <= radius * radius
    t = ((circle_x - start_x) * dx + (circle_y - start_y) * dy) / length_sq
    t = float(np.clip(t, 0.0, 1.0))
    closest_x = start_x + t * dx
    closest_y = start_y + t * dy
    return (closest_x - circle_x) ** 2 + (closest_y - circle_y) ** 2 <= radius * radius


def _segment_aabb_overlap(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> bool:
    """Liang-Barsky segment clipping against an already expanded box."""
    dx = end_x - start_x
    dy = end_y - start_y
    lower, upper = 0.0, 1.0
    for p, q in (
        (-dx, start_x - xmin),
        (dx, xmax - start_x),
        (-dy, start_y - ymin),
        (dy, ymax - start_y),
    ):
        if abs(p) <= 1e-12:
            if q < 0.0:
                return False
            continue
        ratio = q / p
        if p < 0.0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower > upper:
            return False
    return True


def _interpolate_pose(start: StationPose, end: StationPose, fraction: float) -> StationPose:
    return StationPose(
        x=start.x + (end.x - start.x) * fraction,
        y=start.y + (end.y - start.y) * fraction,
        theta=wrap_angle(start.theta + angle_diff(end.theta, start.theta) * fraction),
        vx=start.vx + (end.vx - start.vx) * fraction,
        vy=start.vy + (end.vy - start.vy) * fraction,
    )


def swept_hits_hull(
    previous: ShipState,
    current: ShipState,
    cfg: WorldConfig,
    previous_pose: StationPose | None = None,
    current_pose: StationPose | None = None,
) -> bool:
    """Detect contact anywhere along a simulation step.

    Stationary hulls use exact segment-vs-expanded-AABB intersection. Moving
    or rotating hulls use a conservative temporal sweep whose sampling density
    scales with relative motion; this prevents a ship crossing a thin hull
    between endpoints.
    """
    start_pose = previous_pose or StationPose()
    end_pose = current_pose or start_pose
    if (
        start_pose.x == end_pose.x
        and start_pose.y == end_pose.y
        and start_pose.theta == end_pose.theta
    ):
        start_x, start_y = to_station_local(previous.x, previous.y, cfg, start_pose)
        end_x, end_y = to_station_local(current.x, current.y, cfg, end_pose)
        return _segment_aabb_overlap(
            start_x,
            start_y,
            end_x,
            end_y,
            -cfg.hull_w / 2.0 - cfg.ship_radius,
            -cfg.hull_h / 2.0 - cfg.ship_radius,
            cfg.hull_w / 2.0 + cfg.ship_radius,
            cfg.hull_h / 2.0 + cfg.ship_radius,
        )

    ship_motion = float(np.hypot(current.x - previous.x, current.y - previous.y))
    station_motion = float(np.hypot(end_pose.x - start_pose.x, end_pose.y - start_pose.y))
    rotation_motion = abs(angle_diff(end_pose.theta, start_pose.theta)) * float(
        np.hypot(cfg.hull_w / 2.0, cfg.hull_h / 2.0)
    )
    stride = max(cfg.ship_radius * 0.5, 0.025)
    samples = max(2, int(np.ceil((ship_motion + station_motion + rotation_motion) / stride)))
    for index in range(samples + 1):
        fraction = index / samples
        probe = ShipState(
            x=previous.x + (current.x - previous.x) * fraction,
            y=previous.y + (current.y - previous.y) * fraction,
            vx=0.0,
            vy=0.0,
            theta=0.0,
            omega=0.0,
            fuel=0.0,
        )
        if hits_hull(probe, cfg, _interpolate_pose(start_pose, end_pose, fraction)):
            return True
    return False


def swept_hits_asteroids(
    previous: ShipState,
    current: ShipState,
    asteroids: list[tuple[float, float, float]],
    ship_radius: float,
) -> bool:
    for rock_x, rock_y, radius in asteroids:
        if _segment_circle_overlap(
            previous.x,
            previous.y,
            current.x,
            current.y,
            rock_x,
            rock_y,
            ship_radius + radius,
        ):
            return True
    return False


def step_ship(
    state: ShipState,
    axial_cmd: float,
    lateral_cmd: float,
    yaw_cmd: float,
    cfg: WorldConfig,
    disturbance: tuple[float, float] = (0.0, 0.0),
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
        + abs(lateral_cmd) * cfg.fuel_lateral_rate
        + abs(yaw_cmd) * cfg.fuel_torque_rate
    ) * cfg.dt
    fuel_used = float(min(fuel_used, max(state.fuel, 0.0)))

    axial = axial_cmd * cfg.thrust_max
    lateral = lateral_cmd * cfg.lateral_thrust_max
    torque = yaw_cmd * cfg.torque_max
    cos_t = float(np.cos(state.theta))
    sin_t = float(np.sin(state.theta))
    ax = (axial * cos_t - lateral * sin_t) / cfg.mass + float(disturbance[0])
    ay = (axial * sin_t + lateral * cos_t) / cfg.mass + float(disturbance[1])
    if cfg.dynamics_mode == "orbital_relative" and cfg.orbital_mean_motion:
        mean_motion = float(cfg.orbital_mean_motion)
        # Linearized Hill/Clohessy-Wiltshire terms in the local orbital frame.
        ax += 2.0 * mean_motion * state.vy + 3.0 * mean_motion * mean_motion * state.x
        ay -= 2.0 * mean_motion * state.vx
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
