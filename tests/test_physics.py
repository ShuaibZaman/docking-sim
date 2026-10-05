from docking_sim.env.physics import (
    ShipState,
    StationPose,
    WorldConfig,
    docking_success,
    step_ship,
    swept_hits_asteroids,
    swept_hits_hull,
)


def _still(theta: float = 0.0, fuel: float = 100.0) -> ShipState:
    return ShipState(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=theta, omega=0.0, fuel=fuel)


def test_positive_axial_speeds_up_along_heading():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0, angular_damping=0.0)
    state = _still(theta=0.0)
    nxt, fuel_used = step_ship(state, 1.0, 0.0, 0.0, cfg)
    assert nxt.vx > state.vx
    assert abs(nxt.vy) < 1e-9
    assert nxt.omega == 0.0
    assert fuel_used > 0.0


def test_negative_axial_brakes_along_heading():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0, angular_damping=0.0)
    state = _still(theta=0.0)
    nxt, fuel_used = step_ship(state, -1.0, 0.0, 0.0, cfg)
    assert nxt.vx < 0.0
    assert abs(nxt.vy) < 1e-9
    assert nxt.omega == 0.0
    assert fuel_used > 0.0


def test_lateral_accelerates_perpendicular_to_heading():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0, angular_damping=0.0)
    state = _still(theta=0.0)
    nxt, fuel_used = step_ship(state, 0.0, 1.0, 0.0, cfg)
    assert abs(nxt.vx) < 1e-9
    assert nxt.vy > 0.0
    assert nxt.omega == 0.0
    assert fuel_used > 0.0


def test_lateral_fuel_matches_its_lower_thrust():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0, angular_damping=0.0)
    state = _still()
    _, axial_fuel = step_ship(state, 1.0, 0.0, 0.0, cfg)
    _, lateral_fuel = step_ship(state, 0.0, 1.0, 0.0, cfg)
    assert abs(axial_fuel - cfg.fuel_thrust_rate * cfg.dt) < 1e-9
    assert abs(lateral_fuel - cfg.fuel_lateral_rate * cfg.dt) < 1e-9
    assert abs(lateral_fuel / axial_fuel - cfg.lateral_thrust_max / cfg.thrust_max) < 1e-9


def test_yaw_changes_angular_velocity():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0)
    state = _still()
    nxt, _ = step_ship(state, 0.0, 0.0, 1.0, cfg)
    assert nxt.omega > 0.0
    assert nxt.vx == state.vx
    assert nxt.vy == state.vy


def test_empty_fuel_ignores_commands():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0, angular_damping=0.0)
    state = ShipState(x=1.0, y=-2.0, vx=0.3, vy=-0.1, theta=0.4, omega=0.2, fuel=0.0)
    nxt, fuel_used = step_ship(state, 1.0, -1.0, 1.0, cfg)
    assert fuel_used == 0.0
    assert nxt.vx == state.vx
    assert nxt.vy == state.vy
    assert nxt.omega == state.omega
    assert nxt.fuel == 0.0


def test_moving_target_requires_relative_capture_speed():
    cfg = WorldConfig()
    pose = StationPose(vx=0.4)
    co_moving = ShipState(
        x=cfg.port_cx,
        y=cfg.port_cy,
        vx=0.4,
        vy=0.0,
        theta=cfg.port_approach_angle,
        omega=0.0,
        fuel=100.0,
    )
    stationary = ShipState(
        x=cfg.port_cx,
        y=cfg.port_cy,
        vx=0.0,
        vy=0.0,
        theta=cfg.port_approach_angle,
        omega=0.0,
        fuel=100.0,
    )
    assert docking_success(co_moving, cfg, pose)
    assert not docking_success(stationary, cfg, pose)


def test_capture_requires_a_stop_and_no_spin():
    cfg = WorldConfig()
    stopped = ShipState(
        x=cfg.port_cx,
        y=cfg.port_cy,
        vx=0.0,
        vy=0.0,
        theta=cfg.port_approach_angle,
        omega=0.0,
        fuel=100.0,
    )
    spinning = ShipState(
        x=cfg.port_cx,
        y=cfg.port_cy,
        vx=0.0,
        vy=0.0,
        theta=cfg.port_approach_angle,
        omega=0.4,
        fuel=100.0,
    )
    fast = ShipState(
        x=cfg.port_cx,
        y=cfg.port_cy,
        vx=0.2,
        vy=0.0,
        theta=cfg.port_approach_angle,
        omega=0.0,
        fuel=100.0,
    )
    assert docking_success(stopped, cfg)
    assert not docking_success(spinning, cfg)
    assert not docking_success(fast, cfg)


def test_swept_collisions_catch_tunnelling_between_endpoints():
    cfg = WorldConfig()
    start = ShipState(x=-3.0, y=cfg.hull_cy, vx=0, vy=0, theta=0, omega=0, fuel=100)
    end = ShipState(x=3.0, y=cfg.hull_cy, vx=0, vy=0, theta=0, omega=0, fuel=100)
    assert swept_hits_hull(start, end, cfg)
    assert swept_hits_asteroids(start, end, [(0.0, cfg.hull_cy, 0.4)], cfg.ship_radius)


def test_orbital_relative_mode_adds_hill_dynamics():
    state = ShipState(x=3.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=100.0)
    inertial, _ = step_ship(state, 0.0, 0.0, 0.0, WorldConfig(dt=0.1))
    orbital, _ = step_ship(
        state,
        0.0,
        0.0,
        0.0,
        WorldConfig(dt=0.1, dynamics_mode="orbital_relative", orbital_mean_motion=0.5),
    )
    assert orbital.vx > inertial.vx
