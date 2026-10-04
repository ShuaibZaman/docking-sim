from docking_sim.env.physics import ShipState, WorldConfig, step_ship


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
