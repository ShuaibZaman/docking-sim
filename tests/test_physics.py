from docking_sim.env.physics import ShipState, WorldConfig, step_ship


def test_thrust_increases_speed_along_heading():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0)
    state = ShipState(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=100.0)
    nxt, fuel_used = step_ship(state, thrust_cmd=1.0, torque_cmd=0.0, cfg=cfg)
    assert nxt.vx > state.vx
    assert abs(nxt.vy) < 1e-9
    assert fuel_used > 0.0


def test_torque_changes_angular_velocity():
    cfg = WorldConfig(dt=0.05)
    state = ShipState(x=0.0, y=0.0, vx=0.0, vy=0.0, theta=0.0, omega=0.0, fuel=100.0)
    nxt, _ = step_ship(state, thrust_cmd=0.0, torque_cmd=1.0, cfg=cfg)
    assert nxt.omega > 0.0


def test_empty_fuel_ignores_commands():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0, angular_damping=0.0)
    state = ShipState(x=1.0, y=-2.0, vx=0.3, vy=-0.1, theta=0.4, omega=0.2, fuel=0.0)
    nxt, fuel_used = step_ship(state, thrust_cmd=1.0, torque_cmd=1.0, cfg=cfg)
    assert fuel_used == 0.0
    assert nxt.vx == state.vx
    assert nxt.vy == state.vy
    assert nxt.omega == state.omega
    assert nxt.fuel == 0.0
