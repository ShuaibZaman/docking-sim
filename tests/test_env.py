import numpy as np

from docking_sim.env.docking_env import DockingEnv
from docking_sim.env.physics import ShipState, docking_success, hits_hull


def test_port_success_when_slow_and_aligned():
    env = DockingEnv(max_steps=10)
    cfg = env.cfg
    state = ShipState(
        x=cfg.port_cx,
        y=cfg.port_cy,
        vx=0.02,
        vy=0.02,
        theta=cfg.port_approach_angle,
        omega=0.0,
        fuel=cfg.fuel_capacity,
    )
    assert docking_success(state, cfg)
    env.reset(seed=0, options={"state": state})
    _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert not truncated
    assert info["success"]
    assert not info["crash"]
    env.close()


def test_hull_collision_is_crash():
    env = DockingEnv(max_steps=10)
    cfg = env.cfg
    state = ShipState(
        x=cfg.hull_cx,
        y=cfg.hull_cy,
        vx=0.0,
        vy=0.0,
        theta=0.0,
        omega=0.0,
        fuel=cfg.fuel_capacity,
    )
    assert hits_hull(state, cfg)
    env.reset(seed=0, options={"state": state})
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["crash"]
    assert not info["success"]
    env.close()


def test_reset_seed_is_deterministic():
    env = DockingEnv(randomize_start=True, randomize_velocity=True)
    obs_a, _ = env.reset(seed=7)
    actions = [env.action_space.sample() for _ in range(12)]
    traj_a = [obs_a.copy()]
    for action in actions:
        obs, _, term, trunc, _ = env.step(action)
        traj_a.append(obs.copy())
        if term or trunc:
            break

    obs_b, _ = env.reset(seed=7)
    traj_b = [obs_b.copy()]
    for action in actions[: len(traj_a) - 1]:
        obs, _, term, trunc, _ = env.step(action)
        traj_b.append(obs.copy())
        if term or trunc:
            break

    assert len(traj_a) == len(traj_b)
    for a, b in zip(traj_a, traj_b):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-6)
    env.close()


def test_rgb_array_shape():
    env = DockingEnv(render_mode="rgb_array")
    env.reset(seed=1)
    frame = env.render()
    assert frame is not None
    assert frame.shape == (128, 160, 3)
    assert frame.dtype == np.uint8
    env.close()
