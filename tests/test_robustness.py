import numpy as np

from docking_sim.env.docking_env import DockingEnv
from docking_sim.env.rewards import RewardContext, constrained_docking


def test_actuator_lag_and_scale_are_visible_and_seeded():
    env = DockingEnv(actuator_lag_s=0.2, actuator_scale_std=0.15)
    env.reset(seed=11)
    _obs, _reward, _term, _trunc, first = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    _obs, _reward, _term, _trunc, second = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert first["command_axial"] == 1.0
    assert 0.0 < first["axial"] < 1.0
    assert second["axial"] > first["axial"]
    assert len(first["actuator_scale"]) == 3
    env.close()


def test_persistent_disturbance_has_episode_level_bias():
    env = DockingEnv(disturbance_std=0.5, disturbance_persistence=0.9, disturbance_bias_std=0.2)
    env.reset(seed=7)
    winds = []
    for _ in range(4):
        _obs, _reward, _term, _trunc, info = env.step(np.zeros(3, dtype=np.float32))
        winds.append((info["wind_x"], info["wind_y"]))
    assert winds[0] != winds[1]
    env.reset(seed=7)
    replayed = []
    for _ in range(4):
        _obs, _reward, _term, _trunc, info = env.step(np.zeros(3, dtype=np.float32))
        replayed.append((info["wind_x"], info["wind_y"]))
    assert winds == replayed
    env.close()


def test_sensor_delay_returns_prior_state_observation():
    env = DockingEnv(sensor_delay_steps=1)
    initial, _info = env.reset(seed=0)
    delayed, _reward, _term, _trunc, _info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    current, _reward, _term, _trunc, _info = env.step(np.zeros(3, dtype=np.float32))
    np.testing.assert_allclose(delayed, initial)
    assert not np.allclose(current, initial)
    env.close()


def test_constrained_reward_penalizes_excess_relative_speed():
    context = RewardContext(
        distance=1.0,
        prev_distance=1.1,
        speed=1.2,
        heading_error=0.0,
        fuel_used=0.0,
        success=False,
        crash=False,
        timeout=False,
    )
    reward, parts = constrained_docking(context, {"safety_speed_max": 0.4, "safety_speed": 10.0})
    assert parts["safety"] < 0.0
    assert reward < parts["distance"]
