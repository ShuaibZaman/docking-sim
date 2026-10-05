import numpy as np

from docking_sim.config import env_kwargs_from_config, load_yaml
from docking_sim.env.docking_env import EXTENDED_OBS_DIM, DockingEnv
from docking_sim.env.physics import ShipState, StationPose, WorldConfig, port_world_center, step_ship, step_station


def test_disturbance_adds_sideways_acceleration():
    cfg = WorldConfig(dt=0.05, linear_damping=0.0, angular_damping=0.0)
    state = step_ship(
        ShipState(x=0, y=0, vx=0, vy=0, theta=0, omega=0, fuel=100),
        0.0,
        0.0,
        0.0,
        cfg,
        disturbance=(0.0, 4.0),
    )[0]
    assert state.vy > 0.0


def test_station_motion_and_spin_move_the_port():
    cfg = WorldConfig(dt=0.05, station_omega=0.0)
    bounced = step_station(StationPose(x=7.4, vx=20.0), cfg)
    assert bounced.vx < 0.0

    spinning = step_station(StationPose(theta=0.0, vx=0.0, vy=0.0), WorldConfig(dt=0.05, station_omega=1.0))
    assert spinning.theta != 0.0
    moved = port_world_center(cfg, StationPose(theta=0.4))
    assert abs(moved[0] - cfg.port_cx) > 1e-3


def test_wind_changes_the_trajectory():
    calm = DockingEnv(disturbance_std=0.0, max_steps=30)
    gusty = DockingEnv(disturbance_std=4.0, extended_obs=True, max_steps=30)
    calm.reset(seed=2)
    obs, _ = gusty.reset(seed=2)
    assert obs.shape == (EXTENDED_OBS_DIM,)
    action = np.zeros(3, dtype=np.float32)
    calm_y = []
    gust_y = []
    for _ in range(12):
        _, _, _, _, info = calm.step(action)
        calm_y.append(info["y"])
        _, _, _, _, info = gusty.step(action)
        gust_y.append(info["y"])
    assert calm_y != gust_y
    calm.close()
    gusty.close()


def test_empty_fuel_ends_the_episode_when_limited():
    env = DockingEnv(fuel_capacity=0.3, fail_on_empty_fuel=True, max_steps=20)
    env.reset(seed=0)
    _, _, terminated, _, info = env.step(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert terminated
    assert info["out_of_fuel"]
    assert info["crash"]
    assert not info["success"]
    env.close()


def test_asteroid_contact_is_a_crash_and_spawn_is_stable():
    env = DockingEnv(n_asteroids=5, extended_obs=True)
    first, _ = env.reset(seed=3)
    rocks = list(env._asteroids)
    assert len(rocks) == 5
    assert first.shape == (EXTENDED_OBS_DIM,)
    again, _ = env.reset(seed=3)
    assert env._asteroids == rocks
    np.testing.assert_allclose(first, again)

    env._asteroids = [(env._state.x, env._state.y, 0.45)]
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["hit_asteroid"]
    assert info["crash"]
    assert info["asteroids"]
    env.close()


def test_rotating_station_changes_the_port_pose():
    env = DockingEnv(station_omega=1.0, extended_obs=True, max_steps=20)
    _, info = env.reset(seed=0)
    assert info["station"]["theta"] == 0.0
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["station"]["theta"] != 0.0
    assert abs(info["port_pose"]["cx"] - env.cfg.port_cx) > 1e-3
    env.close()


def test_moving_station_leaves_home():
    env = DockingEnv(station_vx=0.5, extended_obs=True, max_steps=40)
    env.reset(seed=0)
    home = env.cfg.hull_cx
    for _ in range(10):
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["station"]["cx"] != home
    env.close()


def test_pixel_observation_is_an_image():
    env = DockingEnv(obs_mode="pixels", n_asteroids=3, station_omega=0.3)
    obs, info = env.reset(seed=1)
    assert obs.shape == (128, 160, 3)
    assert obs.dtype == np.uint8
    assert info["asteroids"]
    stepped, _, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    assert stepped.shape == obs.shape
    env.close()


def test_level_configs_change_one_difficulty():
    expected = {
        "configs/ppo_level2_spawn.yaml": {"randomize_start": True, "level": 2},
        "configs/ppo_level3_velocity.yaml": {"randomize_velocity": True, "level": 3},
        "configs/ppo_level4_rotate.yaml": {"station_omega": 0.2, "extended": True, "level": 4},
        "configs/ppo_level5_wind.yaml": {"disturbance_std": 0.8, "extended": True, "level": 5},
        "configs/ppo_level6_fuel.yaml": {"fuel": 30.0, "fail_on_empty_fuel": True, "level": 6},
        "configs/ppo_level7_asteroids.yaml": {"n_asteroids": 5, "extended": True, "level": 7},
        "configs/ppo_level8_moving.yaml": {"station_vx": 0.35, "extended": True, "level": 8},
        "configs/ppo_level9_long.yaml": {"start_y": -7.2, "level": 9},
        "configs/ppo_level10_corridor.yaml": {"n_asteroids": 3, "corridor": True, "extended": True, "level": 10},
        "configs/ppo_level11_transfer.yaml": {"second_port": True, "level": 11},
        "configs/ppo_cnn_pixels.yaml": {"pixels": True, "level": 1},
    }
    for path, expect in expected.items():
        env = DockingEnv(**env_kwargs_from_config(load_yaml(path)))
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        assert env.level == expect["level"]
        if "randomize_start" in expect:
            assert env.randomize_start is True
            assert obs.shape == (9,)
        if expect.get("randomize_velocity"):
            assert env.randomize_velocity is True
        if "station_omega" in expect:
            assert env.cfg.station_omega == expect["station_omega"]
        if "disturbance_std" in expect:
            assert env.cfg.disturbance_std == expect["disturbance_std"]
        if "fuel" in expect:
            assert env.cfg.fuel_capacity == expect["fuel"]
            assert env.fail_on_empty_fuel is True
            assert obs.shape == (9,)
        if "n_asteroids" in expect:
            assert len(env._asteroids) == expect["n_asteroids"]
        if "station_vx" in expect:
            assert env.cfg.station_vx == expect["station_vx"]
        if "start_y" in expect:
            assert env._state.y == expect["start_y"]
        if expect.get("corridor"):
            assert env.corridor_obstacles is True
        if expect.get("second_port"):
            assert env.cfg.second_port is True
            assert obs.shape == (10,)
        if expect.get("extended"):
            assert obs.shape == (EXTENDED_OBS_DIM,)
        if expect.get("pixels"):
            assert obs.shape == (128, 160, 3)
            assert env.obs_mode == "pixels"
        env.close()


def test_two_port_hold_switches_target_without_ending():
    env = DockingEnv(**env_kwargs_from_config(load_yaml("configs/ppo_level11_transfer.yaml")))
    env.reset(seed=0)
    env._state = ShipState(
        x=env.cfg.port_cx,
        y=env.cfg.port_cy,
        vx=0.0,
        vy=0.0,
        theta=env.cfg.port_approach_angle,
        omega=0.0,
        fuel=env.cfg.fuel_capacity,
    )
    info = {}
    for _ in range(int(env.cfg.hold_steps)):
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        assert not terminated
        assert not truncated
    assert info["phase"] == "transfer"
    assert info["active_port"] == 1
    assert info["ports"][1]["active"] is True
    assert info["success"] is False
    assert abs(info["port_pose"]["cy"] - env.cfg.port2_cy) < 1e-6
    env.close()
