import numpy as np

from docking_sim.replay.rollout import policy_action_to_commands, rollout


def test_random_rollout_is_seed_deterministic():
    a = rollout(checkpoint="random", seed=3, max_steps=25)
    b = rollout(checkpoint="random", seed=3, max_steps=25)
    assert [frame["x"] for frame in a["frames"]] == [frame["x"] for frame in b["frames"]]
    assert [frame["theta"] for frame in a["frames"]] == [frame["theta"] for frame in b["frames"]]
    assert a["steps"] == b["steps"]
    assert len(a["frames"]) >= 2


def test_rollout_exposes_dock_limits_and_crash_flags():
    data = rollout(checkpoint="random", seed=3, max_steps=25)
    world = data["world"]
    assert world["dock_speed_max"] == 0.35
    assert abs(world["dock_angle_max_deg"] - 12.0) < 1e-6
    assert world["approach_angle"] > 0
    for frame in data["frames"]:
        assert isinstance(frame["hit_hull"], bool)
        assert isinstance(frame["out_of_bounds"], bool)
    assert data["frames"][0]["hit_hull"] is False
    assert data["frames"][0]["out_of_bounds"] is False
    assert "axial" in data["frames"][1]
    assert "lateral" in data["frames"][1]
    assert "yaw" in data["frames"][1]


def test_two_action_policy_maps_to_forward_only():
    commands = policy_action_to_commands(np.array([-1.0, 0.5], dtype=np.float32))
    assert commands.tolist() == [0.0, 0.0, 0.5]
    full = policy_action_to_commands(np.array([1.0, 0.25], dtype=np.float32))
    assert abs(float(full[0]) - 1.0) < 1e-6
    assert float(full[1]) == 0.0
    assert abs(float(full[2]) - 0.25) < 1e-6


def test_three_action_policy_keeps_strafe_and_brake():
    commands = policy_action_to_commands(np.array([-0.4, 0.7, -0.2], dtype=np.float32))
    np.testing.assert_allclose(commands, [-0.4, 0.7, -0.2])
