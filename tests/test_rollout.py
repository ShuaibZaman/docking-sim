import numpy as np

from docking_sim.replay.benchmarks import get_benchmark, get_scenario
from docking_sim.replay.controllers import list_baselines
from docking_sim.replay.rollout import command_fields, policy_action_to_commands, rollout


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


def test_command_fields_fill_missing_pair_from_the_other():
    from_legacy = command_fields({"thrust": 0.4, "torque": -0.2})
    assert from_legacy["axial"] == 0.4
    assert from_legacy["yaw"] == -0.2
    assert from_legacy["thrust"] == 0.4
    assert from_legacy["torque"] == -0.2
    assert from_legacy["lateral"] == 0.0

    from_new = command_fields({"axial": -0.5, "lateral": 0.25, "yaw": 0.1})
    assert from_new["thrust"] == -0.5
    assert from_new["torque"] == 0.1
    assert from_new["lateral"] == 0.25

    both = command_fields({"axial": 1.0, "yaw": 0.3, "thrust": 0.0, "torque": 0.0})
    assert both["axial"] == 1.0
    assert both["thrust"] == 1.0
    assert both["yaw"] == 0.3
    assert both["torque"] == 0.3


def test_three_action_policy_keeps_strafe_and_brake():
    commands = policy_action_to_commands(np.array([-0.4, 0.7, -0.2], dtype=np.float32))
    np.testing.assert_allclose(commands, [-0.4, 0.7, -0.2])


def test_deterministic_controller_baseline_replays_exact_scenario():
    suite = get_benchmark("quick-20")
    scenario = get_scenario("quick-20", "static-v1-000")
    a = rollout(
        checkpoint="pd",
        baseline="pd",
        scenario=scenario,
        env_kwargs_override=suite.env_kwargs,
        max_steps=20,
    )
    b = rollout(
        checkpoint="pd",
        baseline="pd",
        scenario=scenario,
        env_kwargs_override=suite.env_kwargs,
        max_steps=20,
    )
    assert {item["id"] for item in list_baselines()} == {"random", "pd"}
    assert a["baseline"] == "pd"
    assert [frame["x"] for frame in a["frames"]] == [frame["x"] for frame in b["frames"]]
