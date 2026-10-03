from docking_sim.replay.rollout import rollout


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
