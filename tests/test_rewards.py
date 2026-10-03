from docking_sim.env.rewards import RewardContext, distance_only, make_reward, safe_docking


def _ctx(**kwargs) -> RewardContext:
    base = dict(
        distance=4.0,
        speed=1.0,
        heading_error=0.5,
        fuel_used=0.2,
        success=False,
        crash=False,
        timeout=False,
    )
    base.update(kwargs)
    return RewardContext(**base)


def test_distance_only_penalizes_range():
    weights = {"distance": 1.0, "success": 0.0, "crash": 0.0}
    near, _ = distance_only(_ctx(distance=1.0), weights)
    far, _ = distance_only(_ctx(distance=5.0), weights)
    assert near > far


def test_safe_docking_success_bonus():
    weights = {"success": 120.0, "crash": 80.0}
    miss, _ = safe_docking(_ctx(success=False), weights)
    hit, parts = safe_docking(_ctx(success=True, distance=0.2, speed=0.1, heading_error=0.05), weights)
    assert hit > miss
    assert parts["terminal"] == 120.0


def test_make_reward_names():
    fn = make_reward("distance_only", {"distance": 2.0})
    reward, parts = fn(_ctx(distance=3.0))
    assert reward == -6.0
    assert parts["distance"] == -6.0
