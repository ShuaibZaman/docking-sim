from docking_sim.env.rewards import RewardContext, distance_only, make_reward, safe_docking


def _ctx(**kwargs) -> RewardContext:
    base = dict(
        distance=4.0,
        prev_distance=4.0,
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


def test_spin_costs_points_even_far_from_the_port():
    weights = {
        "distance": 0.0,
        "velocity": 0.0,
        "rotation": 0.0,
        "fuel": 0.0,
        "time": 0.0,
        "gate": 0.0,
        "spin": 1.0,
    }
    calm, calm_parts = safe_docking(_ctx(distance=10.0, prev_distance=10.0, speed=0.0, heading_error=0.0, omega=0.0), weights)
    spinning, spinning_parts = safe_docking(
        _ctx(distance=10.0, prev_distance=10.0, speed=0.0, heading_error=0.0, omega=2.0),
        weights,
    )
    assert calm_parts["spin"] == 0.0
    assert spinning_parts["spin"] == -2.0
    assert spinning < calm


def test_hovering_far_is_not_taxed_for_distance():
    weights = {
        "distance": 1.0,
        "velocity": 0.0,
        "rotation": 0.0,
        "fuel": 0.0,
        "time": 0.0,
    }
    reward, parts = safe_docking(_ctx(distance=10.0, prev_distance=10.0, speed=0.0), weights)
    assert parts["distance"] == 0.0
    assert reward == 0.0


def _sequence(distances: list[float], *, speed: float, heading: float, success_on_last: bool, crash_on_last: bool) -> float:
    weights = {
        "distance": 1.0,
        "velocity": 0.4,
        "rotation": 0.35,
        "fuel": 0.02,
        "time": 0.01,
        "success": 120.0,
        "crash": 80.0,
        "crash_speed": 15.0,
        "close_range": 4.0,
    }
    total = 0.0
    prev = distances[0]
    for i, distance in enumerate(distances):
        last = i == len(distances) - 1
        reward, _ = safe_docking(
            _ctx(
                distance=distance,
                prev_distance=prev,
                speed=speed,
                heading_error=heading,
                fuel_used=0.05,
                success=last and success_on_last,
                crash=last and crash_on_last,
            ),
            weights,
        )
        total += reward
        prev = distance
    return total


def test_fast_ram_scores_below_slow_dock():
    ram_distances = [10.0 - (9.4 * i / 46.0) for i in range(47)]
    dock_distances = [10.0 - (9.8 * i / 299.0) for i in range(300)]
    ram = _sequence(ram_distances, speed=7.5, heading=0.05, success_on_last=False, crash_on_last=True)
    dock = _sequence(dock_distances, speed=0.2, heading=0.1, success_on_last=True, crash_on_last=False)
    assert ram < dock


def test_eval_summary_and_train_stats():
    from docking_sim.replay.eval import eval_row, summarize_rows
    from docking_sim.training.callbacks import train_stat_record

    result = {
        "seed": 1000,
        "success": True,
        "crash": False,
        "timeout": False,
        "steps": 40,
        "frames": [
            {"fuel": 100.0, "heading_error": 0.2, "speed": 1.0},
            {"fuel": 90.0, "heading_error": 0.1, "speed": 0.2},
        ],
    }
    row = eval_row(result, "ppo_20000_steps", 20000)
    assert row["fuel_used"] == 10.0
    assert row["success"] == 1
    points = summarize_rows(
        [
            row,
            {**row, "seed": 1001, "success": 0, "crash": 1, "steps": 12},
        ]
    )
    assert points[0]["n"] == 2
    assert points[0]["success_rate"] == 0.5
    assert points[0]["median_steps"] == 40

    stats = train_stat_record({"train/approx_kl": 0.01, "train/entropy_loss": -1.2}, 20000)
    assert stats is not None
    assert stats["approx_kl"] == 0.01
    assert stats["entropy"] == -1.2
    assert train_stat_record({}, 1) is None
    from docking_sim.training.sweep import JOBS

    assert len(JOBS) == 13


def test_make_reward_names():
    fn = make_reward("distance_only", {"distance": 2.0})
    reward, parts = fn(_ctx(distance=3.0))
    assert reward == -6.0
    assert parts["distance"] == -6.0
