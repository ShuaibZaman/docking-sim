from docking_sim.config import env_kwargs_from_config, load_yaml
from docking_sim.env.docking_env import DockingEnv
from docking_sim.replay.benchmarks import (
    compatibility,
    get_benchmark,
    get_preset,
    get_scenario,
    list_presets,
)
from docking_sim.replay.eval import latest_outcomes, paired_summary, summarize_rows
from docking_sim.replay.exports import export_bundle
from docking_sim.replay.rollout import rollout


def test_benchmark_manifests_are_versioned_and_stable():
    quick = get_benchmark("quick-20")
    canonical = get_benchmark("canonical-100")
    assert len(quick.scenarios) == 20
    assert len(canonical.scenarios) == 100
    assert quick.scenarios[0].id == canonical.scenarios[0].id
    assert quick.scenarios[0].scenario_hash == canonical.scenarios[0].scenario_hash
    assert {scenario.stratum for scenario in canonical.scenarios} == {
        "straight_in",
        "lateral_offset",
        "drift",
        "precision",
    }


def test_scenario_replay_uses_exact_initial_world():
    suite = get_benchmark("quick-20")
    scenario = get_scenario("quick-20", "static-v1-003")
    env = DockingEnv(**suite.env_kwargs)
    _obs, info = env.reset(seed=scenario.seed, options=scenario.reset_options())
    assert info["scenario_id"] == scenario.id
    assert info["scenario_hash"] == scenario.scenario_hash
    assert info["x"] == scenario.initial_state["x"]
    assert info["y"] == scenario.initial_state["y"]
    env.close()

    left = rollout(
        checkpoint="random",
        scenario=scenario,
        env_kwargs_override=suite.env_kwargs,
        max_steps=10,
    )
    right = rollout(
        checkpoint="random",
        scenario=scenario,
        env_kwargs_override=suite.env_kwargs,
        max_steps=10,
    )
    assert left["scenario_hash"] == scenario.scenario_hash
    assert [frame["x"] for frame in left["frames"]] == [frame["x"] for frame in right["frames"]]


def test_static_candidates_are_compatible_but_curriculum_is_not():
    static = compatibility(
        env_kwargs_from_config(load_yaml("configs/ppo_64.yaml")),
        "quick-20",
    )
    pixels = compatibility(
        env_kwargs_from_config(load_yaml("configs/ppo_cnn_pixels.yaml")),
        "quick-20",
    )
    moving = compatibility(
        env_kwargs_from_config(load_yaml("configs/ppo_level8_moving.yaml")),
        "quick-20",
    )
    constrained = compatibility(
        env_kwargs_from_config(load_yaml("configs/ppo_64_constrained.yaml")),
        "quick-20",
    )
    actuator = compatibility(
        env_kwargs_from_config(load_yaml("configs/ppo_64_actuator.yaml")),
        "quick-20",
    )
    orbital = compatibility(
        env_kwargs_from_config(load_yaml("configs/ppo_orbital.yaml")),
        "quick-20",
    )
    assert static["compatible"] is True
    assert constrained["compatible"] is True
    assert pixels["compatible"] is False
    assert moving["compatible"] is False
    assert actuator["compatible"] is False
    assert orbital["compatible"] is False


def test_paired_eval_summary_uses_common_scenarios_only():
    left = [
        {
            "benchmark_id": "quick-20",
            "timesteps": 100,
            "checkpoint": "final",
            "scenario_id": "static-v1-000",
            "success": 1,
            "crash": 0,
            "fuel_used": 10.0,
            "steps": 20,
            "time": 1.0,
            "terminal_reason": "docked",
            "stratum": "straight_in",
        },
        {
            "benchmark_id": "quick-20",
            "timesteps": 100,
            "checkpoint": "final",
            "scenario_id": "static-v1-001",
            "success": 0,
            "crash": 1,
            "fuel_used": 15.0,
            "steps": 10,
            "time": 0.5,
            "terminal_reason": "hull",
            "stratum": "lateral_offset",
        },
    ]
    right = [
        {
            **left[0],
            "success": 0,
            "crash": 1,
            "fuel_used": 12.0,
            "terminal_reason": "timeout",
        },
        {
            **left[1],
            "success": 0,
            "crash": 1,
            "fuel_used": 14.0,
            "terminal_reason": "bounds",
        },
        {
            **left[1],
            "scenario_id": "static-v1-999",
            "success": 1,
        },
    ]
    pair = paired_summary(left, right, benchmark_id="quick-20")
    assert pair["n_common"] == 2
    assert pair["outcomes"] == {
        "both_success": 0,
        "left_only": 1,
        "right_only": 0,
        "neither": 1,
    }
    assert pair["mean_fuel_delta_left_minus_right"] == -0.5

    points = summarize_rows(left)
    assert points[0]["success_ci_low"] is not None
    assert points[0]["failure_reasons"]["hull"] == 1
    outcomes = latest_outcomes(left, benchmark_id="quick-20")
    assert outcomes[0]["scenario_id"] == "static-v1-000"


def test_showcase_presets_point_at_real_missions():
    presets = list_presets()
    assert {preset["id"] for preset in presets} == {
        "straight-in",
        "lateral-offset",
        "drifting",
        "precision",
    }
    chosen = get_preset("straight-in")
    scenario = get_scenario(chosen["benchmark_id"], chosen["scenario_id"])
    assert chosen["scenario_hash"] == scenario.scenario_hash
    assert chosen["stratum"] == scenario.stratum


def test_export_bundle_refuses_mixed_scenarios():
    suite = get_benchmark("quick-20")
    scenario = get_scenario("quick-20", "static-v1-000")
    other = get_scenario("quick-20", "static-v1-001")
    replay = rollout(
        checkpoint="random",
        scenario=scenario,
        env_kwargs_override=suite.env_kwargs,
        max_steps=4,
    )
    bundle = export_bundle(
        {
            "benchmark": suite.to_dict(),
            "scenario": scenario.to_dict(),
            "panels": [{"candidate": {"checkpoint": "random"}, "label": "random", "replay": replay}],
        }
    )
    assert bundle["kind"] == "docking-lab-replay"
    assert bundle["scenario"]["id"] == scenario.id
    mixed = dict(replay)
    mixed["scenario_id"] = other.id
    mixed["scenario_hash"] = other.scenario_hash
    try:
        export_bundle(
            {
                "benchmark": suite.to_dict(),
                "scenario": scenario.to_dict(),
                "panels": [{"candidate": {"checkpoint": "random"}, "label": "random", "replay": mixed}],
            }
        )
    except ValueError as exc:
        assert "different" in str(exc)
    else:
        raise AssertionError("mixed-scenario export should be refused")
