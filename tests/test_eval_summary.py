from docking_sim.replay.eval import latest_outcomes, summarize_rows
from docking_sim.training.sweep import MATRIX_JOBS, SUITES, selected_jobs


def test_summary_includes_fuel_and_time():
    rows = [
        {"timesteps": 20, "checkpoint": "ppo_20_steps", "seed": 1000, "success": 1, "crash": 0, "steps": 40, "fuel_used": 10.0, "time": 2.0},
        {"timesteps": 20, "checkpoint": "ppo_20_steps", "seed": 1001, "success": 0, "crash": 1, "steps": 80, "fuel_used": 30.0, "time": 4.0},
    ]
    point = summarize_rows(rows)[0]
    assert point["success_rate"] == 0.5
    assert point["mean_fuel"] == 20.0
    assert point["mean_time"] == 3.0
    assert point["median_steps"] == 40.0


def test_old_rows_estimate_time_from_steps():
    rows = [
        {"timesteps": 10, "checkpoint": "final", "seed": 1000, "success": 1, "crash": 0, "steps": 100, "fuel_used": 5.0},
    ]
    point = summarize_rows(rows)[0]
    assert point["mean_fuel"] == 5.0
    assert point["mean_time"] == 5.0
    outcomes = latest_outcomes(rows)
    assert outcomes[0]["seed"] == 1000
    assert outcomes[0]["success"] is True
    assert outcomes[0]["time"] == 5.0


def test_matrix_suite_is_opt_in():
    assert len(selected_jobs(None, None)) == 13
    assert len(selected_jobs(None, None, suite="matrix")) == len(MATRIX_JOBS)
    names = [name for _cfg, name, _seed in SUITES["all"]]
    assert "td3_64" in names
    assert "ppo_residual" in names
    assert "ppo_level8_moving" in names
    assert "ppo_cnn_pixels" in names
    assert "ppo_64_constrained" in names
    assert "ppo_orbital" in names
    assert len(selected_jobs(None, None, suite="robustness")) == 4
