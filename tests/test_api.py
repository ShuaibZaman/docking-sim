from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_replay_random():
    response = client.post("/api/replay", json={"checkpoint": "random", "seed": 5, "max_steps": 30})
    assert response.status_code == 200
    data = response.json()
    assert data["checkpoint"] == "random"
    assert len(data["frames"]) >= 2
    assert "x" in data["frames"][0]


def test_benchmark_manifest_candidates_and_exact_baseline_replay():
    suites = client.get("/api/benchmarks")
    assert suites.status_code == 200
    assert {item["id"] for item in suites.json()} == {"quick-20", "canonical-100"}

    manifest = client.get("/api/benchmarks/quick-20")
    assert manifest.status_code == 200
    body = manifest.json()
    assert body["n_scenarios"] == 20
    scenario = body["scenarios"][0]
    candidates = client.get("/api/benchmarks/quick-20/candidates")
    assert candidates.status_code == 200
    assert any(item.get("baseline") == "pd" for item in candidates.json())

    replay = client.post(
        "/api/benchmarks/quick-20/replay",
        json={
            "scenario_id": scenario["id"],
            "max_steps": 20,
            "candidates": [{"baseline": "pd", "checkpoint": "pd"}],
        },
    )
    assert replay.status_code == 200
    panel = replay.json()["panels"][0]["replay"]
    assert panel["scenario_id"] == scenario["id"]
    assert panel["scenario_hash"] == scenario["scenario_hash"]
    assert panel["baseline"] == "pd"


def test_presets_and_export_keep_the_same_scenario():
    presets = client.get("/api/presets")
    assert presets.status_code == 200
    chosen = presets.json()[0]
    replay = client.post(
        f"/api/benchmarks/{chosen['benchmark_id']}/export",
        json={
            "scenario_id": chosen["scenario_id"],
            "max_steps": 8,
            "candidates": [{"baseline": "pd", "checkpoint": "pd"}],
        },
    )
    assert replay.status_code == 200
    body = replay.json()
    assert body["kind"] == "docking-lab-replay"
    assert body["scenario"]["id"] == chosen["scenario_id"]
    assert body["panels"][0]["replay"]["scenario_hash"] == chosen["scenario_hash"]


def test_unknown_benchmark_scenario_is_not_replayed():
    response = client.post(
        "/api/benchmarks/quick-20/replay",
        json={
            "scenario_id": "missing",
            "candidates": [{"baseline": "random", "checkpoint": "random"}],
        },
    )
    assert response.status_code == 404


def test_list_runs_and_checkpoint_replay():
    runs = client.get("/api/runs").json()
    if not runs:
        return
    run_id = runs[0]["id"]
    checkpoints = client.get(f"/api/runs/{run_id}/checkpoints").json()
    ids = {c["id"] for c in checkpoints}
    assert "random" in ids
    assert "final" in ids
    replay = client.post(
        "/api/replay",
        json={"run_id": run_id, "checkpoint": "final", "seed": 42, "max_steps": 40},
    )
    assert replay.status_code == 200
    body = replay.json()
    assert body["checkpoint"] == "final"
    assert len(body["frames"]) >= 2
    assert "station" in body["frames"][0]
    assert "port_pose" in body["frames"][0]
    evaluation = client.get(f"/api/runs/{run_id}/eval")
    assert evaluation.status_code == 200
    payload = evaluation.json()
    assert "latest_outcomes" in payload
    if payload["points"]:
        assert "mean_fuel" in payload["points"][-1]
        assert "mean_time" in payload["points"][-1]
