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
