from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from docking_sim.config import ARTIFACTS
from docking_sim.replay.rollout import list_checkpoints, list_run_dirs, load_run_meta, rollout


class ReplayRequest(BaseModel):
    run_id: str | None = None
    checkpoint: str = "random"
    seed: int = 42
    max_steps: int | None = Field(default=None, ge=1, le=5000)


app = FastAPI(title="Docking Lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_dir_or_404(run_id: str) -> Path:
    run_dir = ARTIFACTS / run_id
    if not run_dir.exists() or not (run_dir / "config.yaml").exists():
        raise HTTPException(status_code=404, detail=f"run not found: {run_id}")
    return run_dir


def _metrics_summary(run_dir: Path, limit: int = 400) -> dict[str, Any]:
    path = run_dir / "metrics.jsonl"
    episodes: list[dict[str, Any]] = []
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    episodes.append(json.loads(line))
    window = episodes[-20:] if episodes else []
    success_rate = (
        sum(int(ep.get("success", 0)) for ep in window) / len(window) if window else 0.0
    )
    return {
        "episode_count": len(episodes),
        "success_rate_window": success_rate,
        "episodes": episodes[-limit:],
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def get_runs() -> list[dict[str, Any]]:
    runs = []
    for run_dir in list_run_dirs():
        meta = load_run_meta(run_dir)
        summary = _metrics_summary(run_dir, limit=0)
        runs.append(
            {
                "id": run_dir.name,
                "path": str(run_dir),
                "algorithm": meta.get("algorithm", "PPO"),
                "net_arch": meta.get("net_arch", []),
                "n_params": meta.get("n_params", 0),
                "obs_mode": meta.get("obs_mode", "state"),
                "seed": meta.get("seed", 42),
                "total_timesteps": meta.get("total_timesteps", 0),
                "config_name": meta.get("config_name", ""),
                "episode_count": summary["episode_count"],
                "success_rate_window": summary["success_rate_window"],
            }
        )
    return runs


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir_or_404(run_id)
    meta = load_run_meta(run_dir)
    return {"id": run_id, "meta": meta, "checkpoints": list_checkpoints(run_dir)}


@app.get("/api/runs/{run_id}/checkpoints")
def get_checkpoints(run_id: str) -> list[dict[str, Any]]:
    run_dir = _run_dir_or_404(run_id)
    return list_checkpoints(run_dir)


@app.get("/api/runs/{run_id}/metrics")
def get_metrics(run_id: str) -> dict[str, Any]:
    run_dir = _run_dir_or_404(run_id)
    return _metrics_summary(run_dir)


@app.post("/api/replay")
def post_replay(body: ReplayRequest) -> dict[str, Any]:
    try:
        return rollout(
            run_id=body.run_id,
            checkpoint=body.checkpoint,
            seed=body.seed,
            max_steps=body.max_steps,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
