from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from docking_sim.config import ARTIFACTS, env_kwargs_from_config, load_yaml
from docking_sim.replay.benchmarks import (
    compatibility,
    get_benchmark,
    get_preset,
    get_scenario,
    list_benchmarks,
    list_presets,
)
from docking_sim.replay.controllers import list_baselines
from docking_sim.replay.eval import latest_outcomes, paired_summary, read_eval_rows, summarize_rows
from docking_sim.replay.exports import export_bundle
from docking_sim.replay.rollout import list_checkpoints, list_run_dirs, load_run_meta, rollout


class ReplayRequest(BaseModel):
    run_id: str | None = None
    checkpoint: str = "random"
    seed: int = 42
    max_steps: int | None = Field(default=None, ge=1, le=5000)


class CandidateRequest(BaseModel):
    run_id: str | None = None
    checkpoint: str
    baseline: str | None = None


class BenchmarkReplayRequest(BaseModel):
    scenario_id: str
    candidates: list[CandidateRequest] = Field(min_length=1, max_length=4)
    max_steps: int | None = Field(default=None, ge=1, le=5000)


class BenchmarkCompareRequest(BaseModel):
    candidates: list[CandidateRequest] = Field(min_length=2, max_length=4)


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


def _family(meta: dict[str, Any]) -> str:
    if meta.get("family"):
        return str(meta["family"])
    config_name = str(meta.get("config_name") or "")
    stem = config_name.replace(".yaml", "")
    return stem or "run"


def _legacy(meta: dict[str, Any]) -> bool:
    return _family(meta) in {"ppo_mlp_static", "ppo_mlp_smoke", "ppo_mlp_wide", "ppo_mlp_deep"}


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


def _benchmark_candidate(run_dir: Path, benchmark_id: str) -> dict[str, Any]:
    meta = load_run_meta(run_dir)
    config = load_yaml(run_dir / "config.yaml")
    contract = compatibility(env_kwargs_from_config(config), benchmark_id)
    return {
        "id": run_dir.name,
        "label": _family(meta),
        "algorithm": meta.get("algorithm", "PPO"),
        "architecture": meta.get("architecture"),
        "training_seed": meta.get("seed"),
        "checkpoints": list_checkpoints(run_dir),
        **contract,
    }


def _assert_candidate_compatible(candidate: CandidateRequest, benchmark_id: str) -> Path:
    if candidate.baseline:
        return Path()
    if not candidate.run_id:
        raise HTTPException(status_code=422, detail="A benchmark candidate needs run_id or baseline.")
    run_dir = _run_dir_or_404(candidate.run_id)
    summary = _benchmark_candidate(run_dir, benchmark_id)
    if not summary["compatible"]:
        raise HTTPException(status_code=409, detail=summary["reason"])
    checkpoint_ids = {item["id"] for item in summary["checkpoints"]}
    if candidate.checkpoint not in checkpoint_ids:
        raise HTTPException(
            status_code=404,
            detail=f"checkpoint not found for candidate: {candidate.checkpoint}",
        )
    return run_dir


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/presets")
def get_presets() -> list[dict[str, Any]]:
    return list_presets()


@app.get("/api/presets/{preset_id}")
def get_preset_manifest(preset_id: str) -> dict[str, Any]:
    try:
        return get_preset(preset_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmarks")
def get_benchmarks() -> list[dict[str, Any]]:
    return list_benchmarks()


@app.get("/api/benchmarks/{benchmark_id}")
def get_benchmark_manifest(benchmark_id: str) -> dict[str, Any]:
    try:
        return get_benchmark(benchmark_id).to_dict(include_scenarios=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/benchmarks/{benchmark_id}/candidates")
def get_benchmark_candidates(benchmark_id: str) -> list[dict[str, Any]]:
    try:
        get_benchmark(benchmark_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    candidates = [_benchmark_candidate(run_dir, benchmark_id) for run_dir in list_run_dirs()]
    candidates.extend(
        {
            "id": baseline["id"],
            "baseline": baseline["id"],
            "label": baseline["label"],
            "description": baseline["description"],
            "compatible": True,
            "reason": None,
            "checkpoints": [{"id": baseline["id"], "label": baseline["label"], "steps": 0, "path": None}],
        }
        for baseline in list_baselines()
    )
    return candidates


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
                "family": _family(meta),
                "legacy": _legacy(meta),
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


@app.get("/api/runs/{run_id}/eval")
def get_eval(run_id: str, benchmark_id: str | None = None) -> dict[str, Any]:
    run_dir = _run_dir_or_404(run_id)
    meta = load_run_meta(run_dir)
    rows = read_eval_rows(run_dir / "eval.jsonl")
    if benchmark_id is not None:
        rows = [row for row in rows if row.get("benchmark_id") == benchmark_id]
    points = summarize_rows(rows)
    return {
        "id": run_id,
        "family": _family(meta),
        "legacy": _legacy(meta),
        "seed": meta.get("seed", 42),
        "points": points,
        "latest_outcomes": latest_outcomes(rows, benchmark_id=benchmark_id),
    }


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


@app.post("/api/benchmarks/{benchmark_id}/replay")
def post_benchmark_replay(benchmark_id: str, body: BenchmarkReplayRequest) -> dict[str, Any]:
    try:
        suite = get_benchmark(benchmark_id)
        scenario = get_scenario(benchmark_id, body.scenario_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    panels = []
    for candidate in body.candidates:
        _assert_candidate_compatible(candidate, benchmark_id)
        try:
            result = rollout(
                run_id=candidate.run_id,
                checkpoint=candidate.checkpoint,
                seed=scenario.seed,
                scenario=scenario,
                env_kwargs_override=suite.env_kwargs,
                baseline=candidate.baseline,
                max_steps=body.max_steps,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        panels.append(
            {
                "candidate": candidate.model_dump(),
                "label": candidate.baseline or candidate.run_id,
                "replay": result,
            }
        )
    return {
        "benchmark": suite.to_dict(),
        "scenario": scenario.to_dict(),
        "panels": panels,
    }


@app.post("/api/benchmarks/{benchmark_id}/export")
def post_benchmark_export(benchmark_id: str, body: BenchmarkReplayRequest) -> dict[str, Any]:
    payload = post_benchmark_replay(benchmark_id, body)
    try:
        return export_bundle(payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/benchmarks/{benchmark_id}/compare")
def post_benchmark_compare(benchmark_id: str, body: BenchmarkCompareRequest) -> dict[str, Any]:
    try:
        get_benchmark(benchmark_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    for candidate in body.candidates:
        _assert_candidate_compatible(candidate, benchmark_id)
        if candidate.baseline:
            raise HTTPException(
                status_code=422,
                detail="Stored benchmark comparisons currently require trained-run candidates.",
            )
    rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for candidate in body.candidates:
        assert candidate.run_id is not None
        run_dir = _run_dir_or_404(candidate.run_id)
        rows_by_candidate[candidate.run_id] = read_eval_rows(run_dir / "eval.jsonl")
    pairwise = []
    for index, left in enumerate(body.candidates):
        for right in body.candidates[index + 1 :]:
            assert left.run_id is not None and right.run_id is not None
            pairwise.append(
                {
                    "left": left.model_dump(),
                    "right": right.model_dump(),
                    "summary": paired_summary(
                        rows_by_candidate[left.run_id],
                        rows_by_candidate[right.run_id],
                        benchmark_id=benchmark_id,
                        left_checkpoint=left.checkpoint,
                        right_checkpoint=right.checkpoint,
                    ),
                }
            )
    return {"benchmark_id": benchmark_id, "pairwise": pairwise}
