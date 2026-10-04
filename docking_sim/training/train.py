from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from docking_sim.config import ARTIFACTS, env_kwargs_from_config, load_yaml
from docking_sim.env.docking_env import DockingEnv
from docking_sim.replay.eval import append_eval_rows, evaluate_checkpoint
from docking_sim.replay.rollout import (
    _vecnormalize_for_checkpoint,
    checkpoint_steps,
    load_policy,
    load_run_meta,
)
from docking_sim.training.builders import architecture_label, build_model, canonical_optimizer, make_policy_kwargs
from docking_sim.training.losses import canonical_loss
from docking_sim.training.provenance import run_provenance
from docking_sim.training.callbacks import (
    HeldOutEvalCallback,
    MetricsJsonlCallback,
    TrainStatsCallback,
    VecNormalizeCheckpointCallback,
)


def make_env(env_kwargs: dict[str, Any], seed: int, rank: int = 0):
    def _init() -> Monitor:
        env = DockingEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.policy.parameters()))


def resolve_net_arch(algo: dict[str, Any]) -> list[int]:
    if "net_arch" in algo and algo["net_arch"] is not None:
        return [int(size) for size in algo["net_arch"]]
    return [64, 64]


def build_run_dir(config_path: Path, name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = name or config_path.stem
    run_dir = ARTIFACTS / f"{stamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir


def _resume_run_dir(value: str | Path) -> Path:
    candidate = Path(value)
    run_dir = candidate if candidate.exists() else ARTIFACTS / str(value)
    if not run_dir.exists() or not (run_dir / "config.yaml").exists():
        raise FileNotFoundError(f"Resume run not found: {value}")
    return run_dir


def _latest_checkpoint(run_dir: Path) -> tuple[Path, str]:
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("*.zip"),
        key=lambda path: checkpoint_steps(path.stem),
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint is available to resume: {run_dir}")
    path = checkpoints[-1]
    return path, path.stem


def _write_run_marker(run_dir: Path, state: str, **details: Any) -> None:
    marker = run_dir / "RUNNING"
    if state == "done":
        marker.unlink(missing_ok=True)
        return
    payload = {
        "state": state,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        **details,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(
    config_path: Path,
    name: str | None = None,
    seed_override: int | None = None,
    resume_run: str | Path | None = None,
    additional_timesteps: int | None = None,
) -> Path:
    cfg = load_yaml(config_path)
    env_kwargs = env_kwargs_from_config(cfg)
    algo = dict(cfg.get("algo") or {})
    train_cfg = cfg.get("train") or {}
    obs_mode = str(env_kwargs.get("obs_mode", "state"))
    if obs_mode == "pixels" and str(algo.get("policy", "MlpPolicy")) == "MlpPolicy":
        algo["policy"] = "CnnPolicy"
        algo.setdefault("features", "cnn")

    seed = int(seed_override if seed_override is not None else train_cfg.get("seed", 42))
    normalize = bool(algo.get("normalize_obs", True)) and obs_mode != "pixels"
    total_timesteps = int(
        additional_timesteps
        if additional_timesteps is not None
        else train_cfg.get("total_timesteps", 200_000)
    )
    checkpoint_every = int(train_cfg.get("checkpoint_every", 20_000))
    device = str(train_cfg.get("device", "cpu"))
    net_arch = resolve_net_arch(algo)
    algo_name = str(algo.get("name", "ppo")).lower()
    policy_kwargs = make_policy_kwargs(algo, net_arch)

    resumed = resume_run is not None
    if resumed:
        run_dir = _resume_run_dir(resume_run)
        saved_config_path = run_dir / "config.yaml"
        requested_provenance = run_provenance(config_path, env_kwargs)
        saved_provenance = run_provenance(saved_config_path, env_kwargs)
        if requested_provenance["config_sha256"] != saved_provenance["config_sha256"]:
            raise ValueError(
                "Refusing to resume with a different config. Use the original run config or start a new run."
            )
        meta = load_run_meta(run_dir)
        existing_scene = meta.get("provenance", {}).get("scene_fingerprint")
        if existing_scene and existing_scene != requested_provenance["scene_fingerprint"]:
            raise ValueError("Refusing to resume: the saved scene contract does not match this config.")
    else:
        run_dir = build_run_dir(config_path, name)
        meta = {}

    base_env = DummyVecEnv([make_env(env_kwargs, seed=seed)])
    if normalize:
        if resumed:
            _checkpoint_path, checkpoint_id = _latest_checkpoint(run_dir)
            vecnormalize_path = _vecnormalize_for_checkpoint(run_dir, checkpoint_id)
            if vecnormalize_path is None:
                raise FileNotFoundError(
                    f"Cannot safely resume normalized run without VecNormalize state: {run_dir}"
                )
            env = VecNormalize.load(str(vecnormalize_path), base_env)
            env.training = True
            env.norm_reward = True
        else:
            env = VecNormalize(
                base_env,
                norm_obs=True,
                norm_reward=True,
                clip_obs=10.0,
                gamma=float(algo.get("gamma", 0.99)),
            )
    else:
        env = base_env

    if resumed:
        checkpoint_path, checkpoint_id = _latest_checkpoint(run_dir)
        model = load_policy(checkpoint_path, algo_name)
        model.critic_loss_name = canonical_loss(str(algo.get("critic_loss", "mse")))
        model.set_env(env)
        start_timesteps = int(model.num_timesteps)
        history = list(meta.get("resume_history") or [])
        history.append(
            {
                "checkpoint": checkpoint_id,
                "from_timesteps": start_timesteps,
                "additional_timesteps": total_timesteps,
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
        meta["resume_history"] = history
    else:
        model = build_model(algo_name, env, algo, seed, device, net_arch, run_dir)
        start_timesteps = 0
        meta = {
            "algorithm": algo_name.upper(),
            "policy": algo.get("policy", "MlpPolicy"),
            "net_arch": list(policy_kwargs.get("net_arch") or []),
            "architecture": architecture_label(algo, list(policy_kwargs.get("net_arch") or net_arch), obs_mode),
            "features": str(algo.get("features", "cnn" if obs_mode == "pixels" else "mlp")),
            "critic_loss": canonical_loss(str(algo.get("critic_loss", "mse"))),
            "optimizer": canonical_optimizer(str(algo.get("optimizer", "adam"))),
            "n_params": count_params(model),
            "obs_mode": obs_mode,
            "normalize_obs": normalize,
            "level": (cfg.get("env") or {}).get("level"),
            "seed": seed,
            "config_name": config_path.name,
            "family": config_path.stem,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "provenance": run_provenance(run_dir / "config.yaml", env_kwargs),
            "resume_history": [],
        }
    meta["total_timesteps"] = start_timesteps + total_timesteps
    (run_dir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=checkpoint_every,
                save_path=str(run_dir / "checkpoints"),
                name_prefix=algo_name,
                save_replay_buffer=False,
                save_vecnormalize=False,
            ),
            VecNormalizeCheckpointCallback(checkpoint_every, run_dir / "checkpoints"),
            HeldOutEvalCallback(run_dir, checkpoint_every, algo_name),
            MetricsJsonlCallback(run_dir / "metrics.jsonl"),
            TrainStatsCallback(run_dir / "train_stats.jsonl"),
        ]
    )

    _write_run_marker(
        run_dir,
        "training",
        resumed=resumed,
        from_timesteps=start_timesteps,
        additional_timesteps=total_timesteps,
    )
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=not resumed,
    )
    model.save(str(run_dir / "checkpoints" / "final"))
    if isinstance(env, VecNormalize):
        env.save(str(run_dir / "vecnormalize.pkl"))
        env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))

    try:
        final_rows = evaluate_checkpoint(run_dir.name, "final", timesteps=int(model.num_timesteps))
    except ValueError as exc:
        print(f"Skipping final held-out benchmark eval: {exc}", flush=True)
    else:
        append_eval_rows(run_dir / "eval.jsonl", final_rows)

    (run_dir / "DONE").write_text("ok\n", encoding="utf-8")
    _write_run_marker(run_dir, "done")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a docking policy")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None, help="Run directory or run ID to resume safely.")
    parser.add_argument(
        "--additional-timesteps",
        type=int,
        default=None,
        help="Timesteps to add when resuming; otherwise overrides config total_timesteps.",
    )
    args = parser.parse_args(argv)
    run_dir = train(
        args.config,
        args.name,
        seed_override=args.seed,
        resume_run=args.resume,
        additional_timesteps=args.additional_timesteps,
    )
    print(f"Run artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
