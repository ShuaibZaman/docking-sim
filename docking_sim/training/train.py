from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from docking_sim.config import ARTIFACTS, env_kwargs_from_config, load_yaml
from docking_sim.env.docking_env import DockingEnv
from docking_sim.replay.eval import append_eval_rows, evaluate_checkpoint
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


def build_model(algo_name: str, env, algo: dict[str, Any], seed: int, device: str, net_arch: list[int], run_dir: Path):
    name = algo_name.lower()
    if name == "sac":
        return SAC(
            policy=str(algo.get("policy", "MlpPolicy")),
            env=env,
            learning_rate=float(algo.get("learning_rate", 3e-4)),
            buffer_size=int(algo.get("buffer_size", 1_000_000)),
            learning_starts=int(algo.get("learning_starts", 100)),
            batch_size=int(algo.get("batch_size", 256)),
            tau=float(algo.get("tau", 0.005)),
            gamma=float(algo.get("gamma", 0.99)),
            train_freq=int(algo.get("train_freq", 1)),
            gradient_steps=int(algo.get("gradient_steps", 1)),
            ent_coef=algo.get("ent_coef", "auto"),
            policy_kwargs={"net_arch": net_arch},
            seed=seed,
            device=device,
            tensorboard_log=str(run_dir / "tb"),
            verbose=1,
        )
    if name != "ppo":
        raise ValueError(f"Unsupported algorithm '{algo_name}'. Expected ppo or sac.")
    return PPO(
        policy=str(algo.get("policy", "MlpPolicy")),
        env=env,
        learning_rate=float(algo.get("learning_rate", 3e-4)),
        n_steps=int(algo.get("n_steps", 2048)),
        batch_size=int(algo.get("batch_size", 64)),
        n_epochs=int(algo.get("n_epochs", 10)),
        gamma=float(algo.get("gamma", 0.99)),
        gae_lambda=float(algo.get("gae_lambda", 0.95)),
        clip_range=float(algo.get("clip_range", 0.2)),
        ent_coef=float(algo.get("ent_coef", 0.01)),
        vf_coef=float(algo.get("vf_coef", 0.5)),
        policy_kwargs={"net_arch": net_arch},
        seed=seed,
        device=device,
        tensorboard_log=str(run_dir / "tb"),
        verbose=1,
    )


def build_run_dir(config_path: Path, name: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = name or config_path.stem
    run_dir = ARTIFACTS / f"{stamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "checkpoints").mkdir()
    shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir


def train(config_path: Path, name: str | None = None, seed_override: int | None = None) -> Path:
    cfg = load_yaml(config_path)
    env_kwargs = env_kwargs_from_config(cfg)
    algo = cfg.get("algo") or {}
    train_cfg = cfg.get("train") or {}

    seed = int(seed_override if seed_override is not None else train_cfg.get("seed", 42))
    normalize = bool(algo.get("normalize_obs", True))
    total_timesteps = int(train_cfg.get("total_timesteps", 200_000))
    checkpoint_every = int(train_cfg.get("checkpoint_every", 20_000))
    device = str(train_cfg.get("device", "cpu"))
    net_arch = list(algo.get("net_arch") or [64, 64])
    algo_name = str(algo.get("name", "ppo")).lower()

    run_dir = build_run_dir(config_path, name)
    env = DummyVecEnv([make_env(env_kwargs, seed=seed)])
    if normalize:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=float(algo.get("gamma", 0.99)))

    model = build_model(algo_name, env, algo, seed, device, net_arch, run_dir)

    meta = {
        "algorithm": algo_name.upper(),
        "policy": algo.get("policy", "MlpPolicy"),
        "net_arch": net_arch,
        "n_params": count_params(model),
        "obs_mode": "state",
        "normalize_obs": normalize,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "config_name": config_path.name,
        "family": config_path.stem,
    }
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

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    model.save(str(run_dir / "checkpoints" / "final"))
    if isinstance(env, VecNormalize):
        env.save(str(run_dir / "vecnormalize.pkl"))
        env.save(str(run_dir / "checkpoints" / "vecnormalize_final.pkl"))

    final_rows = evaluate_checkpoint(run_dir.name, "final", timesteps=total_timesteps)
    append_eval_rows(run_dir / "eval.jsonl", final_rows)

    (run_dir / "DONE").write_text("ok\n", encoding="utf-8")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train PPO on Docking-v0")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    run_dir = train(args.config, args.name, seed_override=args.seed)
    print(f"Run artifacts: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
