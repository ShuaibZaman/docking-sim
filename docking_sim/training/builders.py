from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise

from docking_sim.training.losses import attach_critic_loss, canonical_loss
from docking_sim.training.networks import ResidualFeaturesExtractor


OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def canonical_optimizer(name: str) -> str:
    key = str(name or "adam").strip().lower()
    if key not in OPTIMIZERS:
        known = ", ".join(OPTIMIZERS)
        raise ValueError(f"Unknown optimizer '{name}'. Expected one of: {known}")
    return key


def architecture_label(algo: dict[str, Any], net_arch: list[int], obs_mode: str) -> str:
    features = str(algo.get("features", "mlp")).lower()
    policy = str(algo.get("policy", "MlpPolicy"))
    if obs_mode == "pixels" or policy == "CnnPolicy" or features == "cnn":
        head = "-".join(str(size) for size in net_arch) or "linear"
        return f"CNN+MLP {head}"
    if features == "residual":
        dim = int(algo.get("residual_dim", 64))
        blocks = int(algo.get("residual_blocks", 2))
        return f"Residual MLP {dim}x{blocks}"
    arch = "-".join(str(size) for size in net_arch) or "linear"
    return f"MLP {arch}"


def make_policy_kwargs(algo: dict[str, Any], net_arch: list[int]) -> dict[str, Any]:
    features = str(algo.get("features", "mlp")).lower()
    kwargs: dict[str, Any] = {}
    if features == "residual":
        kwargs["features_extractor_class"] = ResidualFeaturesExtractor
        kwargs["features_extractor_kwargs"] = {
            "features_dim": int(algo.get("residual_dim", 64)),
            "n_blocks": int(algo.get("residual_blocks", 2)),
        }
        kwargs["net_arch"] = list(algo.get("head_arch") or [])
    else:
        kwargs["net_arch"] = list(net_arch)
    if algo.get("optimizer"):
        name = canonical_optimizer(str(algo["optimizer"]))
        extra: dict[str, Any] = {}
        if name == "sgd":
            extra["momentum"] = float(algo.get("momentum", 0.0))
        kwargs["optimizer_class"] = OPTIMIZERS[name]
        kwargs["optimizer_kwargs"] = extra
    return kwargs


def _action_noise(env, algo: dict[str, Any]) -> NormalActionNoise:
    n_actions = int(np.prod(env.action_space.shape))
    sigma = float(algo.get("action_noise_sigma", 0.1))
    return NormalActionNoise(mean=np.zeros(n_actions), sigma=np.full(n_actions, sigma))


def build_model(algo_name: str, env, algo: dict[str, Any], seed: int, device: str, net_arch: list[int], run_dir: Path):
    name = algo_name.lower()
    policy_kwargs = make_policy_kwargs(algo, net_arch)
    policy = str(algo.get("policy", "MlpPolicy"))
    learning_rate = float(algo.get("learning_rate", 3e-4))
    gamma = float(algo.get("gamma", 0.99))
    tb_log = str(Path(run_dir) / "tb")
    loss_name = canonical_loss(str(algo.get("critic_loss", "mse")))

    if name == "ppo":
        model = PPO(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_steps=int(algo.get("n_steps", 2048)),
            batch_size=int(algo.get("batch_size", 64)),
            n_epochs=int(algo.get("n_epochs", 10)),
            gamma=gamma,
            gae_lambda=float(algo.get("gae_lambda", 0.95)),
            clip_range=float(algo.get("clip_range", 0.2)),
            ent_coef=float(algo.get("ent_coef", 0.01)),
            vf_coef=float(algo.get("vf_coef", 0.5)),
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            tensorboard_log=tb_log,
            verbose=int(algo.get("verbose", 1)),
        )
    elif name == "sac":
        model = SAC(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=int(algo.get("buffer_size", 1_000_000)),
            learning_starts=int(algo.get("learning_starts", 100)),
            batch_size=int(algo.get("batch_size", 256)),
            tau=float(algo.get("tau", 0.005)),
            gamma=gamma,
            train_freq=int(algo.get("train_freq", 1)),
            gradient_steps=int(algo.get("gradient_steps", 1)),
            ent_coef=algo.get("ent_coef", "auto"),
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            tensorboard_log=tb_log,
            verbose=int(algo.get("verbose", 1)),
        )
    elif name in {"td3", "ddpg"}:
        shared = dict(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=int(algo.get("buffer_size", 1_000_000)),
            learning_starts=int(algo.get("learning_starts", 100)),
            batch_size=int(algo.get("batch_size", 256)),
            tau=float(algo.get("tau", 0.005)),
            gamma=gamma,
            train_freq=int(algo.get("train_freq", 1)),
            gradient_steps=int(algo.get("gradient_steps", 1)),
            action_noise=_action_noise(env, algo),
            policy_kwargs=policy_kwargs,
            seed=seed,
            device=device,
            tensorboard_log=tb_log,
            verbose=int(algo.get("verbose", 1)),
        )
        if name == "td3":
            model = TD3(
                **shared,
                policy_delay=int(algo.get("policy_delay", 2)),
                target_policy_noise=float(algo.get("target_policy_noise", 0.2)),
                target_noise_clip=float(algo.get("target_noise_clip", 0.5)),
            )
        else:
            model = DDPG(**shared)
    else:
        raise ValueError(f"Unsupported algorithm '{algo_name}'. Expected ppo, sac, td3, or ddpg.")

    attach_critic_loss(model, loss_name)
    return model
