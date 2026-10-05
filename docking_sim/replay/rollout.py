from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from docking_sim.config import ARTIFACTS, env_kwargs_from_config, load_yaml
from docking_sim.env.docking_env import DockingEnv
from docking_sim.env.physics import hits_hull, out_of_bounds
from docking_sim.training.losses import _install_loss_hook
from docking_sim.training.networks import ResidualFeaturesExtractor  # noqa: F401

ALGO_CLASSES = {
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
    "ddpg": DDPG,
}


def list_run_dirs() -> list[Path]:
    if not ARTIFACTS.exists():
        return []
    dirs = [p for p in ARTIFACTS.iterdir() if p.is_dir() and (p / "config.yaml").exists()]
    return sorted(dirs, key=lambda p: p.name, reverse=True)


def load_run_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "run.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def checkpoint_steps(name: str) -> int:
    if name == "random":
        return 0
    if name.startswith("final"):
        return 10**12
    digits = "".join(ch for ch in name if ch.isdigit())
    return int(digits) if digits else 0


def list_checkpoints(run_dir: Path | None) -> list[dict[str, Any]]:
    items = [{"id": "random", "label": "Random policy", "steps": 0, "path": None}]
    if run_dir is None:
        return items
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        return items
    zips = sorted(ckpt_dir.glob("*.zip"), key=lambda p: checkpoint_steps(p.stem))
    for zip_path in zips:
        items.append(
            {
                "id": zip_path.stem,
                "label": zip_path.stem.replace("_", " "),
                "steps": checkpoint_steps(zip_path.stem),
                "path": str(zip_path),
            }
        )
    return items


def _vecnormalize_for_checkpoint(run_dir: Path, checkpoint_id: str) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    candidates = [
        ckpt_dir / f"vecnormalize_{checkpoint_id}.pkl",
        ckpt_dir / f"vecnormalize_{checkpoint_id.replace('ppo_', '')}.pkl",
    ]
    if checkpoint_id == "final":
        candidates.append(ckpt_dir / "vecnormalize_final.pkl")
        candidates.append(run_dir / "vecnormalize.pkl")
    steps = checkpoint_steps(checkpoint_id)
    if steps and steps < 10**12:
        candidates.insert(0, ckpt_dir / f"vecnormalize_{steps}_steps.pkl")
    candidates.append(run_dir / "vecnormalize.pkl")
    for path in candidates:
        if path.exists():
            return path
    return None


def policy_action_to_commands(action: np.ndarray) -> np.ndarray:
    """Map a policy action onto axial, lateral, yaw in [-1, 1].

    Two-action checkpoints were trained with forward-only thrust: action[0] in
    [-1, 1] becomes axial in [0, 1], and action[1] is yaw. Lateral stays off.
    """
    values = np.asarray(action, dtype=np.float32).reshape(-1)
    if values.shape[0] == 2:
        axial = float(np.clip(values[0], -1.0, 1.0) + 1.0) * 0.5
        yaw = float(np.clip(values[1], -1.0, 1.0))
        return np.array([axial, 0.0, yaw], dtype=np.float32)
    return np.array(
        [
            float(np.clip(values[0], -1.0, 1.0)),
            float(np.clip(values[1], -1.0, 1.0)),
            float(np.clip(values[2], -1.0, 1.0)),
        ],
        dtype=np.float32,
    )


def command_fields(inf: dict[str, Any]) -> dict[str, float]:
    """Prefer axial/yaw, and copy them onto the thrust/torque aliases.

    An older info dict may only have thrust and torque. A newer one may only
    have axial and yaw. Whichever pair is present, both names on the frame
    carry the same number.
    """

    def pick(primary: str, legacy: str) -> float:
        if primary in inf and inf[primary] is not None:
            return float(inf[primary])
        if legacy in inf and inf[legacy] is not None:
            return float(inf[legacy])
        return 0.0

    axial = pick("axial", "thrust")
    yaw = pick("yaw", "torque")
    lateral = float(inf["lateral"]) if "lateral" in inf and inf["lateral"] is not None else 0.0
    return {
        "axial": axial,
        "lateral": lateral,
        "yaw": yaw,
        "thrust": axial,
        "torque": yaw,
    }


def _normalize_obs(vec: VecNormalize, obs: np.ndarray) -> np.ndarray:
    array = np.asarray(obs)
    if array.ndim >= 3:
        batched = array[None, ...]
        normed = vec.normalize_obs(batched)
        return np.asarray(normed, dtype=np.float32)[0]
    batched = array.reshape(1, -1)
    normed = vec.normalize_obs(batched)
    return np.asarray(normed, dtype=np.float32).reshape(-1)


def load_policy(path: Path, algo_name: str):
    name = str(algo_name or "ppo").lower()
    cls = ALGO_CLASSES.get(name)
    if cls is None:
        known = ", ".join(sorted(ALGO_CLASSES))
        raise ValueError(f"Unsupported algorithm '{algo_name}'. Expected one of: {known}")
    model = cls.load(str(path), device="cpu")
    _install_loss_hook(model)
    return model


def rollout(
    *,
    run_id: str | None = None,
    checkpoint: str = "random",
    seed: int = 42,
    max_steps: int | None = None,
    randomize_start: bool | None = None,
    scenario=None,
    env_kwargs_override: dict[str, Any] | None = None,
    baseline: str | None = None,
    level: int | None = None,
    model=None,
    vecnorm: VecNormalize | None = None,
) -> dict[str, Any]:
    run_dir = ARTIFACTS / run_id if run_id else None
    cfg: dict[str, Any] = {}
    candidate_env_kwargs: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    if run_dir is not None and run_dir.exists():
        cfg = load_yaml(run_dir / "config.yaml")
        candidate_env_kwargs = env_kwargs_from_config(cfg)
        meta = load_run_meta(run_dir)
    env_kwargs = dict(env_kwargs_override or candidate_env_kwargs)
    if level is not None:
        from docking_sim.env.levels import level_env_kwargs

        env_kwargs = dict(level_env_kwargs(int(level)))
    if scenario is None and randomize_start is not None:
        env_kwargs["randomize_start"] = bool(randomize_start)

    env = DockingEnv(**env_kwargs)
    if scenario is not None:
        scenario_seed = int(getattr(scenario, "seed", seed))
        reset_options = scenario.reset_options()
        obs, info = env.reset(seed=scenario_seed, options=reset_options)
    else:
        obs, info = env.reset(seed=int(seed))
    limit = int(max_steps or env.max_steps)
    world = {
        "x_min": env.cfg.x_min,
        "x_max": env.cfg.x_max,
        "y_min": env.cfg.y_min,
        "y_max": env.cfg.y_max,
        "ship_radius": env.cfg.ship_radius,
        "hull": {
            "cx": env.cfg.hull_cx,
            "cy": env.cfg.hull_cy,
            "w": env.cfg.hull_w,
            "h": env.cfg.hull_h,
        },
        "port": {
            "cx": env.cfg.port_cx,
            "cy": env.cfg.port_cy,
            "w": env.cfg.port_w,
            "h": env.cfg.port_h,
        },
        "dock_speed_max": float(env.cfg.dock_speed_max),
        "dock_omega_max": float(env.cfg.dock_omega_max),
        "dock_angle_max_deg": float(np.rad2deg(env.cfg.dock_angle_max)),
        "approach_angle": float(env.cfg.port_approach_angle),
        "scenario_id": info.get("scenario_id"),
        "scenario_hash": info.get("scenario_hash"),
    }
    dt = float(env.cfg.dt)

    owns_vecnorm = False
    if checkpoint != "random" and model is None and baseline is None:
        if run_dir is None:
            raise FileNotFoundError("checkpoint replay requires a run_id")
        zip_path = run_dir / "checkpoints" / f"{checkpoint}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {zip_path}")
        algo_name = str((cfg.get("algo") or {}).get("name") or meta.get("algorithm") or "ppo")
        model = load_policy(zip_path, algo_name)
        expected = tuple(int(size) for size in model.observation_space.shape)
        actual = tuple(int(size) for size in env.observation_space.shape)
        if expected != actual:
            env.close()
            raise ValueError(
                f"Checkpoint observation {expected} does not match this level's observation {actual}."
            )
        vn_path = _vecnormalize_for_checkpoint(run_dir, checkpoint)
        if vn_path is not None:
            dummy = DummyVecEnv([lambda: DockingEnv(**env_kwargs)])
            vecnorm = VecNormalize.load(str(vn_path), dummy)
            vecnorm.training = False
            vecnorm.norm_reward = False
            owns_vecnorm = True

    rng = np.random.default_rng(seed)
    frames: list[dict[str, Any]] = []
    reward_total = 0.0
    terminated = False
    truncated = False

    def snapshot(step: int, reward: float, inf: dict[str, Any], done: bool) -> dict[str, Any]:
        hull_hit = bool(inf.get("hit_hull", hits_hull(env._state, env.cfg, env._pose)))
        left_map = bool(inf.get("out_of_bounds", out_of_bounds(env._state, env.cfg)))
        return {
            "t": step,
            "x": float(inf.get("x", env._state.x)),
            "y": float(inf.get("y", env._state.y)),
            "theta": float(inf.get("theta", env._state.theta)),
            "vx": float(inf.get("vx", env._state.vx)),
            "vy": float(inf.get("vy", env._state.vy)),
            "omega": float(inf.get("omega", env._state.omega)),
            "fuel": float(inf.get("fuel", env._state.fuel)),
            "distance": float(inf.get("distance", 0.0)),
            "speed": float(inf.get("speed", 0.0)),
            "absolute_speed": float(inf.get("absolute_speed", inf.get("speed", 0.0))),
            "relative_vx": float(inf.get("relative_vx", inf.get("vx", 0.0))),
            "relative_vy": float(inf.get("relative_vy", inf.get("vy", 0.0))),
            "heading_error": float(inf.get("heading_error", 0.0)),
            **command_fields(inf),
            "reward": float(reward),
            "reward_total": float(reward_total),
            "components": inf.get("reward_components") or {},
            "done": bool(done),
            "success": bool(inf.get("success", False)),
            "crash": bool(inf.get("crash", False)),
            "timeout": bool(inf.get("timeout", False)),
            "hit_hull": bool(hull_hit),
            "hit_asteroid": bool(inf.get("hit_asteroid", False)),
            "out_of_bounds": bool(left_map),
            "out_of_fuel": bool(inf.get("out_of_fuel", False)),
            "terminal_reason": str(inf.get("terminal_reason", "")),
            "station": inf.get("station"),
            "port_pose": inf.get("port_pose"),
            "ports": inf.get("ports") or [],
            "phase": inf.get("phase") or "approach",
            "active_port": int(inf.get("active_port") or 0),
            "hold": int(inf.get("hold") or 0),
            "hold_steps": int(inf.get("hold_steps") or 0),
            "asteroids": inf.get("asteroids") or [],
        }

    frames.append(snapshot(0, 0.0, info, False))

    for step in range(1, limit + 1):
        if baseline is not None:
            from docking_sim.replay.controllers import controller_action

            action = controller_action(baseline, env)
        elif model is None:
            action = rng.uniform(-1.0, 1.0, size=(3,)).astype(np.float32)
        else:
            policy_obs = _normalize_obs(vecnorm, obs) if vecnorm is not None else obs
            raw, _ = model.predict(policy_obs, deterministic=True)
            action = policy_action_to_commands(raw)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_total += float(reward)
        done = bool(terminated or truncated)
        frames.append(snapshot(step, float(reward), info, done))
        if done:
            break

    env.close()
    if owns_vecnorm and vecnorm is not None:
        vecnorm.close()

    return {
        "run_id": run_id,
        "checkpoint": checkpoint,
        "baseline": baseline,
        "seed": int(seed),
        "scenario_id": info.get("scenario_id"),
        "scenario_hash": info.get("scenario_hash"),
        "success": bool(frames[-1]["success"]),
        "crash": bool(frames[-1]["crash"]),
        "timeout": bool(frames[-1]["timeout"]),
        "steps": len(frames) - 1,
        "reward_total": reward_total,
        "dt": dt,
        "world": world,
        "meta": meta,
        "frames": frames,
    }
