from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from docking_sim.replay.eval import append_eval_rows, evaluate_checkpoint


class MetricsJsonlCallback(BaseCallback):
    def __init__(self, path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._episode = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            episode = info.get("episode")
            if episode is None:
                continue
            self._episode += 1
            record = {
                "episode": self._episode,
                "timesteps": int(self.num_timesteps),
                "reward": float(episode.get("r", 0.0)),
                "length": int(episode.get("l", 0)),
                "success": int(bool(info.get("success", False))),
                "crash": int(bool(info.get("crash", False))),
                "timeout": int(bool(info.get("timeout", False))),
                "fuel_used": float(info.get("fuel_used_total", 0.0)),
                "fuel": float(info.get("fuel", 0.0)),
                "distance": float(info.get("distance", 0.0)),
                "speed": float(info.get("speed", 0.0)),
                "heading_error": float(info.get("heading_error", 0.0)),
            }
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        return True


class VecNormalizeCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_dir: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, tag: str) -> None:
        env = self.model.get_env()
        if isinstance(env, VecNormalize):
            env.save(str(self.save_dir / f"vecnormalize_{tag}.pkl"))
            env.save(str(self.save_dir.parent / "vecnormalize.pkl"))

    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            self._save(f"{self.num_timesteps}_steps")
        return True

    def _on_training_end(self) -> None:
        self._save("final")


STAT_KEYS = {
    "train/approx_kl": "approx_kl",
    "train/clip_fraction": "clip_fraction",
    "train/entropy_loss": "entropy",
    "train/value_loss": "value_loss",
    "train/actor_loss": "actor_loss",
    "train/critic_loss": "critic_loss",
    "train/ent_coef": "ent_coef",
}


def train_stat_record(values: dict, timesteps: int) -> dict | None:
    record: dict = {"timesteps": int(timesteps)}
    found = False
    for source, name in STAT_KEYS.items():
        if source in values and values[source] is not None:
            record[name] = float(values[source])
            found = True
    return record if found else None


class HeldOutEvalCallback(BaseCallback):
    def __init__(self, run_dir: Path, every: int, prefix: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.every = int(every)
        self.prefix = prefix
        self.path = self.run_dir / "eval.jsonl"

    def _eval(self, checkpoint: str, timesteps: int) -> None:
        print(f"Held-out eval {checkpoint}", flush=True)
        try:
            rows = evaluate_checkpoint(
                self.run_dir.name,
                checkpoint,
                timesteps=int(timesteps),
            )
        except ValueError as exc:
            # A curriculum or pixel run may not match the static comparison
            # contract. Training remains valid; it simply needs a matching
            # benchmark suite before it can be ranked with static candidates.
            print(f"Skipping held-out benchmark eval: {exc}", flush=True)
            return
        append_eval_rows(self.path, rows)

    def _on_step(self) -> bool:
        if self.every > 0 and self.n_calls > 0 and self.n_calls % self.every == 0:
            self._eval(f"{self.prefix}_{self.num_timesteps}_steps", self.num_timesteps)
        return True


class TrainStatsCallback(BaseCallback):
    def __init__(self, path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last: tuple | None = None

    def _dump(self) -> None:
        values = getattr(self.logger, "name_to_value", {}) or {}
        record = train_stat_record(values, self.num_timesteps)
        if record is None:
            return
        stamp = tuple(sorted((key, record[key]) for key in record if key != "timesteps"))
        if stamp == self._last:
            return
        self._last = stamp
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    def _on_step(self) -> bool:
        self._dump()
        return True

    def _on_training_end(self) -> None:
        self._dump()
