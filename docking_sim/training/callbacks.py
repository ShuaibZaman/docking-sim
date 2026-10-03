from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize


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
