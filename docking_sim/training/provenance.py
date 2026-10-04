"""Experiment metadata used to make saved runs auditable and resumable."""

from __future__ import annotations

from hashlib import sha256
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import stable_baselines3
import torch

from docking_sim.config import ROOT
from docking_sim.replay.benchmarks import interface_fingerprint, scene_fingerprint


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def runtime_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "stable_baselines3": stable_baselines3.__version__,
    }


def run_provenance(config_path: Path, env_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_revision": git_revision(),
        "config_sha256": sha256_file(config_path),
        "scene_fingerprint": scene_fingerprint(env_kwargs),
        "interface_fingerprint": interface_fingerprint(env_kwargs),
        "runtime": runtime_provenance(),
    }
