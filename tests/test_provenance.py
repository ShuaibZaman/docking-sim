from pathlib import Path

import pytest

from docking_sim.config import env_kwargs_from_config, load_yaml
from docking_sim.training.provenance import run_provenance, sha256_file
from docking_sim.training.train import _resume_run_dir, train


def test_run_provenance_records_source_and_scene_contract(tmp_path: Path):
    config = Path("configs/ppo_64.yaml")
    env_kwargs = env_kwargs_from_config(load_yaml(config))
    payload = run_provenance(config, env_kwargs)
    assert payload["config_sha256"] == sha256_file(config)
    assert payload["source_revision"]
    assert payload["scene_fingerprint"]
    assert payload["interface_fingerprint"]
    assert "python" in payload["runtime"]


def test_resume_refuses_a_different_config(tmp_path: Path):
    run_dir = tmp_path / "fake_run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(Path("configs/ppo_64.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    resolved = _resume_run_dir(run_dir)
    assert resolved == run_dir

    with pytest.raises(ValueError, match="different config"):
        train(Path("configs/ppo_orbital.yaml"), resume_run=run_dir, additional_timesteps=1)
