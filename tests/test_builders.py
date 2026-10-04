import torch
from stable_baselines3 import DDPG, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from docking_sim.env.docking_env import DockingEnv
from docking_sim.replay.rollout import load_policy
from docking_sim.training.builders import architecture_label, build_model
from docking_sim.training.losses import canonical_loss, critic_loss
from docking_sim.training.networks import ResidualFeaturesExtractor


def _env(**kwargs):
    return DummyVecEnv([lambda: DockingEnv(**kwargs)])


def _tiny_ppo(**extra):
    algo = {
        "n_steps": 32,
        "batch_size": 32,
        "n_epochs": 1,
        "net_arch": [32, 32],
        "verbose": 0,
    }
    algo.update(extra)
    return algo


def test_critic_losses_differ():
    pred = torch.tensor([0.0, 2.0])
    target = torch.tensor([0.0, 0.0])
    assert canonical_loss("smooth-l1") == "smooth_l1"
    assert not torch.allclose(critic_loss(pred, target, "mse"), critic_loss(pred, target, "huber"))
    assert not torch.allclose(critic_loss(pred, target, "mse"), critic_loss(pred, target, "smooth_l1"))


def test_optimizer_and_architecture_labels(tmp_path):
    model = build_model("ppo", _env(), {**_tiny_ppo(), "optimizer": "sgd"}, 0, "cpu", [32, 32], tmp_path)
    assert isinstance(model.policy.optimizer, torch.optim.SGD)
    assert architecture_label({"features": "residual", "residual_dim": 64, "residual_blocks": 2}, [], "state") == "Residual MLP 64x2"
    assert architecture_label({"policy": "CnnPolicy"}, [64, 64], "pixels").startswith("CNN+MLP")


def test_td3_and_ddpg_round_trip(tmp_path):
    algo = {
        "net_arch": [32, 32],
        "buffer_size": 200,
        "learning_starts": 0,
        "batch_size": 16,
        "gradient_steps": 1,
        "train_freq": 1,
        "verbose": 0,
        "critic_loss": "huber",
    }
    td3 = build_model("td3", _env(), algo, 0, "cpu", [32, 32], tmp_path / "td3")
    assert isinstance(td3, TD3)
    td3.learn(total_timesteps=4)
    td3.save(str(tmp_path / "td3" / "model"))
    loaded = load_policy(tmp_path / "td3" / "model.zip", "td3")
    obs = td3.observation_space.sample()
    action, _ = loaded.predict(obs, deterministic=True)
    assert action.shape == (3,)

    ddpg = build_model("ddpg", _env(), {**algo, "critic_loss": "mse"}, 0, "cpu", [32, 32], tmp_path / "ddpg")
    assert isinstance(ddpg, DDPG)
    action, _ = ddpg.predict(ddpg.observation_space.sample(), deterministic=True)
    assert action.shape == (3,)


def test_huber_ppo_and_residual_mlp_learn(tmp_path):
    huber = build_model(
        "ppo",
        _env(),
        {**_tiny_ppo(), "critic_loss": "huber", "optimizer": "adamw"},
        0,
        "cpu",
        [32, 32],
        tmp_path / "huber",
    )
    assert isinstance(huber.policy.optimizer, torch.optim.AdamW)
    huber.learn(total_timesteps=32)

    residual = build_model(
        "ppo",
        _env(),
        {
            **_tiny_ppo(),
            "features": "residual",
            "residual_dim": 32,
            "residual_blocks": 2,
            "verbose": 0,
        },
        0,
        "cpu",
        [],
        tmp_path / "residual",
    )
    assert isinstance(residual.policy.features_extractor, ResidualFeaturesExtractor)
    residual.learn(total_timesteps=32)
    residual.save(str(tmp_path / "residual" / "model"))
    loaded = load_policy(tmp_path / "residual" / "model.zip", "ppo")
    assert isinstance(loaded, PPO)
    action, _ = loaded.predict(loaded.observation_space.sample(), deterministic=True)
    assert action.shape == (3,)


def test_pixel_cnn_policy_learns(tmp_path):
    model = build_model(
        "ppo",
        _env(obs_mode="pixels"),
        {
            "policy": "CnnPolicy",
            "features": "cnn",
            "net_arch": [32],
            "n_steps": 32,
            "batch_size": 32,
            "n_epochs": 1,
            "verbose": 0,
        },
        0,
        "cpu",
        [32],
        tmp_path,
    )
    assert model.observation_space.shape[0] == 3
    model.learn(total_timesteps=32)
    obs = DockingEnv(obs_mode="pixels").reset(seed=0)[0]
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (3,)
