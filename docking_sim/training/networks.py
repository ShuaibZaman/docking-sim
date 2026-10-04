from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.fc1(features))
        hidden = self.fc2(hidden)
        return torch.relu(features + hidden)


class ResidualFeaturesExtractor(BaseFeaturesExtractor):
    """Vector observation projected into a stack of residual blocks."""

    def __init__(self, observation_space: gym.Space, features_dim: int = 64, n_blocks: int = 2) -> None:
        super().__init__(observation_space, features_dim)
        if not isinstance(observation_space, gym.spaces.Box) or len(observation_space.shape) != 1:
            raise ValueError("Residual MLP expects a vector observation")
        in_dim = int(observation_space.shape[0])
        blocks = [ResidualBlock(features_dim) for _ in range(int(n_blocks))]
        self.net = nn.Sequential(nn.Linear(in_dim, features_dim), nn.ReLU(), *blocks)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)
