"""Frozen LiDAR PPO interface preflight; no learning starts on import.

Heading is a separately declared simulation-ground-truth self-state. It is
never derived from obstacle geometry or future observations.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HeadingObservation(gym.ObservationWrapper):
    """Add [sin(yaw_ENU), cos(yaw_ENU)] without changing the legacy 7D env."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("LiDAR navigation Dict observation required")
        self.observation_space = spaces.Dict({
            **env.observation_space.spaces,
            "heading": spaces.Box(-1., 1., shape=(2,), dtype=np.float32),
        })

    def observation(self, observation: dict) -> dict:
        quaternion = np.asarray(self.unwrapped.data.qpos[3:7], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if not np.isfinite(quaternion).all() or not .999 < norm < 1.001:
            raise ValueError("current physical yaw quaternion is invalid")
        w, x, y, z = quaternion / norm
        yaw = math.atan2(2. * (w * z + x * y), 1. - 2. * (y * y + z * z))
        result = dict(observation)
        result["heading"] = np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32)
        return result


class LidarStateExtractor(BaseFeaturesExtractor):
    """Small CNN for normalized float32 ranges/masks, plus explicit self-state."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=74)
        if (observation_space["lidar"].shape != (3, 12, 24)
                or observation_space["state"].shape != (7,)
                or observation_space["heading"].shape != (2,)
                or observation_space["lidar_age"].shape != (1,)):
            raise ValueError("unexpected LiDAR PPO observation shape")
        self.lidar_net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 6 * 12, 64),
            torch.nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat((self.lidar_net(observations["lidar"].float()),
                          observations["state"].float(),
                          observations["heading"].float(),
                          observations["lidar_age"].float()), dim=1)


def make_ppo(env: gym.Env, *, n_steps: int = 128, seed: int = 0) -> PPO:
    """Pre-registered single-worker smoke network; does not call learn()."""
    if n_steps < 32 or n_steps > 512 or isinstance(seed, bool) or seed < 0:
        raise ValueError("bounded n_steps and nonnegative seed required")
    return PPO("MultiInputPolicy", env, learning_rate=3e-4, n_steps=n_steps,
               batch_size=64 if n_steps >= 64 else 32, n_epochs=2,
               gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0.,
               vf_coef=.5, max_grad_norm=.5, seed=seed, device="cpu",
               policy_kwargs={"features_extractor_class": LidarStateExtractor,
                              "normalize_images": False,
                              "net_arch": {"pi": [64, 64], "vf": [64, 64]}},
               verbose=0)
