"""Opt-in 10D Reward-V2 environment exposing the existing velocity PI memory."""

import numpy as np
from gymnasium import spaces

from ppo_pi_env import MineUAVPIEnv


class MineUAVPIObservableEnv(MineUAVPIEnv):
    """The original seven observations plus world-frame Ki*integral_error."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = spaces.Box(
            low=np.r_[self.observation_space.low, [-1.5, -1.5, -1.5]]
            .astype(np.float32),
            high=np.r_[self.observation_space.high, [1.5, 1.5, 1.5]]
            .astype(np.float32),
            dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        if not self._state_is_finite():
            return np.zeros(10, dtype=np.float32)
        original = super()._get_obs()
        contribution = self.velocity_controller.integral_acceleration_world
        return np.r_[original, contribution].astype(np.float32)
