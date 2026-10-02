"""Opt-in fixed velocity-PI environments for the single-seed PPO experiment.

The original MineUAVEnv keeps its P-controller default. These wrappers change
only the velocity loop and, when requested, log its hidden state at policy
boundaries. Hidden state never enters the seven-dimensional observation.
"""

from pathlib import Path

import numpy as np
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from audit_velocity_pi import configure_pi_env
from mine_uav_env import MineUAVEnv
from robustness_dynamics import RobustnessEnv


class _PIConfiguredMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        configure_pi_env(self)
        self._integral_capture_enabled = False
        self._integral_completed = []
        self._integral_active = None

    def begin_integral_capture(self) -> None:
        self._integral_capture_enabled = True
        self._integral_completed = []
        self._integral_active = None

    def reset(self, *, seed=None, options=None):
        if self._integral_capture_enabled and self._integral_active is not None:
            self._integral_completed.append(self._integral_active)
        observation, info = super().reset(seed=seed, options=options)
        initial = self.velocity_controller.integral_error
        if not np.array_equal(initial, np.zeros(3)):
            raise RuntimeError("PI integral memory leaked across reset")
        if self._integral_capture_enabled:
            self._integral_active = {
                "seed": seed,
                "target_m": info["target_position_m"].tolist(),
                "initial_integral_error_m": initial.tolist(),
                "steps": [],
            }
        return observation, info

    def step(self, action):
        result = super().step(action)
        if self._integral_capture_enabled:
            if self._integral_active is None:
                raise RuntimeError("Start PI capture before resetting an episode")
            _, _, terminated, truncated, info = result
            controller = self.velocity_controller
            self._integral_active["steps"].append({
                "policy_step": self.episode_steps - 1,
                "time_s": float(self.data.time),
                "integral_error_m": controller.integral_error.tolist(),
                "integral_acceleration_m_s2":
                    controller.integral_acceleration_world.tolist(),
                "desired_acceleration_m_s2":
                    controller.last_desired_acceleration_world.tolist(),
                "distance_m": float(info["distance_m"]),
                "speed_m_s": (float(info["speed_m_s"])
                              if np.isfinite(info["speed_m_s"]) else None),
                "termination_reason": info["termination_reason"]
                    if terminated or truncated else None,
            })
        return result

    def take_integral_capture(self) -> list[dict]:
        if not self._integral_capture_enabled:
            raise RuntimeError("PI capture was not started")
        if self._integral_active is not None:
            self._integral_completed.append(self._integral_active)
        completed = self._integral_completed
        self._integral_completed = []
        self._integral_active = None
        self._integral_capture_enabled = False
        return completed


class MineUAVPIEnv(_PIConfiguredMixin, MineUAVEnv):
    """Nominal Reward-V2 waypoint environment with fixed PI velocity loop."""


class PIRobustnessEnv(_PIConfiguredMixin, RobustnessEnv):
    """Private, single-factor disturbed plant with the same fixed PI loop."""


def make_pi_training_envs(monitor_dir: Path, n_envs: int = 8,
                          env_type=MineUAVPIEnv) -> DummyVecEnv:
    """Eight independent monitored/headless PI environments in one process."""
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def factory(rank: int):
        def create():
            return Monitor(
                env_type(render_mode=None, reward_version="v2",
                         target_distribution="full"),
                filename=str(monitor_dir / f"train_{rank}"),
                info_keywords=("distance_m", "termination_reason"),
            )
        return create

    return DummyVecEnv([factory(rank) for rank in range(n_envs)])
