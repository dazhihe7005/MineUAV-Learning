"""Offline-only 7D PI-policy inference and PX4-format action conversion.

This module has no transport, Offboard, arming, or simulator entry point.
Its output is data for audit, never a command sent to PX4.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from px4_bridge.frame_conversions import enu_to_ned
from uav_bc_policy import load as load_bc, parameter_hash as bc_parameter_hash


ACTION_SCALE = np.array([1.5, 1.5, 1.0, 1.0], dtype=np.float64)


def _finite_vector(value, size: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite {size}D")
    return result


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def build_observation(position_enu_m, velocity_enu_m_s, yaw_enu_rad,
                      target_enu_m, target_yaw_enu_rad) -> np.ndarray:
    """Match mine_uav_env._get_obs: target-minus-position, world velocity, yaw error."""
    position = _finite_vector(position_enu_m, 3, "position")
    velocity = _finite_vector(velocity_enu_m_s, 3, "velocity")
    target = _finite_vector(target_enu_m, 3, "target")
    if not np.isfinite([yaw_enu_rad, target_yaw_enu_rad]).all():
        raise ValueError("yaw must be finite")
    return np.r_[target - position, velocity,
                 _wrap(float(target_yaw_enu_rad) - float(yaw_enu_rad))].astype(np.float32)


def _check_time(sample_time_us: int, now_time_us: int, max_age_s: float) -> None:
    if not isinstance(sample_time_us, (int, np.integer)) or not isinstance(now_time_us, (int, np.integer)):
        raise ValueError("timestamp must be integer microseconds")
    if sample_time_us < 0 or now_time_us < sample_time_us:
        raise ValueError("timestamp reversal")
    if (now_time_us - sample_time_us) > max_age_s * 1_000_000:
        raise ValueError("stale state")


class ObservationGate:
    """Per-episode causal validity gate; reset by constructing a new instance."""
    def __init__(self, max_age_s: float = .12, max_yaw_rate_rad_s: float = 4.):
        if max_age_s <= 0 or max_yaw_rate_rad_s <= 0:
            raise ValueError("positive validity limits required")
        self.max_age_s = max_age_s
        self.max_yaw_rate_rad_s = max_yaw_rate_rad_s
        self._last_time_us: int | None = None
        self._last_yaw_error: float | None = None

    def accept(self, observation, sample_time_us: int, now_time_us: int) -> np.ndarray:
        obs = _finite_vector(observation, 7, "observation")
        if abs(obs[6]) > math.pi + 1e-6:
            raise ValueError("yaw error outside wrapped range")
        _check_time(sample_time_us, now_time_us, self.max_age_s)
        if self._last_time_us is not None:
            if sample_time_us <= self._last_time_us:
                raise ValueError("non-increasing timestamp")
            dt = (sample_time_us - self._last_time_us) * 1e-6
            # One sample of angular quantization slack, not a permission to jump.
            if abs(_wrap(float(obs[6]) - self._last_yaw_error)) > self.max_yaw_rate_rad_s * dt + .02:
                raise ValueError("yaw discontinuity")
        self._last_time_us = int(sample_time_us)
        self._last_yaw_error = float(obs[6])
        return obs.astype(np.float32)


@dataclass(frozen=True)
class PX4VelocityCommand:
    velocity_ned_m_s: np.ndarray
    yaw_rate_ned_rad_s: float
    source_time_us: int


class ActionAdapter:
    """Convert normalized world-ENU PI actions into a *non-transmitted* NED value."""
    def __init__(self, max_age_s: float = .12):
        if max_age_s <= 0:
            raise ValueError("positive age limit required")
        self.max_age_s = max_age_s

    def convert(self, normalized_action, sample_time_us: int, now_time_us: int) -> PX4VelocityCommand:
        action = _finite_vector(normalized_action, 4, "action")
        _check_time(sample_time_us, now_time_us, self.max_age_s)
        if np.any(np.abs(action) > 1. + 1e-6):
            raise ValueError("action outside normalized [-1,1] range")
        action = np.clip(action, -1., 1.)  # only float tolerance, never a safety envelope
        physical = action * ACTION_SCALE
        return PX4VelocityCommand(enu_to_ned(physical[:3]), float(-physical[3]), int(sample_time_us))


def _torch_module_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class FrozenPolicy:
    def __init__(self, model, kind: str, path: Path, checkpoint_sha256: str):
        self.model = model
        self.kind = kind
        self.path = path
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def load(cls, path: str | Path, *, kind: str, expected_sha256: str) -> "FrozenPolicy":
        path = Path(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"checkpoint SHA256 mismatch: {path.name}")
        if kind == "bc":
            model, _ = load_bc(path)
        elif kind == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(path, device="cpu")
            if model.observation_space.shape != (7,) or model.action_space.shape != (4,):
                raise ValueError("PPO observation/action contract mismatch")
            if not np.all(model.action_space.low == -1) or not np.all(model.action_space.high == 1):
                raise ValueError("PPO normalized action bounds mismatch")
            model.policy.set_training_mode(False)
            model.policy.requires_grad_(False)
        else:
            raise ValueError("unknown policy kind")
        return cls(model, kind, path, actual)

    def parameter_hash(self) -> str:
        module = self.model.actor if self.kind == "bc" else self.model.policy
        return bc_parameter_hash(module) if self.kind == "bc" else _torch_module_hash(module)

    def predict(self, observation) -> np.ndarray:
        obs = _finite_vector(observation, 7, "observation").astype(np.float32)
        with torch.inference_mode():
            action = (self.model.predict(obs) if self.kind == "bc"
                      else self.model.predict(obs, deterministic=True)[0])
        action = _finite_vector(action, 4, "predicted action")
        if np.any(np.abs(action) > 1. + 1e-6):
            raise ValueError("model predicted out-of-range action")
        return action.astype(np.float32)


def shadow_step(policies: dict, observation, sample_time_us: int, now_time_us: int, *,
                gate: ObservationGate) -> dict[str, PX4VelocityCommand]:
    """Offline inference only; intentionally accepts no PX4 transport object."""
    obs = gate.accept(observation, sample_time_us, now_time_us)
    adapter = ActionAdapter()
    return {name: adapter.convert(policy.predict(obs), sample_time_us, now_time_us)
            for name, policy in policies.items()}
