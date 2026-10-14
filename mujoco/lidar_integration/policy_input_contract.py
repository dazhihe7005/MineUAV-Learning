"""Typed future-policy input boundary; no legacy BC/PPO or PX4 execution."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .async_observation import ObservationRead


_STATE_SOURCES = {"SIM_GROUND_TRUTH_STATE", "PX4_ESTIMATED_STATE", "OFFLINE_TEST_STATE"}
_TARGET_SOURCES = {"MISSION_TARGET", "OFFLINE_TEST_TARGET"}


def _xyz(value: np.ndarray, name: str) -> np.ndarray:
    out = np.array(value, dtype=np.float64, copy=True)
    if out.shape != (3,) or not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite 3D ENU")
    return np.frombuffer(out.tobytes(), dtype=np.float64).reshape(3)


@dataclass(frozen=True)
class MotionState:
    """ENU motion input with explicit source provenance, not a PX4 adapter."""

    position_enu_m: np.ndarray
    velocity_enu_m_s: np.ndarray
    yaw_enu_rad: float
    sample_time_s: float
    clock_domain: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_enu_m", _xyz(self.position_enu_m, "position"))
        object.__setattr__(self, "velocity_enu_m_s", _xyz(self.velocity_enu_m_s, "velocity"))
        if (not math.isfinite(self.yaw_enu_rad) or not math.isfinite(self.sample_time_s)
                or self.sample_time_s < 0 or self.clock_domain != "MUJOCO_SIM_TIME"):
            raise ValueError("motion state time or clock domain invalid")
        if self.source not in _STATE_SOURCES:
            raise ValueError("motion state source provenance is unknown")


@dataclass(frozen=True)
class TargetReference:
    """Target stays separate from LiDAR and motion state."""

    position_enu_m: np.ndarray
    yaw_enu_rad: float
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_enu_m", _xyz(self.position_enu_m, "target"))
        if not math.isfinite(self.yaw_enu_rad) or self.source not in _TARGET_SOURCES:
            raise ValueError("target yaw or source invalid")


@dataclass(frozen=True)
class FuturePolicyInput:
    lidar_tensor: np.ndarray  # [3,12,24] only local ray-derived input
    lidar_age_s: float
    lidar_source_frame_identity: str
    motion_state: MotionState
    target: TargetReference
    policy_time_s: float
    clock_domain: str


def build_policy_input(lidar_read: ObservationRead, motion_state: MotionState,
                       target: TargetReference, *, policy_time_s: float) -> FuturePolicyInput:
    """Join *fresh* independent inputs without inventing a legacy 7D policy call."""
    if lidar_read.status != "FRESH" or lidar_read.observation is None:
        raise ValueError("FRESH LiDAR input required; stale/missing payload is withheld")
    policy_time = float(policy_time_s)
    obs = lidar_read.observation
    if (not math.isfinite(policy_time) or policy_time < 0
            or obs.clock_domain != motion_state.clock_domain
            or policy_time + 1e-9 < obs.frame_timestamp
            or policy_time + 1e-9 < motion_state.sample_time_s):
        raise ValueError("future sample, invalid policy time or clock mismatch")
    age = max(0., policy_time - obs.frame_timestamp)
    if (lidar_read.age_s is None or abs(lidar_read.age_s - age) > 1e-9
            or lidar_read.source_frame_timestamp != obs.frame_timestamp):
        raise ValueError("LiDAR read age or source time disagrees with policy time")
    return FuturePolicyInput(lidar_tensor=obs.network_tensor,
                             lidar_age_s=age,
                             lidar_source_frame_identity=obs.source_frame_identity,
                             motion_state=motion_state, target=target,
                             policy_time_s=policy_time, clock_domain=obs.clock_domain)
