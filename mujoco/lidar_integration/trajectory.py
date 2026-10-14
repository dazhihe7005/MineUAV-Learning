"""Kinematic pose replay on MuJoCo's integrated 500 Hz clock.

qpos is prescribed every tick; this is not physically consistent flight or IMU.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np

from .pointcloud import PointCloudFrame
from .scanner import Mid360Scanner
from .scene import set_uav_pose


TRAJECTORIES = ("x_line", "y_line", "z_triangle", "yaw_spin", "combined", "shaft_vertical")


@dataclass(frozen=True)
class TickSchedule:
    physics_hz: int = 500
    lidar_hz: int = 10
    imu_hz: int = 250

    def __post_init__(self) -> None:
        rates = (self.physics_hz, self.lidar_hz, self.imu_hz)
        if (any(isinstance(rate, bool) or not isinstance(rate, int) or rate <= 0 for rate in rates)
                or self.physics_hz % self.lidar_hz or self.physics_hz % self.imu_hz
                or 1_000_000 % self.physics_hz):
            raise ValueError("sensor frequencies must be integer divisors of the physics clock")

    @property
    def physics_dt_s(self) -> float:
        return 1 / self.physics_hz

    @property
    def lidar_stride(self) -> int:
        return self.physics_hz // self.lidar_hz

    @property
    def imu_stride(self) -> int:
        return self.physics_hz // self.imu_hz

    def _check_tick(self, tick: int) -> None:
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("tick must be a nonnegative integer")

    def is_lidar_tick(self, tick: int) -> bool:
        self._check_tick(tick)
        return tick % self.lidar_stride == 0

    def is_imu_tick(self, tick: int) -> bool:
        self._check_tick(tick)
        return tick % self.imu_stride == 0

    def time_usec(self, tick: int) -> int:
        self._check_tick(tick)
        return tick * (1_000_000 // self.physics_hz)


def trajectory_pose(name: str, t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Pre-registered one-second paths in world ENU, body quaternion wxyz."""
    t = float(t_s)
    if name not in TRAJECTORIES or not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("unknown trajectory or time outside fixed [0,1] s protocol")
    xyz = np.array([0., 0., .7])
    yaw = 0.0
    if name == "x_line":
        xyz[0] = .4 * t
    elif name == "y_line":
        xyz[1] = .3 * t
    elif name == "z_triangle":
        xyz[2] += .4 * min(t, 1 - t)
    elif name == "yaw_spin":
        yaw = math.pi / 2 * t
    elif name == "combined":
        xyz += [.25 * t, .1 * t, .1 * t]
        yaw = math.pi / 3 * t
    elif name == "shaft_vertical":
        xyz[2] += t
    return xyz, np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])


@dataclass(frozen=True)
class DynamicScan:
    frame: PointCloudFrame
    time_usec: int
    body_position_enu_m: np.ndarray
    body_quaternion_wxyz: np.ndarray
    sensor_pose_world_enu: np.ndarray
    probe_ranges_m: dict[str, float]
    scan_trace_ms: float
    scan_conversion_ms: float
    scan_total_ms: float
    provenance: str = "KINEMATIC_POSE_REPLAY"


def replay_scans(model: mujoco.MjModel, data: mujoco.MjData, scanner: Mid360Scanner,
                 name: str, frames: int = 11, downsample: int = 1) -> list[DynamicScan]:
    """Advance real `mj_step` time; overwrite pose at each tick before raycasting.

    The physics integrator is used as the clock, *not* as a trajectory solver.
    No IMU values are sampled or implied by this kinematic qpos/qvel replay.
    """
    schedule = TickSchedule()
    if (name not in TRAJECTORIES or isinstance(frames, bool) or not isinstance(frames, int)
            or not 1 <= frames <= 11 or isinstance(downsample, bool)
            or not isinstance(downsample, int) or downsample < 1
            or downsample > scanner.pattern_samples or model is not scanner.model
            or abs(model.opt.timestep - schedule.physics_dt_s) > 1e-12
            or abs(data.time) > 1e-12):
        raise ValueError("invalid replay protocol, model or initial clock")
    results: list[DynamicScan] = []
    last_tick = (frames - 1) * schedule.lidar_stride
    angles_theta = np.array([0, math.pi / 2, -math.pi / 2, 0, 0], dtype=float)
    angles_phi = np.array([0, 0, 0, -math.pi / 2, math.pi / 2], dtype=float)
    names = ("front", "left", "right", "down", "up")
    for tick in range(last_tick + 1):
        prescribed_time = schedule.time_usec(tick) / 1_000_000
        if abs(float(data.time) - prescribed_time) > 1e-9:
            raise RuntimeError("MuJoCo integrated clock disagrees with integer tick schedule")
        xyz, quat = trajectory_pose(name, prescribed_time)
        set_uav_pose(model, data, xyz, quat)
        if schedule.is_lidar_tick(tick):
            start = time.perf_counter()
            frame = scanner.scan_next(data, downsample=downsample)
            elapsed_ms = (time.perf_counter() - start) * 1000
            trace_ms, conversion_ms = scanner.last_trace_ms, scanner.last_conversion_ms
            # Fixed-angle probes use the same CPU ray backend, outside the pattern timing.
            probe = scanner.scan_angles(data, angles_theta, angles_phi)
            results.append(DynamicScan(
                frame=frame, time_usec=schedule.time_usec(tick),
                body_position_enu_m=data.qpos[:3].copy(),
                body_quaternion_wxyz=data.qpos[3:7].copy(),
                sensor_pose_world_enu=frame.sensor_pose.copy(),
                probe_ranges_m=dict(zip(names, map(float, probe.ranges))),
                scan_trace_ms=trace_ms, scan_conversion_ms=conversion_ms,
                scan_total_ms=elapsed_ms))
        if tick < last_tick:
            mujoco.mj_step(model, data)
    return results
