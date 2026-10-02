"""World-velocity/yaw-rate commands into the existing thrust/attitude cascade.

The policy boundary is [vx, vy, vz] in world coordinates plus body/world-Z
yaw rate. No action directly reaches MuJoCo rotor controls or body wrench.
"""

import math

import numpy as np

from hover_controller import HoverController
from position_controller import (
    PositionCommand, acceleration_to_thrust_vector, attitude_to_wrench,
    thrust_vector_to_attitude,
)


ACTION_SCALE = np.array([1.5, 1.5, 1.0, 1.0], dtype=float)
DEFAULT_KV = np.array([1.5, 1.5, 2.0], dtype=float)  # 1/s
DEFAULT_HORIZONTAL_ACCEL_LIMIT = 3.0  # m/s²
DEFAULT_VERTICAL_ACCEL_LIMIT = 3.0    # m/s²


def map_normalized_action(action) -> tuple[np.ndarray, float]:
    """Normalized [-1,1]^4 → world velocity (m/s), yaw rate (rad/s)."""
    value = np.asarray(action, dtype=float)
    if (value.shape != (4,) or not np.isfinite(value).all()
            or np.any(value < -1) or np.any(value > 1)):
        raise ValueError("action must be four finite values in [-1, 1]")
    scaled = value * ACTION_SCALE
    return scaled[:3].copy(), float(scaled[3])


def velocity_error_to_acceleration(velocity_command, velocity, kv,
                                   horizontal_limit, vertical_limit,
                                   integral_acceleration=None) -> np.ndarray:
    """World velocity error plus optional integral term → limited acceleration."""
    command = np.asarray(velocity_command, dtype=float)
    current = np.asarray(velocity, dtype=float)
    gain = np.asarray(kv, dtype=float)
    extra = (np.zeros(3, dtype=float) if integral_acceleration is None
             else np.asarray(integral_acceleration, dtype=float))
    if (any(v.shape != (3,) or not np.isfinite(v).all()
            for v in (command, current, gain, extra)) or np.any(gain < 0)
            or not math.isfinite(horizontal_limit) or horizontal_limit <= 0
            or not math.isfinite(vertical_limit) or vertical_limit <= 0):
        raise ValueError("velocity inputs must be finite 3-vectors with valid gains/limits")
    acceleration = gain * (command - current) + extra
    horizontal_norm = float(np.linalg.norm(acceleration[:2]))
    if horizontal_norm > horizontal_limit:
        acceleration[:2] *= horizontal_limit / horizontal_norm
    acceleration[2] = np.clip(acceleration[2], -vertical_limit, vertical_limit)
    return acceleration


class VelocityCommandController:
    """Velocity P/PI outer loop with yaw target; reuses all lower loops."""

    def __init__(self, attitude_controller: HoverController, gravity_vector,
                 kv=DEFAULT_KV, horizontal_accel_limit=DEFAULT_HORIZONTAL_ACCEL_LIMIT,
                 vertical_accel_limit=DEFAULT_VERTICAL_ACCEL_LIMIT, *,
                 ki_xy: float = 0.0, ki_z: float = 0.0,
                 integral_accel_limit_xy: float = 1.5,
                 integral_accel_limit_z: float = 1.5):
        self.attitude_controller = attitude_controller
        self.gravity_vector = np.asarray(gravity_vector, dtype=float).copy()
        self.kv = np.asarray(kv, dtype=float).copy()
        self.horizontal_accel_limit = float(horizontal_accel_limit)
        self.vertical_accel_limit = float(vertical_accel_limit)
        self.ki_xy = float(ki_xy)
        self.ki_z = float(ki_z)
        self.integral_accel_limit_xy = float(integral_accel_limit_xy)
        self.integral_accel_limit_z = float(integral_accel_limit_z)
        if (self.gravity_vector.shape != (3,) or not np.isfinite(self.gravity_vector).all()
                or self.kv.shape != (3,) or not np.isfinite(self.kv).all()
                or np.any(self.kv < 0) or not math.isfinite(self.horizontal_accel_limit)
                or not math.isfinite(self.vertical_accel_limit)
                or min(self.horizontal_accel_limit, self.vertical_accel_limit) <= 0
                or not all(math.isfinite(value) for value in (
                    self.ki_xy, self.ki_z, self.integral_accel_limit_xy,
                    self.integral_accel_limit_z))
                or min(self.ki_xy, self.ki_z) < 0
                or min(self.integral_accel_limit_xy,
                       self.integral_accel_limit_z) <= 0):
            raise ValueError("Invalid gravity, velocity gains or acceleration limits")
        self.reset()

    @property
    def integral_error(self) -> np.ndarray:
        """Accumulated world-frame velocity error, in metres (copy)."""
        return self._integral_error.copy()

    @property
    def integral_acceleration_world(self) -> np.ndarray:
        """World-frame Ki contribution, in m/s² (copy)."""
        return self._integral_error * [self.ki_xy, self.ki_xy, self.ki_z]

    @property
    def last_desired_acceleration_world(self) -> np.ndarray:
        return self._last_desired_acceleration_world.copy()

    def reset(self, yaw_target: float = 0.0) -> None:
        if not math.isfinite(yaw_target):
            raise ValueError("yaw_target must be finite")
        self.yaw_target = math.atan2(math.sin(yaw_target), math.cos(yaw_target))
        self._integral_error = np.zeros(3, dtype=float)
        self._last_desired_acceleration_world = np.zeros(3, dtype=float)
        self.anti_windup_freeze_count_xy = 0
        self.anti_windup_freeze_count_z = 0

    def compute(self, velocity, quaternion, angular_velocity, velocity_command,
                yaw_rate_command: float, control_dt: float) -> PositionCommand:
        if (not math.isfinite(yaw_rate_command) or not math.isfinite(control_dt)
                or control_dt <= 0):
            raise ValueError("yaw_rate_command and control_dt must be finite; dt > 0")
        # Validate before touching the integrator so malformed inputs cannot
        # pollute the next episode's controller state.
        velocity_error_to_acceleration(
            velocity_command, velocity, self.kv,
            self.horizontal_accel_limit, self.vertical_accel_limit)
        error = np.asarray(velocity_command, dtype=float) - np.asarray(velocity, dtype=float)
        candidate = self._integral_error.copy()
        if self.ki_xy > 0:
            candidate[:2] += error[:2] * control_dt
            contribution_xy = candidate[:2] * self.ki_xy
            magnitude = float(np.linalg.norm(contribution_xy))
            if magnitude > self.integral_accel_limit_xy:
                candidate[:2] *= self.integral_accel_limit_xy / magnitude
        if self.ki_z > 0:
            candidate[2] += error[2] * control_dt
            candidate[2] = np.clip(
                candidate[2], -self.integral_accel_limit_z / self.ki_z,
                self.integral_accel_limit_z / self.ki_z)
        gains = np.array([self.ki_xy, self.ki_xy, self.ki_z])
        candidate_extra = gains * candidate
        delta_extra = gains * (candidate - self._integral_error)
        raw = self.kv * error + candidate_extra
        if (np.linalg.norm(raw[:2]) > self.horizontal_accel_limit
                and np.dot(raw[:2], delta_extra[:2]) > 0):
            candidate[:2] = self._integral_error[:2]
            self.anti_windup_freeze_count_xy += 1
        if (abs(raw[2]) > self.vertical_accel_limit
                and raw[2] * delta_extra[2] > 0):
            candidate[2] = self._integral_error[2]
            self.anti_windup_freeze_count_z += 1
        self._integral_error[:] = candidate
        acceleration = velocity_error_to_acceleration(
            velocity_command, velocity, self.kv,
            self.horizontal_accel_limit, self.vertical_accel_limit,
            self.integral_acceleration_world)
        self._last_desired_acceleration_world[:] = acceleration
        self.yaw_target = math.atan2(
            math.sin(self.yaw_target + yaw_rate_command * control_dt),
            math.cos(self.yaw_target + yaw_rate_command * control_dt))
        force = acceleration_to_thrust_vector(
            acceleration, self.attitude_controller.mass_kg, self.gravity_vector)
        rotation, desired_quat = thrust_vector_to_attitude(force, self.yaw_target)
        wrench = attitude_to_wrench(
            float(np.linalg.norm(force)), quaternion, angular_velocity,
            desired_quat, self.attitude_controller)
        return PositionCommand(acceleration, force, rotation, desired_quat, wrench)
