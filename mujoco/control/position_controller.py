"""World-position → body-attitude/thrust → body-wrench cascade for MineUAV v2.

World: right-handed, +Z up, model gravity [0, 0, -9.81]. Body: +Z is
rotor thrust, +X/+Y follow the MJCF body axes. Quaternions are MuJoCo
wxyz, rotating body vectors into world. No integral, motor lag or XY yaw
shortcut is hidden in this controller.
"""

import math
from dataclasses import dataclass

import numpy as np

from hover_controller import HoverController


DEFAULT_KP_POSITION = (0.5, 0.5, 2.0)  # s^-2; XY below attitude-loop bandwidth
DEFAULT_KD_POSITION = (1.2, 1.2, 2.0)  # s^-1
DEFAULT_HORIZONTAL_ACCEL_LIMIT = 3.0  # m/s^2, horizontal vector norm
DEFAULT_VERTICAL_ACCEL_LIMIT = 3.0    # m/s^2, absolute vertical command


def _vector3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be three finite values")
    return array


def position_error_to_acceleration(
    position, velocity, target_position, target_velocity,
    kp_position, kd_position, horizontal_accel_limit, vertical_accel_limit,
) -> np.ndarray:
    """World p/v errors → world acceleration, with horizontal-norm and Z limits."""
    p = _vector3(position, "position")
    v = _vector3(velocity, "velocity")
    p_d = _vector3(target_position, "target_position")
    v_d = _vector3(target_velocity, "target_velocity")
    kp = _vector3(kp_position, "kp_position")
    kd = _vector3(kd_position, "kd_position")
    if (np.any(kp < 0) or np.any(kd < 0)
            or not math.isfinite(horizontal_accel_limit) or horizontal_accel_limit <= 0
            or not math.isfinite(vertical_accel_limit) or vertical_accel_limit <= 0):
        raise ValueError("Gains must be nonnegative and acceleration limits positive")
    accel = kp * (p_d - p) + kd * (v_d - v)
    horizontal_norm = float(np.linalg.norm(accel[:2]))
    if horizontal_norm > horizontal_accel_limit:
        accel[:2] *= horizontal_accel_limit / horizontal_norm
    accel[2] = np.clip(accel[2], -vertical_accel_limit, vertical_accel_limit)
    return accel


def acceleration_to_thrust_vector(acceleration, mass_kg, gravity_vector) -> np.ndarray:
    """World acceleration → world force m(a_des - g); gravity has negative Z."""
    accel = _vector3(acceleration, "acceleration")
    gravity = _vector3(gravity_vector, "gravity_vector")
    if not math.isfinite(mass_kg) or mass_kg <= 0:
        raise ValueError("mass_kg must be finite and positive")
    return mass_kg * (accel - gravity)


def _rotation_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Proper rotation matrix → normalized body-to-world MuJoCo wxyz quaternion."""
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quat = np.array([0.25 * scale,
                         (rotation[2, 1] - rotation[1, 2]) / scale,
                         (rotation[0, 2] - rotation[2, 0]) / scale,
                         (rotation[1, 0] - rotation[0, 1]) / scale])
    else:
        index = int(np.argmax(np.diag(rotation)))
        following = (index + 1) % 3
        last = (index + 2) % 3
        scale = 2.0 * math.sqrt(1.0 + rotation[index, index]
                                - rotation[following, following] - rotation[last, last])
        quat = np.zeros(4)
        quat[1 + index] = 0.25 * scale
        quat[0] = (rotation[last, following] - rotation[following, last]) / scale
        quat[1 + following] = (rotation[following, index] + rotation[index, following]) / scale
        quat[1 + last] = (rotation[last, index] + rotation[index, last]) / scale
    return quat / np.linalg.norm(quat)


def thrust_vector_to_attitude(thrust_world, target_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """World thrust direction + yaw heading → body-to-world R and wxyz quaternion.

    R columns are desired body X/Y/Z axes in world coordinates. Desired
    body Z follows thrust. Project the horizontal yaw heading onto the
    plane perpendicular to Z, then complete the right-handed basis.
    """
    thrust = _vector3(thrust_world, "thrust_world")
    if not math.isfinite(target_yaw):
        raise ValueError("target_yaw must be finite")
    norm = float(np.linalg.norm(thrust))
    if norm < 1e-12:
        raise ValueError("thrust vector must be nonzero")
    body_z_world = thrust / norm
    heading_world = np.array([math.cos(target_yaw), math.sin(target_yaw), 0.0])
    body_y_world = np.cross(body_z_world, heading_world)
    lateral_norm = float(np.linalg.norm(body_y_world))
    if lateral_norm < 1e-12:
        raise ValueError("Yaw heading is parallel to desired thrust direction")
    body_y_world /= lateral_norm
    body_x_world = np.cross(body_y_world, body_z_world)
    rotation = np.column_stack((body_x_world, body_y_world, body_z_world))
    return rotation, _rotation_to_quaternion(rotation)


def attitude_to_wrench(thrust_magnitude, quaternion, angular_velocity,
                       desired_quaternion, attitude_controller: HoverController) -> np.ndarray:
    """Desired scalar body-Z thrust + attitude target → [Fz,Tx,Ty,Tz]."""
    if not math.isfinite(thrust_magnitude) or thrust_magnitude < 0:
        raise ValueError("thrust_magnitude must be finite and nonnegative")
    torque = attitude_controller.attitude_torque(
        quaternion, angular_velocity, desired_quaternion)
    return np.r_[min(thrust_magnitude, attitude_controller.max_total_thrust_n), torque]


@dataclass(frozen=True)
class PositionCommand:
    acceleration_world: np.ndarray
    thrust_vector_world: np.ndarray
    desired_rotation_body_to_world: np.ndarray
    desired_quaternion_wxyz: np.ndarray
    desired_wrench_body: np.ndarray


class PositionController:
    """Four-stage outer loop; reuses HoverController's quaternion attitude PD."""

    def __init__(self, attitude_controller: HoverController, gravity_vector,
                 kp_position=DEFAULT_KP_POSITION, kd_position=DEFAULT_KD_POSITION,
                 horizontal_accel_limit=DEFAULT_HORIZONTAL_ACCEL_LIMIT,
                 vertical_accel_limit=DEFAULT_VERTICAL_ACCEL_LIMIT):
        self.attitude_controller = attitude_controller
        self.gravity_vector = _vector3(gravity_vector, "gravity_vector")
        self.kp_position = _vector3(kp_position, "kp_position")
        self.kd_position = _vector3(kd_position, "kd_position")
        self.horizontal_accel_limit = float(horizontal_accel_limit)
        self.vertical_accel_limit = float(vertical_accel_limit)
        if (np.any(self.kp_position < 0) or np.any(self.kd_position < 0)
                or not math.isfinite(self.horizontal_accel_limit)
                or not math.isfinite(self.vertical_accel_limit)
                or min(self.horizontal_accel_limit, self.vertical_accel_limit) <= 0):
            raise ValueError("Invalid position gains or acceleration limits")

    def compute(self, position, velocity, quaternion, angular_velocity,
                target_position, target_velocity=(0, 0, 0), target_yaw=0.0) -> PositionCommand:
        accel = position_error_to_acceleration(
            position, velocity, target_position, target_velocity,
            self.kp_position, self.kd_position,
            self.horizontal_accel_limit, self.vertical_accel_limit)
        force = acceleration_to_thrust_vector(
            accel, self.attitude_controller.mass_kg, self.gravity_vector)
        rotation, desired_quat = thrust_vector_to_attitude(force, target_yaw)
        wrench = attitude_to_wrench(
            float(np.linalg.norm(force)), quaternion, angular_velocity,
            desired_quat, self.attitude_controller)
        return PositionCommand(accel, force, rotation, desired_quat, wrench)
