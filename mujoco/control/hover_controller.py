"""Height and quaternion-attitude PD controller that requests a body wrench.

Position/linear velocity are world-frame; quaternion is MuJoCo wxyz and maps
body to world; angular velocity is in the local body frame. Output is
[Fz, Tx, Ty, Tz] about ESTIMATED_COM_V2, never motor commands.
"""

import math

import numpy as np


DEFAULT_KP_Z = 14.0          # N/m
DEFAULT_KD_Z = 14.0          # N/(m/s)
DEFAULT_KP_ATTITUDE = (6.0, 6.0, 2.0)  # 1/s^2, roll/pitch/yaw
DEFAULT_KD_ATTITUDE = (4.0, 4.0, 2.8)  # 1/s, roll/pitch/yaw


def _quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.r_[a[0] * b[0] - np.dot(a[1:], b[1:]),
                 a[0] * b[1:] + b[0] * a[1:] + np.cross(a[1:], b[1:])]


def _attitude_error_body(quaternion: np.ndarray, desired: np.ndarray) -> np.ndarray:
    """Shortest current-to-desired axis-angle error, expressed in body frame."""
    conjugate = quaternion * np.array([1.0, -1.0, -1.0, -1.0])
    error = _quaternion_multiply(conjugate, desired)
    if error[0] < 0:
        error = -error
    vector_norm = float(np.linalg.norm(error[1:]))
    if vector_norm < 1e-12:
        return 2 * error[1:]
    angle = 2 * math.atan2(vector_norm, float(error[0]))
    return angle * error[1:] / vector_norm


class HoverController:
    """Open-loop wrench request from state errors; no integral or x/y control."""

    def __init__(self, mass_kg: float, inertia_kg_m2: np.ndarray,
                 max_total_thrust_n: float, gravity_m_s2: float = 9.81,
                 kp_z: float = DEFAULT_KP_Z, kd_z: float = DEFAULT_KD_Z,
                 kp_attitude: tuple[float, float, float] = DEFAULT_KP_ATTITUDE,
                 kd_attitude: tuple[float, float, float] = DEFAULT_KD_ATTITUDE):
        inertia = np.asarray(inertia_kg_m2, dtype=float)
        gains_p = np.asarray(kp_attitude, dtype=float)
        gains_d = np.asarray(kd_attitude, dtype=float)
        scalars = (mass_kg, max_total_thrust_n, gravity_m_s2, kp_z, kd_z)
        if (inertia.shape != (3, 3) or not np.isfinite(inertia).all()
                or not np.allclose(inertia, inertia.T, atol=1e-10)
                or np.min(np.linalg.eigvalsh(inertia)) <= 0):
            raise ValueError("inertia_kg_m2 must be a finite positive-definite 3x3 tensor")
        if (gains_p.shape != (3,) or gains_d.shape != (3,)
                or not np.isfinite(gains_p).all() or not np.isfinite(gains_d).all()
                or np.any(gains_p < 0) or np.any(gains_d < 0)
                or any(not math.isfinite(x) or x <= 0 for x in scalars[:3])
                or any(not math.isfinite(x) or x < 0 for x in scalars[3:])):
            raise ValueError("mass, thrust, gravity and PD gains must be finite and nonnegative")
        self.mass_kg = float(mass_kg)
        self.inertia_kg_m2 = inertia.copy()
        self.max_total_thrust_n = float(max_total_thrust_n)
        self.gravity_m_s2 = float(gravity_m_s2)
        self.kp_z = float(kp_z)
        self.kd_z = float(kd_z)
        self.kp_attitude = gains_p.copy()
        self.kd_attitude = gains_d.copy()

    def attitude_torque(self, quaternion: np.ndarray, angular_velocity: np.ndarray,
                        desired_quaternion: np.ndarray) -> np.ndarray:
        """Shared quaternion PD attitude loop; all quaternions are body-to-world wxyz."""
        quat = np.asarray(quaternion, dtype=float)
        desired = np.asarray(desired_quaternion, dtype=float)
        omega = np.asarray(angular_velocity, dtype=float)
        if (quat.shape != (4,) or desired.shape != (4,) or omega.shape != (3,)
                or not all(np.isfinite(v).all() for v in (quat, desired, omega))):
            raise ValueError("Attitude inputs must be finite quaternion/quaternion/body rates")
        norms = (float(np.linalg.norm(quat)), float(np.linalg.norm(desired)))
        if min(norms) < 1e-12:
            raise ValueError("Quaternion norm must be nonzero")
        attitude_error = _attitude_error_body(quat / norms[0], desired / norms[1])
        desired_angular_accel = self.kp_attitude * attitude_error - self.kd_attitude * omega
        angular_momentum = self.inertia_kg_m2 @ omega
        # Rigid-body equation: tau = I*omega_dot + omega x (I*omega).
        return self.inertia_kg_m2 @ desired_angular_accel + np.cross(omega, angular_momentum)

    def compute(self, position: np.ndarray, velocity: np.ndarray,
                quaternion: np.ndarray, angular_velocity: np.ndarray,
                target_z: float = 1.0, target_yaw: float = 0.0) -> np.ndarray:
        pos = np.asarray(position, dtype=float)
        vel = np.asarray(velocity, dtype=float)
        quat = np.asarray(quaternion, dtype=float)
        omega = np.asarray(angular_velocity, dtype=float)
        if (pos.shape != (3,) or vel.shape != (3,) or quat.shape != (4,)
                or omega.shape != (3,) or not all(np.isfinite(v).all() for v in (pos, vel, quat, omega))
                or not math.isfinite(target_z) or not math.isfinite(target_yaw)):
            raise ValueError("State and targets must have correct shapes and finite values")
        qnorm = float(np.linalg.norm(quat))
        if qnorm < 1e-12:
            raise ValueError("quaternion norm must be nonzero")
        quat = quat / qnorm

        # World-Z component of body +Z. Never divide by zero or command
        # positive thrust while body +Z points into the ground.
        tilt_cosine = 1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2)
        vertical_force = (self.mass_kg * self.gravity_m_s2
                          + self.kp_z * (target_z - pos[2])
                          - self.kd_z * vel[2])
        if tilt_cosine <= 0:
            thrust = 0.0
        else:
            thrust = float(np.clip(max(0.0, vertical_force) / max(tilt_cosine, 0.25),
                                   0.0, self.max_total_thrust_n))

        desired = np.array([math.cos(target_yaw / 2), 0.0, 0.0, math.sin(target_yaw / 2)])
        torque = self.attitude_torque(quat, omega, desired)
        return np.r_[thrust, torque]
