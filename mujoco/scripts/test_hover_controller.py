"""Unit checks for the height/attitude PD wrench controller."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))


def axis_quat(axis: str, degrees: float) -> np.ndarray:
    theta = math.radians(degrees) / 2
    vector = np.zeros(3)
    vector["xyz".index(axis)] = math.sin(theta)
    return np.r_[math.cos(theta), vector]


class HoverControllerTests(unittest.TestCase):
    def setUp(self):
        from hover_controller import HoverController

        self.controller = HoverController(
            mass_kg=7.0,
            inertia_kg_m2=np.diag([0.16, 0.14, 0.25]),
            max_total_thrust_n=158.0,
        )
        self.position = np.array([0.0, 0.0, 1.0])
        self.zero = np.zeros(3)
        self.level = np.array([1.0, 0.0, 0.0, 0.0])

    def command(self, position=None, velocity=None, quaternion=None, angular_velocity=None,
                target_yaw=0.0):
        return self.controller.compute(
            position=self.position if position is None else position,
            velocity=self.zero if velocity is None else velocity,
            quaternion=self.level if quaternion is None else quaternion,
            angular_velocity=self.zero if angular_velocity is None else angular_velocity,
            target_z=1.0, target_yaw=target_yaw,
        )

    def test_level_target_outputs_gravity_feedforward_and_zero_moments(self):
        np.testing.assert_allclose(self.command(), [68.67, 0, 0, 0], atol=1e-10)

    def test_height_pd_and_tilt_compensation(self):
        low = self.command(position=[0, 0, 0.5])
        high = self.command(position=[0, 0, 1.5])
        tilted = self.command(quaternion=axis_quat("x", 15))
        self.assertGreater(low[0], 68.67)
        self.assertLess(high[0], 68.67)
        self.assertGreater(tilted[0], 68.67)
        self.assertLessEqual(tilted[0], 158.0)
        upside_down = self.command(quaternion=axis_quat("x", 180))
        self.assertEqual(upside_down[0], 0.0)

    def test_attitude_error_has_restoring_sign_without_euler_subtraction(self):
        self.assertLess(self.command(quaternion=axis_quat("x", 15))[1], 0)
        self.assertGreater(self.command(quaternion=axis_quat("y", -15))[2], 0)
        self.assertLess(self.command(quaternion=axis_quat("z", 15))[3], 0)
        self.assertLess(self.command(angular_velocity=[0.5, 0, 0])[1], 0)

    def test_yaw_wrap_chooses_shortest_rotation(self):
        wrench = self.command(quaternion=axis_quat("z", 179), target_yaw=math.radians(-179))
        self.assertGreater(wrench[3], 0)
        self.assertLess(wrench[3], 0.1)

    def test_general_target_quaternion_uses_existing_attitude_loop(self):
        torque = self.controller.attitude_torque(
            self.level, self.zero, axis_quat("y", 10))
        self.assertGreater(torque[1], 0)
        np.testing.assert_allclose(
            self.controller.attitude_torque(self.level, self.zero, self.level),
            self.zero, atol=1e-12)

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.command(quaternion=[0, 0, 0, 0])
        with self.assertRaises(ValueError):
            self.command(velocity=[0, float("nan"), 0])


if __name__ == "__main__":
    unittest.main()
