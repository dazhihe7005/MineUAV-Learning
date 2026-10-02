"""Frame, sign, bound and composition tests for the position cascade."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))


class PositionControllerTests(unittest.TestCase):
    def test_outer_loop_axis_limits(self):
        from position_controller import position_error_to_acceleration

        accel = position_error_to_acceleration(
            [0, 0, 1], [0, 0, 0], [10, 10, 10], [0, 0, 0],
            [2, 2, 2], [2, 2, 2], 3.0, 2.0)
        self.assertAlmostEqual(np.linalg.norm(accel[:2]), 3.0)
        self.assertAlmostEqual(accel[2], 2.0)
        np.testing.assert_allclose(
            position_error_to_acceleration(
                [1, 2, 3], [0, 0, 0], [1, 2, 3], [0, 0, 0],
                [2, 2, 2], [2, 2, 2], 3.0, 2.0),
            [0, 0, 0])

    def test_gravity_sign_and_thrust_direction(self):
        from position_controller import acceleration_to_thrust_vector

        np.testing.assert_allclose(
            acceleration_to_thrust_vector([0, 0, 0], 7.0, [0, 0, -9.81]),
            [0, 0, 68.67])
        np.testing.assert_allclose(
            acceleration_to_thrust_vector([1, -2, 0], 7.0, [0, 0, -9.81]),
            [7, -14, 68.67])

    def test_attitude_is_orthonormal_and_has_expected_tilt_sign(self):
        from position_controller import thrust_vector_to_attitude

        rotation, quat = thrust_vector_to_attitude([0, 0, 68.67], 0.0)
        np.testing.assert_allclose(rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(quat, [1, 0, 0, 0], atol=1e-12)
        for force, yaw in (([7, 0, 68.67], 0), ([0, 7, 68.67], 0),
                           ([7, -5, 68.67], math.pi / 3)):
            rotation, quat = thrust_vector_to_attitude(force, yaw)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)
            np.testing.assert_allclose(rotation[:, 2], np.array(force) / np.linalg.norm(force))
            self.assertAlmostEqual(np.linalg.norm(quat), 1.0)
        self.assertGreater(thrust_vector_to_attitude([7, 0, 68.67], 0)[1][2], 0)
        self.assertLess(thrust_vector_to_attitude([0, 7, 68.67], 0)[1][1], 0)

    def test_attitude_to_wrench_reuses_hover_attitude_loop(self):
        from hover_controller import HoverController
        from position_controller import attitude_to_wrench

        attitude = HoverController(7.0, np.diag([0.16, 0.14, 0.25]), 158.0)
        current = np.array([1, 0, 0, 0])
        desired = np.array([math.cos(0.05), 0, math.sin(0.05), 0])
        wrench = attitude_to_wrench(70.0, current, np.zeros(3), desired, attitude)
        self.assertAlmostEqual(wrench[0], 70.0)
        self.assertGreater(wrench[2], 0)
        np.testing.assert_allclose(wrench[1::2], [0, 0], atol=1e-12)


if __name__ == "__main__":
    unittest.main()
