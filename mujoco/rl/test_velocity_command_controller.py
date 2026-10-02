"""Speed-command boundary tests; actions must never become rotor inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))


class VelocityCommandTests(unittest.TestCase):
    @staticmethod
    def _controller(**kwargs):
        from hover_controller import HoverController
        from velocity_command_controller import VelocityCommandController

        attitude = HoverController(7, np.diag([0.16, 0.14, 0.25]), 158)
        return VelocityCommandController(attitude, [0, 0, -9.81], **kwargs)

    @staticmethod
    def _compute(controller, command, velocity=(0, 0, 0), dt=0.1):
        return controller.compute(
            velocity=velocity, quaternion=[1, 0, 0, 0],
            angular_velocity=[0, 0, 0], velocity_command=command,
            yaw_rate_command=0.0, control_dt=dt)

    def test_normalized_action_maps_to_metric_commands(self):
        from velocity_command_controller import map_normalized_action

        velocity, yaw_rate = map_normalized_action(np.array([1, -1, 1, -1]))
        np.testing.assert_allclose(velocity, [1.5, -1.5, 1.0])
        self.assertEqual(yaw_rate, -1.0)
        with self.assertRaises(ValueError):
            map_normalized_action([1.1, 0, 0, 0])

    def test_speed_error_commands_tilt_and_integrates_yaw_target(self):
        from hover_controller import HoverController
        from velocity_command_controller import VelocityCommandController

        attitude = HoverController(7, np.diag([0.16, 0.14, 0.25]), 158)
        controller = VelocityCommandController(attitude, [0, 0, -9.81])
        command = controller.compute(
            velocity=[0, 0, 0], quaternion=[1, 0, 0, 0],
            angular_velocity=[0, 0, 0], velocity_command=[1.5, 0, 0],
            yaw_rate_command=1.0, control_dt=0.01)
        self.assertGreater(command.acceleration_world[0], 0)
        self.assertGreater(command.desired_quaternion_wxyz[2], 0)
        self.assertGreater(command.desired_wrench_body[0], 68.67)
        self.assertAlmostEqual(controller.yaw_target, 0.01)
        controller.reset()
        self.assertEqual(controller.yaw_target, 0.0)

    def test_pi_accumulates_separate_xy_and_z_velocity_error(self):
        """Removing integral action must lose the static-error compensation."""
        controller = self._controller(ki_xy=0.5, ki_z=0.8)
        for _ in range(10):
            command = self._compute(controller, [0.5, 0.0, 0.5])
        np.testing.assert_allclose(controller.integral_error, [0.5, 0, 0.5])
        np.testing.assert_allclose(controller.integral_acceleration_world,
                                   [0.25, 0, 0.4])
        np.testing.assert_allclose(command.acceleration_world, [1.0, 0, 1.4])
        np.testing.assert_allclose(controller.last_desired_acceleration_world,
                                   [1.0, 0, 1.4])

    def test_default_controller_remains_exact_p_for_legacy_baseline(self):
        controller = self._controller()
        for _ in range(10):
            command = self._compute(controller, [0.5, 0, 0.5])
        np.testing.assert_allclose(command.acceleration_world, [0.75, 0, 1.0])
        np.testing.assert_array_equal(controller.integral_error, [0, 0, 0])

    def test_integral_acceleration_is_bounded_separately_in_xy_and_z(self):
        """Persistent error must not build unbounded acceleration demand."""
        controller = self._controller(ki_xy=0.5, ki_z=0.8)
        for _ in range(100):
            self._compute(controller, [0.5, 0, 0.5], dt=0.1)
        np.testing.assert_allclose(controller.integral_acceleration_world,
                                   [1.5, 0, 1.5])
        np.testing.assert_allclose(controller.integral_error, [3.0, 0, 1.875])

    def test_acceleration_saturation_freezes_only_worsening_integral(self):
        """Integration must stop at both limits but permit later unwinding."""
        controller = self._controller(ki_xy=0.5, ki_z=0.8)
        for _ in range(20):
            command = self._compute(controller, [1.5, 0, 1.0],
                                    velocity=[-1.5, 0, -1.0])
        np.testing.assert_array_equal(controller.integral_error, [0, 0, 0])
        np.testing.assert_allclose(command.acceleration_world, [3, 0, 3])
        self.assertEqual(controller.anti_windup_freeze_count_xy, 20)
        self.assertEqual(controller.anti_windup_freeze_count_z, 20)
        for _ in range(5):
            self._compute(controller, [0.5, 0, 0])
        self.assertGreater(controller.integral_error[0], 0)
        before = controller.integral_error[0]
        self._compute(controller, [-0.5, 0, 0])
        self.assertLess(controller.integral_error[0], before)

    def test_reset_clears_pi_state_and_diagnostics(self):
        controller = self._controller(ki_xy=0.5, ki_z=0.8)
        self._compute(controller, [0.5, 0, 0.5])
        self.assertGreater(np.linalg.norm(controller.integral_error), 0)
        controller.reset(yaw_target=0.2)
        np.testing.assert_array_equal(controller.integral_error, [0, 0, 0])
        np.testing.assert_array_equal(controller.integral_acceleration_world, [0, 0, 0])
        np.testing.assert_array_equal(controller.last_desired_acceleration_world, [0, 0, 0])
        self.assertEqual(controller.anti_windup_freeze_count_xy, 0)
        self.assertEqual(controller.anti_windup_freeze_count_z, 0)
        self.assertAlmostEqual(controller.yaw_target, 0.2)

    def test_pi_rejects_negative_gain_and_invalid_dt(self):
        with self.assertRaises(ValueError):
            self._controller(ki_xy=-0.1)
        with self.assertRaises(ValueError):
            self._controller(ki_z=float("nan"))
        controller = self._controller(ki_xy=0.5, ki_z=0.8)
        with self.assertRaises(ValueError):
            self._compute(controller, [0, 0, 0], dt=0)


if __name__ == "__main__":
    unittest.main()
