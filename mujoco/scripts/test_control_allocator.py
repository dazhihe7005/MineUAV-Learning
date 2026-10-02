"""Behavior checks for bounded four-rotor wrench allocation."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "control"))


class ControlAllocatorTests(unittest.TestCase):
    def test_hover_trim_has_zero_moments_and_is_inside_rpm_limit(self):
        from control_allocator import ControlAllocator

        allocator = ControlAllocator.from_v2_model()
        result = allocator.allocate([68.67, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result.u, [320188.62675, 321133.06346,
                                              366819.10192, 365874.66520], atol=0.01)
        np.testing.assert_allclose(result.achieved_wrench, [68.67, 0, 0, 0], atol=1e-9)
        self.assertTrue(np.all(result.u >= 0))
        self.assertTrue(np.all(result.rpm <= 8500))
        self.assertTrue(result.feasible)
        self.assertFalse(result.saturated)

    def test_five_requested_wrenches_allocate_without_saturation(self):
        from control_allocator import ControlAllocator

        allocator = ControlAllocator.from_v2_model()
        requests = (
            [68.67, 0, 0, 0], [68.67, 0.5, 0, 0],
            [68.67, 0, 0.5, 0], [68.67, 0, 0, 0.05],
            [68.67, 0.2, -0.2, 0.02],
        )
        for requested in requests:
            with self.subTest(requested=requested):
                result = allocator.allocate(requested)
                np.testing.assert_allclose(result.achieved_wrench, requested, atol=1e-9)
                np.testing.assert_allclose(allocator.B @ result.u, requested, atol=1e-9)
                self.assertFalse(result.saturated)

    def test_unreachable_wrench_saturates_without_negative_squared_speed(self):
        from control_allocator import ControlAllocator

        u_max = (8500 * 2 * math.pi / 60) ** 2
        allocator = ControlAllocator(np.eye(4), max_rpm=8500)
        result = allocator.allocate([2 * u_max, 0.2 * u_max, 0.3 * u_max, 0.4 * u_max])
        np.testing.assert_allclose(result.u, [u_max, 0.2 * u_max, 0.3 * u_max, 0.4 * u_max], atol=1e-8)
        self.assertFalse(result.feasible)
        self.assertTrue(result.saturated)
        self.assertEqual(result.active_limits, ("motor_1_upper",))
        self.assertTrue(np.all(result.u >= 0))
        self.assertTrue(np.all(result.u <= u_max))

    def test_nonfinite_or_wrong_shape_requests_are_rejected(self):
        from control_allocator import ControlAllocator

        allocator = ControlAllocator(np.eye(4))
        with self.assertRaises(ValueError):
            allocator.allocate([1, 2, 3])
        with self.assertRaises(ValueError):
            allocator.allocate([1, 2, float("nan"), 4])


if __name__ == "__main__":
    unittest.main()
