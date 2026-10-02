"""Regression: XY outer-loop bandwidth must settle with the existing attitude loop."""

import sys
import unittest
from pathlib import Path

import mujoco


sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_position_loop import (CASES, MODEL, build_controllers, simulate_case,
                               summarize_case)  # noqa: E402


class PositionLoopTests(unittest.TestCase):
    def test_x_step_settles_without_saturation(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        controller, allocator = build_controllers(model)
        config = CASES["A_x_plus_2"]
        rows, extra = simulate_case(model, controller, allocator, config)
        result = summarize_case(rows, extra, config)
        self.assertLess(result["final_position_error_norm_m"], 0.05)
        self.assertLess(result["final_speed_norm_m_s"], 0.05)
        self.assertIsNotNone(result["settling_time_s"])
        self.assertEqual(result["saturation_steps"], 0)

    def test_all_six_required_cases_converge_and_keep_rotor_bounds(self):
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        controller, allocator = build_controllers(model)
        for name, config in CASES.items():
            with self.subTest(case=name):
                rows, extra = simulate_case(model, controller, allocator, config)
                result = summarize_case(rows, extra, config)
                self.assertLess(result["final_position_error_norm_m"], 0.05)
                self.assertLess(result["final_speed_norm_m_s"], 0.05)
                self.assertIsNotNone(result["settling_time_s"])
                self.assertLess(result["max_motor_rpm"], allocator.max_rpm)
                self.assertEqual(result["saturation_steps"], 0)
                self.assertFalse(result["has_nonfinite_state"])


if __name__ == "__main__":
    unittest.main()
