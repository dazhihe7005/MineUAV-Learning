"""End-to-end checks for v2 PD hover recovery and logged evidence."""

import csv
import json
import math
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_closed_loop_hover.py"
REPORT = ROOT / "reports" / "closed_loop_hover_summary.json"


class ClosedLoopHoverTests(unittest.TestCase):
    def test_five_initial_recoveries_and_external_impulse(self):
        completed = subprocess.run([sys.executable, str(SCRIPT)],
                                   capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(REPORT.read_text(encoding="utf-8"))
        self.assertEqual(set(report["cases"]),
                         {"height_low", "height_high", "roll_15", "pitch_minus_15",
                          "combined", "external_vertical_force"})
        self.assertEqual(report["controller"]["integral_gain"], 0.0)
        for name, case in report["cases"].items():
            with self.subTest(case=name):
                self.assertFalse(case["has_nonfinite_state"])
                self.assertLess(abs(case["final_z_error_m"]), 0.05)
                self.assertLess(max(abs(x) for x in case["final_attitude_error_deg"]), 2.0)
                self.assertLess(case["max_motor_rpm"], 8500.0)
                self.assertLess(case["saturation_fraction"], 0.01)
                self.assertIsNotNone(case["settling_time_s"])
                self.assertLess(case["settling_time_s"], 10.0)
                path = ROOT / "reports" / f"closed_loop_{name}.csv"
                self.assertTrue(path.exists())
                with path.open(newline="", encoding="utf-8") as stream:
                    reader = csv.DictReader(stream)
                    self.assertTrue({"time", "x", "y", "z", "vx", "vy", "vz",
                                     "roll", "pitch", "yaw", "wx", "wy", "wz",
                                     "Fz", "Tx", "Ty", "Tz", "rpm1", "rpm2",
                                     "rpm3", "rpm4", "allocator_saturated"}
                                    <= set(reader.fieldnames))
        external = report["cases"]["external_vertical_force"]
        self.assertGreater(external["peak_post_disturbance_height_error_m"], 0.01)
        self.assertGreaterEqual(external["settling_time_s"], 2.1)
        for name in ("height_response.png", "attitude_response.png", "motor_rpm_response.png"):
            self.assertGreater((ROOT / "reports" / name).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
