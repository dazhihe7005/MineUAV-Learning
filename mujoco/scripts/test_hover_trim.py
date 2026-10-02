"""Open-loop trim regression against the previous equal-speed baseline."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "hover_trim_test.py"


class HoverTrimTests(unittest.TestCase):
    def test_five_second_trim_and_equal_speed_comparison(self):
        result = subprocess.run([sys.executable, str(SCRIPT), "--json"],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertAlmostEqual(report["trim_simulation"]["duration_s"], 5.0, places=8)
        np.testing.assert_allclose(report["trim"]["achieved_wrench"],
                                   [68.67, 0, 0, 0], atol=1e-9)
        np.testing.assert_allclose(report["trim_simulation"]["initial_mujoco_wrench_about_com"],
                                   [0, 0, 68.67, 0, 0, 0], atol=1e-8)
        self.assertLess(np.linalg.norm(report["trim_simulation"]["initial_angular_acceleration_rad_s2"]), 1e-5)
        final = report["trim_simulation"]["final_state"]
        self.assertLess(np.linalg.norm(final["position_m"][:2]), 0.05)
        self.assertLess(abs(final["position_m"][2] - 1.0), 0.05)
        self.assertLess(abs(final["euler_rad"][1]), 0.01)
        trimmed_pitch = abs(report["trim_simulation"]["pitch_at_comparison_s_rad"])
        equal_pitch = abs(report["equal_speed_comparison"]["pitch_at_comparison_s_rad"])
        self.assertGreater(equal_pitch, max(0.01, 10 * trimmed_pitch))
        self.assertTrue((ROOT / "reports" / "hover_trim_report.json").exists())
        self.assertTrue((ROOT / "reports" / "hover_trim_trajectory.csv").exists())


if __name__ == "__main__":
    unittest.main()
