"""Integration checks for the four-rotor single-input wrench model."""

import csv
import json
import subprocess
import sys
import unittest
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL = ROOT / "models" / "mine_uav_visual.xml"


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class RotorWrenchTests(unittest.TestCase):
    def test_yaw_sanity_input_follows_rotation_config(self):
        import verify_rotor_wrench

        self.assertTrue(hasattr(verify_rotor_wrench, "make_cases"))
        cases = verify_rotor_wrench.make_cases(100.0, [-1, 1, -1, 1])
        self.assertEqual(cases["yaw"].tolist(), [90.0, 110.0, 90.0, 110.0])

    def test_official_u7_thrust_fit_is_reproducible(self):
        path = ROOT / "references" / "tmotor_u7_kv490_15x5cf_static_tests.csv"
        self.assertTrue(path.exists())
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertEqual((int(rows[0]["rpm"]), float(rows[0]["thrust_g"])), (5300, 1600.0))
        self.assertEqual((int(rows[-1]["rpm"]), float(rows[-1]["thrust_g"])), (8500, 4100.0))
        result = run_script("fit_thrust_coefficient.py", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["samples"], 5)
        self.assertAlmostEqual(report["k_f_n_per_rad_s_squared"], 4.997760369666161e-5, places=10)

    def test_each_rotor_has_one_combined_site_wrench_input(self):
        result = run_script("build_rotor_model.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        self.assertEqual(model.nu, 4)
        self.assertEqual(
            [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(4)],
            [f"motor_{i}_wrench" for i in range(1, 5)],
        )
        expected_xy = [
            (-0.218964, 0.219884),
            (-0.218964, -0.219118),
            (0.220039, -0.219118),
            (0.220039, 0.219884),
        ]
        for i, (x, y) in enumerate(expected_xy, start=1):
            site = model.site(f"motor_{i}_site")
            self.assertAlmostEqual(site.pos[0], x, places=6)
            self.assertAlmostEqual(site.pos[1], y, places=6)
            self.assertAlmostEqual(site.pos[2], -0.025425, places=6)
            self.assertGreater(model.actuator_gear[i - 1, 2], 0.0)
            self.assertAlmostEqual(abs(model.actuator_gear[i - 1, 5]), 8.6e-7, places=12)

        data = mujoco.MjData(model)
        data.ctrl[0] = 400.0**2
        mujoco.mj_forward(model, data)
        self.assertAlmostEqual(data.qfrc_actuator[2], 7.9964165915, places=5)
        self.assertAlmostEqual(data.qfrc_actuator[5], 0.1376, places=6)
        self.assertNotIn("yaw_reaction", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(4)])

    def test_mixer_cases_match_mujoco_and_short_time_response(self):
        result = run_script("verify_rotor_wrench.py", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["rotation_signs"], [1, -1, 1, -1])
        self.assertEqual(len(report["mixing_matrix"]), 4)
        self.assertEqual(len(report["mixing_matrix"][0]), 4)
        cases = report["tests"]
        collective = cases["collective"]
        self.assertGreater(collective["theoretical"][2], 0.0)
        self.assertLess(abs(collective["theoretical"][3]), 0.01)
        self.assertLess(abs(collective["theoretical"][4]), 0.01)
        self.assertLess(abs(collective["theoretical"][5]), 1e-9)
        for name, torque_index, velocity_index in (("roll", 3, 3), ("pitch", 4, 4), ("yaw", 5, 5)):
            case = cases[name]
            self.assertAlmostEqual(case["theoretical"][2], collective["theoretical"][2], places=8)
            self.assertGreater(case["theoretical"][torque_index], 0.0)
            self.assertGreater(case["simulated_qvel"][velocity_index], 0.0)
            for actual, expected in zip(case["mujoco"], case["theoretical"]):
                self.assertAlmostEqual(actual, expected, places=6)
        self.assertTrue((ROOT / "reports" / "mixing_matrix.txt").exists())


if __name__ == "__main__":
    unittest.main()
