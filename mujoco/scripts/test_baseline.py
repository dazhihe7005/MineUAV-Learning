"""Behavior checks for the MuJoCo learning artifacts."""

import json
import csv
import subprocess
import sys
import unittest
from pathlib import Path

import mujoco
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CF2 = ROOT / "references" / "crazyflie" / "cf2.xml"


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True,
        text=True,
        check=False,
    )


class BaselineTests(unittest.TestCase):
    def test_inspection_reports_free_joint_and_all_actuators(self):
        result = run_script("inspect_model.py", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual((report["nq"], report["nv"], report["nu"]), (7, 6, 4))
        self.assertEqual(report["qpos_xyz"], [0.0, 0.0, 0.1])
        self.assertEqual(report["qpos_quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(report["actuator_names"], ["body_thrust", "x_moment", "y_moment", "z_moment"])

    def test_fixed_thrust_gives_fall_hover_and_rise(self):
        result = run_script("crazyflie_thrust_test.py", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        cases = json.loads(result.stdout)
        self.assertLess(cases["zero"]["final_vz"], -0.5)
        self.assertLess(cases["zero"]["final_z"], 0.1)
        self.assertLess(abs(cases["hover"]["final_vz"]), 0.02)
        self.assertLess(abs(cases["hover"]["final_z"] - 0.1), 0.005)
        self.assertGreater(cases["high"]["final_vz"], 0.2)
        self.assertGreater(cases["high"]["final_z"], 0.1)
        self.assertLessEqual(cases["high"]["ctrl"][0], 0.35)
        for case in cases.values():
            self.assertEqual(case["ctrl"][1:], [0.0, 0.0, 0.0])

    def test_visual_model_has_one_free_body_and_placeholder_inertia(self):
        result = run_script("prepare_mine_uav_visual.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        path = ROOT / "models" / "mine_uav_visual.xml"
        self.assertTrue(path.exists())
        self.assertIn("TEMPORARY_PLACEHOLDER", path.read_text())
        model = mujoco.MjModel.from_xml_path(str(path))
        self.assertEqual((model.nq, model.nv, model.nu), (7, 6, 4))
        self.assertEqual(model.nbody, 2)
        self.assertEqual(model.nmesh, 3)
        body = trimesh.load(ROOT / "assets" / "mine_uav_body_visual.stl", force="mesh")
        self.assertGreater(body.extents[0], 0.7)
        self.assertGreater(body.extents[1], 0.7)
        self.assertLess(body.extents[0], 1.0)
        self.assertLess(body.extents[1], 1.0)

    def test_yaw_estimate_preserves_source_rows_and_reports_fit_gap(self):
        path = ROOT / "references" / "tmotor_15x5cf_mn5212_kv340_static_tests.csv"
        self.assertTrue(path.exists())
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 7)
        self.assertEqual((int(rows[0]["rpm"]), float(rows[0]["torque_nm"])), (3821, 0.142))
        self.assertEqual((int(rows[-1]["rpm"]), float(rows[-1]["torque_nm"])), (7167, 0.498))
        result = run_script("fit_yaw_torque.py", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["provenance"], "ESTIMATED_FROM_TMOTOR_15x5CF_STATIC_TESTS")
        self.assertAlmostEqual(report["used_k_m"], 8.6e-7, places=11)
        self.assertAlmostEqual(report["through_origin_fit_k_m"], 8.7872411373e-7, places=11)
        self.assertGreater(report["difference_percent"], 2.0)
        self.assertLess(report["difference_percent"], 3.0)


if __name__ == "__main__":
    unittest.main()
