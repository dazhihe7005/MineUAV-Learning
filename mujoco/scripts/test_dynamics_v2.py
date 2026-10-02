"""Regression tests for the 7 kg geometry-based inertia estimate."""

import json
import subprocess
import sys
import unittest
from pathlib import Path

import mujoco
import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MODEL = ROOT / "models" / "mine_uav_dynamics_v2.xml"


def run_script(name, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name), *args],
        capture_output=True, text=True, check=False,
    )


class DynamicsV2Tests(unittest.TestCase):
    def test_mass_budget_and_all_stl_roles(self):
        import estimate_dynamics_v2 as v2

        report = v2.estimate_dynamics()
        self.assertAlmostEqual(report["known_mass_kg"], 6.2298, places=8)
        self.assertAlmostEqual(report["structure_mass_kg"], 0.7702, places=8)
        self.assertEqual(report["stl_count"], 177)
        self.assertEqual(report["structure_mesh_count"], 156)
        self.assertEqual(report["geometry_method_counts"]["watertight_volume"], 171)
        self.assertAlmostEqual(sum(item["mass_kg"] for item in report["components"]), 7.0, places=9)
        self.assertEqual(len({item["filename"] for item in report["components"]}), 177)
        self.assertTrue(all(item["geometry_method"] == "watertight_volume"
                            for item in report["components"] if item["role"] == "structure"))

    def test_mesh_inertia_uses_shape_not_only_centroid(self):
        import estimate_dynamics_v2 as v2

        box = trimesh.creation.box(extents=[0.2, 0.4, 0.6])
        moments = v2.geometry_moments(box)
        self.assertEqual(moments.method, "watertight_volume")
        np.testing.assert_allclose(moments.center_m, [0, 0, 0], atol=1e-12)
        np.testing.assert_allclose(
            np.diag(moments.inertia_per_kg),
            [(0.4**2 + 0.6**2) / 12,
             (0.2**2 + 0.6**2) / 12,
             (0.2**2 + 0.4**2) / 12],
            atol=1e-12,
        )

    def test_generated_model_has_estimated_full_inertia_and_four_inputs(self):
        result = run_script("estimate_dynamics_v2.py")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((ROOT / "reports" / "dynamics_v2_report.json").read_text())
        model = mujoco.MjModel.from_xml_path(str(MODEL))
        body = model.body("mine_uav")
        self.assertAlmostEqual(body.mass[0], 7.0, places=10)
        np.testing.assert_allclose(body.ipos, report["estimated_com_v2_m"], atol=1e-9)
        self.assertEqual(model.nu, 4)
        self.assertGreater(min(np.linalg.eigvalsh(report["estimated_inertia_v2_kg_m2"])), 0)
        self.assertTrue((ROOT / "reports" / "dynamics_v2_component_manifest.csv").exists())
        self.assertNotIn("TEMPORARY mass/COM/inertia", MODEL.read_text(encoding="utf-8"))

    def test_v2_mixer_matches_mujoco_about_estimated_com(self):
        result = run_script("verify_dynamics_v2.py", "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["rotation_signs"], [1, -1, 1, -1])
        collective = report["tests"]["collective"]
        self.assertAlmostEqual(collective["theoretical"][2], 7.0 * 9.81, places=7)
        for case in report["tests"].values():
            np.testing.assert_allclose(case["mujoco"], case["theoretical"], atol=1e-7)
            self.assertAlmostEqual(case["theoretical"][2], collective["theoretical"][2], places=7)
        for name, torque_axis in (("roll", 3), ("pitch", 4), ("yaw", 5)):
            self.assertGreater(report["tests"][name]["theoretical"][torque_axis],
                               collective["theoretical"][torque_axis])
        self.assertTrue((ROOT / "reports" / "mixing_matrix_v2.txt").exists())


if __name__ == "__main__":
    unittest.main()
