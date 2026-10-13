"""Offline checks for the Phase 3E-L1 evidence and design boundaries."""

import hashlib
import json
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "mujoco/reports/mid360s_interface_audit.json"
MODEL = ROOT / "mujoco/models/mine_uav_dynamics_v2.xml"


class Mid360sAuditContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(REPORT.read_text(encoding="utf-8"))

    def test_source_identity_is_pinned_and_truth_levels_are_distinct(self):
        sources = self.report["sources"]
        self.assertEqual(sources["frontier"]["commit"],
                         "db897c4e3a1607f5852b484afd0a01523cb08c6b")
        for name in ("driver_reference", "fast_lio_reference", "mujoco_lidar_candidate"):
            self.assertEqual(len(sources[name]["commit"]), 40)
            self.assertFalse(sources[name]["deployed_version_verified"])
        self.assertEqual(self.report["real_hardware"]["network"]["radar_ip"], "192.168.1.157")
        self.assertEqual(self.report["real_hardware"]["network"]["nuc_ip"], "192.168.1.10")

    def test_mid360_candidate_is_not_claimed_mid360s_equivalent(self):
        candidate = self.report["reuse_candidate"]
        self.assertEqual(candidate["scan_pattern_name"], "mid360")
        self.assertFalse(candidate["mid360s_equivalence_verified"])
        self.assertFalse(candidate["per_point_timing_verified"])
        self.assertFalse(candidate["imu_simulation_verified"])
        self.assertEqual(candidate["backend"], "MuJoCo")

    def test_ros_and_udp_interfaces_are_not_confused(self):
        interfaces = self.report["interfaces"]
        self.assertEqual(interfaces["ros_lidar_reference"]["type"],
                         "livox_ros_driver2/CustomMsg")
        self.assertEqual(interfaces["ros_imu_reference"]["type"], "sensor_msgs/Imu")
        self.assertEqual(interfaces["fast_lio_reference_custom_subscription"],
                         "livox_ros_driver/CustomMsg")
        self.assertFalse(interfaces["deployed_fast_lio_message_compatibility_verified"])
        self.assertEqual(interfaces["udp_packet_layout"]["status"], "UNVERIFIED")

    def test_mujoco_visual_mesh_is_not_a_sensor_or_calibration(self):
        tree = ET.parse(MODEL)
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                         self.report["sources"]["mineuav"]["model_sha256"])
        self.assertIsNotNone(tree.find(".//geom[@name='lidar_visual']"))
        self.assertIsNone(tree.find(".//sensor"))
        self.assertIsNone(tree.find(".//site[@name='lidar_site']"))
        self.assertFalse(self.report["extrinsics"]["real_aircraft_calibration_verified"])
        self.assertEqual(self.report["extrinsics"]["legacy_sdf_source_status"],
                         "CONFIGURED_ONLY")

    def test_localization_path_requires_timing_extrinsics_and_imu(self):
        path = self.report["integration_plan"]["fast_lio2_path"]
        self.assertEqual(path["status"], "NOT TESTED")
        for field in ("per_point_time", "synchronized_imu", "measured_extrinsics",
                      "deployed_message_type_check", "clock_domain_check",
                      "tag_line_semantics"):
            self.assertIn(field, path["entry_gates"])

    def test_no_runtime_or_training_claims(self):
        execution = self.report["execution"]
        self.assertEqual(execution["sitl_runs"], 0)
        self.assertEqual(execution["px4_runs"], 0)
        self.assertEqual(execution["flights"], 0)
        self.assertEqual(execution["training_runs"], 0)


if __name__ == "__main__":
    unittest.main()
