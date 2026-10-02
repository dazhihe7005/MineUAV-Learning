"""Synthetic checks for the read-only braking diagnostic math."""

import csv
import gzip
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_ppo_braking import (  # noqa: E402
    crosses_target_plane, evaluate_paired, first_entry, make_step_record,
    save_diagnostic_plots, select_representative_seed, summarize_distance_bins,
    summarize_policy, write_step_log,
)


def row(t, x, vx=0.0, cmd_x=0.0):
    return make_step_record(t, [x, 0, 0], [0, 0, 0], [vx, 0, 0],
                            [0, 0, 0, 0], [cmd_x, 0, 0], 0.0)


class BrakingMathTests(unittest.TestCase):
    def test_radial_tangential_and_reverse_command(self):
        record = make_step_record(0.0, [1, 0, 0], [0, 0, 0],
                                  [-2, 3, 0], [0, 0, 0, 0], [1, 0, 0], 0)
        self.assertAlmostEqual(record["v_radial_m_s"], 2.0)
        self.assertAlmostEqual(record["v_tangent_m_s"], 3.0)
        self.assertAlmostEqual(record["v_cmd_radial_m_s"], -1.0)
        self.assertAlmostEqual(record["v_cmd_tangent_m_s"], 0.0)
        self.assertTrue(record["active_braking"])
        self.assertTrue(record["deceleration_requested"])
        self.assertAlmostEqual(record["speed_error_m_s"], math.sqrt(18))

    def test_zero_distance_has_finite_projections(self):
        record = make_step_record(0, [0, 0, 0], [0, 0, 0],
                                  [1, 2, 0], [0, 0, 0, 0], [1, 0, 0], 0)
        self.assertEqual(record["v_radial_m_s"], 0)
        self.assertEqual(record["v_tangent_m_s"], math.sqrt(5))
        self.assertTrue(math.isfinite(record["v_cmd_radial_m_s"]))

    def test_bins_do_not_mix_regions(self):
        bins = summarize_distance_bins([row(0, 0.4, vx=-1, cmd_x=0.2),
                                        row(0.04, 0.15, vx=-3, cmd_x=0.4)])
        self.assertEqual(bins["0_3_to_0_5"]["steps"], 1)
        self.assertAlmostEqual(bins["0_3_to_0_5"]["mean_actual_speed_m_s"], 1)
        self.assertEqual(bins["0_1_to_0_2"]["steps"], 1)
        self.assertAlmostEqual(bins["0_1_to_0_2"]["mean_actual_speed_m_s"], 3)
        self.assertEqual(bins["lt_0_1"]["steps"], 0)
        self.assertIsNone(bins["lt_0_1"]["mean_actual_speed_m_s"])

    def test_near_summary_preserves_signed_radial_and_command_tangent(self):
        step = make_step_record(0, [0.1, 0, 0], [0, 0, 0],
                                [-1, 2, 0], [0, 0, 0, 0], [1, 3, 0], 0)
        episode = {"seed": 1, "target_m": [0, 0, 0], "steps": [step],
                   "episode_length": 1, "episode_reward": 0,
                   "final_distance_m": 0.1, "termination_reason": "time_limit"}
        stats = summarize_policy([episode])["near_target"]["lt_0_2"]
        self.assertAlmostEqual(stats["mean_v_radial_m_s"], 1)
        self.assertAlmostEqual(stats["mean_abs_v_radial_m_s"], 1)
        self.assertAlmostEqual(stats["mean_v_tangent_m_s"], 2)
        self.assertAlmostEqual(stats["mean_v_cmd_radial_m_s"], -1)
        self.assertAlmostEqual(stats["mean_v_cmd_tangent_m_s"], 3)

    def test_crossing_requires_signed_plane_flip_not_only_distance_minimum(self):
        passed = [row(0, 0.6), row(0.04, 0.4), row(0.08, 0.01), row(0.12, -0.03)]
        missed = [row(0, 0.6), row(0.04, 0.4), row(0.08, 0.01), row(0.12, 0.04)]
        self.assertTrue(crosses_target_plane(passed, 1, 0.5))
        self.assertFalse(crosses_target_plane(missed, 1, 0.5))

    def test_first_entry_half_second_lookahead_stops_at_episode_end(self):
        trace = [row(0, 0.6, vx=-2), row(0.04, 0.4, vx=-1.5, cmd_x=0.2),
                 row(0.08, 0.2, vx=-0.5, cmd_x=0.1), row(0.12, -0.03, vx=1.2, cmd_x=0.1)]
        entry = first_entry(trace, 0.5)
        self.assertEqual(entry["step_index"], 1)
        self.assertAlmostEqual(entry["minimum_speed_next_0_5_s_m_s"], 0.5)
        self.assertAlmostEqual(entry["maximum_speed_next_0_5_s_m_s"], 1.5)
        self.assertTrue(entry["crossed_target_plane_next_0_5_s"])
        self.assertTrue(entry["crossed_target_plane_before_episode_end"])
        self.assertTrue(entry["active_braking_next_0_5_s"])
        self.assertAlmostEqual(entry["v_cmd_tangent_m_s"], 0.0)
        self.assertIsNone(first_entry(trace, 0.02))

    def test_paired_evaluation_reuses_seeds_and_logs_precommand_state(self):
        class OneStepEnv:
            max_episode_steps = 1

            def __init__(self):
                self.data = SimpleNamespace(qpos=np.array([0.0, 0.0, 1.0]),
                                            qvel=np.zeros(3), time=0.0)

            def reset(self, *, seed):
                self.target_position = np.array([seed * 0.001, 0.0, 1.0])
                self.data.qpos[:] = [0, 0, 1]
                self.data.time = 0
                return np.zeros(7, dtype=np.float32), {}

            def step(self, action):
                self.data.time = 0.04
                self.data.qpos[0] = 0.1
                return np.zeros(7, dtype=np.float32), 0.0, False, True, {
                    "velocity_command_m_s": [0.75, 0, 0],
                    "yaw_rate_command_rad_s": 0.0,
                    "distance_m": abs(self.target_position[0] - 0.1),
                    "speed_m_s": 0.0,
                    "termination_reason": "time_limit",
                    "success_streak": 1,
                }

            def close(self):
                pass

        action = lambda obs: np.array([0.5, 0, 0, 0], dtype=np.float32)
        paired = evaluate_paired({"scripted": action, "ppo_v2": action},
                                 OneStepEnv, [101, 102])
        self.assertEqual([ep["seed"] for ep in paired["scripted"]], [101, 102])
        self.assertEqual([ep["seed"] for ep in paired["ppo_v2"]], [101, 102])
        self.assertEqual(paired["scripted"][0]["target_m"],
                         paired["ppo_v2"][0]["target_m"])
        record = paired["scripted"][0]["steps"][0]
        self.assertEqual(record["time_s"], 0.0)
        self.assertEqual(record["position_m"], [0, 0, 1])
        self.assertEqual(record["velocity_command_m_s"], [0.75, 0, 0])
        self.assertAlmostEqual(record["post_time_s"], 0.04)
        self.assertEqual(record["success_streak"], 1)

    def test_policy_summary_includes_first_entry_and_pooled_near_means(self):
        steps = [row(0, 0.6, vx=-1), row(0.04, 0.4, vx=-2, cmd_x=0.2),
                 row(0.08, 0.15, vx=-3, cmd_x=0.1)]
        episode = {"seed": 101, "target_m": [0, 0, 0], "steps": steps,
                   "episode_length": 3, "episode_reward": 1.0,
                   "final_distance_m": 0.15, "termination_reason": "time_limit"}
        result = summarize_policy([episode])
        self.assertAlmostEqual(result["near_target"]["lt_0_5"]["mean_actual_speed_m_s"], 2.5)
        self.assertEqual(result["entries"]["lt_0_2"]["episode_count"], 1)
        self.assertAlmostEqual(result["entries"]["lt_0_2"]["mean_entry_speed_m_s"], 3.0)
        self.assertEqual(result["distance_bins"]["0_1_to_0_2"]["steps"], 1)

    def test_representative_prefers_median_crossing_entry(self):
        def episode(seed, entry_speed, crossed):
            return {"seed": seed, "analysis": {"first_entry_lt_0_2": {
                "speed_m_s": entry_speed,
                "crossed_target_plane_next_0_5_s": crossed}}}

        candidates = [episode(1, 2.0, True), episode(2, 1.0, True),
                      episode(3, 3.0, True), episode(4, 0.1, False)]
        self.assertEqual(select_representative_seed(candidates), 1)

    def test_persisted_step_log_and_five_paired_figures(self):
        scripted = {"seed": 1, "target_m": [0, 0, 0],
                    "steps": [row(0, 0.4, vx=-1, cmd_x=-0.2),
                              row(0.04, 0.2, vx=-0.2, cmd_x=-0.1)]}
        ppo = {"seed": 1, "target_m": [0, 0, 0],
               "steps": [row(0, 0.4, vx=-1, cmd_x=-1),
                         row(0.04, -0.1, vx=-1, cmd_x=-0.8)]}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            log_path = destination / "steps.csv.gz"
            write_step_log({"scripted": [scripted], "ppo_v2": [ppo]}, log_path)
            with gzip.open(log_path, "rt", newline="") as stream:
                logged = list(csv.DictReader(stream))
            self.assertEqual(len(logged), 4)
            self.assertEqual({record["policy"] for record in logged}, {"scripted", "ppo_v2"})
            self.assertIn("v_cmd_radial_m_s", logged[0])
            figures = save_diagnostic_plots(scripted, ppo, destination)
            self.assertEqual({path.name for path in figures}, {
                "diagnostic_distance_vs_time.png", "diagnostic_speed_vs_time.png",
                "diagnostic_radial_velocity.png", "diagnostic_velocity_command.png",
                "diagnostic_xy_trajectory.png"})
            for path in figures:
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
