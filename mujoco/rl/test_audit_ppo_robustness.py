"""Frozen-checkpoint robustness evaluator must preserve holdout metrics."""

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402
from ppo_exploration_audit import evaluate_exploration  # noqa: E402
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


def audit_module():
    try:
        return importlib.import_module("audit_ppo_robustness")
    except ImportError as error:
        raise AssertionError("Robustness evaluation implementation is missing") from error


class RobustnessAuditTests(unittest.TestCase):
    def test_loads_exactly_five_saved_lowstd_models_without_training(self):
        """Catches evaluating only the best seed or a non-100k checkpoint."""
        audit = audit_module()
        policies = audit.load_frozen_policies()
        self.assertEqual([seed for seed, _ in policies],
                         [20260929, 0, 8, 16, 24])
        self.assertTrue(all(model.num_timesteps == 100352 for _, model in policies))
        self.assertTrue(all(model.policy.log_std_init == -2.0 for _, model in policies))

    def test_nominal_one_waypoint_matches_existing_diagnostics(self):
        """Catches a new success/near-speed/crossing metric convention."""
        audit = audit_module()
        from robustness_dynamics import RobustnessEnv, Scenario
        seed, model = audit.load_frozen_policies()[0]
        self.assertEqual(seed, 20260929)
        waypoint = build_holdout_waypoints()[:1]
        nominal_env = MineUAVEnv(reward_version="v2")
        audit_env = RobustnessEnv(Scenario("nominal", 0.0))
        try:
            previous = evaluate_exploration(model, nominal_env, waypoint, True)
            result = audit.evaluate_policy(model, audit_env, waypoint)
            self.assertEqual(result["metrics"]["success_rate"],
                             previous["task_metrics"]["success_rate"])
            self.assertEqual(result["metrics"]["mean_final_distance_m"],
                             previous["task_metrics"]["mean_final_distance_m"])
            self.assertEqual(result["metrics"]["mean_episode_length"],
                             previous["task_metrics"]["mean_episode_length"])
            self.assertEqual(result["metrics"]["crossing_rate"],
                             previous["crossing_after_first_0_2_m"]
                             ["before_episode_end_fraction_all_episodes"])
            self.assertEqual(result["metrics"]["near_target_speed_m_s"],
                             previous["braking_diagnostic"]["near_target"]["lt_0_1"]
                             ["mean_actual_speed_m_s"])
            self.assertEqual(result["metrics"]["action_saturation"],
                             previous["task_metrics"]
                             ["action_near_boundary_fraction_overall"])
            self.assertEqual(result["episodes"][0]["target_position_m"], waypoint[0][1])
        finally:
            nominal_env.close()
            audit_env.close()

    def test_timeout_has_null_completion_time_and_one_timeout(self):
        """Catches converting no-success completion time to zero."""
        audit = audit_module()
        model = audit.load_frozen_policies()[0][1]
        env = MineUAVEnv(reward_version="v2", max_episode_seconds=0.04)
        try:
            result = audit.evaluate_policy(model, env, build_holdout_waypoints()[:1])
            self.assertEqual(result["metrics"]["successes"], 0)
            self.assertIsNone(result["metrics"]["mean_successful_completion_time_s"])
            self.assertEqual(result["metrics"]["time_limit_count"], 1)
            self.assertEqual(result["metrics"]["failure_count"], 0)
        finally:
            env.close()

    def test_large_force_counts_terminal_failure_separately_from_timeout(self):
        """Catches non-success terminations hidden inside timeout count."""
        audit = audit_module()
        from robustness_dynamics import RobustnessEnv, Scenario
        model = audit.load_frozen_policies()[0][1]
        env = RobustnessEnv(Scenario("force_x_n", 1000.0))
        try:
            result = audit.evaluate_policy(model, env, build_holdout_waypoints()[:1])
            self.assertEqual(result["metrics"]["successes"], 0)
            self.assertEqual(result["metrics"]["failure_count"], 1)
            self.assertEqual(result["metrics"]["time_limit_count"], 0)
            self.assertNotEqual(result["episodes"][0]["termination_reason"], "time_limit")
        finally:
            env.close()

    def test_scenario_result_path_preserves_existing_data(self):
        """Catches accidental overwrite of completed audit conditions."""
        audit = audit_module()
        from robustness_dynamics import Scenario
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            path = audit.scenario_output_path(Scenario("mass", 0.8), root)
            audit.write_new_result(path, {"condition": "first"})
            with self.assertRaises(FileExistsError):
                audit.write_new_result(path, {"condition": "second"})
            self.assertEqual(json.loads(path.read_text()), {"condition": "first"})


if __name__ == "__main__":
    unittest.main()
