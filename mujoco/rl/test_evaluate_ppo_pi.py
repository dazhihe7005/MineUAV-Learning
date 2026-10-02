"""PI policy evaluation must retain paired task metrics and hidden-state logs."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_ppo_pi import (compact_pi_evaluation, evaluate_pi_policy,
                             nominal_gate)  # noqa: E402
from ppo_pi_env import MineUAVPIEnv  # noqa: E402
from train_ppo_lowstd_multiseed import build_holdout_waypoints  # noqa: E402


MODEL = (Path(__file__).resolve().parent / "models" /
         "ppo_waypoint_v2_lowstd_multiseed" / "seed_0" /
         "ppo_waypoint_v2_lowstd_100k.zip")


class PPOPIEvaluationTests(unittest.TestCase):
    def test_nominal_gate_requires_both_complete_fixed_100_sets(self):
        def point(successes, episodes=100):
            return {"task_metrics": {"successes": successes, "episodes": episodes}}
        self.assertTrue(nominal_gate(point(90), point(91)))
        self.assertFalse(nominal_gate(point(89), point(100)))
        self.assertFalse(nominal_gate(point(100), point(89)))
        with self.assertRaises(ValueError):
            nominal_gate(point(100, 99), point(100))

    def test_real_pi_episode_records_integral_state_without_changing_observation(self):
        model = PPO.load(MODEL, device="cpu")
        env = MineUAVPIEnv(render_mode=None, reward_version="v2")
        try:
            waypoint = build_holdout_waypoints()[:1]
            with tempfile.TemporaryDirectory() as scratch:
                path = Path(scratch) / "integral_trace.json"
                evaluation = evaluate_pi_policy(model, env, waypoint, path)
                saved = json.loads(path.read_text())
                self.assertEqual(evaluation["seeds"], [waypoint[0][0]])
                self.assertEqual(evaluation["targets"], [waypoint[0][1]])
                self.assertEqual(len(saved["episodes"]), 1)
                episode = saved["episodes"][0]
                self.assertEqual(episode["seed"], waypoint[0][0])
                self.assertEqual(episode["target_m"], waypoint[0][1])
                self.assertEqual(len(episode["steps"]), evaluation["braking_diagnostic"]
                                 ["episode_summaries"][0]["episode_length"])
                self.assertEqual(episode["steps"][-1]["termination_reason"],
                                 evaluation["braking_diagnostic"]
                                 ["episode_summaries"][0]["termination_reason"])
                self.assertEqual(env.observation_space.shape, (7,))
                np.testing.assert_array_equal(episode["initial_integral_error_m"],
                                              [0, 0, 0])
                compact = compact_pi_evaluation(evaluation)
                self.assertEqual(compact["episodes"], 1)
                self.assertEqual(compact["integral"]["episode_count"], 1)
                self.assertLessEqual(compact["integral"]["max_xy_accel_m_s2"], 1.5)
                self.assertLessEqual(compact["integral"]["max_z_accel_m_s2"], 1.5)
                with self.assertRaises(FileExistsError):
                    evaluate_pi_policy(model, env, waypoint, path)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
