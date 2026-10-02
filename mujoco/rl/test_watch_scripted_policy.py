"""The desktop episode must use the validated policy and fixed waypoint."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402


class WatchScriptTests(unittest.TestCase):
    def test_fixed_waypoint_episode_reaches_target_through_gym_step(self):
        from watch_scripted_policy import run_episode

        env = MineUAVEnv(render_mode=None)
        try:
            result = run_episode(env, status_sink=lambda _line: None)
            np.testing.assert_allclose(result["start_position_m"], [0, 0, 1])
            np.testing.assert_allclose(result["target_position_m"], [2, 2, 1.5])
            self.assertEqual(result["reason"], "success")
            self.assertLess(np.linalg.norm(
                np.asarray(result["final_position_m"]) - [2, 2, 1.5]), 0.10)
            self.assertGreater(result["peak_speed_m_s"], 0.2)
            self.assertGreater(result["peak_tilt_deg"], 1.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
