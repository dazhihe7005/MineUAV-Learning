"""Opt-in real-X11 regression for passive viewer lifecycle.

Run with DISPLAY=:0 MINE_UAV_GUI_TEST=1; excluded from normal headless CI.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402


@unittest.skipUnless(os.environ.get("MINE_UAV_GUI_TEST") == "1", "requires desktop X11")
class RealViewerTests(unittest.TestCase):
    def test_passive_viewer_opens_advances_and_closes_without_glx_error(self):
        env = MineUAVEnv(render_mode="human")
        try:
            env.reset(seed=0, options={"target_position": [2, 2, 1.5]})
            self.assertTrue(env.viewer_is_running)
            before = env.data.time
            env.step(np.zeros(4, dtype=np.float32))
            self.assertAlmostEqual(env.data.time - before, 0.04)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
