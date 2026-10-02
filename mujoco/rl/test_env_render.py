"""Rendering must share the live simulation and leave headless stepping untouched."""

import sys
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco.viewer
import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
from mine_uav_env import MineUAVEnv  # noqa: E402


class FakeViewer:
    """Only the OS window is replaced; model, data and dynamics stay real."""

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.running = True
        self.synced_positions = []
        self.close_count = 0
        self.cam = SimpleNamespace(lookat=np.zeros(3), distance=10.0,
                                   azimuth=0.0, elevation=0.0)

    def lock(self):
        return nullcontext()

    def is_running(self):
        return self.running

    def sync(self):
        self.synced_positions.append(self.data.qpos[:3].copy())

    def close(self):
        self.running = False
        self.close_count += 1


class RenderTests(unittest.TestCase):
    def test_headless_never_launches_viewer_or_sleeps(self):
        with (patch("mujoco.viewer.launch_passive", side_effect=AssertionError("viewer launched")),
              patch("time.sleep", side_effect=AssertionError("headless slept"))):
            env = MineUAVEnv(render_mode=None)
            try:
                env.reset(seed=1)
                for _ in range(3):
                    env.step(np.zeros(4, dtype=np.float32))
                self.assertAlmostEqual(env.data.time, 0.12)
            finally:
                env.close()

    def test_human_lazily_uses_same_model_data_and_reuses_window(self):
        opened = []
        slept = []
        clock = [100.0]

        def launch(model, data):
            viewer = FakeViewer(model, data)
            opened.append(viewer)
            return viewer

        def sleep(seconds):
            slept.append(seconds)
            clock[0] += seconds

        with (patch("mujoco.viewer.launch_passive", side_effect=launch),
              patch("time.sleep", side_effect=sleep),
              patch("time.monotonic", side_effect=lambda: clock[0])):
            env = MineUAVEnv(render_mode="human")
            try:
                self.assertEqual(opened, [])
                env.reset(seed=1, options={"target_position": [2, 2, 1.5]})
                self.assertEqual(len(opened), 1)
                viewer = opened[0]
                self.assertIs(viewer.model, env.model)
                self.assertIs(viewer.data, env.data)
                np.testing.assert_allclose(viewer.cam.lookat, [1.0, 1.0, 1.25])
                self.assertLess(viewer.cam.distance, 5.0)
                self.assertTrue(env.viewer_is_running)
                env.step(np.zeros(4, dtype=np.float32))
                np.testing.assert_allclose(viewer.synced_positions[-1], env.data.qpos[:3])
                self.assertEqual(len(viewer.synced_positions), 2)
                self.assertAlmostEqual(sum(slept), 0.04, places=8)
                env.reset(seed=2)
                self.assertEqual(len(opened), 1)
            finally:
                env.close()
                env.close()
            self.assertEqual(viewer.close_count, 1)

    def test_manual_viewer_close_does_not_break_simulation(self):
        viewer_box = []

        def launch(model, data):
            viewer = FakeViewer(model, data)
            viewer_box.append(viewer)
            return viewer

        with (patch("mujoco.viewer.launch_passive", side_effect=launch),
              patch("time.sleep") as sleep):
            env = MineUAVEnv(render_mode="human")
            try:
                env.reset(seed=1)
                viewer_box[0].running = False
                before = env.data.time
                env.step(np.zeros(4, dtype=np.float32))
                self.assertAlmostEqual(env.data.time - before, 0.04)
                self.assertFalse(env.viewer_is_running)
                sleep.assert_not_called()
            finally:
                env.close()
            sleep.assert_called_once_with(0.5)

    def test_invalid_render_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            MineUAVEnv(render_mode="rgb_array")


if __name__ == "__main__":
    unittest.main()
