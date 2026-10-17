"""R29 read-only tensors; no environment construction, physics or optimizer."""
import copy
import importlib
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_training_runner import _weights_sha
from lidar_integration.ppo_formal_validation import _optimizer_states_equal
from .policy_function_probe import references


class PolicyFunctionTensorTests(unittest.TestCase):
    def module(self):
        name = 'research_regression.authorized_rl.policy_function_run'
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing no-physics R29 tensor probe')
        return importlib.import_module(name)

    def test_saved_real_context_load_finite_deterministic_batch_no_updates(self):
        m = self.module(); root = Path(__file__).resolve().parents[2]
        prereg = load_l6_prereg(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        trusted = references(root)
        def forbidden(*args, **kwargs): raise AssertionError('R29_NO_TRAINING_OR_PHYSICS')
        with patch.object(PPO, 'learn', forbidden), patch.object(PPO, 'train', forbidden), \
             patch.object(torch.optim.Adam, 'step', forbidden), patch.object(mujoco, 'mj_step', forbidden):
            model = m.load_model(trusted['models'][0], prereg, trusted)
            before = _weights_sha(model.policy); opt = copy.deepcopy(model.policy.optimizer.state_dict())
            original = copy.deepcopy(model._last_obs)
            states = np.array([[.2, 0, 0, 0, 0, 0, 0], [0, 0, 0, .2, 0, 0, 0]], dtype=np.float32)
            obs = m.make_observation(original, states)
            self.assertEqual(obs['lidar'].shape, (2, 3, 12, 24))
            np.testing.assert_array_equal(obs['lidar'][0], original['lidar'][0])
            np.testing.assert_array_equal(obs['heading'][1], original['heading'][0])
            with torch.inference_mode():
                a = m.forward(model, obs); b = m.forward(model, obs)
            self.assertEqual(a, b)
            self.assertTrue(np.isfinite(a['raw_mean']).all())
            self.assertEqual(np.asarray(a['clipped_action']).shape, (2, 4))
            self.assertEqual(_weights_sha(model.policy), before)
            self.assertEqual(model.num_timesteps, 0)
            self.assertTrue(_optimizer_states_equal(opt, model.policy.optimizer.state_dict()))
            for key in original: np.testing.assert_array_equal(original[key], model._last_obs[key])
            self.assertTrue(all(v.grad is None for v in model.policy.parameters()))

    def test_invalid_saved_context_and_counterfactual_state_rejected(self):
        m = self.module()
        # Synthetic unit fixture only; never represented as a real scanned input.
        base = {'lidar': np.zeros((1, 3, 12, 24), dtype=np.float32),
                'heading': np.array([[0., 1.]], dtype=np.float32),
                'state': np.zeros((1, 7), dtype=np.float32), 'lidar_age': np.zeros((1, 1), dtype=np.float32)}
        states = np.zeros((2, 7), dtype=np.float32)
        self.assertEqual(m.make_observation(base, states)['state'].shape, (2, 7))
        for key, index, value in [('heading', (0, 0), 1.), ('lidar', (0, 0, 0, 0), float('nan')),
                                  ('lidar', (0, 2, 0, 0), 1.), ('lidar_age', (0, 0), .2)]:
            bad = copy.deepcopy(base); bad[key][index] = value
            with self.assertRaises(ValueError): m.make_observation(bad, states)
        for bad in (np.zeros((1, 8), dtype=np.float32), np.full((1, 7), np.inf, dtype=np.float32)):
            with self.assertRaises(ValueError): m.make_observation(base, bad)


if __name__ == '__main__':
    unittest.main()
