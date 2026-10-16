"""R22 actual SB3 model/buffer/save-load contracts; optimizer forbidden."""
import importlib
import importlib.util
from pathlib import Path
import tempfile
import os
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from research_ops.storage import OpsError
from . import temporal_credit_tests as previous_tests


class GAECreditTensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*_a, **_k):
            raise AssertionError('NO_OPTIMIZER_R22_TEST')
        for cls, name in ((PPO, 'learn'), (PPO, 'train'), (torch.optim.Adam, 'step')):
            p = patch.object(cls, name, forbidden); p.start(); self.addCleanup(p.stop)

    def module(self):
        name = 'research_regression.authorized_rl.gae_credit_training'
        self.assertIsNotNone(importlib.util.find_spec(name), 'missing R22 GAE adapter')
        return importlib.import_module(name)

    def setup_model(self):
        return previous_tests.TemporalCreditTensorTests.setup_model(self)

    def test_lambda_model_buffer_change_without_weights_predictions_or_optimizer_change(self):
        adapter = self.module(); m, s, e = self.setup_model()
        from lidar_integration.ppo_training_runner import _weights_sha
        obs, _ = e.reset(); before = _weights_sha(m.policy)
        action = m.predict(obs, deterministic=True)[0].copy()
        value = m.policy.predict_values(m.policy.obs_to_tensor(obs)[0]).detach().clone()
        objects = list(map(id, m.policy.parameters()))
        adapter.configure_gae_credit(m)
        self.assertEqual((m.gamma, m.rollout_buffer.gamma, m.gae_lambda, m.rollout_buffer.gae_lambda), (.99, .99, .99, .99))
        self.assertEqual(_weights_sha(m.policy), before)
        self.assertEqual(list(map(id, m.policy.parameters())), objects)
        self.assertFalse(m.policy.optimizer.state)
        np.testing.assert_array_equal(m.predict(obs, deterministic=True)[0], action)
        torch.testing.assert_close(m.policy.predict_values(m.policy.obs_to_tensor(obs)[0]), value, rtol=0, atol=0)
        with self.assertRaises(OpsError): adapter.configure_gae_credit(m)
        m.rollout_buffer.gae_lambda = .95
        with self.assertRaises(OpsError): adapter.verify_gae_configuration(m)

    def test_real_sb3_gae_recursion_changes_lambda_but_not_td_discount(self):
        adapter = self.module(); m, s, e = self.setup_model(); adapter.configure_gae_credit(m)
        b = m.rollout_buffer; obs, _ = e.reset()
        for i in range(128):
            b.add({k: np.expand_dims(v, 0) for k, v in obs.items()}, np.zeros((1, 4), np.float32),
                  np.array([1. if i == 127 else 0.]), np.array([i == 0]), torch.zeros(1), torch.zeros(1))
        b.compute_returns_and_advantage(last_values=torch.zeros(1), dones=np.ones(1))
        self.assertTrue(np.isfinite(b.advantages).all())
        self.assertAlmostEqual(float(b.advantages[-2, 0]), .9801, places=6)
        self.assertNotAlmostEqual(float(b.advantages[-2, 0]), .9405, places=4)
        self.assertAlmostEqual(float(b.advantages[0, 0]), (.99*.99)**127, places=6)
        # Independent one-step TD check with a nonzero final value: .99*2-1=.98.
        b.reset()
        for i in range(128):
            b.add({k: np.expand_dims(v, 0) for k, v in obs.items()}, np.zeros((1, 4), np.float32),
                  np.zeros(1), np.array([i == 0]), torch.ones(1), torch.zeros(1))
        b.compute_returns_and_advantage(last_values=torch.full((1,), 2.), dones=np.zeros(1))
        self.assertAlmostEqual(float(b.advantages[-1, 0]), .98, places=6)
        self.assertEqual(m.num_timesteps, 0); self.assertFalse(m.policy.optimizer.state)

    def test_zero_update_checkpoint_preserves_new_metadata_and_same_initial_policy(self):
        adapter = self.module(); m, s, e = self.setup_model(); adapter.configure_gae_credit(m)
        from .centered_noise_training import initialize_noise_training
        from lidar_integration.ppo_formal_checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as d:
            manager = CheckpointManager(Path(d), self.prereg, execution_protocol_sha256='b'*64)
            init, event, probe = initialize_noise_training(m, s, manager, identity='R22_NO_UPDATE_ONLY', seed=0)
            self.assertEqual(init['assigned_policy_sha256'], '650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
            restored = manager.load_verified(None, run_id='R22_NO_UPDATE_ONLY_STEP0')
            adapter.verify_gae_configuration(restored)
            self.assertEqual(restored.num_timesteps, 0); self.assertFalse(restored.policy.optimizer.state)
            np.testing.assert_array_equal(restored.predict(probe[0], deterministic=True)[0], probe[1])
            _, _, _, _, info = e.step(np.zeros(4, dtype=np.float32))
            self.assertEqual((e.policy_steps, e.physics_ticks, info['distance_stage']), (1, 20, 'SHORT'))

    def test_actual_worker_dispatch_is_correct_before_any_sampling_or_optimizer(self):
        # Stop at the learn METHOD BOUNDARY; its implementation is never called.
        # This catches missing schema branches that adapter-only tests cannot.
        from .runtime import ROOT
        from . import worker
        from .budget import BudgetStore, grant
        from .gae_credit_profile import gae_experiment
        from research_ops.storage import digest, read_json, process_identity
        from lidar_integration.ppo_training_runner import _weights_sha
        observed = []
        class SamplingForbidden(Exception): pass
        with tempfile.TemporaryDirectory() as d:
            home = Path(d)/'grant'; store = BudgetStore(home)
            store.initialize(expected_grant_sha256=digest(grant()))
            p = gae_experiment(); p.update(registration_sha256='a'*64, source_identity_sha256='b'*64)
            store.reserve(p)
            output = Path(d)/'output'; output.mkdir()
            permit = {'experiment_id': p['experiment_id'], 'parent_pid': os.getppid(),
                      'parent_identity': process_identity(os.getppid()), 'protocol_sha256': digest(p)}
            def boundary(model, steps, **kwargs):
                self.assertEqual((model.gamma, model.gae_lambda, model.rollout_buffer.gae_lambda), (.99, .99, .99))
                self.module().verify_gae_configuration(model)
                self.assertEqual(_weights_sha(model.policy), '650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
                self.assertEqual((steps, model.num_timesteps, model._n_updates), (32768, 0, 0))
                self.assertFalse(model.policy.optimizer.state)
                self.assertEqual(read_json(output/'initialization.json')['execution_profile'], p['execution_profile'])
                observed.append('STOP_BEFORE_SAMPLING')
                raise SamplingForbidden()
            with patch.object(worker, 'HOME', home), patch.object(PPO, 'learn', boundary):
                with self.assertRaises(SamplingForbidden):
                    worker.train_worker(output, permit, p, {}, p['experiment_id'])
            self.assertEqual(observed, ['STOP_BEFORE_SAMPLING'])
            self.assertFalse((output/'ppo_updates.jsonl').exists())
            self.assertFalse((output/'summary.json').exists())
