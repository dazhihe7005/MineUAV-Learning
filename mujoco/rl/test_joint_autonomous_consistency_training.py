"""v3: literal objective, detached targets and actual composed-state BPTT."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_joint_latent_world_model import fixture
from joint_latent_world_model import initialize_joint, component_hashes, predict_window, encode_episode
from joint_latent_training import predict_windows, observation_loss
from joint_consistency_training import consistency_mse, initial_scale

try:
    import joint_autonomous_consistency_training as trainer
except ImportError:
    trainer = None


class AutonomousConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(trainer, 'v3 trainer missing')
        torch.set_num_threads(1)

    def test_literal_loss_fixed_lambda_and_target_detach(self):
        pred = torch.ones(2, 10, 64, requires_grad=True)
        target = torch.zeros_like(pred, requires_grad=True)
        loss = trainer.autonomous_mse(pred, target, [2.] * 64)
        self.assertAlmostEqual(loss.item(), .25)
        loss.backward()
        self.assertGreater(pred.grad.norm().item(), 0.)
        self.assertIsNone(target.grad)
        self.assertAlmostEqual(trainer.total_loss(torch.tensor(2.), torch.tensor(3.)).item(), 2.3, places=6)
        with self.assertRaises(ValueError):
            trainer.autonomous_mse(pred[:, :9], target[:, :9], [2.] * 64)

    def test_actual_autonomous_states_not_local_sources_and_full_bptt(self):
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            model, _, _ = initialize_joint(*paths)
            starts = [np.array([2])]
            with patch('joint_consistency_training.local_loss', side_effect=AssertionError('no local term')):
                row = trainer.loss_batch(model, [ep], stats, [.2] * 64, 10, starts)
            base = predict_windows(model, [ep], stats, 10, starts)
            self.assertTrue(torch.equal(row['latents'], base['latents']))
            self.assertTrue(torch.equal(row['normalized_observations'], base['normalized_observations']))
            self.assertEqual(row['window_count'], 1)
            self.assertAlmostEqual(row['observation_loss'].item(), observation_loss(base['normalized_observations'], base['truth'], stats).item(), places=12)
            ref = encode_episode(model, np.vstack([ep['obs'], ep['next_obs'][-1]]),
                                 np.vstack([ep['previous_action'], ep['action'][-1]]), stats)
            torch.testing.assert_close(row['reference_targets'][0], ref[3:13], rtol=0, atol=0)
            self.assertFalse(row['reference_targets'].requires_grad)
            self.assertAlmostEqual(row['consistency_loss'].item(), consistency_mse(base['latents'][:, 1:], ref[None, 3:13], [.2] * 64).item(), places=12)
            # Last-step-only loss must traverse earlier composed states (not detach).
            row['steps'][1].retain_grad()
            last = consistency_mse(row['steps'][-1], row['reference_targets'][:, -1], [.2] * 64)
            last.backward()
            self.assertGreater(row['steps'][1].grad.norm().item(), 0.)
            for module in (model.encoder, model.transition):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None), 0.)
            self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
            with self.assertRaisesRegex(ValueError, 'boundary'):
                trainer.loss_batch(model, [ep], stats, [.2] * 64, 10, [np.array([3])])

    def test_future_reference_changes_loss_not_prediction_or_inference(self):
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            model, _, _ = initialize_joint(*paths)
            starts = [np.array([2])]
            first = trainer.loss_batch(model, [ep], stats, [.2] * 64, 10, starts)
            changed = copy.deepcopy(ep)
            changed['obs'][3:] += 100.
            changed['next_obs'] += 200.
            second = trainer.loss_batch(model, [changed], stats, [.2] * 64, 10, starts)
            self.assertTrue(torch.equal(first['latents'], second['latents']))
            self.assertTrue(torch.equal(first['normalized_observations'], second['normalized_observations']))
            self.assertNotEqual(first['consistency_loss'].item(), second['consistency_loss'].item())
            reference = predict_window(model, ep, 2, 10, stats)
            changed['obs'] = changed['obs'][:3]
            changed['previous_action'] = changed['previous_action'][:3]
            del changed['next_obs']
            with patch.object(trainer, 'reference_targets', side_effect=AssertionError('reference forbidden in inference')):
                inferred = predict_window(model, changed, 2, 10, stats)
            self.assertTrue(torch.equal(reference['latents'], inferred['latents']))

    def test_reference_branch_has_no_grad_and_source_total_all_modules_have_grad(self):
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            model, _, _ = initialize_joint(*paths)
            calls = []
            handle = model.encoder.register_forward_hook(lambda m, args, out: calls.append(out[0]))
            row = trainer.loss_batch(model, [ep], stats, [.2] * 64, 10)
            handle.remove()
            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0].requires_grad)
            self.assertFalse(calls[1].requires_grad)
            row['loss'].backward()
            for module in (model.encoder, model.transition, model.decoder):
                self.assertTrue(all(p.grad is not None for p in module.parameters()))
                self.assertGreater(sum(p.grad.square().sum().item() for p in module.parameters()), 0.)

    def test_scale_episode_reset_selection_reload_and_deterministic_training(self):
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            config = dict(seed=0, epochs=3, batch_size=2, learning_rate=.0003, horizon=10)
            runs = []
            for i in range(2):
                model, _, _ = initialize_joint(*paths)
                before = component_hashes(model)
                scale = initial_scale(model, [ep], stats)
                self.assertEqual(before, component_hashes(model))
                target1 = trainer.reference_targets(model, [ep], stats, 10)
                other = copy.deepcopy(ep); other['obs'] += 50.
                trainer.reference_targets(model, [other], stats, 10)
                self.assertTrue(torch.equal(target1, trainer.reference_targets(model, [ep], stats, 10)))
                path = Path(td) / f'v3_{i}.pt'
                result = trainer.train_autonomous(model, [ep], [ep], stats, scale['std'], config, path)
                loaded, ss, meta = trainer.load_autonomous(path)
                self.assertEqual(stats, ss)
                self.assertEqual(result['parameter_count'], 74119)
                self.assertEqual(meta['lambda_autonomous_consistency'], .1)
                self.assertFalse(meta['local_consistency_enabled'])
                self.assertEqual(meta['best_epoch'], int(np.argmin([r['validation']['observation'] for r in result['history']])) + 1)
                self.assertEqual(meta['initial_latent_std'], scale['std'])
                check = trainer.validation_losses(loaded, [ep], stats, scale['std'], 10, 2)
                self.assertAlmostEqual(check['observation'], result['best_validation']['observation'], places=12)
                self.assertAlmostEqual(check['total'], check['observation'] + .1 * check['consistency'], places=12)
                self.assertEqual(result['history'][0]['objective_gradient_diagnostics']['decoder']['weighted_consistency_norm'], 0.)
                self.assertTrue(all(before[k] != component_hashes(loaded)[k] for k in before))
                runs.append((component_hashes(loaded), result['history']))
            self.assertEqual(runs[0], runs[1])
            self.assertEqual(trainer.selection_epoch([dict(observation=1., total=2.), dict(observation=2., total=.1)]), 1)


if __name__ == '__main__':
    unittest.main()
