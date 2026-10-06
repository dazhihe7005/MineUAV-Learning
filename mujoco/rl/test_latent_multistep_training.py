"""Catches future-truth injection, detached BPTT, action misalignment and leakage."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from latent_dynamics_models import make_model, load_model
from latent_dynamics_multistep import rollout_windows
from test_latent_dynamics_multistep import episode, statistics

try:
    import latent_multistep_training as training
except ImportError:
    training = None


class EchoEncoder(nn.Module):
    def forward(self, x, state=None):
        return x[..., :7], x[:, -1, :7].unsqueeze(0)


class EchoHead(nn.Module):
    def forward(self, x):
        return x[..., :7]


class EchoDynamics(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = EchoEncoder(); self.head = EchoHead()

    def predict_step(self, x, action, state=None):
        z, state = self.encoder(x[:, None], state)
        return self.head(torch.cat([z[:, 0], action], -1)), state


class MemoryOnlyDynamics(EchoDynamics):
    """Ignores predicted obs, so state gradient cannot use the residual shortcut."""
    def predict_step(self, x, action, state=None):
        state = state * 2
        return state[0], state


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(training, 'multi-step differentiable trainer is missing')
        torch.set_num_threads(1); torch.manual_seed(0)

    def test_ten_step_recursion_uses_predictions_not_truth(self):
        out = training.predict_windows(EchoDynamics(), [episode(12)], statistics(), 10,
                                       starts_by_episode=[[0]])
        np.testing.assert_array_equal(out['predictions'].detach()[0, :, 0],
                                      [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
        self.assertEqual(tuple(out['predictions'].shape), (1, 10, 7))

    def test_future_truth_and_pi_are_never_model_inputs(self):
        net = make_model('history'); ep = episode(12); changed = copy.deepcopy(ep)
        changed['obs'][4:] += 100; changed['next_obs'] += 200; changed['pi'] += 1000
        a = training.predict_windows(net, [ep], statistics(), 5, [[3]])
        b = training.predict_windows(net, [changed], statistics(), 5, [[3]])
        torch.testing.assert_close(a['predictions'], b['predictions'], rtol=0, atol=0)

    def test_late_loss_reaches_early_prediction_and_recurrent_state(self):
        out = training.predict_windows(make_model('history'), [episode(12)], statistics(), 10, [[1]])
        grad_pred, grad_state = torch.autograd.grad(out['steps'][-1].square().sum(),
            (out['steps'][0], out['states'][0]), retain_graph=True)
        self.assertGreater(grad_pred.abs().sum().item(), 0)
        self.assertGreater(grad_state.abs().sum().item(), 0)
        self.assertTrue(all(step.requires_grad for step in out['steps']))

    def test_hidden_recurrence_gradient_has_no_detach_even_without_observation_path(self):
        ep = episode(12)
        # The prefix has a gradient through its actual input, but recurrent steps
        # ignore observations. A hidden detach alone would break this derivative.
        net = MemoryOnlyDynamics()
        net.encoder = nn.GRU(11, 7, batch_first=True)
        out = training.predict_windows(net, [ep], statistics(), 10, [[0]])
        grad = torch.autograd.grad(out['states'][-1].sum(), out['states'][0])[0]
        torch.testing.assert_close(grad, torch.ones_like(grad) * 512)

    def test_warmup_and_recorded_action_index_match_frozen_evaluator(self):
        net = make_model('history'); ep = episode(12); stats = statistics()
        stats['obs']['mean'] = [.5] * 7; stats['obs']['std'] = [2.] * 7
        stats['action']['std'] = [.4] * 4
        stats['delta']['mean'] = [.2] * 7; stats['delta']['std'] = [.3] * 7
        before = copy.deepcopy(stats)
        train = training.predict_windows(net, [ep], stats, 10, [[2]])
        frozen = rollout_windows(net, 'history', ep, stats, [2], 10)
        np.testing.assert_allclose(train['predictions'].detach(), frozen['predictions'], atol=3e-6)
        self.assertEqual(stats, before)

    def test_windows_are_episode_local_and_a_b_a_reset(self):
        net = make_model('history'); ep = episode(12); other = episode(11, 2)
        other['obs'] *= 10
        a = training.predict_windows(net, [ep], statistics(), 10)
        training.predict_windows(net, [other], statistics(), 10)
        b = training.predict_windows(net, [ep], statistics(), 10)
        torch.testing.assert_close(a['predictions'], b['predictions'], rtol=0, atol=0)
        combined = training.predict_windows(net, [ep, other], statistics(), 10)
        self.assertEqual(len(combined['predictions']), 5)  # 3 + 2 legal starts
        torch.testing.assert_close(combined['predictions'][:3], a['predictions'])
        with self.assertRaisesRegex(ValueError, 'window'):
            training.predict_windows(net, [ep], statistics(), 10, [[3]])

    def test_loss_is_uniform_observation_scaled_not_delta_scaled(self):
        stats = statistics(); stats['obs']['std'] = [2.] * 7
        stats['delta']['std'] = [100.] * 7
        self.assertEqual(training.observation_loss(torch.ones(2, 10, 7) * 4,
                         torch.zeros(2, 10, 7), stats).item(), 4.)

    def test_best_checkpoint_is_validation_selected_reload_and_reproducible(self):
        config = dict(seed=0, epochs=3, batch_size=2, learning_rate=.001, horizon=2)
        train = [episode(4), episode(3, 2)]; val = [episode(3, 3)]
        with tempfile.TemporaryDirectory() as td:
            a_path = Path(td)/'a.pt'; b_path = Path(td)/'b.pt'
            a = training.train_multistep(train, val, statistics(), config, a_path)
            b = training.train_multistep(train, val, statistics(), config, b_path)
            net, stats, metadata = load_model(a_path)
            second, _, _ = load_model(b_path)
            self.assertEqual(a['best_epoch'], np.argmin([r['validation_loss'] for r in a['history']])+1)
            self.assertEqual(metadata['best_epoch'], a['best_epoch'])
            self.assertAlmostEqual(training.validation_loss(net, val, stats, 2, 2), a['best_validation_loss'])
            for p, q in zip(net.parameters(), second.parameters()):
                torch.testing.assert_close(p, q, rtol=0, atol=0)
            self.assertEqual(a['initial_parameter_sha256'], b['initial_parameter_sha256'])


if __name__ == '__main__':
    unittest.main()
