import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import episodes_from_arrays, fit_statistics, pad_episodes
from test_latent_dynamics_data import fixture_arrays

try:
    import latent_dynamics_models as models
except ImportError:
    models = None


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(models, 'supervised models are not implemented')
        torch.manual_seed(0)
        torch.set_num_threads(1)
        self.eps = episodes_from_arrays(fixture_arrays())
        self.stats = fit_statistics(self.eps)
        self.batch = pad_episodes(self.eps, self.stats)

    def test_shapes_and_actual_architectures(self):
        for kind in ('markov', 'history', 'oracle'):
            net = models.make_model(kind)
            self.assertEqual(tuple(net(self.batch).shape), (2, 2, 7))
            self.assertTrue(torch.isfinite(net(self.batch)).all())
        net = models.make_model('history')
        self.assertEqual(net.encoder.input_size, 11)
        self.assertEqual(net.encoder.hidden_size, 64)
        self.assertEqual(net.encoder.num_layers, 1)

    def test_masked_loss_ignores_padding_without_changing_denominator(self):
        p = torch.zeros(1, 2, 7); y = torch.ones_like(p)
        y[:, 1] = 10000
        self.assertEqual(models.masked_mse(p, y, torch.tensor([[True, False]])).item(), 1.)

    def test_history_outputs_and_loss_are_independent_of_pi_labels(self):
        net = models.make_model('history')
        b = {k: v.clone() for k, v in self.batch.items()}
        b['pi'].fill_(1e6)
        torch.testing.assert_close(net(b), net(self.batch), rtol=0, atol=0)
        self.assertEqual(models.masked_mse(net(b), b['delta'], b['mask']).item(),
                         models.masked_mse(net(self.batch), self.batch['delta'], self.batch['mask']).item())

    def test_episode_A_B_A_has_no_memory_leakage(self):
        net = models.make_model('history').eval()
        x = torch.randn(1, 4, 11); a = torch.randn(1, 4, 4)
        first, z = net.predict_sequence(x, a)
        net.predict_sequence(torch.ones_like(x) * 20, torch.ones_like(a))
        again, _ = net.predict_sequence(x, a)  # implicit reset=None for new episode
        torch.testing.assert_close(first, again, rtol=0, atol=0)
        state = None; rows = []
        for t in range(4):
            p, state = net.predict_step(x[:, t], a[:, t], state)
            rows.append(p)
        torch.testing.assert_close(torch.stack(rows, 1), first, atol=1e-6, rtol=1e-6)
        self.assertEqual(tuple(z.shape), (1, 4, 64))

    def test_current_action_does_not_enter_encoder_latent(self):
        net = models.make_model('history')
        x = torch.randn(1, 3, 11)
        _, z1 = net.predict_sequence(x, torch.zeros(1, 3, 4))
        _, z2 = net.predict_sequence(x, torch.ones(1, 3, 4))
        torch.testing.assert_close(z1, z2, rtol=0, atol=0)

    def test_model_reload_is_exact_for_all_models(self):
        with tempfile.TemporaryDirectory() as td:
            for kind in ('markov', 'history', 'oracle'):
                net = models.make_model(kind).eval()
                path = Path(td) / f'{kind}.pt'
                models.save_model(path, net, kind, self.stats, dict(epoch=0))
                loaded, stat, _ = models.load_model(path)
                self.assertEqual(stat, self.stats)
                torch.testing.assert_close(net(self.batch), loaded(self.batch), rtol=0, atol=0)

    def test_training_selects_validation_best_and_runs_full_budget(self):
        with tempfile.TemporaryDirectory() as td:
            result = models.train_model('markov', self.eps, self.eps, self.stats,
                                        dict(seed=0, epochs=3, batch_size=2, learning_rate=.001),
                                        Path(td) / 'model.pt')
            self.assertEqual(len(result['history']), 3)
            self.assertEqual(result['final_epoch'], 3)
            losses = [row['validation_loss'] for row in result['history']]
            self.assertEqual(result['best_epoch'], int(np.argmin(losses)) + 1)
            self.assertEqual(result['best_validation_loss'], min(losses))


if __name__ == '__main__':
    unittest.main()
