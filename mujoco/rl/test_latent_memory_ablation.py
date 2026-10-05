"""Functional guards for the sole intervention: zero recurrent state every step."""
import tempfile
import unittest
from pathlib import Path

import torch

import latent_dynamics_models as models
from latent_dynamics_data import episodes_from_arrays, fit_statistics, pad_episodes
from test_latent_dynamics_data import fixture_arrays


class NoMemoryTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(hasattr(models, 'NoMemoryLatentDynamics'), 'no-memory GRU is not implemented')
        torch.set_num_threads(1)

    def networks(self):
        torch.manual_seed(0); full = models.make_model('history')
        torch.manual_seed(0); independent = models.make_model('no_memory')
        return full, independent

    def test_parameter_shapes_values_and_one_step_outputs_match_full_history(self):
        full, independent = self.networks()
        self.assertEqual(list(full.state_dict()), list(independent.state_dict()))
        for name, value in full.state_dict().items():
            torch.testing.assert_close(value, independent.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in independent.parameters()), 23815)
        x = torch.randn(2, 1, 11); action = torch.randn(2, 1, 4)
        torch.testing.assert_close(full.predict_sequence(x, action)[0],
                                   independent.predict_sequence(x, action)[0], rtol=0, atol=0)

    def test_past_input_changes_full_history_but_not_no_memory_current_output(self):
        full, independent = self.networks()
        x = torch.randn(2, 6, 11); changed = x.clone(); changed[:, :-1] += 2
        action = torch.randn(2, 6, 4)
        for net, affected in [(full, True), (independent, False)]:
            p = net.predict_sequence(x, action)[0][:, -1]
            q = net.predict_sequence(changed, action)[0][:, -1]
            if affected:
                self.assertGreater((p - q).abs().max().item(), 1e-5)
            else:
                torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_no_cross_timestep_input_gradient(self):
        _, net = self.networks()
        x = torch.randn(1, 5, 11, requires_grad=True)
        net.predict_sequence(x, torch.zeros(1, 5, 4))[0][:, -1].sum().backward()
        torch.testing.assert_close(x.grad[:, :-1], torch.zeros_like(x.grad[:, :-1]), rtol=0, atol=0)
        self.assertGreater(x.grad[:, -1].abs().sum().item(), 0)

    def test_each_lane_matches_an_independent_zero_state_gru_call(self):
        _, net = self.networks()
        x = torch.randn(2, 4, 11); action = torch.randn(2, 4, 4)
        p, z = net.predict_sequence(x, action)
        self.assertEqual(tuple(p.shape), (2, 4, 7))
        for t in range(4):
            state, _ = net.encoder(x[:, t:t + 1], None)
            expected = net.head(torch.cat([state[:, 0], action[:, t]], -1))
            torch.testing.assert_close(p[:, t], expected, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(z[:, t], state[:, 0], rtol=1e-6, atol=1e-7)

    def test_streaming_supplied_memory_is_ignored(self):
        _, net = self.networks()
        x = torch.randn(2, 11); action = torch.randn(2, 4)
        p, _ = net.predict_step(x, action, None)
        q, _ = net.predict_step(x, action, torch.ones(1, 2, 64) * 100)
        torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_reload_and_shared_trainer_preserve_no_memory_behavior(self):
        eps = episodes_from_arrays(fixture_arrays()); stats = fit_statistics(eps)
        b = pad_episodes(eps, stats)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'model.pt'
            result = models.train_model('no_memory', eps, eps, stats,
                                        dict(seed=0, epochs=2, batch_size=16, learning_rate=.001), path)
            net, saved, _ = models.load_model(path)
            self.assertEqual(saved, stats)
            self.assertEqual(result['final_epoch'], 2)
            self.assertIsInstance(net, models.NoMemoryLatentDynamics)
            first = net(b)
            net({k: v * 3 for k, v in b.items()})
            torch.testing.assert_close(first, net(b), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
