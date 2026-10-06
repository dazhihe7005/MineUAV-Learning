"""Breaks caught: truth injection, prefix double-counting, boundary/mask leaks."""
import copy
import unittest

import numpy as np
import torch
from torch import nn

from latent_dynamics_models import make_model

try:
    import latent_dynamics_multistep as multi
except ImportError:
    multi = None


def statistics():
    return {key: dict(mean=[0.] * dim, std=[1.] * dim)
            for key, dim in [('obs', 7), ('action', 4), ('delta', 7), ('pi', 3)]}


def episode(length=6, target_id=1):
    observations = np.zeros((length + 1, 7), np.float32)
    observations[:, 0] = np.arange(length + 1) + 1
    actions = np.zeros((length, 4), np.float32)
    actions[:, 0] = np.arange(length) / 10
    previous = np.vstack([np.zeros((1, 4)), actions[:-1]]).astype(np.float32)
    return dict(obs=observations[:-1], next_obs=observations[1:],
                delta=np.diff(observations, axis=0), action=actions,
                previous_action=previous, pi=np.zeros((length, 3), np.float32),
                metadata=dict(target_id=target_id, source='fixture'), source_rows=[0, length])


class ObservationDelta(nn.Module):
    """Analytical delta = current normalized observation; no mutable state."""
    def forward(self, batch):
        return batch['obs']


class PiDelta(nn.Module):
    def forward(self, batch):
        result = torch.zeros_like(batch['obs'])
        result[:, 0] = batch['pi'][:, 0]
        return result


class MultistepTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(multi, 'frozen multi-step evaluator is not implemented')
        torch.set_num_threads(1)
        torch.manual_seed(0)

    def test_autoregression_doubles_prediction_not_recorded_truth(self):
        out = multi.rollout_windows(ObservationDelta(), 'markov', episode(), statistics(), [0], 3)
        np.testing.assert_array_equal(out['predictions'][0, :, 0], [2., 4., 8.])
        self.assertEqual(out['predictions'].shape, (1, 3, 7))

    def test_invalid_window_cannot_cross_episode_or_use_missing_action(self):
        with self.assertRaisesRegex(ValueError, 'window'):
            multi.rollout_windows(ObservationDelta(), 'markov', episode(3), statistics(), [2], 2)
        with self.assertRaises(ValueError):
            multi.rollout_windows(ObservationDelta(), 'markov', episode(3), statistics(), [-1], 1)

    def test_manifest_enumerates_all_legal_starts_without_cross_episode(self):
        manifest = multi.start_manifest([episode(3), episode(2, 2)], [1, 2, 4], 'testhash')
        self.assertEqual(manifest['window_counts'], {'1': 5, '2': 3, '4': 0})
        self.assertEqual(manifest['episodes'][0]['valid_start_count'], {'1': 3, '2': 2, '4': 0})
        self.assertEqual(manifest, multi.start_manifest([episode(3), episode(2, 2)], [1, 2, 4], 'testhash'))
        changed = multi.start_manifest([episode(3), episode(2, 3)], [1, 2, 4], 'testhash')
        self.assertNotEqual(manifest['manifest_sha256'], changed['manifest_sha256'])

    def test_normalize_input_and_inverse_delta_before_recursion(self):
        stats = statistics()
        stats['obs']['mean'] = [1.] * 7; stats['obs']['std'] = [2.] * 7
        stats['delta']['mean'] = [3.] * 7; stats['delta']['std'] = [4.] * 7
        ep = episode(); ep['obs'][:] = 5
        out = multi.rollout_windows(ObservationDelta(), 'markov', ep, stats, [0], 2)
        # 5 + ((5-1)/2)*4 + 3 =16; 16 + ((16-1)/2)*4 +3 =49.
        np.testing.assert_array_equal(out['predictions'][0, :, 0], [16., 49.])
        self.assertEqual(stats['obs']['mean'], [1.] * 7)

    def test_full_history_first_step_matches_real_prefix_exactly_once(self):
        net = make_model('history').eval(); ep = episode()
        x = torch.tensor(np.concatenate([ep['obs'][:4], ep['previous_action'][:4]], -1))[None]
        with torch.no_grad():
            latent, _ = net.encoder(x, None)
            expected = net.head(torch.cat([latent[:, -1], torch.tensor(ep['action'][3:4])], -1))
        out = multi.rollout_windows(net, 'history', ep, statistics(), [3], 1)
        np.testing.assert_allclose(out['predictions'][0, 0], ep['obs'][3] + expected.numpy()[0], atol=1e-7)
        np.testing.assert_allclose(out['latent_autoregressive'][0, 0], latent.numpy()[0, -1], atol=1e-7)

    def test_history_future_truth_is_not_injected_into_predicted_state_or_latent(self):
        net = make_model('history').eval(); ep = episode(); changed = copy.deepcopy(ep)
        changed['obs'][2:] += 100  # include the very first future observation
        first = multi.rollout_windows(net, 'history', ep, statistics(), [1], 4)
        second = multi.rollout_windows(net, 'history', changed, statistics(), [1], 4)
        np.testing.assert_array_equal(first['predictions'], second['predictions'])
        np.testing.assert_array_equal(first['latent_autoregressive'], second['latent_autoregressive'])

    def test_no_memory_each_step_matches_manual_zero_hidden_call(self):
        net = make_model('no_memory').eval(); ep = episode(); predicted = ep['obs'][2].copy()
        expected = []
        with torch.no_grad():
            for index in range(2, 5):
                x = torch.tensor(np.r_[predicted, ep['previous_action'][index]])[None, None]
                z, _ = net.encoder(x.float(), None)
                delta = net.head(torch.cat([z[:, 0], torch.tensor(ep['action'][index:index+1])], -1))
                predicted = predicted + delta.numpy()[0]; expected.append(predicted.copy())
        out = multi.rollout_windows(net, 'no_memory', ep, statistics(), [2], 3)
        np.testing.assert_allclose(out['predictions'][0], expected, atol=3e-7)

    def test_pi_baseline_teacher_forces_pi_not_observation(self):
        ep = episode(); ep['pi'][:, 0] = [1, 2, 3, 4, 5, 6]
        out = multi.rollout_windows(PiDelta(), 'pi_state', ep, statistics(), [0], 3)
        np.testing.assert_array_equal(out['predictions'][0, :, 0], [2., 4., 7.])

    def test_a_b_a_reset_and_batch_partition_are_deterministic(self):
        net = make_model('history').eval(); ep = episode(); other = episode()
        other['obs'] *= 10
        a = multi.rollout_windows(net, 'history', ep, statistics(), [0, 2], 3)
        multi.rollout_windows(net, 'history', other, statistics(), [1], 3)
        b = multi.rollout_windows(net, 'history', ep, statistics(), [0, 2], 3)
        np.testing.assert_array_equal(a['predictions'], b['predictions'])
        c = multi.rollout_windows(net, 'history', ep, statistics(), [2], 3)
        np.testing.assert_allclose(a['predictions'][1], c['predictions'][0], atol=1e-6)

    def test_endpoint_metrics_use_observation_not_delta_scale(self):
        accumulator = multi.ErrorAccumulator(np.array([2.] * 7))
        accumulator.add(np.ones((2, 7)) * 4)
        row = accumulator.result()
        self.assertEqual(row['normalized_observation_rmse'], 2.)
        self.assertEqual(row['physical']['vx']['rmse'], 4.)
        self.assertEqual(row['physical']['vz']['mae'], 4.)
        self.assertEqual(row['horizontal_velocity_rmse'], 4.)
        self.assertEqual(row['window_count'], 2)

    def test_nonfinite_errors_are_reported_not_clipped_or_silently_accepted(self):
        accumulator = multi.ErrorAccumulator(np.ones(7))
        error = np.ones((3, 7)); error[1, 2] = np.nan; error[2, 3] = np.inf
        accumulator.add(error)
        row = accumulator.result()
        self.assertEqual(row['nonfinite_windows'], 2)
        self.assertEqual(row['window_count'], 1)
        self.assertEqual(row['normalized_observation_rmse'], 1)


if __name__ == '__main__':
    unittest.main()
