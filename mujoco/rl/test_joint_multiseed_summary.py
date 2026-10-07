"""All five paired identities, honest failure accounting and seven summary plots."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

import joint_multiseed_replication as replication


def synthetic_branches():
    """Literal toy metrics for aggregation only; NOT physical experiment results."""
    rows = []
    for seed in range(5):
        for condition, coefficient in [('v1', 1.), ('v3', .8)]:
            value = coefficient*(.1 + seed*.02)
            observation = dict(normalized_observation_rmse=value, horizontal_velocity_rmse=value*2,
                               physical={'yaw_error': {'rmse': value/2}})
            periodic = {str(c): {'observation': {'normalized_observation_rmse': value/c}}
                        for c in (1, 5, 10, 25)}
            periodic['never'] = {'observation': observation}
            # Smaller correction intervals should reduce error in this toy case.
            periodic['1']['observation']['normalized_observation_rmse'] = value*.1
            periodic['10']['observation']['normalized_observation_rmse'] = value*.4
            rows.append(dict(condition=condition, seed=seed, status='completed', reused=seed==0,
                             evaluation=dict(horizons={str(h): {'observation': copy.deepcopy(observation)} for h in (1, 5, 10, 25, 50)},
                                 current_reconstruction={'normalized_observation_rmse': value/10},
                                 periodic_correction=periodic, potential_latent_collapse=False)))
    return rows


class SummaryTests(unittest.TestCase):
    def test_all_five_seed_identity_pairing_and_short_horizon_win_counts(self):
        self.assertTrue(hasattr(replication, 'summarize_branches'), 'paired summary missing')
        rows = synthetic_branches()[::-1]
        summary = replication.summarize_branches(rows)
        self.assertEqual(summary['horizons']['50']['v3_win_count'], 5)
        np.testing.assert_allclose(summary['horizons']['50']['deltas'], [.02, .024, .028, .032, .036])
        self.assertAlmostEqual(summary['horizons']['50']['v1']['mean'], .14)
        self.assertAlmostEqual(summary['horizons']['50']['v3']['mean'], .112)
        self.assertEqual(summary['horizons']['1']['v3_win_count'], 5)
        self.assertEqual(summary['anomalies'], [])
        # Dropping the worst seed or duplicating an identity must be rejected.
        with self.assertRaises(ValueError):
            replication.summarize_branches(rows[:-1])
        with self.assertRaises(ValueError):
            replication.summarize_branches(rows + [rows[0]])

    def test_failed_and_collapsed_seeds_remain_in_robustness_result(self):
        self.assertTrue(hasattr(replication, 'summarize_branches'), 'paired summary missing')
        rows = synthetic_branches()
        rows[-1] = dict(condition='v3', seed=4, status='failed', failure={'type': 'FloatingPointError'})
        rows[1]['evaluation']['potential_latent_collapse'] = True
        summary = replication.summarize_branches(rows)
        self.assertEqual(summary['horizons']['50']['missing_pair_count'], 1)
        self.assertEqual(summary['horizons']['50']['requested_pair_count'], 5)
        self.assertEqual(summary['horizons']['50']['delta']['count'], 4)
        self.assertEqual(len(summary['anomalies']), 2)
        self.assertIsNone(summary['horizons']['50']['deltas'][4])

    def test_seven_required_plots_show_actual_seed_values(self):
        try:
            import joint_multiseed_plots as plots
        except ImportError:
            plots = None
        self.assertIsNotNone(plots, 'paired plots missing')
        rows = synthetic_branches()
        summary = replication.summarize_branches(rows)
        with tempfile.TemporaryDirectory() as td:
            paths = plots.render_figures(rows, summary, Path(td))
            self.assertEqual(len(paths), 7)
            self.assertEqual(len(set(paths)), 7)
            self.assertTrue(all(Path(p).stat().st_size>1000 for p in paths))


if __name__ == '__main__':
    unittest.main()
