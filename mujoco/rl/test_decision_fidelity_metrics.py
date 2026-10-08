"""Hand-checked rankings distinguish both Top5 directions and true regret."""
import importlib
import importlib.util
import unittest
import numpy as np


class RankingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('decision_fidelity_metrics'),'ranking metrics missing')
        return importlib.import_module('decision_fidelity_metrics')

    def test_reverse_ranking_regret_and_analytic_random_selection(self):
        r=self.api().ranking(np.arange(5,-1,-1),np.arange(6))
        self.assertAlmostEqual(r['spearman'],-1); self.assertAlmostEqual(r['kendall'],-1)
        self.assertFalse(r['top1_exact']); self.assertFalse(r['pred_best_in_true_top5']); self.assertFalse(r['true_best_in_pred_top5'])
        self.assertEqual(r['regret'],5); self.assertAlmostEqual(r['normalized_regret'],1)
        self.assertEqual(r['random_expected_regret'],2.5); self.assertAlmostEqual(r['random_normalized_regret'],.5)
        self.assertEqual(r['model_selected_true_rank_percentile'],1)

    def test_oracle_zero_and_directed_top5_are_not_conflated(self):
        api=self.api(); t=np.arange(7,dtype=float)
        self.assertEqual(api.ranking(t,t)['regret'],0)
        r=api.ranking(np.array([9,0,1,2,3,4,5]),t)
        self.assertTrue(r['pred_best_in_true_top5']); self.assertFalse(r['true_best_in_pred_top5'])
        self.assertEqual(r['true_best_index'],0); self.assertEqual(r['pred_best_index'],1)

    def test_ties_constant_costs_and_percentile_direction(self):
        api=self.api(); r=api.ranking(np.ones(6),np.array([0,0,1,2,3,4]))
        self.assertIsNone(r['spearman']); self.assertIsNone(r['kendall'])
        self.assertEqual(r['true_best_margin'],0); self.assertAlmostEqual(r['model_selected_true_rank_percentile'],.1)
        p=api.reference_percentile(0,np.array([0,0,1,2,3,4]))
        self.assertAlmostEqual(p['cost_percentile'],1/6)
        self.assertAlmostEqual(p['strictly_better_than_fraction'],2/3)
        z=api.ranking(np.arange(6),np.ones(6))
        self.assertEqual(z['regret'],0); self.assertEqual(z['normalized_regret'],0)

    def test_nonfinite_cost_is_not_silently_dropped(self):
        with self.assertRaises(ValueError): self.api().ranking(np.array([0,np.nan]),np.ones(2))

    def test_tau_b_tie_denominator_and_average_spearman_ranks(self):
        api=self.api()
        self.assertAlmostEqual(api.correlation([1,1,2],[1,2,3],'kendall'),2/np.sqrt(6))
        self.assertAlmostEqual(api.correlation([1,1,2],[1,2,3]),np.sqrt(3)/2)
        np.testing.assert_array_equal(api.rankdata([3,1,1,2]),[4,1.5,1.5,3])


if __name__=='__main__': unittest.main()
