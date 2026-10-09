"""Cohort composition, paired differences and full frozen-path contracts."""
import importlib
import importlib.util
import unittest
import numpy as np

class AnalysisTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('later_ranking_analysis'))
        return importlib.import_module('later_ranking_analysis')

    def test_paired_selection_excludes_missing_stage_not_failed_episodes(self):
        records=[]
        for episode,stages in [(0,['S0','S1','S2','S3']),(1,['S0','S1'])]:
            for j,s in enumerate(stages):
                records.append(dict(index=episode,split='benchmark',stage=s,termination_reason='time_limit',
                    metrics=dict(spearman=.8-.1*j,kendall=None,normalized_regret=.1*j,
                        model_selected_true_rank_percentile=.2*j)))
        out=self.api().paired_changes(records)
        self.assertEqual(out['episodes'],1)
        self.assertAlmostEqual(out['S3_minus_S0']['spearman']['mean'],-.3)
        self.assertEqual(out['S3_minus_S0']['kendall']['missing'],1)

    def test_condition_full_cohort_validation_rejects_dropped_episode(self):
        api=self.api(); records=[]
        episodes=[]
        for c in ('unconstrained','train_supported'):
            for split in ('benchmark','holdout'):
                for i in range(100):
                    episodes.append(dict(condition=c,split=split,index=i,episode_steps=1))
                    records.append(dict(condition=c,split=split,index=i,stage='S0',decision_index=0))
        self.assertEqual(api.validate_cohort(records,episodes),400)
        with self.assertRaises(AssertionError): api.validate_cohort(records[:-1],episodes)

    def test_first_stage_uses_original_double_precision_task_distance(self):
        import run_later_decision_ranking as runner
        from types import SimpleNamespace
        self.assertTrue(hasattr(runner,'task_distance'))
        snap=SimpleNamespace(environment={'previous_distance':1.327907828759674})
        self.assertEqual(runner.task_distance(snap),1.327907828759674)

    def test_pure_prediction_terminal_matches_original_and_no_future_argument(self):
        from pathlib import Path
        from run_world_model_decision_fidelity import audit_context
        from decision_fidelity_rollouts import frozen_prediction
        from later_ranking_core import prediction_trajectory
        ctx=audit_context(Path(__file__).resolve().parents[2]); z=np.zeros((1,64),np.float32)
        a=np.random.default_rng(42).uniform(-.5,.5,(512,10,4)).astype(np.float32)
        out=prediction_trajectory(ctx['model'],ctx['stats'],z,a)
        np.testing.assert_array_equal(out[:,-1],frozen_prediction(ctx['model'],ctx['stats'],z,a))
        np.testing.assert_array_equal(out,prediction_trajectory(ctx['model'],ctx['stats'],z,a))
        with self.assertRaises(TypeError): prediction_trajectory(ctx['model'],ctx['stats'],z,a,future_observations=np.ones((10,7)))

if __name__=='__main__': unittest.main()
