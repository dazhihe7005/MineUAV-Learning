import importlib.util,unittest

class AdaptationAnalysisTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('onpolicy_adaptation_analysis'))
        import onpolicy_adaptation_analysis as api
        return api
    def test_matched_state_aggregation_not_cross_state_comparison(self):
        api=self.api();rows=[]
        for i in range(2):
            for model,value in [('original',.5),('replay',.4),('mpc_state',.2)]:
                rows.append(dict(target_id=str(i),stage='S3',model=model,metrics=dict(spearman=value,kendall=value,
                    normalized_regret=1-value,model_selected_true_rank_percentile=1-value,top1_exact=False,
                    pred_best_in_true_top5=False,true_best_in_pred_top5=False,best_tail=dict(spearman=value),
                    prediction={key:{str(h):dict(normalized_observation_rmse=1-value) for h in (1,5,10)} for key in ('whole','selected','true_best')})))
        r=api.aggregate(rows);self.assertAlmostEqual(r['S3']['paired_mpc_minus_replay']['spearman']['mean'],-.2)
        self.assertEqual(r['S3']['models']['mpc_state']['state_count'],2)
        with self.assertRaises(ValueError):api.aggregate(rows[:-1])
    def test_expected_artifact_names(self):
        self.api()
        self.assertIsNotNone(importlib.util.find_spec('onpolicy_adaptation_plots'))
        from onpolicy_adaptation_plots import FIGURES
        self.assertEqual(len(FIGURES),8);self.assertIn('nominal_regression_comparison.png',FIGURES)

if __name__=='__main__':unittest.main()
