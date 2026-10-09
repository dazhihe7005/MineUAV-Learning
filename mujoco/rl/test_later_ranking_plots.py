"""Required scientific figures render from aggregate audit metrics."""
import importlib
from pathlib import Path
import tempfile
import unittest
from latent_dynamics_data import file_hash

class PlotTests(unittest.TestCase):
    def test_all_ten_figures_have_verified_nonempty_outputs(self):
        api=importlib.import_module('later_ranking_plots')
        # Handwritten summary fixture. No raw traces or generated ranking inputs.
        scalar={'mean':.3}
        metrics=dict(spearman=scalar,normalized_regret=scalar,random_normalized_regret=scalar,
            model_selected_true_rank_percentile=scalar,script_true_percentile={'cost_percentile':scalar},
            script_pred_percentile={'cost_percentile':scalar},oracle={'distance_reduction':scalar},model_selected={'distance_reduction':scalar})
        prediction={k:scalar for k in ('whole_set_h10_nrmse','selected_h10_nrmse','true_best_h10_nrmse','scripted_h10_nrmse')}
        ood={k:scalar for k in ('observation_standardized_rms','mahalanobis_rms','action_standardized_rms')}
        stage=dict(metrics=metrics,prediction=prediction,ood=ood,best_tail={'spearman':scalar},scripted={'distance_reduction':scalar})
        report=dict(results={c:{'all':dict(stages={s:stage for s in ('S0','S1','S2','S3')},
            paired={'individual':[dict(spearman_delta=-.4,regret_delta=.2)]})} for c in ('unconstrained','train_supported')},states=[])
        with tempfile.TemporaryDirectory() as tmp:
            figures=api.render(report,tmp)
            self.assertEqual(len(figures),10)
            for fig in figures:
                self.assertEqual(file_hash(fig['path']),fig['sha256'])
                self.assertGreater(Path(fig['path']).stat().st_size,1000)

if __name__=='__main__': unittest.main()
