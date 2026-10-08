"""Reject altered derived metrics even when all source identities still match."""
import importlib
import importlib.util
import unittest
import copy
import tempfile
from pathlib import Path
import numpy as np
from decision_fidelity_metrics import candidate_metrics
from latent_dynamics_data import file_hash


class VerificationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('decision_fidelity_verification'),'independent verifier missing')
        return importlib.import_module('decision_fidelity_verification')

    def test_every_derived_metric_is_recomputed_and_tampering_rejected(self):
        api=self.api(); n=512; pred=np.arange(n,dtype=float); true=pred[::-1].copy()
        terminal=np.zeros((n,7)); terminal[:,0]=np.linspace(0,1,n); p=terminal+.01
        distance=terminal[:,0]; speed=np.zeros(n); failure=np.zeros(n,bool)
        script=dict(cost=.1); stats=dict(obs=dict(std=np.ones(7)))
        metrics=candidate_metrics(pred,true,terminal,p,distance,speed,failure,1.,script,.2,stats)
        api.verify_derived(metrics,pred,true,terminal,p,distance,speed,failure,1.,script,.2,stats)
        for path in [('normalized_regret',),('oracle','terminal_distance'),('script_true_percentile','cost_percentile'),('prediction_error','selected_h10_nrmse')]:
            bad=copy.deepcopy(metrics); parent=bad
            for key in path[:-1]: parent=parent[key]
            parent[path[-1]]+=.1
            with self.assertRaises(AssertionError): api.verify_derived(bad,pred,true,terminal,p,distance,speed,failure,1.,script,.2,stats)

    def test_schema_does_not_allow_candidate_reduction(self):
        api=self.api()
        with self.assertRaises(AssertionError): api.require_shapes(dict(terminal=np.zeros((511,7))))

    def test_postprocessing_source_and_figure_hashes_are_verified(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'figure.png'; path.write_bytes(b'original')
            report=dict(postprocessing_sources={str(path):file_hash(path)},figures=[dict(path=str(path),sha256=file_hash(path))])
            api.verify_postprocessing(report)
            path.write_bytes(b'changed')
            with self.assertRaises(AssertionError): api.verify_postprocessing(report)

    def test_tampered_interpretation_statistics_are_rejected(self):
        with self.assertRaises(AssertionError): self.api().verify_postprocessing(dict(postprocessing_sources={},figures=[],analysis={},states=[]))

    def test_undefined_paired_correlations_survive_final_postprocessing(self):
        from finalize_decision_fidelity import supplement
        api=self.api()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'costs.npz'
            np.savez(path,unconstrained_pred_cost=np.zeros(512),unconstrained_true_cost=np.ones(512),
                train_supported_pred_cost=np.zeros(512),train_supported_true_cost=np.ones(512))
            condition=dict(spearman=.5,kendall=.4,regret=.2,normalized_regret=.1,model_selected_true_rank_percentile=.2,
                true_cost_spread=1.,oracle=dict(true_cost=.8),model_selected=dict(true_cost=1.))
            for key in ('spearman','kendall'):
                for a,b in ((None,.5),(.5,None),(None,None)):
                    with self.subTest(key=key,pair=(a,b)):
                        u=copy.deepcopy(condition); s=copy.deepcopy(condition); u[key]=a; s[key]=b
                        report=dict(states=[dict(conditions=dict(unconstrained=u,train_supported=s),
                            local_arrays=dict(path=str(path)),scripted=dict(cost=.9))])
                        result=supplement(report); paired=result['paired_support'][key]
                        self.assertEqual(paired['supported_minus_unconstrained']['count'],0)
                        self.assertEqual(paired['supported_minus_unconstrained']['missing'],1)
                        self.assertIsNone(paired['supported_minus_unconstrained']['mean'])
                        self.assertEqual(paired['positive_count'],0); self.assertEqual(paired['negative_count'],0)
                        api.verify_postprocessing(dict(report,analysis=result))


if __name__=='__main__': unittest.main()
