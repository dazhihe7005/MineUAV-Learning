"""Categorical reason counts cannot be treated as identical numeric schemas."""
import importlib
import importlib.util
import unittest
import numpy as np
from decision_fidelity_metrics import candidate_metrics

class ReportingTests(unittest.TestCase):
    def test_heterogeneous_candidate_reasons_preserve_all_counts(self):
        self.assertIsNotNone(importlib.util.find_spec('later_ranking_reporting'))
        api=importlib.import_module('later_ranking_reporting')
        records=[]
        p=np.arange(512,dtype=float); t=p.copy()
        terminals=np.zeros((512,7)); script=dict(cost=0,terminal_distance=1,terminal_speed=0,distance_reduction=0,physical_failure=False)
        m=candidate_metrics(p,t,terminals,terminals,np.ones(512),np.zeros(512),np.zeros(512,bool),1,script,0,{'obs':{'std':np.ones(7)}})
        import copy
        for index,reasons in [(0,{'none':512}),(1,{'none':510,'success':2})]:
            metrics=copy.deepcopy(m); metrics['termination_reason_counts']=reasons
            records.append(dict(condition='unconstrained',split='benchmark',index=index,stage='S0',decision_index=0,
                termination_reason='time_limit',initial_distance=1,current_distance=1,distance=1,realized_progress=0,
                metrics=metrics,best_tail={'spearman':.5,'true_cost_spread':1},prediction=dict(whole_set_h10_nrmse=0,selected_h10_nrmse=0),
                ood=dict(observation_standardized_rms=0,action_standardized_rms=0,mahalanobis_rms=0),scripted=script))
        saved=copy.deepcopy(records); result=api.aggregate_for_report(records)
        self.assertEqual(records,saved)
        counts=result['unconstrained']['all']['stages']['S0']['candidate_termination_reason_totals']
        self.assertEqual(counts,{'none':1022,'success':2})

if __name__=='__main__': unittest.main()
