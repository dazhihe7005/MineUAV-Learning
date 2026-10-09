"""Shared exact Final cohort and nominal regression contracts."""
import copy,importlib.util,unittest
import numpy as np
import torch
from decision_fidelity_metrics import ranking
from joint_latent_world_model import JointLatentWorldModel
from test_onpolicy_adaptation_training import fixture

class AdaptationEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('onpolicy_adaptation_evaluation'))
        import onpolicy_adaptation_evaluation as api
        return api
    def test_endpoint_metrics_and_common_true_ranking(self):
        api=self.api();truth=np.zeros((512,10,7));pred=truth+2.;cost=np.arange(512,dtype=float)
        m=api.prediction_metrics(pred,truth,np.ones(7),3,0)
        self.assertEqual(m['whole']['10']['normalized_observation_rmse'],2.)
        self.assertEqual(m['selected']['1']['normalized_observation_rmse'],2.)
        self.assertEqual(m['true_best']['5']['physical']['vx']['mae'],2.)
        self.assertEqual(ranking(cost,cost)['normalized_regret'],0.)
    def test_scripted_prediction_all_three_horizons(self):
        api=self.api();torch.set_num_threads(1);torch.manual_seed(0);model=JointLatentWorldModel().eval().requires_grad_(False)
        data,stats=fixture();state=data.states[0];actions=np.repeat(state['branch_actions'][:1],512,0)
        prediction=api.predict_from_prefix(model,stats,state['prefix_observations'],state['prefix_actions'],actions)
        from latent_mpc_core import planning_cost
        previous=state['prefix_actions'][-1];costs,_=planning_cost(prediction[:,-1],actions,previous)
        raw=dict(candidates=actions,prefix_observations=state['prefix_observations'],prefix_actions=state['prefix_actions'],
            truth=prediction.copy(),true_costs=costs,distances=np.ones(512),speeds=np.ones(512),failures=np.zeros(512,bool),
            script_actions=actions[0],script_truth=prediction[0])
        metadata=dict(initial_distance=2.,script=dict(cost=costs[0]))
        result,_=api.evaluate_state(model,stats,raw,metadata)
        self.assertEqual(set(result['scripted_prediction']),{'1','5','10'})
        # Same sequence inferred in batch512 vs batch1: float32 GEMM roundoff.
        self.assertLess(result['scripted_prediction']['10']['normalized_observation_rmse'],1e-6)
    def test_real_prefix_only_no_future_feedback_and_same_actions(self):
        api=self.api();torch.set_num_threads(1);torch.manual_seed(0);model=JointLatentWorldModel().eval().requires_grad_(False)
        data,stats=fixture();state=data.states[0];actions=np.repeat(state['branch_actions'][:1],512,0)
        one=api.predict_from_prefix(model,stats,state['prefix_observations'],state['prefix_actions'],actions)
        altered=copy.deepcopy(state);altered['branch_observations']+=1000.
        two=api.predict_from_prefix(model,stats,altered['prefix_observations'],altered['prefix_actions'],actions)
        np.testing.assert_array_equal(one,two);self.assertEqual(one.shape,(512,10,7))
    def test_final_selection_independent_and_nominal_cohort(self):
        api=self.api();self.assertEqual(api.final_stages(101),dict(S0=0,S2=50,S3=75))
        self.assertEqual(api.final_stages(1),dict(S0=0))
        data,stats=fixture();ep,starts=data.batch([0]);e=ep[0]
        e['metadata']=dict(target_id=1,source='fixture');e['source_rows']=[0,len(e['obs'])]
        torch.set_num_threads(1);torch.manual_seed(0);model=JointLatentWorldModel().eval().requires_grad_(False)
        r=api.nominal_prediction(model,[e],stats,[1,10])
        self.assertEqual(r['1']['window_count'],12);self.assertEqual(r['10']['window_count'],3)
        self.assertEqual(r,api.nominal_prediction(model,[e],stats,[1,10]))

if __name__=='__main__':unittest.main()
