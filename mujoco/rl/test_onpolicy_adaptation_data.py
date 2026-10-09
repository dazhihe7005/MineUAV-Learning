"""Budget, target isolation and branch-history contracts before collection."""
import importlib
import importlib.util
import unittest
import numpy as np

class AdaptationDataTests(unittest.TestCase):
    def test_collection_and_runner_import(self):
        self.assertTrue(callable(importlib.import_module('onpolicy_adaptation_collection').collect_branches))
        self.assertTrue(callable(importlib.import_module('run_onpolicy_adaptation_data').collection_context))

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('onpolicy_adaptation_data'))
        return importlib.import_module('onpolicy_adaptation_data')

    def test_fresh_targets_disjoint_and_deterministic(self):
        api=self.api(); old=[[-1.,0.,1.],[0.,0.,1.]]
        one=api.make_targets(old,{1,2,3}); two=api.make_targets(old,{1,2,3})
        self.assertEqual(one,two)
        self.assertEqual({k:len(v) for k,v in one.items()},dict(train=60,val=20,final=100))
        api.validate_splits(one,old,{1,2,3})
        bad={k:[dict(r) for r in v] for k,v in one.items()}
        bad['final'][0]['target']=bad['train'][0]['target']
        with self.assertRaises(ValueError): api.validate_splits(bad,old,{1,2,3})

    def test_balanced_quotas_and_eight_original_candidate_indices(self):
        api=self.api(); q=api.quotas(800,60)
        self.assertEqual(q[:20],[14]*20); self.assertEqual(q[20:],[13]*40)
        self.assertEqual(sum(q),800)
        ids=api.branch_indices(3,12)
        np.testing.assert_array_equal(ids,api.branch_indices(3,12))
        self.assertEqual(ids[:2].tolist(),[0,1]); self.assertEqual(len(set(ids)),8)
        self.assertTrue(np.all((ids>=0)&(ids<512)))

    def test_no_early_failure_padding_is_a_training_window(self):
        api=self.api(); good=dict(executed_steps=10,invalid_state=False,observations=np.zeros((11,7)))
        self.assertTrue(api.valid_branch(good))
        self.assertFalse(api.valid_branch(dict(good,executed_steps=9)))
        self.assertFalse(api.valid_branch(dict(good,invalid_state=True)))
        self.assertFalse(api.valid_branch(dict(good,observations=np.full((11,7),np.nan))))

    def test_exact_prefix_previous_action_and_one_start(self):
        api=self.api(); po=np.arange(21,dtype=np.float32).reshape(3,7)
        pa=np.array([[.1,.2,.3,.4],[.5,.6,.7,.8]],np.float32)
        a=np.arange(40,dtype=np.float32).reshape(10,4)/40
        future=np.vstack([po[-1],np.arange(70,dtype=np.float32).reshape(10,7)])
        ep,start=api.branch_episode(dict(prefix_observations=po,prefix_actions=pa,
            branch_actions=a[None],branch_observations=future[None]),0)
        self.assertEqual(start,2); self.assertEqual(len(ep['action']),12)
        np.testing.assert_array_equal(ep['previous_action'][0],np.zeros(4))
        np.testing.assert_array_equal(ep['previous_action'][1:],ep['action'][:-1])
        np.testing.assert_array_equal(ep['next_obs'][start:start+10],future[1:])
        np.testing.assert_array_equal(ep['obs'][start],po[-1])
        bad=dict(prefix_observations=po,prefix_actions=pa,branch_actions=a[None],branch_observations=future[None].copy())
        bad['branch_observations'][0,0,0]+=1
        with self.assertRaises(ValueError): api.branch_episode(bad,0)

if __name__=='__main__': unittest.main()
