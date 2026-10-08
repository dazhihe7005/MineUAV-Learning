"""Source/identity/cohort rejection and undefined physical-terminal disclosure."""
import importlib
import importlib.util
import unittest
import numpy as np
from ppo_pi_env import MineUAVPIEnv
from decision_fidelity_snapshot import capture
from decision_fidelity_rollouts import true_rollout


class IntegrityTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_world_model_decision_fidelity'),'audit runner missing')
        return importlib.import_module('run_world_model_decision_fidelity')

    def test_cache_rejects_executable_change_and_wrong_initial_state_identity(self):
        api=self.api(); identity=dict(checkpoint='fixed',sources={'model_forward':'old'})
        saved=dict(identity=identity,row=dict(split='benchmark',index=0))
        api.validate_cache(saved,identity,'benchmark',0)
        with self.assertRaises(ValueError): api.validate_cache(saved,dict(identity,sources={'model_forward':'new'}),'benchmark',0)
        with self.assertRaises(ValueError): api.validate_cache(saved,identity,'holdout',0)
        with self.assertRaises(ValueError): api.validate_cache(saved,identity,'benchmark',1)

    def test_complete_200_state_and_both_512_cohorts_required(self):
        api=self.api(); rows=[dict(split=s,index=i,conditions={c:dict(candidate_count=512) for c in ('unconstrained','train_supported')})
                              for s in ('benchmark','holdout') for i in range(100)]
        self.assertEqual(api.verify_cohort(rows),204800)
        with self.assertRaises(AssertionError): api.verify_cohort(rows[:-1])
        rows[0]['conditions']['train_supported']['candidate_count']=511
        with self.assertRaises(AssertionError): api.verify_cohort(rows)

    def test_nonfinite_physical_failure_is_kept_and_sanitization_explicit(self):
        self.api(); e=MineUAVPIEnv(reward_version='v2')
        try:
            e.reset(seed=2,options={'target_position':[1,0,1]}); e.data.qvel[0]=np.nan
            result=true_rollout(e,capture(e,np.zeros(4,np.float32)),np.zeros((10,4),np.float32))
            self.assertTrue(result['invalid_state']); self.assertTrue(result['physical_failure'])
            self.assertEqual(result['cost'],0,'no new penalty; original env sanitizes invalid obs')
            self.assertEqual(result['executed_steps'],1)
        finally: e.close()


if __name__=='__main__': unittest.main()
