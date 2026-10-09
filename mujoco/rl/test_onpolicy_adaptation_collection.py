"""Exact real state, original proposals and sealed Final evaluation."""
import importlib.util,unittest
import numpy as np
from ppo_pi_env import MineUAVPIEnv
from decision_fidelity_snapshot import capture,fingerprint
from decision_fidelity_rollouts import true_rollout
from onpolicy_adaptation_data import make_targets
from onpolicy_adaptation_collection import collect_visited
from run_onpolicy_adaptation_data import collection_context
from pathlib import Path

class AdaptationCollectionTests(unittest.TestCase):
    def test_exact_branch_restore_full_hidden_state(self):
        env=MineUAVPIEnv(reward_version='v2')
        try:
            env.reset(seed=710090000,options={'target_position':[1.,1.,1.]})
            previous=np.array([.2,-.1,.1,.2],np.float32);env.step(previous);snapshot=capture(env,previous)
            a=np.repeat(previous[None],10,0);one=true_rollout(env,snapshot,a);two=true_rollout(env,snapshot,a)
            np.testing.assert_array_equal(one['observations'],two['observations'])
            self.assertEqual(one['final_snapshot_sha256'],two['final_snapshot_sha256'])
            self.assertEqual(fingerprint(snapshot),fingerprint(snapshot))
        finally:env.close()
    def test_final_collection_gate(self):
        self.assertIsNotNone(importlib.util.find_spec('onpolicy_adaptation_final_data'))
        from onpolicy_adaptation_final_data import require_completed_training
        with self.assertRaisesRegex(ValueError,'completed'):require_completed_training({}, {})
        rows={s:dict(final_step=1000,train_windows=6400,validation_windows=1600,order_sha256='same',initial_hashes=dict(e='same')) for s in ('replay','mpc_state')}
        require_completed_training(rows,dict(replay=True,mpc_state=True))
        bad=dict(rows);bad['mpc_state']=dict(rows['mpc_state'],order_sha256='different')
        with self.assertRaisesRegex(ValueError,'paired'):require_completed_training(bad,dict(replay=True,mpc_state=True))

if __name__=='__main__':unittest.main()
