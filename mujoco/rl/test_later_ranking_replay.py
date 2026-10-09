"""Real canonical replay must fail on archived action corruption."""
import importlib
import importlib.util
from pathlib import Path
import unittest
import numpy as np
from ppo_pi_env import MineUAVPIEnv

class ReplayTests(unittest.TestCase):
    def test_original_episode_candidates_and_snapshot_repeat(self):
        self.assertIsNotNone(importlib.util.find_spec('later_ranking_replay'))
        api=importlib.import_module('later_ranking_replay')
        from run_world_model_decision_fidelity import audit_context
        from decision_fidelity_rollouts import true_rollout
        from joint_latent_world_model import component_hashes
        ctx=audit_context(Path(__file__).resolve().parents[2]); env=MineUAVPIEnv(reward_version='v2')
        try:
            replay=api.replay_episode(ctx,env,'unconstrained','benchmark',0)
            self.assertTrue(replay['verification']['all_actions_costs_noise_exact'])
            self.assertEqual(set(replay['decisions']),{'S0','S1','S2','S3'})
            d=replay['decisions']['S2']; a=d['plan']['candidates'][d['plan']['index']]
            one=true_rollout(env,d['snapshot'],a); two=true_rollout(env,d['snapshot'],a)
            np.testing.assert_array_equal(one['observations'],two['observations'])
            self.assertEqual(one['final_snapshot_sha256'],two['final_snapshot_sha256'])
            self.assertEqual(component_hashes(ctx['model']),ctx['before'])
            self.assertEqual(d['snapshot'].environment['episode_steps'],d['decision_index'])
        finally: env.close()

    def test_supported_episode_replay_and_corrupted_archive_rejection(self):
        api=importlib.import_module('later_ranking_replay')
        from run_world_model_decision_fidelity import audit_context
        ctx=audit_context(Path(__file__).resolve().parents[2]); env=MineUAVPIEnv(reward_version='v2')
        try:
            result=api.replay_episode(ctx,env,'train_supported','holdout',0)
            self.assertTrue(result['verification']['all_actions_costs_noise_exact'])
            arrays=ctx['_original_arrays'][('train_supported','holdout')]
            arrays['actions']=arrays['actions'].copy(); arrays['actions'][0,0]+=.01
            with self.assertRaises(AssertionError): api.replay_episode(ctx,env,'train_supported','holdout',0)
        finally: env.close()

if __name__=='__main__': unittest.main()
