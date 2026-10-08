"""Real environment/model tests, no fake candidate physics."""
import importlib
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import torch
from ppo_pi_env import MineUAVPIEnv
from decision_fidelity_snapshot import capture,restore
from latent_mpc_core import RandomShootingMPC,planning_cost
from joint_autonomous_consistency_training import load_autonomous
from joint_latent_world_model import component_hashes


class RolloutTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('decision_fidelity_rollouts'),'ground-truth rollout missing')
        return importlib.import_module('decision_fidelity_rollouts')

    def test_same_recorded_actions_and_cost_formula_with_exact_restore(self):
        api=self.api(); e=MineUAVPIEnv(reward_version='v2')
        try:
            e.reset(seed=2,options={'target_position':[1,.4,1.2]}); s=capture(e,np.zeros(4,np.float32))
            a=np.random.default_rng(8).uniform(-.2,.2,(10,4)).astype(np.float32)
            result=api.true_rollout(e,s,a)
            restore(e,s); obs=[e.step(x)[0] for x in a]
            np.testing.assert_array_equal(result['observations'][1:],obs)
            np.testing.assert_array_equal(result['executed_actions'],a)
            cost,_=planning_cost(np.asarray(obs[-1:]),a[None],np.zeros(4))
            self.assertEqual(result['cost'],cost[0]); self.assertEqual(result['executed_steps'],10)
            again=api.true_rollout(e,s,a)
            np.testing.assert_array_equal(result['observations'],again['observations'])
            self.assertEqual(result['final_snapshot_sha256'],again['final_snapshot_sha256'])
        finally: e.close()

    def test_scripted_reference_matches_open_loop_replay_not_added_to_candidates(self):
        api=self.api(); e=MineUAVPIEnv(reward_version='v2')
        try:
            e.reset(seed=2,options={'target_position':[1,.4,1.2]}); s=capture(e,np.zeros(4,np.float32))
            scripted=api.scripted_reference(e,s)
            self.assertEqual(scripted['actions'].shape,(10,4))
            replay=api.true_rollout(e,s,scripted['actions'])
            np.testing.assert_array_equal(scripted['observations'],replay['observations'])
            self.assertEqual(scripted['cost'],replay['cost'])
        finally: e.close()

    def test_early_failure_retained_with_no_added_penalty_or_post_done_step(self):
        api=self.api(); e=MineUAVPIEnv(reward_version='v2')
        try:
            e.reset(seed=2,options={'target_position':[1,0,1]}); e.data.qpos[0]=6.1
            s=capture(e,np.zeros(4,np.float32)); a=np.zeros((10,4),np.float32)
            r=api.true_rollout(e,s,a)
            self.assertTrue(r['physical_failure']); self.assertEqual(r['executed_steps'],1)
            self.assertEqual(r['termination_reason'],'outside_flight_area')
            self.assertEqual(r['observations'].shape,(11,7))
            np.testing.assert_array_equal(r['observations'][1:],np.repeat(r['observations'][1:2],10,0))
            self.assertAlmostEqual(r['cost'],5.1**2,places=5)
        finally: e.close()

    def test_frozen_prediction_matches_original_candidate_costs_and_has_no_future_input(self):
        api=self.api(); torch.set_num_threads(1)
        model,stats,_=load_autonomous(Path(__file__).parent/'models/joint_latent_world_model_v3_autonomous_consistency.pt')
        planner=RandomShootingMPC(model,stats,-np.ones(4),np.ones(4),11); before=component_hashes(model)
        z=planner.encode_current(np.array([1,.2,.1,0,0,0,0],np.float32),np.zeros(4))
        p=planner.plan(z,np.zeros(4)); terminal=api.frozen_prediction(model,stats,z,p['candidates'])
        cost,_=planning_cost(terminal,p['candidates'],np.zeros(4))
        np.testing.assert_array_equal(cost,p['costs'])
        np.testing.assert_array_equal(terminal[p['index']],p['terminal_observation'])
        np.testing.assert_array_equal(terminal,api.frozen_prediction(model,stats,z,p['candidates']))
        self.assertEqual(before,component_hashes(model)); self.assertTrue(all(not p.requires_grad for p in model.parameters()))


if __name__=='__main__': unittest.main()
