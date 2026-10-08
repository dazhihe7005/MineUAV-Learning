"""Behavioral contracts for frozen random-shooting action selection."""
import unittest
from pathlib import Path
import numpy as np
import torch

from joint_autonomous_consistency_training import load_autonomous
from joint_latent_world_model import component_hashes, encode_episode, predict_latent
from latent_mpc_core import RandomShootingMPC, sample_candidates, planning_cost, select_lowest


class PlannerCoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.model, cls.stats, _ = load_autonomous(Path(__file__).parent/'models/joint_latent_world_model_v3_autonomous_consistency.pt')

    def planner(self, seed=0):
        return RandomShootingMPC(self.model, self.stats, -np.ones(4), np.ones(4), seed)

    def test_sampler_preserves_zero_current_and_bounds_for_all_512_sequences(self):
        a = np.array([.9, -.8, .4, .2], np.float32)
        x = sample_candidates(a, np.ones(4)*.4, -np.ones(4), np.ones(4), np.random.default_rng(0))
        self.assertEqual(x.shape, (512, 10, 4))
        np.testing.assert_array_equal(x[0], np.zeros((10,4)))
        np.testing.assert_array_equal(x[1], np.tile(a,(10,1)))
        self.assertTrue(np.all(np.abs(x)<=1))
        self.assertGreater(np.std(x[2:]), .1)

    def test_sampler_half_train_std_random_walk_and_fixed_seed(self):
        rng = np.random.default_rng(7)
        noise = rng.normal(size=(510,10,4)) * (.5*np.array([.2,.3,.4,.5]))
        expected = np.empty_like(noise); previous = np.zeros((510,4))
        for k in range(10):
            previous=np.clip(previous+noise[:,k],-1,1); expected[:,k]=previous
        result=sample_candidates(np.zeros(4),np.array([.2,.3,.4,.5]),-np.ones(4),np.ones(4),np.random.default_rng(7))
        np.testing.assert_allclose(result[2:],expected,rtol=1e-6,atol=1e-7)

    def test_physical_terminal_cost_and_first_previous_action_smoothing(self):
        terminal = np.array([[1,2,2, 1,2,2, 2]],np.float32)
        seq=np.zeros((1,10,4),np.float32); seq[:,0,0]=1
        # 9 position +4.5 velocity +0.4 yaw +.05*(1+1)/10.
        cost, parts=planning_cost(terminal,seq,np.zeros(4))
        self.assertAlmostEqual(cost[0],13.91,places=5)
        self.assertAlmostEqual(parts['smoothness'][0],.2,places=6)
        self.assertEqual(select_lowest(np.array([4.,1.,1.,7.])),1)
        with self.assertRaises(FloatingPointError): select_lowest(np.array([0.,np.nan]))

    def test_online_history_matches_whole_real_prefix_and_A_B_A_reset(self):
        p=self.planner(); a=np.zeros(4,np.float32)
        obs=np.array([[1,0,0,0,0,0,0],[.9,0,0,.1,0,0,0]],np.float32)
        previous=np.vstack([a,[.2,0,0,0]]).astype(np.float32)
        p.encode_current(obs[0],previous[0]); z=p.encode_current(obs[1],previous[1])
        with torch.no_grad(): expected=encode_episode(self.model,obs,previous,self.stats)[-1:]
        torch.testing.assert_close(z,expected,atol=1e-6,rtol=1e-6)
        p.reset(0); first=p.encode_current(obs[0],a).clone()
        p.reset(0); p.encode_current(-obs[0],a)
        p.reset(0); again=p.encode_current(obs[0],a)
        torch.testing.assert_close(first,again,atol=0,rtol=0)

    def test_plan_matches_pure_latent_reference_selects_first_and_no_mutation(self):
        p=self.planner(); before=component_hashes(self.model)
        z=p.encode_current(np.array([.6,-.3,.1,0,0,0,0],np.float32),np.zeros(4))
        result=p.plan(z,np.zeros(4))
        with torch.no_grad():
            out=predict_latent(self.model,z.expand(512,-1),torch.from_numpy(result['candidates']),self.stats)
        obs=out['normalized_observations'][:,-1].numpy()*self.stats['obs']['std']+self.stats['obs']['mean']
        expected,_=planning_cost(obs,result['candidates'],np.zeros(4))
        np.testing.assert_allclose(result['costs'],expected,rtol=1e-5,atol=1e-6)
        self.assertEqual(result['index'],int(np.argmin(expected)))
        np.testing.assert_array_equal(result['action'],result['candidates'][result['index'],0])
        np.testing.assert_allclose(result['terminal_observation'],obs[result['index']],rtol=1e-5,atol=1e-6)
        self.assertGreater(result['planning_seconds'],0)
        self.assertEqual(before,component_hashes(self.model))
        self.assertFalse(self.model.training)
        self.assertTrue(all(not v.requires_grad for v in self.model.parameters()))

    def test_seeded_plan_reproduces_predictions_without_future_interface(self):
        p=self.planner(3); z=p.encode_current(np.array([1,0,0,0,0,0,0],np.float32),np.zeros(4))
        first=p.plan(z,np.zeros(4)); p.reset(3)
        z=p.encode_current(np.array([1,0,0,0,0,0,0],np.float32),np.zeros(4)); second=p.plan(z,np.zeros(4))
        np.testing.assert_array_equal(first['costs'],second['costs'])
        np.testing.assert_array_equal(first['action'],second['action'])


if __name__=='__main__': unittest.main()
