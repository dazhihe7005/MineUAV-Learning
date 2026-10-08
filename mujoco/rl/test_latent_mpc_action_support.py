"""Support projection must be the sole paired planner intervention."""
import importlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from latent_mpc_core import RandomShootingMPC, sample_candidates
from latent_dynamics_data import file_hash, json_hash
from joint_autonomous_consistency_training import load_autonomous
from joint_latent_world_model import component_hashes


class SupportPlannerTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('latent_mpc_action_support'),
                             'missing Train support projection implementation')
        return importlib.import_module('latent_mpc_action_support')

    def test_support_reads_only_full_train_actions_and_canonical_hash(self):
        api=self.api()
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'train.npz'
            # Four executed rows, including terminal commands: literal quantiles.
            actions=np.tile(np.array([[-1.],[-.5],[.5],[1.]],np.float32),(1,4))
            np.savez(path,actions=actions)
            np.savez(Path(d)/'test.npz',actions=np.full((100,4),.123))
            first=api.compute_support(path,file_hash(path))
            self.assertEqual(first['count'],4)
            np.testing.assert_allclose(first['lower'],[-.9625]*4,atol=1e-12)
            np.testing.assert_allclose(first['upper'],[.9625]*4,atol=1e-12)
            payload={k:v for k,v in first.items() if k!='sha256'}
            self.assertEqual(first['sha256'],json_hash(payload))
            np.savez(Path(d)/'test.npz',actions=np.full((100,4),-.9))
            self.assertEqual(api.compute_support(path,file_hash(path)),first)
            with self.assertRaises(ValueError): api.compute_support(Path(d)/'test.npz',file_hash(Path(d)/'test.npz'))
            with self.assertRaises(ValueError): api.compute_support(path,'changed')

    def test_recursive_projection_anchors_noise_and_bounds(self):
        api=self.api(); prev=np.array([.9,-.8,.4,.2],np.float32)
        low=np.array([.1,-.2,-.3,-.4]); high=np.array([.2,-.1,.3,.4]); std=np.array([.2,.3,.4,.5])
        rng=np.random.default_rng(7); noise=rng.normal(size=(510,10,4))*(.5*std)
        want=np.empty((512,10,4),np.float32)
        want[0]=np.clip(np.zeros(4),low,high); want[1]=np.clip(prev,low,high)
        cur=np.broadcast_to(prev,(510,4)).copy()
        for k in range(10):
            cur=np.clip(np.clip(cur+noise[:,k],-1,1),low,high); want[2:,k]=cur
        result,audit=api.sample_projected_candidates(prev,std,-np.ones(4),np.ones(4),low,high,np.random.default_rng(7))
        np.testing.assert_array_equal(result,want)
        self.assertEqual(result.shape,(512,10,4))
        self.assertTrue(np.all(result>=low-1e-7) and np.all(result<=high+1e-7))
        self.assertEqual(audit['anchor_projected_candidates'],2)
        self.assertGreater(audit['projection_magnitudes'].max(),0)

    def test_full_environment_support_is_exact_original_sampler_for_many_decisions(self):
        api=self.api(); a=np.array([.9,-.8,.4,.2],np.float32); std=np.array([.2,.3,.4,.5])
        r1=np.random.default_rng(17); r2=np.random.default_rng(17)
        for _ in range(4):
            original=sample_candidates(a,std,-np.ones(4),np.ones(4),r1)
            projected,audit=api.sample_projected_candidates(a,std,-np.ones(4),np.ones(4),-np.ones(4),np.ones(4),r2)
            np.testing.assert_array_equal(original,projected)
            self.assertEqual(audit['anchor_projected_candidates'],0)
            self.assertEqual(float(audit['projection_magnitudes'].max()),0.)
            a=original[5,0]

    def test_paired_noise_does_not_depend_on_state_or_support(self):
        api=self.api(); r1=np.random.default_rng(11); r2=np.random.default_rng(11)
        for i in range(5):
            _,u=api.audited_candidates(np.full(4,.1*i),np.ones(4)*.4,-np.ones(4),np.ones(4),r1,None)
            _,s=api.audited_candidates(np.full(4,-.1*i),np.ones(4)*.4,-np.ones(4),np.ones(4),r2,(-np.ones(4)*.2,np.ones(4)*.2))
            self.assertEqual(u['epsilon_sha256'],s['epsilon_sha256'])

    def test_supported_plan_full_box_matches_original_cost_action_and_reset(self):
        api=self.api(); torch.set_num_threads(1)
        model,stats,_=load_autonomous(Path(__file__).parent/'models/joint_latent_world_model_v3_autonomous_consistency.pt')
        before=component_hashes(model)
        original=RandomShootingMPC(model,stats,-np.ones(4),np.ones(4),23)
        support=api.ActionSupportMPC(model,stats,-np.ones(4),np.ones(4),23,(-np.ones(4),np.ones(4)))
        obs=np.array([.6,-.3,.1,0,0,0,0],np.float32); prev=np.zeros(4,np.float32)
        z1=original.encode_current(obs,prev); z2=support.encode_current(obs,prev)
        u=original.plan(z1,prev); s=support.plan(z2,prev)
        np.testing.assert_array_equal(u['candidates'],s['candidates'])
        np.testing.assert_array_equal(u['costs'],s['costs'])
        np.testing.assert_array_equal(u['action'],s['action'])
        self.assertEqual(u['index'],s['index'])
        self.assertEqual(before,component_hashes(model))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        support.reset(23); again=support.plan(support.encode_current(obs,prev),prev)
        np.testing.assert_array_equal(s['action'],again['action'])
        self.assertEqual(support.planning_count,1)

    def test_invalid_support_rejected_not_silently_reordered(self):
        api=self.api()
        for low,high in [(np.ones(4),-np.ones(4)),(-np.ones(4)*2,np.ones(4)),(np.full(4,np.nan),np.ones(4))]:
            with self.subTest(low=low),self.assertRaises(ValueError):
                api.sample_projected_candidates(np.zeros(4),np.ones(4),-np.ones(4),np.ones(4),low,high,np.random.default_rng(0))


if __name__=='__main__': unittest.main()
