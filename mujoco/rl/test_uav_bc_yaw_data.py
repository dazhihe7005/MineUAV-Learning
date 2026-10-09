"""Tests catch duplicated labels, yaw/frame mistakes, quota and split leakage."""
import importlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np

class YawDataTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_bc_yaw_data'),'yaw data implementation missing')
        return importlib.import_module('uav_bc_yaw_data')
    def test_pairing_yaw_balance_and_unique_base_variations(self):
        a=self.api();base=[dict(target_id=i,env_seed=i,target=[.5,.5,1.]) for i in range(84)]
        b=a.expert_tasks(base,'train','nominal_expanded');c=a.expert_tasks(base,'train','yaw_augmented')
        self.assertEqual(len(b),840);self.assertEqual(len(c),840)
        self.assertEqual(sum(x['yaw_group']==0 for x in c),336)
        for deg in (10,30):
            self.assertEqual(sum(x['yaw_group']==deg and x['yaw_sign']==1 for x in c),126)
            self.assertEqual(sum(x['yaw_group']==deg and x['yaw_sign']==-1 for x in c),126)
        for x,y in zip(b,c):
            self.assertEqual(x['target'],y['target']);self.assertEqual(x['env_seed'],y['env_seed'])
            self.assertEqual(x['perturbation']['position_m'],y['perturbation']['position_m'])
            self.assertEqual(x['perturbation']['velocity_m_s'],y['perturbation']['velocity_m_s'])
            self.assertEqual(x['perturbation']['yaw_rad'],0)
        self.assertEqual(len({tuple(x['perturbation']['position_m']) for x in b}),840)
        self.assertEqual(c,a.expert_tasks(base,'train','yaw_augmented'))
    def test_real_yaw_frame_wrap_and_label_indexing(self):
        a=self.api();from ppo_pi_env import MineUAVPIEnv
        from uav_bc_robustness import prepare_snapshot,rollout
        from test_env_scripted_policy import scripted_action
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            t=dict(target_id=1,env_seed=7,target=[1.,.5,1.2]);p=dict(position_m=[.1,0,0],velocity_m_s=[.15,0,0],yaw_rad=np.pi/6)
            s,m=prepare_snapshot(env,t,p)
            np.testing.assert_allclose(m['observation'],[.9,.5,.2,.15,0,0,-np.pi/6],atol=1e-7)
            r,z=rollout(env,s,scripted_action);a.validate_expert_trace(z)
            np.testing.assert_allclose(z['actions'][0],[.2,1/9,1/15,-np.pi/6],atol=1e-7)
            z['actions'][0,0]+=.01
            with self.assertRaises(ValueError):a.validate_expert_trace(z)
            from mine_uav_env import _wrapped_angle
            self.assertAlmostEqual(_wrapped_angle(3*np.pi/2),-np.pi/2)
        finally:env.close()
    def test_phase_priority_includes_braking_without_overlap(self):
        a=self.api();obs=np.array([[1,0,0,.7,0,0,0],[.1,0,0,.7,0,0,0],[1,0,0,.7,0,0,0],[.3,0,0,.2,0,0,0]])
        np.testing.assert_array_equal(a.phase_labels(obs,np.array([0,10,10,10])),['early','near_braking','high_speed','approach_other'])
    def test_stratified_unique_selection_and_shortage_rejection(self):
        a=self.api();pool=[]
        # Hand-built quotas for200 transitions:20early60near50high70approach.
        for phase,n in [('early',20),('near_braking',60),('high_speed',50),('approach_other',70)]:
            for deg,signs in [(0,[0]),(10,[-1,1]),(30,[-1,1])]:
                count=int(n*(.4 if deg==0 else .3))/len(signs)
                for sign in signs:
                    for j in range(int(count)+2):pool.append(dict(key=f'{phase}-{deg}-{sign}-{j}',phase=phase,yaw_group=deg,yaw_sign=sign))
        selected=a.choose_samples(pool,200,123)
        self.assertEqual(len(selected),200);self.assertEqual(len({r['key'] for r in selected}),200)
        self.assertEqual(selected,a.choose_samples(pool,200,123))
        for deg,want in [(0,80),(10,60),(30,60)]:self.assertEqual(sum(r['yaw_group']==deg for r in selected),want)
        with self.assertRaises(ValueError):a.choose_samples(pool[:2],200,123)
        with self.assertRaises(ValueError):a.choose_samples(pool+pool[:1],200,123)
    def test_target_overlap_and_new_seed_rejected(self):
        a=self.api();old=[dict(target_id='old',env_seed=2,target=[.5,0,1.])]
        fresh=a.final_targets(old);self.assertEqual(len(fresh),100)
        self.assertEqual(fresh,a.final_targets(old));a.assert_isolated(fresh,old)
        for bad in [old,[dict(fresh[0],target=old[0]['target'])],[dict(fresh[0],env_seed=2)]]:
            with self.assertRaises(ValueError):a.assert_isolated(bad,old)
    def test_resume_rejects_corrupted_raw_and_identity(self):
        a=self.api();from uav_bc_safety import atomic_npz,atomic_json
        from latent_dynamics_data import file_hash,json_hash
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'a.npz';atomic_npz(p,values=np.ones(1))
            r=dict(key='a',identity='one',trace_path=str(p),trace_sha256=file_hash(p));r['record_sha256']=json_hash(r)
            a.check_record(r,'one','a')
            with self.assertRaises(ValueError):a.check_record(r,'two','a')
            p.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):a.check_record(r,'one','a')
    def test_archived_evaluation_without_target_ids_still_excluded(self):
        a=self.api();self.assertTrue(hasattr(a,'exclusion_targets'),'archived evaluation identity adapter missing')
        bc=dict(groups=dict(train=[dict(target_id=7,env_seed=2,target=[1.,0,1.])]),
            provenance=dict(evaluation=dict(benchmark=dict(targets=[dict(env_seed=8,target=[.5,0,1.])]))))
        old=a.exclusion_targets(bc,dict(splits={}))
        self.assertEqual(len(old),2);self.assertEqual(old[1]['env_seed'],8)
        self.assertEqual(old[1]['target'],[.5,0,1.]);self.assertIn('target_id',old[1])
        with self.assertRaises(ValueError):a.assert_isolated([dict(target_id='new',env_seed=99,target=[.5,0,1.])],old)

if __name__=='__main__':unittest.main()
