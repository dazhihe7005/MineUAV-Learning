"""Force exposure/censoring, complete failure accounting and verified resume."""
import importlib,importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

class ExternalEvaluationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_bc_external_evaluation'),'external-force evaluator missing')
        return importlib.import_module('uav_bc_external_evaluation')
    def trace(self,times,distance,speed):
        o=np.zeros((len(times),7),np.float32);o[:,0]=distance;o[:,3]=speed
        force=np.zeros((round(times[-1]/.002),3));ticks=np.arange(len(force))
        force[(ticks>=1000)&(ticks<2000),0]=1.
        return dict(observations=o,physical_state_valid=np.ones(len(times),bool),policy_times=np.asarray(times,float),
            physics_force_times=np.arange(round(times[-1]/.002))*.002,
            physics_forces=force)
    def test_gust_recovery_requires_real_post_end_sustained_samples(self):
        a=self.api();z=self.trace([0,4,4.04,4.08,4.12,4.16],[1,.05,.05,.05,.05,.05],[0,.1,.1,.1,.1,.1])
        r=a.metrics({},z,'gust_high');self.assertEqual(r['gust_recovery_onset_lag_s'],0.)
        self.assertAlmostEqual(r['gust_recovery_confirmation_lag_s'],.16)
        z['physical_state_valid'][-1]=False
        self.assertIsNone(a.metrics({},z,'gust_high')['gust_recovery_onset_lag_s'])
    def test_unexposed_or_incomplete_gust_is_not_recovery(self):
        a=self.api();z=self.trace([0,.04,.08,.12,.16],.05,.1)
        r=a.metrics({},z,'gust_medium');self.assertIsNone(r['gust_recovery_onset_lag_s'])
        self.assertTrue(r['terminated_before_gust_start']);self.assertFalse(r['survived_gust_end'])
        z=self.trace([0,2,3,3.5,3.9],.05,.1)
        self.assertIsNone(a.metrics({},z,'gust_medium')['gust_recovery_onset_lag_s'])
    def test_partial_or_failed_terminal_sample_cannot_complete_gust_recovery(self):
        a=self.api()
        for end in (4.122,4.16):
            z=self.trace([4.,4.04,4.08,4.12,end],.05,.1)
            self.assertIsNone(a.metrics({'physical_failure':True},z,'gust_high')['gust_recovery_onset_lag_s'])
        z=self.trace([4.,4.04,4.08,4.12,4.122],.05,.1)
        self.assertIsNone(a.metrics({},z,'gust_high')['gust_recovery_onset_lag_s'])
    def test_recovery_requires_consecutive_complete_boundaries_and_gust_exposure(self):
        a=self.api();z=self.trace([4.,4.04,4.12,4.16,4.20],.05,.1)
        self.assertIsNone(a.metrics({},z,'gust_medium')['gust_recovery_onset_lag_s'])
        z=self.trace([4.,4.04,4.08,4.12,4.16],.05,.1);z['physics_forces'][:]=0
        self.assertIsNone(a.metrics({},z,'gust_medium')['gust_recovery_onset_lag_s'])
        z['physics_forces'][1000:1500,0]=1.
        self.assertIsNone(a.metrics({},z,'gust_medium')['gust_recovery_onset_lag_s'])
    def test_postprocessing_provenance_preserves_acquisition_and_rejects_unknown_revision(self):
        a=self.api();self.assertTrue(hasattr(a,'evaluation_provenance'))
        from latent_dynamics_data import file_hash
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'mujoco/rl';source.mkdir(parents=True)
            path=source/'uav_bc_external_evaluation.py';path.write_text('old evaluator')
            old=file_hash(path);identity=dict(source_sha256={path.name:old},manifest_sha256='original manifest')
            path.write_text('revised metric processor');new=file_hash(path)
            with self.assertRaises(ValueError):a.evaluation_provenance(root,identity,{})
            result=a.evaluation_provenance(root,identity,{path.name:{old:'Exclude partial recovery samples'}})
            self.assertEqual(result['acquisition_identity'],identity)
            self.assertEqual(result['metric_verifier_source_sha256'][path.name],new)
            self.assertEqual(result['source_revisions'][path.name]['acquisition_sha256'],old)
            self.assertEqual(identity['source_sha256'][path.name],old)
            identity['source_sha256'][path.name]='unrecognized hash'
            with self.assertRaises(ValueError):a.evaluation_provenance(root,identity,{path.name:{old:'Reviewed revision'}})
    def test_constant_steady_proxy_requires_full_eligible_window(self):
        a=self.api();z=self.trace(np.arange(151)*.04,.2,.01)
        r=a.metrics({},z,'constant_high');self.assertTrue(r['steady_window_eligible'])
        self.assertAlmostEqual(r['steady_position_error_m'],.2,places=6)
        self.assertAlmostEqual(r['steady_velocity_m_s'],.01,places=6)
        self.assertEqual(r['steady_target_hold_fraction'],0.)
        self.assertAlmostEqual(r['steady_distance_slope_m_s'],0.,places=10)
        self.assertFalse(a.metrics({},self.trace([0,1,2],.1,.01),'constant_low')['steady_window_eligible'])
    def test_real_frozen_independent_inference_and_valid_resume(self):
        a=self.api();from uav_bc_external_disturbance import DisturbanceEnv,initial_snapshot
        from uav_bc_policy import load,parameter_hash
        root=Path(__file__).resolve().parents[2];p,_=load(root/'mujoco/rl/models/uav_bc_yaw_augmented_seed0.pt');h=parameter_hash(p.actor)
        env=DisturbanceEnv(max_episode_seconds=.08);self.addCleanup(env.close)
        env.configure('constant_high',[1.,0.,0.])
        state=dict(key='constant_high/final/000',condition='constant_high',split='final',target_id='new',env_seed=7,target=[1.,0,1.],direction_world=[1.,0.,0.])
        snap,state['initial_state']=initial_snapshot(env,state);expected=p.predict(env._get_obs())
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'r.json'
            with patch('test_env_scripted_policy.scripted_action',side_effect=AssertionError('teacher leaked')):
                r=a.evaluate_task(env,snap,p.predict,state,'yaw_augmented',path,dict(source='fixed'))
            with np.load(r['trace_path']) as z:
                np.testing.assert_array_equal(z['actions'][0],expected)
                np.testing.assert_allclose(z['policy_times'],[0,.04,.08],atol=1e-12)
                self.assertEqual(len(z['physics_forces']),40)
            cached=a.evaluate_task(env,snap,lambda _:self.fail('duplicate execution'),state,'yaw_augmented',path,dict(source='fixed'))
            self.assertEqual(cached,r);self.assertTrue(r['timeout']);self.assertFalse(r['success'])
            self.assertIsNone(r['completion_time_s']);self.assertEqual(parameter_hash(p.actor),h)
            with self.assertRaises(ValueError):a.evaluate_task(env,snap,p.predict,state,'yaw_augmented',path,dict(source='changed'))
    def test_resume_rejects_trace_tampering_and_relabelled_records(self):
        a=self.api();from uav_bc_external_disturbance import DisturbanceEnv,initial_snapshot
        from uav_bc_safety import atomic_json
        from latent_dynamics_data import json_hash
        env=DisturbanceEnv(max_episode_seconds=.04);self.addCleanup(env.close)
        s=dict(key='nominal/final/000',condition='nominal',split='final',target_id=1,env_seed=7,target=[1.,0,1.],direction_world=[1.,0.,0.])
        snap,s['initial_state']=initial_snapshot(env,s)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'r.json';r=a.evaluate_task(env,snap,lambda o:np.zeros(4,np.float32),s,'original',path,{})
            bad=dict(r,maximum_target_distance_m=9.);bad['record_sha256']=json_hash({k:v for k,v in bad.items() if k!='record_sha256'});atomic_json(path,bad)
            with self.assertRaises((ValueError,AssertionError)):a.evaluate_task(env,snap,lambda o:np.zeros(4,np.float32),s,'original',path,{})
            atomic_json(path,r);Path(r['trace_path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError):a.evaluate_task(env,snap,lambda o:np.zeros(4,np.float32),s,'original',path,{})
    def test_paired_metrics_retain_both_failed_and_timeout_discordance(self):
        a=self.api()
        def row(key,ok,d):return dict(key=key,success=ok,timeout=not ok,completion_time_s=1. if ok else None,
            final_distance_m=d,termination_reason='success' if ok else 'time_limit',maximum_speed_m_s=.5,
            near_steps=1,near_actual_sum=.1)
        b=[row('a',True,.05),row('b',False,.5)];c=[row('a',False,.8),row('b',False,.4)]
        r=a.paired(b,c);self.assertEqual(r['success_delta'],-.5);self.assertEqual(r['discordance']['both_failed'],1)
        self.assertEqual(r['timeout_discordance']['second_only_timeout'],1)
        self.assertIsNone(r['completion_time_delta_s'])
    def test_partial_physics_failure_time_and_force_count(self):
        a=self.api();from uav_bc_external_disturbance import DisturbanceEnv,initial_snapshot
        from uav_bc_robustness import rollout
        env=DisturbanceEnv(max_episode_seconds=.12);self.addCleanup(env.close)
        s,_=initial_snapshot(env,dict(env_seed=7,target=[1.,0,1.]))
        original=DisturbanceEnv._before_physics_step
        def fail(e):
            original(e)
            if e.data.time>=.04-1e-12:e.data.qpos[0]=7.
        with patch.object(DisturbanceEnv,'_before_physics_step',fail):r,z=rollout(env,s,lambda o:np.zeros(4,np.float32))
        z=a.attach_physics(env,z);a.verify_arrays(r,z,'nominal',[1.,0.,0.])
        self.assertEqual(r['steps'],2);self.assertEqual(len(z['physics_forces']),21)
        self.assertAlmostEqual(r['simulated_seconds'],.042);self.assertTrue(r['physical_failure'])
    def test_cohort_rejects_missing_or_duplicate_episode(self):
        self.api();from uav_bc_robustness import validate_cohort
        for rows in ([dict(task_key='a')],[dict(task_key='a'),dict(task_key='a')]):
            with self.assertRaises(ValueError):validate_cohort(rows,['a','b'])

if __name__=='__main__':unittest.main()
