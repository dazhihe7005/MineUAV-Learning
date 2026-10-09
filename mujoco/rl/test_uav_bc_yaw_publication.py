"""Publication must validate real selected labels, and not replace a regressed baseline."""
import importlib,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
from unittest.mock import patch

class YawPublicationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_uav_bc_yaw_coverage'),'publication verifier missing')
        return importlib.import_module('run_uav_bc_yaw_coverage')
    def test_selected_arrays_reconstructed_from_hashed_real_expert_indices(self):
        a=self.api();from uav_bc_safety import atomic_npz
        from latent_dynamics_data import file_hash
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'raw.npz';obs=np.array([[1.,0,0,0,0,0,0],[.8,0,0,.1,0,0,0],[.7,0,0,.2,0,0,0]],np.float32)
            actions=np.array([[2/9,0,0,0],[.8/4.5,0,0,0]],np.float32)
            atomic_npz(path,observations=obs,actions=actions,physical_state_valid=np.ones(3,bool))
            r=dict(key='train/b/0/0',source='nominal_expanded',target_id=4,yaw_group=0,initial_yaw_degrees=0.,trace_path=str(path),trace_sha256=file_hash(path))
            selected=dict(observations=obs[:-1].copy(),actions=actions.copy(),sample_key=np.array(['train/b/0/0/000','train/b/0/0/001']),
                target_id=np.array([4,4]),step_index=np.array([0,1]),phase=np.array(['early','early']),yaw_group=np.array([0,0]),initial_yaw_degrees=np.array([0.,0.]))
            a.verify_selected_arrays(selected,{r['key']:r},'nominal_expanded',{4})
            selected['actions'][0,0]=.8
            with self.assertRaises((AssertionError,ValueError)):a.verify_selected_arrays(selected,{r['key']:r},'nominal_expanded',{4})
    def test_duplicate_sample_or_cross_target_is_rejected(self):
        a=self.api();s=dict(sample_key=np.array(['same','same']),target_id=np.array([4,4]))
        with self.assertRaises(ValueError):a.verify_selected_arrays(s,{},'nominal_expanded',{4})
    def test_engineering_pass_does_not_hide_nominal_regression(self):
        a=self.api();metrics={}
        for c in ['nominal','yaw_small','yaw_large','position_small','position_large','velocity_small','velocity_large','combined_small','combined_large']:
            metrics[c]={n:dict(successes=x) for n,x in [('original',100),('nominal_expanded',100 if c=='nominal' else 5),('yaw_augmented',100)]}
        self.assertTrue(a.interpret(metrics)['engineering_targets_met'])
        metrics['nominal']['yaw_augmented']['successes']=70
        self.assertFalse(a.interpret(metrics)['engineering_targets_met']);self.assertTrue(a.interpret(metrics)['nominal_regression'])
    def test_real_aggregate_figures_use_atomic_png_writer(self):
        from uav_bc_yaw_plots import plots
        from uav_bc_yaw_evaluation import CONTROLLERS
        from uav_bc_robustness import conditions,wilson
        from uav_bc_safety import atomic_npz
        obs=np.array([[1.,0,0,0,0,0,.2],[.1,0,0,.1,0,0,.1]],np.float32)
        selected=dict(observations=obs,initial_yaw_degrees=np.array([0.,30.]))
        metrics={c:{n:dict(success_rate=1.,success_wilson_95_ci=wilson(100,100),maximum_speed_m_s=dict(mean=.5),
            near_target_actual_speed_m_s=.1,near_target_command_speed_m_s=.1,final_distance_m=dict(mean=.1)) for n in CONTROLLERS} for c in conditions()}
        metrics['yaw_large']['original'].update(success_rate=0.,success_wilson_95_ci=wilson(0,100))
        offline={n:dict(by_initial_yaw={str(g):{kind:dict(rmse=.1) for kind in ['raw','clipped']} for g in [0,10,30]}) for n in CONTROLLERS[1:]}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reports=root/'mujoco/reports';reports.mkdir(parents=True)
            (reports/'uav_bc_dataset_manifest_seed0.json').write_text('{}')
            historical=reports/'nominal_regression_comparison.png';historical.write_bytes(b'historical adaptation figure, must remain unchanged')
            raw=reports/'trace.npz';atomic_npz(raw,observations=obs,positions=np.array([[0,0,1],[.9,0,1]]),physical_state_valid=np.ones(2,bool))
            rows=[dict(key='yaw_large/final/000',controller=n,termination_reason='success',target=[1,0,1],trace_path=str(raw)) for n in CONTROLLERS]
            with patch('uav_bc_yaw_plots.load_selected',return_value=selected),patch('uav_bc_data.load_split',return_value=dict(observations=obs)):
                names=plots(root,{}, {}, {},dict(metrics=metrics,offline_test=offline,rows=rows))
            self.assertEqual(len(names),8)
            self.assertEqual(historical.read_bytes(),b'historical adaptation figure, must remain unchanged')
            for name in names:self.assertEqual((reports/name).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
    def test_wilson_endpoint_roundoff_only_is_clipped_for_rendering(self):
        from uav_bc_robustness import wilson
        import uav_bc_yaw_plots as p
        self.assertTrue(hasattr(p,'confidence_error_lengths'),'endpoint-safe renderer missing')
        values=np.array([0.,1.]);ci=np.array([wilson(0,100),wilson(100,100)])
        errors=p.confidence_error_lengths(values,ci)
        self.assertTrue((errors>=0).all());np.testing.assert_allclose(errors[0,1],1-ci[1,0])
        with self.assertRaises(ValueError):p.confidence_error_lengths(np.array([0.]),np.array([[.1,.2]]))
    def test_scalar_outcomes_verified_against_real_physical_trace(self):
        a=self.api();self.assertTrue(hasattr(a,'verify_rollout_metrics'),'independent scalar verification missing')
        from ppo_pi_env import MineUAVPIEnv
        from uav_bc_robustness import prepare_snapshot,rollout
        from test_env_scripted_policy import scripted_action
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.08)
        try:
            s,_=prepare_snapshot(env,dict(target_id=1,env_seed=7,target=[1.,0,1.]),dict(position_m=[0,0,0],velocity_m_s=[.15,0,0],yaw_rad=np.pi/6))
            r,z=rollout(env,s,scripted_action);a.verify_rollout_metrics(r,z)
            with self.assertRaises((AssertionError,ValueError)):a.verify_rollout_metrics(dict(r,maximum_speed_m_s=8.),z)
        finally:env.close()
    def test_partial_terminal_policy_interval_is_valid_physical_failure(self):
        a=self.api();from ppo_pi_env import MineUAVPIEnv
        from uav_bc_robustness import prepare_snapshot,rollout
        env=MineUAVPIEnv(reward_version='v2',max_episode_seconds=.12)
        try:
            snap,_=prepare_snapshot(env,dict(target_id=1,env_seed=7,target=[1.,0,1.]),dict(position_m=[0,0,0],velocity_m_s=[0,0,0],yaw_rad=0))
            def leave_flight_area():
                if env.data.time>=.04-1e-12:env.data.qpos[1]=7.
            with patch.object(type(env),'_before_physics_step',lambda _:leave_flight_area()):
                r,z=rollout(env,snap,lambda o:np.zeros(4,np.float32))
            self.assertEqual(r['termination_reason'],'outside_flight_area');self.assertEqual(r['steps'],2)
            self.assertAlmostEqual(r['simulated_seconds'],.042);self.assertIsNone(r['completion_time_s'])
            a.verify_rollout_metrics(r,z)
            for invalid in [.02,.051,.082,float('nan')]:
                with self.assertRaises((AssertionError,ValueError)):a.verify_rollout_metrics(dict(r,simulated_seconds=invalid),z)
        finally:env.close()

if __name__=='__main__':unittest.main()
