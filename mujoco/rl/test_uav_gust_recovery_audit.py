"""Read-only NumPy fixtures: no MuJoCo episode, policy call or training."""
import importlib,importlib.util,json,tempfile,unittest
from pathlib import Path
import numpy as np


class GustAuditTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_gust_recovery_audit'),'pure gust analysis missing')
        return importlib.import_module('uav_gust_recovery_audit')

    def trace(self,seconds=15.):
        n=round(seconds/.04);times=np.arange(n+1)*.04
        o=np.zeros((n+1,7));o[:,0]=.5;o[:,3]=.3
        a=np.zeros((n,4));a[:,0]=-.1
        ft=np.arange(round(seconds/.002))*.002;forces=np.zeros((len(ft),3))
        forces[(ft>=2)&(ft<4),0]=6.867
        return dict(observations=o,positions=np.array([1.,0.,1.])-o[:,:3],actions=a,
            previous_actions=np.vstack((np.zeros((1,4)),a[:-1])),step_index=np.arange(n),
            success_streak=np.zeros(n+1,int),physical_state_valid=np.ones(n+1,bool),
            policy_times=times,physics_force_times=ft,physics_forces=forces,
            allocator_saturation_counts=np.zeros(n,int),control_update_counts=np.full(n,4))

    def row(self,z,**overrides):
        r=dict(condition='gust_medium',controller='scripted',steps=len(z['actions']),
            simulated_seconds=float(z['policy_times'][-1]),success=False,physical_failure=False,
            timeout=True,termination_reason='time_limit',target=[1.,0.,1.],direction_world=[1.,0.,0.])
        return dict(r,**overrides)

    def test_phase_intervals_use_pre_action_alignment_and_real_duration(self):
        a=self.api();z=self.trace(8.);r=a.analyze_episode(self.row(z),z)
        self.assertEqual([v['samples'] for v in r['phases'].values()],[50,50,50,50])
        self.assertAlmostEqual(r['phases']['post_4_6']['duration_s'],2.)
        self.assertAlmostEqual(r['phases']['post_4_6']['command_actual_rmse_m_s'],.45)
        short=self.trace(5.);r=a.analyze_episode(self.row(short),short)
        self.assertIsNone(r['phases']['late_6_end']);self.assertEqual(r['phases']['post_4_6']['samples'],25)

    def test_opposing_command_is_measured_not_named_wrong_control(self):
        a=self.api();z=self.trace();r=a.analyze_episode(self.row(z),z)
        self.assertEqual(r['phases']['post_4_6']['opposing_fraction_given_gated'],1.)
        self.assertAlmostEqual(r['phases']['post_4_6']['longest_opposing_run_s'],2.)
        z['actions'][:,:3]=0;r=a.analyze_episode(self.row(z),z)
        self.assertIsNone(r['phases']['post_4_6']['opposing_fraction_given_gated'])

    def test_distance_can_grow_after_force_removal_without_final_divergence(self):
        a=self.api();z=self.trace();t=z['policy_times'];z['observations'][:,0]=np.interp(t,[0,4,6,15],[.5,.5,1.,.2])
        r=a.analyze_episode(self.row(z),z)
        self.assertAlmostEqual(r['post_gust']['max_distance_excess_over_at4_m'],.5)
        self.assertAlmostEqual(r['post_gust']['final_minus_at4_distance_m'],-.3)
        self.assertTrue(r['post_gust']['distance_grows_after4'])
        self.assertEqual(r['post_gust']['post_distance_peak_time_s'],6.)
        self.assertEqual(r['post_gust']['post_speed_peak_time_s'],4.)
        self.assertAlmostEqual(r['post_gust']['post_speed_peak_m_s'],.3)

    def test_analysis_has_no_simulator_or_training_imports(self):
        import ast
        a=self.api();tree=ast.parse(Path(a.__file__).read_text())
        imports=[n.module if isinstance(n,ast.ImportFrom) else alias.name
            for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))
            for alias in (n.names if isinstance(n,ast.Import) else [None])]
        self.assertFalse(any(name and any(x in name for x in ('mujoco','mine_uav_env','torch')) for name in imports))

    def test_recovery_requires_joint_consecutive_complete_samples(self):
        a=self.api();z=self.trace();t=z['policy_times'];z['observations'][t>=4,0]=.05;z['observations'][t>=4,3]=.1
        r=a.analyze_episode(self.row(z),z)
        self.assertEqual(r['post_gust']['joint_recovery_onset_s'],4.)
        self.assertAlmostEqual(r['post_gust']['joint_recovery_confirmation_s'],4.16)
        z['observations'][t>=4,3]=.2;r=a.analyze_episode(self.row(z),z)
        self.assertIsNone(r['post_gust']['joint_recovery_onset_s'])
        self.assertEqual(r['post_gust']['distance_only_longest_run_s'],11.)
        self.assertEqual(r['post_gust']['joint_longest_run_s'],0.)

    def test_failed_partial_boundary_cannot_confirm_recovery(self):
        a=self.api();z=self.trace(4.16);z['observations'][-5:,0]=.05;z['observations'][-5:,3]=.1
        z['policy_times'][-1]=4.122
        r=a.analyze_episode(self.row(z,physical_failure=True,timeout=False,termination_reason='outside_flight_area'),z)
        self.assertIsNone(r['post_gust']['joint_recovery_onset_s'])

    def test_success_verifier_resets_streak_on_failed_partial_terminal_step(self):
        a=self.api();z=self.trace(4.16);z['observations'][-5:,0]=.05;z['observations'][-5:,3]=.1
        z['policy_times'][-1]=4.122;z['success_streak'][-5:-1]=[1,2,3,4];z['success_streak'][-1]=0
        row=self.row(z,physical_failure=True,timeout=False,termination_reason='outside_flight_area')
        self.assertEqual(a.verify_success(row,z),0)
        row['success']=True
        with self.assertRaises(ValueError):a.verify_success(row,z)

    def test_success_verifier_accepts_complete_valid_fifth_step_and_rejects_bad_streak(self):
        a=self.api();z=self.trace(4.16);z['observations'][-5:,0]=.05;z['observations'][-5:,3]=.1
        z['success_streak'][-5:]=[1,2,3,4,5];row=self.row(z,success=True,timeout=False)
        self.assertEqual(a.verify_success(row,z),5)
        z['success_streak'][-2]=0
        with self.assertRaises(ValueError):a.verify_success(row,z)

    def test_success_verifier_excludes_invalid_complete_terminal_state(self):
        a=self.api();z=self.trace(4.16);z['observations'][-5:,0]=.05;z['observations'][-5:,3]=.1
        z['success_streak'][-5:-1]=[1,2,3,4];z['physical_state_valid'][-1]=False
        self.assertEqual(a.verify_success(self.row(z,physical_failure=True),z),0)

    def test_missing_pi_telemetry_is_explicit_and_not_reconstructed(self):
        a=self.api();s=a.signal_availability(self.trace())
        self.assertEqual(s['pi_integral_state'],'not recorded')
        self.assertEqual(s['pi_output'],'not recorded')
        self.assertEqual(s['anti_windup_freeze_count'],'not recorded')
        self.assertEqual(s['yaw_error'],'recorded')

    def test_saved_actions_indexing_and_force_buffer_tampering_are_rejected(self):
        a=self.api();z=self.trace();row=self.row(z)
        a.validate_arrays(row,z)
        z['previous_actions'][2,0]=.5
        with self.assertRaises((ValueError,AssertionError)):a.validate_arrays(row,z)
        z=self.trace();z['physics_forces'][2000,0]=6.867
        with self.assertRaises((ValueError,AssertionError)):a.validate_arrays(row,z)

    def test_validated_trace_requires_actual_hash_and_readonly_input(self):
        a=self.api();z=self.trace()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'trace.npz';np.savez_compressed(p,**z);digest=a.sha(p)
            row=dict(self.row(z),trace_path=str(p),trace_sha256=digest)
            with a.verified_trace(row) as data:self.assertEqual(data['actions'].shape,(375,4))
            self.assertEqual(a.sha(p),digest)
            row['trace_sha256']='wrong'
            with self.assertRaises(ValueError):
                with a.verified_trace(row):pass

    def test_paired_comparison_requires_identical_target_ids(self):
        a=self.api()
        def item(i,condition,final):return dict(index=i,target_id='target'+str(i),condition=condition,controller='scripted',
            success=condition=='constant_medium',final_distance_m=final,final_speed_m_s=.2,peak_speed_m_s=.5,
            maximum_distance_m=1.,final_yaw_error_rad=0.,phases={},post_gust=None)
        r=a.paired_metrics([item(0,'constant_medium',.05),item(0,'gust_medium',.2)],'scripted','constant_medium','scripted','gust_medium')
        self.assertAlmostEqual(r['final_distance_delta_m']['mean'],.15)
        self.assertEqual(r['first_only_success'],1)
        with self.assertRaises(ValueError):a.paired_metrics([item(0,'constant_medium',.05),item(1,'gust_medium',.2)],'scripted','constant_medium','scripted','gust_medium')

    def plots_api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_gust_recovery_plots'),'read-only plots missing')
        return importlib.import_module('uav_gust_recovery_plots')

    def test_plot_contract_excludes_unrecorded_pi_and_keeps_five_figures(self):
        p=self.plots_api();names=p.figure_names(self.api().signal_availability(self.trace()))
        self.assertEqual(len(names),5)
        self.assertNotIn('pi_integral_diagnostics.png',names)
        self.assertIn('force_vs_time.png',names);self.assertIn('termination_timeline.png',names)

    def test_plot_alignment_never_extends_successful_episode(self):
        p=self.plots_api();z=self.trace(5.)
        series=p.aligned_series(z,'command_projection',[1.,0.,0.])
        self.assertEqual(series.shape,(376,));self.assertAlmostEqual(series[124],-.15)
        self.assertTrue(np.isnan(series[125:]).all())
        distance=p.aligned_series(z,'distance',[1.,0.,0.])
        self.assertAlmostEqual(distance[125],.5);self.assertTrue(np.isnan(distance[126:]).all())

    def test_plot_representatives_use_manifest_indices_not_acquisition_row_fields(self):
        p=self.plots_api()
        result={'episode_metrics':[dict(condition='gust_medium',controller='scripted',index=i,task_key=str(i)) for i in [33,0,66]]}
        self.assertEqual([r['index'] for r in p.cohort(result,'gust_medium','scripted')],[0,33,66])
        raw={'0':{'task_key':'0','condition':'gust_medium','controller':'scripted'}}
        self.assertEqual(raw[p.cohort(result,'gust_medium','scripted')[0]['task_key']]['task_key'],'0')

    def test_initial_condition_pairing_checks_physical_and_controller_fields(self):
        a=self.api()
        base=dict(index=0,target_id='t0',env_seed=3,target=[1,0,1],direction_world=[1,0,0],
            initial_state=dict(snapshot_sha256='varies-with-condition-config',qpos=[0,0,1],qvel=[0]*6,pi_integral=[0]*3,previous_action=[0]*4))
        states=[dict(base,condition=c) for c in a.CONDITIONS]
        self.assertEqual(a.validate_initial_pairs(states),1)
        states[-1]=dict(states[-1],initial_state=dict(base['initial_state'],pi_integral=[1,0,0]))
        with self.assertRaises(ValueError):a.validate_initial_pairs(states)

    def test_resources_use_completed_publication_peak_not_early_self_sample(self):
        p=self.plots_api()
        def record(peak):return dict(exit_code=0,failure=None,oom_count_delta=0,process_tree_peak_hwm_bytes=peak)
        self.assertEqual(p.completed_resource_peak([record(80),record(160)],100),160)
        bad=dict(record(200),exit_code=1)
        with self.assertRaises(ValueError):p.completed_resource_peak([bad],100)

if __name__=='__main__':unittest.main()
