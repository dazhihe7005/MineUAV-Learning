"""Publication refuses incomplete science, unsafe phases or large raw arrays."""
import importlib,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np

class ExternalPublicationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_uav_bc_external_disturbance'),'disturbance publication missing')
        return importlib.import_module('run_uav_bc_external_disturbance')
    def test_resources_reject_incomplete_or_oom_phases(self):
        a=self.api();good={n:dict(exit_code=0,failure=None,oom_count_delta=0,workers=1,
            process_tree_peak_sampled_rss_bytes=1000000,process_tree_peak_hwm_bytes=1100000,
            minimum_available_bytes=8000000000) for n in ['baseline','related','smoke','evaluation','resume','verification','tests']}
        self.assertFalse(a.resource_summary(good)['oom_occurred'])
        bad={k:dict(v) for k,v in good.items()};bad['evaluation']['oom_count_delta']=1
        with self.assertRaises(ValueError):a.resource_summary(bad)
        bad={k:dict(v) for k,v in good.items()};bad.pop('evaluation')
        with self.assertRaises(ValueError):a.resource_summary(bad)
    def test_report_rows_omit_raw_arrays_and_local_paths(self):
        a=self.api();r=dict(key='constant_high/final/000',success=False,trace_path='/local/raw.npz',
            trace_sha256='hash',identity={'huge':'repeat'},simulated_seconds=15.,completion_time_s=None)
        p=a.public_rows([r]);self.assertEqual(p[0]['trace_sha256'],'hash')
        self.assertNotIn('trace_path',p[0]);self.assertNotIn('identity',p[0]);self.assertIsNone(p[0]['completion_time_s'])
    def test_publication_retains_one_copy_of_acquisition_and_processing_hashes(self):
        a=self.api();self.assertTrue(hasattr(a,'publication_provenance'))
        from latent_dynamics_data import file_hash
        root=Path(__file__).resolve().parents[2]
        identity=dict(source_sha256={'uav_bc_external_evaluation.py':file_hash(root/'mujoco/rl/uav_bc_external_evaluation.py')},manifest_sha256='manifest')
        result=a.publication_provenance(root,dict(identity=identity))
        self.assertEqual(result['acquisition_identity'],identity)
        self.assertEqual(result['source_revisions'],{})
        for name in ['run_uav_bc_external_disturbance.py','uav_bc_external_plots.py','uav_bc_yaw_data.py']:
            self.assertEqual(result['publication_source_sha256'][name],file_hash(root/'mujoco/rl'/name))
    def test_paired_failure_not_named_as_expert_defect_when_all_fail(self):
        a=self.api();metrics={}
        for c in ['nominal','constant_low','constant_medium','constant_high','gust_medium','gust_high']:
            metrics[c]={n:dict(successes=100 if c=='nominal' else 0) for n in ['scripted','original','yaw_augmented']}
            metrics[c]['paired']={'yaw_augmented_minus_scripted':dict(discordance=dict(first_only_success=0,second_only_success=0,both_failed=100 if c!='nominal' else 0,both_success=100 if c=='nominal' else 0))}
        result=a.interpret(metrics)
        self.assertIn('constant_high',result['shared_failed_conditions'])
        self.assertEqual(result['expert_only_success_count']['constant_high'],0)
        self.assertEqual(len(result['next_step']),1)
    def test_three_way_shared_failures_require_same_episode(self):
        a=self.api();self.assertTrue(hasattr(a,'triple_outcomes'),'three-way paired outcomes missing')
        rows=[dict(condition='gust_high',key=str(i),controller=n,success=i!=j)
              for i in range(3) for j,n in enumerate(['scripted','original','yaw_augmented'])]
        result=a.triple_outcomes(rows)['gust_high']
        self.assertEqual(result['all_failed'],0);self.assertEqual(result['all_success'],0)
        for r in rows:
            if r['key']=='0':r['success']=False
        self.assertEqual(a.triple_outcomes(rows)['gust_high']['all_failed'],1)
    def test_eight_real_figures_are_namespaced_and_atomic(self):
        a=self.api();from uav_bc_external_plots import plots
        from uav_bc_safety import atomic_npz
        from uav_bc_external_disturbance import conditions
        s=dict(success_rate=1.,success_wilson_95_ci=[.963,1.],successes=100,
            maximum_speed_m_s=dict(mean=.5),action_saturation_fraction=0.,
            steady_position_error_m=dict(measured_episodes=100,statistics=dict(mean=.05)))
        metrics={c:{n:dict(s) for n in ['scripted','original','yaw_augmented']} for c in conditions()}
        for c in metrics:metrics[c]['paired']={n:dict(success_delta_percentage_points=0.) for n in ['yaw_augmented_minus_original','yaw_augmented_minus_scripted']}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);reports=root/'mujoco/reports';reports.mkdir(parents=True)
            old=reports/'success_vs_external_force.png';old.write_bytes(b'old experiment')
            raw=reports/'raw.npz';atomic_npz(raw,observations=np.array([[1.,0,0,0,0,0,0],[.05,0,0,.1,0,0,0]]),
                positions=np.array([[0.,0,1],[.95,0,1]]),policy_times=np.array([0.,.04]),physical_state_valid=np.ones(2,bool))
            rows=[dict(condition=c,controller=n,key=c+'/final/000',target=[1.,0,1.],trace_path=str(raw),success=True)
                for c in ['constant_low','constant_medium','constant_high','gust_medium','gust_high'] for n in ['scripted','original','yaw_augmented']]
            names=plots(root,dict(metrics=metrics,rows=rows))
            self.assertEqual(len(names),8);self.assertEqual(old.read_bytes(),b'old experiment')
            for name in names:self.assertEqual((reports/name).read_bytes()[:8],b'\x89PNG\r\n\x1a\n')

if __name__=='__main__':unittest.main()
