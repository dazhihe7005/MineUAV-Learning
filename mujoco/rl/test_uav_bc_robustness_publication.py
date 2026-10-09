"""Publication must preserve all outcomes and select only real representative traces."""
import importlib
import importlib.util
import unittest
import numpy as np


class PublicationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('run_uav_bc_robustness'),'publication implementation missing')
        return importlib.import_module('run_uav_bc_robustness')

    def test_truthful_representatives_include_only_existing_cohorts(self):
        api=self.api()
        rows=[dict(controller='bc',condition='combined_large',key='a',success=True),
              dict(controller='scripted',condition='combined_large',key='a',success=True)]
        r=api.representative_pairs(rows)
        self.assertEqual(r['success'][0]['key'],'a');self.assertIsNone(r['bc_gap'])
        rows.extend([dict(controller='bc',condition='yaw_large',key='b',success=False),
            dict(controller='scripted',condition='yaw_large',key='b',success=True)])
        r=api.representative_pairs(rows);self.assertEqual(r['bc_gap'][0]['key'],'b')

    def test_aggregate_preserves_failing_cohort_and_direction_counts(self):
        api=self.api()
        state=dict(key='combined_large/benchmark/000',condition='combined_large',split='benchmark',index=0,
            perturbation=dict(position_m=[.3,0,0],velocity_m_s=[0,.4,0],yaw_rad=.5))
        common=dict(key=state['key'],condition='combined_large',split='benchmark',target_id=0,env_seed=1,
            steps=2,simulated_seconds=.08,final_distance_m=.3,near_steps=0,near_actual_sum=0,near_command_sum=0,
            action_saturation_elements=0,crossings=0,initial_distance_m=1.,distance_reduction_m=.7,
            max_position_excursion_m=1.,maximum_speed_m_s=.4,final_abs_yaw_error_rad=.2,maximum_abs_yaw_error_rad=.5,
            final_speed_m_s=.1,velocity_settled_below_015_s=None,yaw_settled_below_2deg_s=None,reached_target_region=False)
        rows=[dict(common,task_key=state['key']+'/scripted',controller='scripted',success=True,
                   termination_reason='success',completion_time_s=.08),
              dict(common,task_key=state['key']+'/bc',controller='bc',success=False,
                   termination_reason='time_limit',completion_time_s=None)]
        result=api.aggregate(rows,[state])
        group=result['combined_large']['benchmark']
        self.assertEqual(group['bc']['episodes'],1);self.assertEqual(group['bc']['timeouts'],1)
        self.assertIsNone(group['bc']['completion_time_s'])
        self.assertEqual(group['paired']['discordance']['scripted_only_success'],1)
        self.assertEqual(group['direction_groups']['yaw_sign']['positive']['bc_successes'],0)

    def test_nominal_reproduction_checks_actual_action_observation_arrays(self):
        api=self.api()
        import tempfile
        from pathlib import Path
        from uav_bc_safety import atomic_npz
        from latent_dynamics_data import file_hash
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/'a.npz';b=Path(d)/'b.npz'
            arrays=dict(observations=np.zeros((2,7),np.float32),actions=np.zeros((1,4),np.float32))
            atomic_npz(a,**arrays);atomic_npz(b,**arrays)
            new=dict(trace_path=str(a),trace_sha256=file_hash(a));old=dict(path=str(b),sha256=file_hash(b))
            api.assert_nominal_trace(new,old)
            arrays['actions'][0,0]=.1;atomic_npz(b,**arrays);old['sha256']=file_hash(b)
            with self.assertRaises(AssertionError):api.assert_nominal_trace(new,old)


if __name__=='__main__':unittest.main()
