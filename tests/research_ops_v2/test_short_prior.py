"""Pure evidence contract tests; SYNTHETIC, never experimental outcomes."""
import copy
import importlib
import importlib.util
import math
import unittest
from research_ops.storage import OpsError


class ShortPriorContractTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.short_prior'),'missing short-prior evidence contract')
        return importlib.import_module('research_regression.authorized_rl.short_prior')

    def example(self):
        # Literal synthetic observations obey 5x20 ticks, .04s, .08m/.10m/s.
        tasks=[{'identity_sha256':str(i)*64,'base_identity_sha256':str(i+20)*32,
                'start_xyz':[0.,0.,2.],'target_xyz':[.2,0.,2.]} for i in range(12)]
        rows=[{'task_identity_sha256':t['identity_sha256'],'base_identity_sha256':t['base_identity_sha256'],
               'terminal_class':'success','raw_termination_reason':'success','start_xyz':t['start_xyz'],
               'target_xyz':t['target_xyz'],'steps':5,'trace':[{'sim_time':j*.04,'distance_m':.08,
               'speed_m_s':.1,'position':[.12,0.,2.],'velocity':[.1,0.,0.],
               'physics_ticks':20,'action':[0.,0.,0.,0.],'lidar_age_s':.04} for j in range(1,6)]} for t in tasks]
        return {'status':'PASS','experiment_id':'AUTORL_R9_SHORT_PRIOR_FEASIBILITY_ONLY','rows':rows,
                'optimizer_updates':0,'training_timesteps':0,'validation_episodes':0,'test_episodes':0,'physics_ticks':1200}, {'train':tasks}

    def test_contract_accepts_complete_observed_stop(self):
        self.module().validate_short_prior_result(*self.example())

    def test_rejects_claimed_success_without_five_valid_samples(self):
        module=self.module();result,manifest=self.example()
        result['rows'][0]['trace'][-1]['speed_m_s']=.16
        with self.assertRaises(OpsError):module.validate_short_prior_result(result,manifest)

    def test_rejects_mislabelled_identity_or_missing_coverage(self):
        module=self.module()
        for mutation in ('identity','coverage'):
            result,manifest=self.example()
            if mutation=='identity':result['rows'][0]['task_identity_sha256']='a'*64
            else:result['rows'].pop()
            with self.assertRaises(OpsError):module.validate_short_prior_result(result,manifest)

    def test_rejects_invented_time_or_physics_or_nonfinite(self):
        module=self.module()
        for key,value in [('sim_time',.201),('physics_ticks',19),('distance_m',float('nan'))]:
            result,manifest=self.example();result['rows'][0]['trace'][-1][key]=value
            with self.assertRaises(OpsError):module.validate_short_prior_result(result,manifest)

    def test_rejects_training_or_test_data_as_feasibility(self):
        module=self.module()
        for key in ('optimizer_updates','training_timesteps','test_episodes','validation_episodes'):
            result,manifest=self.example();result[key]=1
            with self.assertRaises(OpsError):module.validate_short_prior_result(result,manifest)

    def test_protocol_freezes_train_only_original_bounds(self):
        module=self.module()
        spec=module.short_prior_spec()
        self.assertEqual(spec['experiment_id'],'AUTORL_R9_SHORT_PRIOR_FEASIBILITY_ONLY')
        self.assertEqual((spec['train_episodes'],spec['optimizer_updates'],spec['training_timesteps']),(12,0,0))
        self.assertEqual(spec['max_policy_steps_per_episode'],375)
        self.assertEqual(spec['success'],{'distance_m_strictly_below':.1,'speed_m_s_strictly_below':.15,'consecutive_policy_steps':5})
        self.assertEqual(len(spec['short_manifest']['train']),12)
        self.assertNotIn('test',spec['short_manifest'])

    def test_only_ordinary_terminal_results_can_continue_coverage(self):
        module=self.module()
        self.assertTrue(hasattr(module,'require_safe_coverage_terminal'),'missing R9 terminal safety boundary')
        for kind in ('success','collision','time_limit'):
            module.require_safe_coverage_terminal(kind)

    def test_physical_sensor_and_unknown_outcome_stop_coverage(self):
        module=self.module()
        self.assertTrue(hasattr(module,'require_safe_coverage_terminal'),'missing R9 terminal safety boundary')
        for kind in ('physical_failure','sensor_failure','mystery',None):
            with self.assertRaises(OpsError):module.require_safe_coverage_terminal(kind)


if __name__=='__main__':unittest.main()
