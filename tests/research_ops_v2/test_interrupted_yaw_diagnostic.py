"""R26 fixed read-only diagnostic contracts; no model/simulator imports."""
import copy
import importlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from research_ops.storage import OpsError


class InterruptedYawDiagnosticTests(unittest.TestCase):
    def module(self):
        name='research_regression.authorized_rl.interrupted_yaw_diagnostic'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing fixed R26 diagnostic contract')
        return importlib.import_module(name)

    def test_fixed_scope_is_eight_new_diagnostics_not_failed_training_resume(self):
        c=self.module().configuration()
        self.assertEqual((c['model_count'],c['case_count'],c['episodes'],c['max_policy_steps_per_episode']),(2,4,8,375))
        self.assertEqual((c['training_timesteps'],c['optimizer_updates'],c['test_episodes']),(0,0,0))
        self.assertEqual(c['checkpoint_steps'],[0,32768])
        self.assertFalse(c['formal_completion_claim'])

    def test_missing_released_failure_or_repair_proof_cannot_create_output(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):self.module().references(Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_missing_current_admission_refuses_before_model_or_output(self):
        name='research_regression.authorized_rl.interrupted_yaw_diagnostic_run'
        self.assertIsNotNone(importlib.util.find_spec(name))
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(OpsError):importlib.import_module(name).verify_checks({'source_identity':{}},Path(d))
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_actual_fixed_references_only_empty_validation_not_test(self):
        r=self.module().references(Path(__file__).resolve().parents[2])
        self.assertEqual([m['step'] for m in r['models']],[0,32768])
        self.assertEqual(len(r['cases']),4)
        self.assertEqual({v['distance_stage'] for v in r['cases']},{'SHORT','ORIGINAL'})
        self.assertTrue(all(v['scenario']['layout']=='empty' and v['scenario']['split']=='validation' for v in r['cases']))
        self.assertEqual(len({v['scenario']['identity_sha256'] for v in r['cases']}),4)

    def fixture(self):
        spec={'models':[{'step':0},{'step':32768}],
              'cases':[{'distance_stage':s,'scenario':{'identity_sha256':s+str(i)}} for s in ('SHORT','ORIGINAL') for i in range(2)]}
        rows=[]
        for model in spec['models']:
            for case in spec['cases']:
                rows.append({'checkpoint_step':model['step'],'distance_stage':case['distance_stage'],
                    'scenario_identity_sha256':case['scenario']['identity_sha256'],'outcome':'success','policy_steps':25,
                    'physics_ticks':500,'optimizer_unchanged':True,'training_timesteps_before_after':model['step'],
                    'policy_sha256_before_after':'a'*64,'stop_metrics':{'max_consecutive_joint_samples':5}})
        return spec,rows

    def test_paired_short_retention_loss_not_claimed_as_formal_final_result(self):
        spec,rows=self.fixture();m=self.module()
        self.assertEqual(m.evaluate_retention(rows,spec)['short_retention_loss'],'NOT_SUPPORTED_IN_THIS_SUBSET')
        rows[4].update(outcome='time_limit',policy_steps=375,physics_ticks=7500,stop_metrics={'max_consecutive_joint_samples':0})
        self.assertEqual(m.evaluate_retention(rows,spec)['short_retention_loss'],'SUPPORTED_IN_THIS_SUBSET')
        rows[0].update(outcome='time_limit',policy_steps=375,physics_ticks=7500,stop_metrics={'max_consecutive_joint_samples':0})
        self.assertEqual(m.evaluate_retention(rows,spec)['short_retention_loss'],'NOT_TESTED_PRIOR_NOT_ESTABLISHED')

    def test_missing_duplicate_wrong_case_or_training_change_fails_closed(self):
        spec,rows=self.fixture();fn=self.module().evaluate_retention
        for bad in (rows[:-1],rows+[rows[-1]],rows[:1]+rows[:1]+rows[2:]):
            with self.assertRaises(ValueError):fn(bad,spec)
        for key,value in (('scenario_identity_sha256','foreign'),('optimizer_unchanged',False),
                          ('training_timesteps_before_after',65536),('policy_steps',True),('physics_ticks',501),
                          ('physics_ticks',480),('training_timesteps_before_after',False),('checkpoint_step',False),('outcome','physical_failure')):
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):fn(bad,spec)

    def test_false_success_and_mixed_model_fingerprint_rejected(self):
        spec,rows=self.fixture();fn=self.module().evaluate_retention
        bad=copy.deepcopy(rows);bad[0]['stop_metrics']['max_consecutive_joint_samples']=4
        with self.assertRaises(ValueError):fn(bad,spec)
        bad=copy.deepcopy(rows);bad[1]['policy_sha256_before_after']='b'*64
        with self.assertRaises(ValueError):fn(bad,spec)
