"""R31 fixed independent no-learning feedback comparison, not a training gate."""
import copy,importlib,importlib.util,unittest

class VelocityDampingContractTests(unittest.TestCase):
 def module(self):
  name='research_regression.authorized_rl.velocity_damping_diagnostic'
  self.assertIsNotNone(importlib.util.find_spec(name),'missing fixed R31 velocity-feedback diagnostic')
  return importlib.import_module(name)
 def test_fixed_feedback_units_scope_and_zero_learning(self):
  p=self.module().configuration()
  self.assertEqual(p['feedback_gain'],.3);self.assertEqual(p['action_scales'],[1.5,1.5,1.,1.])
  self.assertEqual(p['feedback_frame'],'WORLD_ENU_VELOCITY_FROM_CURRENT_SIM_STATE7')
  self.assertEqual(p['new_episodes'],2)
  self.assertEqual((p['training_timesteps'],p['optimizer_updates'],p['test_episodes']),(0,0,0))
 def test_exact_two_R30_yawzero_timeout_controls_and_same_final_weights(self):
  s=self.module().references()
  self.assertEqual(s['model']['step'],65536);self.assertEqual(len(s['cases']),2)
  self.assertTrue(all(c['outcome']=='time_limit' and c['applied_yaw_zero_verified'] for c in s['controls']))
  self.assertEqual([c['task_identity_sha256'] for c in s['controls']],[t['identity_sha256'] for t in s['cases']])
 def test_actual_stop_count_is_required_for_success(self):
  m=self.module();spec=m.references()
  rows=[{'task_identity_sha256':t['identity_sha256'],'outcome':'success','policy_steps':100,'physics_ticks':2000,
   'stop_metrics':{'max_consecutive_joint_samples':5},'optimizer_unchanged':True,
   'policy_sha256_before_after':'f7349c40158b1f064542bdf8180f22cf9b68f11aee9ab563f732e678bfc85475',
   'training_timesteps_before_after':65536,'applied_feedback_verified':True,'applied_yaw_zero_verified':True}
   for t in spec['cases']]
  self.assertEqual(m.evaluate(rows,spec)['velocity_feedback_contrast'],'PASS')
  rows[0]['stop_metrics']={'max_consecutive_joint_samples':4}
  with self.assertRaises(ValueError):m.evaluate(rows,spec)
 def test_wrong_case_and_unverified_feedback_failclosed(self):
  m=self.module();spec=m.references()
  with self.assertRaises(ValueError):m.evaluate([],spec)
  rows=[{'task_identity_sha256':t['identity_sha256'],'outcome':'time_limit','policy_steps':375,'physics_ticks':7500,
   'stop_metrics':{'max_consecutive_joint_samples':0},'optimizer_unchanged':True,
   'policy_sha256_before_after':'f7349c40158b1f064542bdf8180f22cf9b68f11aee9ab563f732e678bfc85475',
   'training_timesteps_before_after':65536,'applied_feedback_verified':True,'applied_yaw_zero_verified':True}
   for t in spec['cases']]
  self.assertEqual(m.evaluate(rows,spec)['velocity_feedback_contrast'],'FAIL')
  wrong=copy.deepcopy(rows);wrong[1]['task_identity_sha256']=wrong[0]['task_identity_sha256']
  with self.assertRaises(ValueError):m.evaluate(wrong,spec)
  rows[0]['applied_feedback_verified']=False
  with self.assertRaises(ValueError):m.evaluate(rows,spec)

if __name__=='__main__':unittest.main()
