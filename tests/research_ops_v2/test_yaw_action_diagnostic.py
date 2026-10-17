"""No tensor import: new no-learning action ablation is not a task-goal change."""
import importlib,importlib.util,copy,unittest

class YawActionContractTests(unittest.TestCase):
 def module(self):
  name='research_regression.authorized_rl.yaw_action_diagnostic'
  self.assertIsNotNone(importlib.util.find_spec(name),'missing explicit R30 zero-yaw-action diagnostic')
  return importlib.import_module(name)
 def test_budget_and_ablation_do_not_change_task_goal_or_authorize_learning(self):
  p=self.module().configuration()
  self.assertEqual(p['new_episodes'],2);self.assertEqual(p['max_policy_steps_per_episode'],375)
  self.assertEqual(p['ablation'],'APPLIED_YAW_RATE_ACTION_ZERO_TRANSLATION_UNCHANGED_FOR_SAME_OBSERVATION')
  self.assertEqual(p['goal_yaw_mode'],'HOLD_INITIAL_ENU_YAW')
  self.assertEqual((p['training_timesteps'],p['optimizer_updates'],p['test_episodes']),(0,0,0))
 def test_exact_final_model_two_actual_timeout_controls_not_test_scenes(self):
  s=self.module().references()
  self.assertEqual(s['model']['step'],65536);self.assertEqual(len(s['cases']),2)
  self.assertTrue(all(t['scenario']['layout']=='empty' and t['scenario']['split']=='validation' for t in s['cases']))
  self.assertTrue(all(v['terminal_class']=='time_limit' and v['policy_steps']==375 for v in s['controls']))
  self.assertEqual(s['model']['checkpoint_sha256'],'0bcbdca60b131b749ed3a3b0101fbd149e47b58244f99e13e92c0916da5dbd0a')
 def test_incomplete_or_wrong_task_cannot_support_yaw_contrast(self):
  m=self.module();spec=m.references()
  with self.assertRaises(ValueError):m.evaluate([],spec)
  rows=[{'task_identity_sha256':t['identity_sha256'],'outcome':'success','policy_steps':100,'physics_ticks':2000,
   'stop_metrics':{'max_consecutive_joint_samples':5},'optimizer_unchanged':True,
   'policy_sha256_before_after':'f7349c40158b1f064542bdf8180f22cf9b68f11aee9ab563f732e678bfc85475',
   'training_timesteps_before_after':65536,'applied_yaw_zero_verified':True,'translation_unchanged_given_observation':True}
   for t in spec['cases']]
  self.assertEqual(m.evaluate(rows,spec)['yaw_action_contrast'],'PASS')
  wrong=copy.deepcopy(rows);wrong[1]['task_identity_sha256']=wrong[0]['task_identity_sha256']
  with self.assertRaises(ValueError):m.evaluate(wrong,spec)
 def test_stopping_samples_and_real_tick_counts_are_required(self):
  m=self.module();spec=m.references()
  row={'task_identity_sha256':spec['cases'][0]['identity_sha256'],'outcome':'success','policy_steps':100,'physics_ticks':2000,
   'stop_metrics':{'max_consecutive_joint_samples':4},'optimizer_unchanged':True,'policy_sha256_before_after':'f7349c40158b1f064542bdf8180f22cf9b68f11aee9ab563f732e678bfc85475',
   'training_timesteps_before_after':65536,'applied_yaw_zero_verified':True,'translation_unchanged_given_observation':True}
  rows=[row,{**row,'task_identity_sha256':spec['cases'][1]['identity_sha256']}]
  with self.assertRaises(ValueError):m.evaluate(rows,spec)
  rows[0]['stop_metrics']={'max_consecutive_joint_samples':5};rows[1]['stop_metrics']={'max_consecutive_joint_samples':5}
  rows[1]['physics_ticks']=True
  with self.assertRaises(ValueError):m.evaluate(rows,spec)

if __name__=='__main__':unittest.main()
