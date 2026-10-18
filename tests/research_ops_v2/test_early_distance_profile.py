import copy,importlib.util,tempfile,unittest
from pathlib import Path
from research_ops.storage import OpsError,digest
class EarlyDistanceProfileTests(unittest.TestCase):
 def api(self):
  name='research_regression.authorized_rl.early_distance_profile'
  self.assertIsNotNone(importlib.util.find_spec(name),'missing reviewed early-distance profile')
  return __import__(name,fromlist=['early_mix_experiment'])
 def test_profile_rejects_algorithm_budget_task_and_distribution_changes(self):
  m=self.api();p=m.early_mix_experiment();m.validate_early_mix_profile(p)
  for key,value in (('reward_version','other'),('gamma',.995),('timesteps',65537),('seed',1),
   ('workers',2),('held_goal_manifest_sha256','a'*64),('initialization','CHECKPOINT'),('early_original_probability_per_reset',.6)):
   q=copy.deepcopy(p);q[key]=value
   with self.subTest(key=key),self.assertRaises(OpsError):m.validate_early_mix_profile(q)
  q=copy.deepcopy(p);q.update(source_identity_sha256='a'*64,registration_sha256='b'*64);m.validate_early_mix_profile(q)
 def test_completed_control_proof_is_real_and_missing_evidence_blocks(self):
  m=self.api();p=m.early_mix_experiment()
  self.assertEqual(m.verify_early_mix_proof(p)['status'],'PASS')
  with tempfile.TemporaryDirectory() as directory,self.assertRaises(OpsError):m.verify_early_mix_proof(p,Path(directory))
 def test_new_profile_is_accepted_by_budget_but_cannot_reuse_attempt(self):
  m=self.api()
  from research_regression.authorized_rl.budget import BudgetStore,grant
  p=m.early_mix_experiment();p.update(source_identity_sha256='a'*64,registration_sha256='b'*64)
  with tempfile.TemporaryDirectory() as directory:
   s=BudgetStore(Path(directory)/'budget');s.initialize(expected_grant_sha256=digest(grant()));s.reserve(p)
   self.assertEqual(s.status()['remaining_timesteps'],1934464);self.assertEqual(s.status()['remaining_wall_seconds'],82800)
   with self.assertRaises(OpsError):s.reserve(p)
 def test_execution_must_equal_frozen_preregistration_including_source_identity(self):
  m=self.api();self.assertTrue(hasattr(m,'require_preregistered'),'missing frozen R33 preregistration binding')
  p=m.early_mix_experiment();p.update(source_identity_sha256='a'*64,registration_sha256='b'*64)
  m.require_preregistered(p,copy.deepcopy(p))
  for key,value in (('timesteps',65535),('source_identity_sha256','c'*64),('experiment_id','ARL_OLD_ID')):
   declared=copy.deepcopy(p);declared[key]=value
   with self.subTest(key=key),self.assertRaises(OpsError):m.require_preregistered(p,declared)
if __name__=='__main__':unittest.main()
