"""R30 no-learning ablation interface tests, not the two diagnostic episodes."""
import importlib,importlib.util,unittest,copy
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_training_runner import _weights_sha
from .policy_function_probe import references
from .policy_function_run import load_model

class YawActionTests(unittest.TestCase):
 def module(self):
  name='research_regression.authorized_rl.yaw_action_trace'
  self.assertIsNotNone(importlib.util.find_spec(name),'missing R30 explicit yaw-only action adapter')
  return importlib.import_module(name)
 def test_adapter_changes_only_yaw_and_rejects_stochastic_or_invalid_actions(self):
  cls=self.module().ZeroAppliedYaw
  class Source:
   policy=object();num_timesteps=65536
   def predict(self,obs,deterministic):return np.array([.2,-.3,.4,.7],np.float32),None
  source=Source();adapter=cls(source);action,_=adapter.predict({},deterministic=True)
  np.testing.assert_array_equal(action,np.array([.2,-.3,.4,0.],np.float32));self.assertIs(adapter.policy,source.policy)
  self.assertEqual(adapter.num_timesteps,65536)
  with self.assertRaises(ValueError):adapter.predict({},deterministic=False)
  source.predict=lambda *args,**kwargs:(np.array([0,0,np.nan,0],np.float32),None)
  with self.assertRaises(ValueError):adapter.predict({},deterministic=True)
 def test_actual_final_policy_readonly_inference_preserves_translation_and_weights(self):
  cls=self.module().ZeroAppliedYaw;root=Path(__file__).resolve().parents[2]
  def forbidden(*args,**kwargs):raise AssertionError('R30_TEST_NO_OPTIMIZER')
  with patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
   trusted=references();p=load_l6_prereg(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
   model=load_model(trusted['models'][-1],p,trusted);before=_weights_sha(model.policy)
   obs={k:v[0].copy() for k,v in model._last_obs.items()};original=copy.deepcopy(obs)
   with torch.inference_mode():
    expected,_=model.predict(obs,deterministic=True);action,_=cls(model).predict(obs,deterministic=True)
   np.testing.assert_array_equal(action[:3],expected[:3]);self.assertEqual(float(action[3]),0.)
   self.assertEqual(_weights_sha(model.policy),before)
   for k in obs:np.testing.assert_array_equal(obs[k],original[k])
   from lidar_integration.navigation_env import LidarNavigationEnv
   from lidar_integration.ppo_preflight import HeadingObservation
   from lidar_integration.ppo_scenarios import build_ppo_scene
   from .goal_heading_trace import HoldInitialHeading
   from .yaw_action_diagnostic import references as ablation_inputs
   task=ablation_inputs()['cases'][0];row=task['scenario']
   env=HoldInitialHeading(HeadingObservation(LidarNavigationEnv('empty',row['scene_seed'],root/'.research_local/dependencies/MuJoCo-LiDAR',scene=build_ppo_scene(row))))
   try:
    obs,_=env.reset(seed=row['scene_seed'],options={'start_seed':row['start_seed'],'target_seed':row['target_seed'],'start_position':row['start_xyz'],'target_position':row['target_xyz'],'start_yaw_rad':row['start_yaw_rad']})
    with torch.inference_mode():action,_=cls(model).predict(obs,deterministic=True)
    _,_,t,tr,info=env.step(action)
    self.assertFalse(t or tr);self.assertEqual(info['physics_steps'],20)
    self.assertEqual(float(info['yaw_rate_command_rad_s']),0.)
    np.testing.assert_allclose(info['velocity_command_m_s'],action[:3]*np.array([1.5,1.5,1.]),rtol=0,atol=1e-8)
    self.assertAlmostEqual(float(env.unwrapped.velocity_controller.yaw_target),task['goal_yaw_rad'])
   finally:env.close()

if __name__=='__main__':unittest.main()
