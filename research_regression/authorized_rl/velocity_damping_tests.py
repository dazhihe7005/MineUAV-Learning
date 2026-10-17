"""R31 feedback units/real single-step interface,not science or learning episodes."""
import copy,importlib,importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_training_runner import _weights_sha
from .policy_function_probe import references
from .policy_function_run import load_model

class VelocityDampingTests(unittest.TestCase):
 def module(self):
  name='research_regression.authorized_rl.velocity_damping_trace'
  self.assertIsNotNone(importlib.util.find_spec(name),'missing explicit R31 physical-unit velocity feedback adapter')
  return importlib.import_module(name)
 def test_feedback_units_sign_clip_and_invalid_inputs(self):
  m=self.module();base=np.array([.2,-.3,.4,0.],np.float32)
  state=np.array([0.,0.,0.,1.,-1.,1.,0.],np.float32)
  np.testing.assert_allclose(m.apply_feedback(base,state),[0.,-.1,.1,0.],atol=1e-7,rtol=0)
  np.testing.assert_array_equal(m.apply_feedback(base,np.zeros(7,np.float32)),base)
  state[3:6]=[-10.,10.,-10.]
  np.testing.assert_array_equal(m.apply_feedback(base,state),[1.,-1.,1.,0.])
  for bad in (np.zeros(8,np.float32),np.full(7,np.nan,np.float32)):
   with self.assertRaises(ValueError):m.apply_feedback(base,bad)
  with self.assertRaises(ValueError):m.apply_feedback(np.array([2.,0.,0.,0.],np.float32),np.zeros(7,np.float32))
 def test_actual_model_and_physics_step_use_current_world_velocity_without_weight_edits(self):
  cls=self.module().DampedTranslationPolicy;root=Path(__file__).resolve().parents[2]
  def forbidden(*args,**kwargs):raise AssertionError('R31_TEST_NO_TRAINING')
  with patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
   trusted=references();p=load_l6_prereg(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json');model=load_model(trusted['models'][-1],p,trusted)
   from lidar_integration.navigation_env import LidarNavigationEnv
   from lidar_integration.ppo_preflight import HeadingObservation
   from lidar_integration.ppo_scenarios import build_ppo_scene
   from .goal_heading_trace import HoldInitialHeading
   from .velocity_damping_diagnostic import references as diagnostic_inputs
   task=diagnostic_inputs()['cases'][0];row=task['scenario']
   env=HoldInitialHeading(HeadingObservation(LidarNavigationEnv('empty',row['scene_seed'],root/'.research_local/dependencies/MuJoCo-LiDAR',scene=build_ppo_scene(row))))
   before=_weights_sha(model.policy)
   try:
    obs,_=env.reset(seed=row['scene_seed'],options={'start_seed':row['start_seed'],'target_seed':row['target_seed'],'start_position':row['start_xyz'],'target_position':row['target_xyz'],'start_yaw_rad':row['start_yaw_rad']})
    original=copy.deepcopy(obs)
    with torch.inference_mode():
     raw,_=model.predict(obs,deterministic=True);action,_=cls(model).predict(obs,deterministic=True)
    # Reset has true zero velocity;the additional feedback vanishes exactly.
    np.testing.assert_array_equal(action[:3],raw[:3]);self.assertEqual(float(action[3]),0.)
    _,_,t,tr,info=env.step(action);self.assertFalse(t or tr);self.assertEqual(info['physics_steps'],20)
    self.assertEqual(float(info['yaw_rate_command_rad_s']),0.)
    np.testing.assert_allclose(info['velocity_command_m_s'],action[:3]*np.array([1.5,1.5,1.]),atol=1e-8,rtol=0)
    self.assertEqual(_weights_sha(model.policy),before)
    for key in original:np.testing.assert_array_equal(obs[key],original[key])
   finally:env.close()

if __name__=='__main__':unittest.main()
