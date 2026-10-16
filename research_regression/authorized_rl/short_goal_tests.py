"""Actual distance-task resets and physics, with optimizer tripwires."""
import importlib.util
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from .runtime import ROOT
from .short_goal import short_manifest


class ShortGoalEnvironmentTests(unittest.TestCase):
    def setUp(self):
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_SHORT_TASK_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')

    def scheduler(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.short_goal_env'),'missing distance scheduler')
        from .short_goal_env import make_short_scheduler
        env=make_short_scheduler(self.prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
        self.addCleanup(env.close);return env

    def test_reset_task_has_new_identity_same_physics_geometry_and_actual_sensor(self):
        from lidar_integration.ppo_scenarios import build_ppo_scene,path_exists
        for rows in short_manifest(self.prereg).values():
            for row in rows:
                scene=build_ppo_scene(row)
                self.assertEqual(scene.geometry_hash,row['obstacle_configuration_hash'])
                self.assertFalse(scene.overlaps_obstacle(row['target_xyz'],clearance_m=.2))
                self.assertTrue(path_exists(scene,row['start_xyz'],row['target_xyz']))
        e=self.scheduler();b=self.prereg['scenario_manifest']['train'][0]
        obs,info=e.reset(options={'scenario_identity_sha256':b['identity_sha256']});p=e.active.unwrapped
        self.assertAlmostEqual(float(np.linalg.norm(obs['state'][:3])),.20,places=6)
        self.assertEqual(info['base_scenario_identity_sha256'],b['identity_sha256'])
        self.assertNotEqual(info['scenario_identity_sha256'],b['identity_sha256'])
        self.assertEqual(float(p.data.time),0);self.assertEqual(p.last_frame.ray_count,24000)
        np.testing.assert_allclose(p.data.qpos[:3],b['start_xyz'],atol=1e-12)
        np.testing.assert_allclose(obs['heading'],[np.sin(b['start_yaw_rad']),np.cos(b['start_yaw_rad'])],atol=1e-6)

    def test_actual_step_and_repeated_reset_clear_physics_controller_and_lidar(self):
        e=self.scheduler();obs,_=e.reset();p=e.active.unwrapped
        e.step(np.array([.2,-.1,0,.1],dtype=np.float32));self.assertAlmostEqual(float(p.data.time),.04)
        self.assertEqual(e.completed_policy_steps,1);self.assertEqual(p.lidar_frame_count,1)
        obs,info=e.reset();p=e.active.unwrapped
        self.assertEqual(float(p.data.time),0);np.testing.assert_array_equal(p.data.qvel,np.zeros(6))
        self.assertEqual(p.lidar_frame_count,1);self.assertEqual(float(obs['lidar_age'][0]),0)
        np.testing.assert_array_equal(p.velocity_controller.integral_error,np.zeros(3))

    def test_forced_counter_changes_only_next_reset_without_extra_train_rng_draw(self):
        e=self.scheduler();obs,info=e.reset();goal=e.active.unwrapped.target_position.copy()
        # FORCED_TEST_COUNTER, not a claim of 32768 actual training steps.
        e.completed_policy_steps=32767;e.step(np.zeros(4,dtype=np.float32))
        np.testing.assert_array_equal(e.active.unwrapped.target_position,goal)
        rng=copy.deepcopy(e._rng.bit_generator.state)
        base=self.prereg['scenario_manifest']['train'][0]
        obs,info=e.reset(options={'scenario_identity_sha256':base['identity_sha256']})
        self.assertEqual(e._rng.bit_generator.state,rng)
        np.testing.assert_array_equal(e.active.unwrapped.target_position,base['target_xyz'])
        self.assertEqual(info['distance_stage'],'ORIGINAL');self.assertEqual(e.phase_switch_step,32768)
        for split in ('validation','test'):
            with self.assertRaises(ValueError):e.reset(options={'scenario_identity_sha256':self.prereg['scenario_manifest'][split][0]['identity_sha256']})


if __name__=='__main__':unittest.main()
