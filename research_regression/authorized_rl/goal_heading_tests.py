"""One controlled reset/step interface test, NOT a scientific episode."""
import importlib,importlib.util,math,unittest
from pathlib import Path
import numpy as np
from lidar_integration.navigation_env import LidarNavigationEnv
from lidar_integration.ppo_preflight import HeadingObservation
from lidar_integration.ppo_scenarios import build_ppo_scene
from research_ops.storage import read_json

class GoalHeadingTests(unittest.TestCase):
    def test_goal_changes_state_yaw_only_without_extra_scan_or_physics_reset(self):
        name='research_regression.authorized_rl.goal_heading_trace'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing explicit heading-only wrapper')
        cls=importlib.import_module(name).HoldInitialHeading
        root=Path(__file__).resolve().parents[2];p=read_json(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        row=next(v for v in p['scenario_manifest']['validation'] if v['layout']=='empty')
        env=cls(HeadingObservation(LidarNavigationEnv('empty',row['scene_seed'],root/'.research_local/dependencies/MuJoCo-LiDAR',scene=build_ppo_scene(row))))
        try:
            o,i=env.reset(seed=row['scene_seed'],options={'start_seed':row['start_seed'],'target_seed':row['target_seed'],
                'start_position':row['start_xyz'],'target_position':row['target_xyz'],'start_yaw_rad':row['start_yaw_rad']})
            self.assertAlmostEqual(env.unwrapped.data.time,0.);self.assertEqual(env.unwrapped.lidar_frame_count,1)
            np.testing.assert_array_equal(o['state'][:3],np.asarray(np.array(row['target_xyz'])-row['start_xyz'],dtype=np.float32))
            np.testing.assert_array_equal(o['state'][3:6],np.zeros(3,dtype=np.float32));self.assertAlmostEqual(float(o['state'][6]),0.)
            self.assertAlmostEqual(i['target_yaw_rad'],row['start_yaw_rad']);self.assertEqual(i['goal_yaw_mode'],'HOLD_INITIAL_ENU_YAW')
            np.testing.assert_allclose(o['heading'],[math.sin(row['start_yaw_rad']),math.cos(row['start_yaw_rad'])],atol=1e-7)
            np.testing.assert_array_equal(env.unwrapped.velocity_controller.integral_error,np.zeros(3))
            source=env.unwrapped.last_frame.frame_timestamp
            o,r,t,tr,i=env.step(np.zeros(4,dtype=np.float32))
            self.assertFalse(t or tr);self.assertEqual(i['physics_steps'],20);self.assertAlmostEqual(env.unwrapped.data.time,.04)
            self.assertEqual(env.unwrapped.last_frame.frame_timestamp,source);self.assertEqual(env.unwrapped.lidar_frame_count,1)
            self.assertAlmostEqual(i['target_yaw_rad'],row['start_yaw_rad']);self.assertTrue(np.isfinite(o['state']).all())
        finally:env.close()

if __name__=='__main__':unittest.main()
