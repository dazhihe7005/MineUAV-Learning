"""Physical-state perturbations; catches observation-only or inconsistent resets."""
import importlib
import importlib.util
import unittest
from pathlib import Path
import numpy as np
from ppo_pi_env import MineUAVPIEnv
from decision_fidelity_snapshot import capture, restore, fingerprint


class PhysicalRobustnessTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_bc_robustness'), 'physical robustness implementation missing')
        return importlib.import_module('uav_bc_robustness')

    def setUp(self):
        self.env = MineUAVPIEnv(reward_version='v2', max_episode_seconds=.08)
        self.target = dict(target_id=0, env_seed=31, target=[1., -.8, 1.2])

    def tearDown(self):
        self.env.close()

    def test_real_position_velocity_yaw_combination(self):
        api=self.api()
        p=dict(position_m=[.1, -.2, .1], velocity_m_s=[.2, .3, -.1], yaw_rad=np.pi/6)
        snap,meta=api.prepare_snapshot(self.env,self.target,p)
        np.testing.assert_allclose(self.env.data.qpos[:3],[.1,-.2,1.1],atol=1e-15)
        np.testing.assert_allclose(self.env.data.qvel[:3],[.2,.3,-.1],atol=1e-15)
        np.testing.assert_allclose(self.env.data.qpos[3:7],[np.cos(np.pi/12),0,0,np.sin(np.pi/12)],atol=1e-15)
        np.testing.assert_allclose(self.env._get_obs(),[.9,-.6,.1,.2,.3,-.1,-np.pi/6],atol=1e-7)
        self.assertAlmostEqual(np.linalg.norm(self.env.data.qpos[3:7]),1)
        self.assertAlmostEqual(self.env.previous_distance,np.sqrt(.81+.36+.01))
        self.assertEqual(self.env.velocity_controller.yaw_target,0.)
        np.testing.assert_array_equal(self.env.velocity_controller.integral_error,[0,0,0])
        self.assertEqual(self.env.episode_steps,0);self.assertFalse(self.env._done)
        self.assertEqual(meta['snapshot_sha256'],fingerprint(snap))
        self.assertGreater(meta['ground_clearance_m'],0)

    def test_snapshot_restores_physical_controller_rng_bitwise(self):
        api=self.api();p=api.sample_perturbation('combined_large','benchmark',0)
        snap,meta=api.prepare_snapshot(self.env,self.target,p)
        noise=self.env.np_random.normal(size=5)
        a=np.array([.2,-.1,.3,.2],np.float32)
        first=[self.env.step(a)[0] for _ in range(2)]
        restore(self.env,snap)
        self.assertEqual(fingerprint(capture(self.env,np.zeros(4,np.float32))),meta['snapshot_sha256'])
        np.testing.assert_array_equal(self.env.np_random.normal(size=5),noise)
        np.testing.assert_array_equal([self.env.step(a)[0] for _ in range(2)],first)

    def test_component_only_perturbations_leave_other_fields_nominal(self):
        api=self.api()
        for name,expected in [('position_small',(.1,0,0)),('velocity_large',(0,.4,0)),('yaw_small',(0,0,np.pi/18)),('nominal',(0,0,0))]:
            p=api.sample_perturbation(name,'holdout',8)
            api.prepare_snapshot(self.env,self.target,p)
            self.assertAlmostEqual(np.linalg.norm(self.env.data.qpos[:3]-[0,0,1]),expected[0])
            self.assertAlmostEqual(np.linalg.norm(self.env.data.qvel[:3]),expected[1])
            self.assertAlmostEqual(abs(float(self.env._get_obs()[6])),expected[2],places=6)

    def test_seeded_directions_norm_balanced_yaw_and_combined_reuse(self):
        api=self.api()
        for split in ['benchmark','holdout']:
            signs=[]
            for i in range(100):
                small=api.sample_perturbation('combined_small',split,i)
                large=api.sample_perturbation('combined_large',split,i)
                self.assertEqual(small,api.sample_perturbation('combined_small',split,i))
                np.testing.assert_allclose(large['position_m'],np.asarray(small['position_m'])*3,atol=1e-15)
                np.testing.assert_allclose(large['velocity_m_s'],np.asarray(small['velocity_m_s'])*(.4/.15),atol=1e-15)
                self.assertAlmostEqual(np.linalg.norm(small['position_m']),.1)
                self.assertAlmostEqual(np.linalg.norm(large['velocity_m_s']),.4)
                signs.append(small['yaw_rad']>0)
                self.assertEqual(api.sample_perturbation('yaw_large',split,i)['yaw_rad'],large['yaw_rad'])
            self.assertEqual(sum(signs),50)

    def test_invalid_physical_initial_state_is_not_repaired_or_filtered(self):
        api=self.api()
        for p in [dict(position_m=[0,0,-1],velocity_m_s=[0,0,0],yaw_rad=0),
                  dict(position_m=[0,0,0],velocity_m_s=[float('nan'),0,0],yaw_rad=0)]:
            with self.assertRaises(ValueError):api.prepare_snapshot(self.env,self.target,p)

    def test_complete_manifest_reuses_fixed_targets_and_predeclared_conditions(self):
        # Catches resampling evaluation targets or silently dropping inconvenient states.
        import json
        api=self.api();root=Path(__file__).resolve().parents[2]
        manifest=api.build_manifest(root)
        old=json.loads((root/'mujoco/reports/uav_bc_target_splits_seed0.json').read_text())['provenance']['evaluation']
        self.assertEqual(len(manifest['states']),1800)
        self.assertEqual(len({r['key'] for r in manifest['states']}),1800)
        for r in manifest['states']:
            expected=old[r['split']]['targets'][r['index']]
            self.assertEqual(r['target'],expected['target']);self.assertEqual(r['env_seed'],expected['env_seed'])
            self.assertEqual(r['initial_state']['pi_integral'],[0.,0.,0.])
            self.assertGreaterEqual(r['initial_state']['ground_clearance_m'],.65-1e-12)


if __name__=='__main__':unittest.main()
