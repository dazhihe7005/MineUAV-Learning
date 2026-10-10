"""Protect physical-force injection, not an observation-level perturbation."""
import importlib.util
import math
import unittest
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]

class ExternalForceTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_bc_external_disturbance'),
            'physical external-disturbance implementation missing')
        import uav_bc_external_disturbance as impl
        self.impl=impl;self.env=impl.DisturbanceEnv();self.addCleanup(self.env.close)
        self.env.reset(seed=3,options={'target_position':[1.,0.,1.]})

    def test_world_com_wrench_without_direct_torque(self):
        e=self.env;e.configure('constant_high',[0.,1.,0.])
        e.data.qpos[3:7]=[math.cos(.5),0.,0.,math.sin(.5)]
        mujoco.mj_forward(e.model,e.data);e._before_physics_step()
        np.testing.assert_allclose(e.data.xfrc_applied[e.uav_body_id], [0,13.734,0,0,0,0],atol=1e-12)
        self.assertEqual(e.model.body('mine_uav').id,e.uav_body_id)
        np.testing.assert_array_equal(e.data.xfrc_applied[0],np.zeros(6))
        # Independent API check: same generalized force as force applied at world COM.
        expected=np.zeros(e.model.nv)
        mujoco.mj_applyFT(e.model,e.data,np.array([0.,13.734,0.]),np.zeros(3),
                         e.data.xipos[e.uav_body_id],e.uav_body_id,expected)
        e.data.ctrl[:]=0;mujoco.mj_forward(e.model,e.data)
        with_force=e.data.qfrc_smooth.copy();e.data.xfrc_applied[:]=0
        mujoco.mj_forward(e.model,e.data)
        np.testing.assert_allclose(with_force-e.data.qfrc_smooth,expected,atol=1e-10)
        self.assertAlmostEqual(float(expected[1]),13.734,places=10)
        self.assertAlmostEqual(float(expected[0]),0.,places=10)
        self.assertLess(np.linalg.norm(expected[3:]),1.) # COM vs freejoint offset, no direct torque.

    def test_gust_exact_tick_edges_and_no_force_accumulation(self):
        e=self.env;e.configure('gust_medium',[1.,0.,0.])
        for tick,force in [(999,0.),(1000,6.867),(1999,6.867),(2000,0.)]:
            np.testing.assert_allclose(e.force_at_tick(tick),[force,0,0],atol=1e-12)
        e.data.time=2.;e._before_physics_step();e._before_physics_step()
        self.assertAlmostEqual(e.data.xfrc_applied[e.uav_body_id,0],6.867,places=10)

    def test_every_physics_tick_recorded_and_physically_integrated(self):
        e=self.env;e.configure('constant_low',[1.,0.,0.]);initial=e.data.qvel.copy()
        _,_,_,_,info=e.step(np.zeros(4,np.float32))
        self.assertEqual(len(e.force_times),20)
        np.testing.assert_allclose(e.force_times,np.arange(20)*.002,atol=1e-12)
        np.testing.assert_allclose(e.force_vectors,np.tile([3.4335,0,0],(20,1)),atol=1e-12)
        self.assertGreater(e.data.qvel[0],initial[0])
        self.assertEqual(info['physics_steps'],20)

    def test_reset_and_nominal_remove_all_residual_force(self):
        e=self.env;e.configure('constant_high',[1.,0.,0.]);e.step(np.zeros(4,np.float32))
        e.configure('nominal',[1.,0.,0.]);e.reset(seed=3)
        np.testing.assert_array_equal(e.data.xfrc_applied,np.zeros((e.model.nbody,6)))
        self.assertEqual(e.force_times,[]);e.step(np.zeros(4,np.float32))
        np.testing.assert_array_equal(e.force_vectors,np.zeros((20,3)))

    def test_full_snapshot_restore_replays_forces_pi_and_physics(self):
        from decision_fidelity_snapshot import capture,restore,fingerprint
        e=self.env;e.configure('gust_high',[.6,.8,0.]);e.data.time=1.96
        snap=capture(e,np.zeros(4,np.float32));digest=fingerprint(snap)
        e.step(np.array([.1,-.2,0.,.1],np.float32));e.step(np.zeros(4,np.float32))
        expected=capture(e,np.zeros(4,np.float32));restore(e,snap)
        self.assertEqual(fingerprint(capture(e,np.zeros(4,np.float32))),digest)
        e.step(np.array([.1,-.2,0.,.1],np.float32));e.step(np.zeros(4,np.float32))
        self.assertEqual(fingerprint(capture(e,np.zeros(4,np.float32))),fingerprint(expected))

    def test_fixed_strength_feasibility_and_parameter_identity(self):
        p=self.impl.preflight(self.env)
        self.assertEqual(p['mass_kg'],7.)
        self.assertAlmostEqual(p['forces_n']['high'],13.734)
        self.assertTrue(p['high_static_thrust_tilt_feasible'])
        self.assertTrue(p['high_exceeds_pi_integral_acceleration_limit'])

    def test_directions_cover_circle_deterministically(self):
        d=self.impl.directions();np.testing.assert_array_equal(d,self.impl.directions())
        np.testing.assert_allclose(np.linalg.norm(d,axis=1),np.ones(100),atol=1e-12)
        np.testing.assert_array_equal(d[:,2],np.zeros(100))
        angles=np.mod(np.arctan2(d[:,1],d[:,0]),2*np.pi)
        self.assertEqual(len(np.unique(np.floor(angles/(2*np.pi)*100))),100)

    def test_new_targets_disjoint_from_all_prior_cohorts(self):
        from uav_bc_yaw_data import assert_isolated
        old=self.impl.excluded_targets(ROOT);new=self.impl.targets(old)
        assert_isolated(new,old);self.assertEqual(len(new),100)
        self.assertEqual(new,self.impl.targets(old))

    def test_reject_invalid_directions_and_conditions(self):
        for direction in ([0,0,0],[1,0,1],[2,0,0],[np.nan,0,0]):
            with self.assertRaises(ValueError):self.env.configure('constant_low',direction)
        with self.assertRaises(ValueError):self.env.configure('unknown',[1,0,0])

    def test_nominal_hook_preserves_original_pi_trajectory_bitwise(self):
        from ppo_pi_env import MineUAVPIEnv
        from test_env_scripted_policy import scripted_action
        other=MineUAVPIEnv(reward_version='v2');self.addCleanup(other.close)
        a,_=self.env.reset(seed=8,options={'target_position':[.8,-.4,1.2]})
        b,_=other.reset(seed=8,options={'target_position':[.8,-.4,1.2]})
        for _ in range(30):
            np.testing.assert_array_equal(a,b)
            action=scripted_action(a);a=self.env.step(action)[0];b=other.step(action)[0]
            np.testing.assert_array_equal(self.env.data.qpos,other.data.qpos)
            np.testing.assert_array_equal(self.env.data.qvel,other.data.qvel)
            np.testing.assert_array_equal(self.env.velocity_controller.integral_error,other.velocity_controller.integral_error)

    def test_both_canonical_checkpoints_hashes_and_inference_frozen(self):
        from uav_bc_policy import load,parameter_hash
        from latent_dynamics_data import file_hash
        for name,sha in self.impl.MODELS.values():
            path=ROOT/'mujoco/rl/models'/name;p,_=load(path);before=parameter_hash(p.actor)
            self.assertEqual(file_hash(path),sha);self.assertFalse(p.actor.training)
            self.assertTrue(all(not x.requires_grad for x in p.actor.parameters()))
            self.assertTrue(np.isfinite(p.predict(np.array([1.,0,0,0,0,0,0],np.float32))).all())
            self.assertEqual(parameter_hash(p.actor),before);self.assertEqual(file_hash(path),sha)

if __name__=='__main__':unittest.main()
