"""Controller-only extraction tests; never integrate a new flight episode."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'control'))
try:
    import uav_pi_telemetry as telemetry
except ModuleNotFoundError:
    telemetry = None
try:
    import run_uav_pi_internal_telemetry as runner
except ModuleNotFoundError:
    runner = None
try:
    import uav_pi_telemetry_analysis as analysis
except ModuleNotFoundError:
    analysis = None


def fixture():
    from hover_controller import HoverController
    from velocity_command_controller import VelocityCommandController
    from control_allocator import ControlAllocator
    c = VelocityCommandController(HoverController(7, np.diag([.16, .14, .25]), 158),
                                  [0, 0, -9.81], ki_xy=.5, ki_z=.8)
    b = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [-1, 1, 1, -1], [1, -1, 1, -1]], float)
    return SimpleNamespace(velocity_controller=c, allocator=ControlAllocator(b),
                          data=SimpleNamespace(time=0., qpos=np.array([0., 0., 1., 1., 0., 0., 0.]),
                            qvel=np.zeros(6), qacc=np.zeros(6)),
                          target_position=np.array([.5, 0., 1.]), episode_steps=0, success_streak=0)


def compute(env, cmd, vel=(0, 0, 0), dt=.01):
    return env.velocity_controller.compute(velocity=vel, quaternion=[1, 0, 0, 0],
        angular_velocity=[0, 0, 0], velocity_command=cmd, yaw_rate_command=0., control_dt=dt)


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(telemetry, 'passive telemetry implementation missing')

    def test_passive_observer_preserves_outputs_and_state_bitwise(self):
        from dataclasses import fields
        a, b = fixture(), fixture()
        before = dict(vars(b.velocity_controller))
        with telemetry.PassiveTelemetry(b) as recorder:
            for i in range(8):
                b.data.time = i * .01
                actual = compute(b, [.5, -.2, .1])
                expected = compute(a, [.5, -.2, .1])
                for f in fields(actual):
                    np.testing.assert_array_equal(getattr(actual, f.name), getattr(expected, f.name))
        self.assertEqual(len(recorder.control_rows), 8)
        np.testing.assert_array_equal(a.velocity_controller.integral_error, b.velocity_controller.integral_error)
        self.assertEqual(set(vars(b.velocity_controller)), set(before))
        self.assertIsNone(sys.gettrace())

    def test_extracts_actual_integrator_and_pre_post_limit_values(self):
        e = fixture()
        with telemetry.PassiveTelemetry(e) as r:
            compute(e, [.5, 0, .5], dt=.1)
        row = r.control_rows[0]
        np.testing.assert_allclose(row['integral_before_m'], [0, 0, 0], atol=0)
        np.testing.assert_allclose(row['integral_after_m'], [.05, 0, .05])
        np.testing.assert_allclose(row['i_contribution_m_s2'], [.025, 0, .04])
        np.testing.assert_allclose(row['p_contribution_m_s2_derived'], [.75, 0, 1.])
        np.testing.assert_allclose(row['output_before_limiting_m_s2'], [.775, 0, 1.04])
        np.testing.assert_allclose(row['output_after_limiting_m_s2'], [.775, 0, 1.04])

    def test_records_true_clamp_branch_and_not_large_integral_only(self):
        e = fixture()
        e.velocity_controller._integral_error[:] = [2.999, 0, 1.874]
        with telemetry.PassiveTelemetry(e) as r:
            compute(e, [.5, 0, .5])
        row = r.control_rows[0]
        self.assertTrue(row['integral_clamp_xy'])
        self.assertTrue(row['integral_clamp_z'])
        np.testing.assert_allclose(row['integral_after_m'], [3., 0, 1.875])

    def test_separates_antiwindup_trial_from_committed_output(self):
        e = fixture()
        with telemetry.PassiveTelemetry(e) as r:
            compute(e, [1.5, 0, 1.], vel=[-1.5, 0, -1.])
        row = r.control_rows[0]
        self.assertTrue(row['anti_windup_freeze_xy'])
        self.assertTrue(row['anti_windup_freeze_z'])
        np.testing.assert_allclose(row['anti_windup_trial_m_s2'], [4.515, 0, 4.016])
        np.testing.assert_allclose(row['output_before_limiting_m_s2'], [4.5, 0, 4.])
        np.testing.assert_allclose(row['output_after_limiting_m_s2'], [3., 0, 3.])
        np.testing.assert_array_equal(row['integral_after_m'], [0, 0, 0])

    def test_allocation_outputs_are_returned_command_not_guessed_thrust(self):
        e = fixture()
        with telemetry.PassiveTelemetry(e) as r:
            command = compute(e, [0, 0, 0])
            a = e.allocator.allocate(command.desired_wrench_body)
        row = r.control_rows[0]
        np.testing.assert_array_equal(row['allocator_achieved_wrench'], a.achieved_wrench)
        np.testing.assert_array_equal(row['rotor_command_u'], a.u)
        self.assertAlmostEqual(row['desired_thrust_n'], 68.67)
        self.assertAlmostEqual(row['commanded_thrust_n'], a.achieved_wrench[0])

    def test_exception_removes_observer_without_integrator_mutation(self):
        e = fixture()
        with self.assertRaises(ValueError):
            with telemetry.PassiveTelemetry(e):
                compute(e, [float('nan'), 0, 0])
        self.assertIsNone(sys.gettrace())
        np.testing.assert_array_equal(e.velocity_controller.integral_error, [0, 0, 0])

    def test_timestamp_is_pre_integration_control_time(self):
        e = fixture(); e.data.time = 4.
        with telemetry.PassiveTelemetry(e) as r:
            compute(e, [0, 0, 0])
        self.assertEqual(r.control_rows[0]['time_s'], 4.)

    def test_comparison_rejects_control_change_and_missing_signal(self):
        x = dict(actions=np.zeros((1, 4)), positions=np.zeros((2, 3)))
        y = {k:v.copy() for k,v in x.items()}
        self.assertTrue(telemetry.verify_passive(x, y)['bitwise_equal'])
        y['positions'][1, 0] = 1e-5
        with self.assertRaises(ValueError): telemetry.verify_passive(x, y)
        with self.assertRaises(ValueError): telemetry.verify_passive(x, dict(actions=x['actions']))


class AcquisitionContractTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, 'two-episode runner implementation missing')

    def test_fixed_selection_ignores_outcomes_and_rejects_duplicates(self):
        states=[dict(condition=c,index=i,target_id=f't{i}',key=f'{c}/{i}')
                for c in ('gust_medium','constant_medium') for i in (1,0)]
        selected=runner.select_states(dict(states=states))
        self.assertEqual([s['key'] for s in selected], ['constant_medium/0','gust_medium/0'])
        with self.assertRaises(ValueError): runner.select_states(dict(states=states+[states[-1]]))

    def test_force_alignment_uses_current_tick_not_previous_buffer(self):
        e=fixture();e.data.time=4.
        with telemetry.PassiveTelemetry(e) as r: compute(e,[0,0,0])
        times=np.arange(2001)*.002;forces=np.zeros((2001,3));forces[1000:2000,0]=6.867
        a=r.arrays(times,forces)
        np.testing.assert_array_equal(a['control_external_force_world_n'],[[0,0,0]])
        e.data.time=2.
        with telemetry.PassiveTelemetry(e) as r: compute(e,[0,0,0])
        np.testing.assert_array_equal(r.arrays(times,forces)['control_external_force_world_n'],[[6.867,0,0]])

    def test_resume_rejects_raw_hash_mismatch_and_incomplete_record(self):
        import hashlib,json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'r.npz';np.savez(p,actions=np.zeros((1,4)))
            identity=dict(test='fixed')
            row=dict(identity=identity,task_id='constant_medium',raw_path=str(p),raw_sha256='wrong',
                     completed=True,passivity=dict(bitwise_equal=True))
            row['record_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            with self.assertRaisesRegex(ValueError,'telemetry raw hash mismatch'):
                runner.validate_completed(row,identity,'constant_medium')
            row['completed']=False
            with self.assertRaises(ValueError): runner.validate_completed(row,identity,'constant_medium')

    def test_100hz_validator_does_not_confuse_legacy_25hz_count_array(self):
        from uav_bc_external_disturbance import DisturbanceEnv
        env=DisturbanceEnv()
        try:
            env.reset(seed=1,options=dict(target_position=[.5,0,1]))
            with telemetry.PassiveTelemetry(env) as r:
                for i in range(4):
                    env.data.time=i*.01
                    c=compute(env,[0,0,0]);a=env.allocator.allocate(c.desired_wrench_body)
                    env._apply_rotor_command(a.u)
            z=r.arrays(np.arange(20)*.002,np.zeros((20,3)))
            z.update(control_update_counts=np.array([4]),physics_forces=np.zeros((20,3)),
                     positions=np.array([[0,0,1],[0,0,1]]),actions=np.zeros((1,4)))
            runner.verify_telemetry(z,1)
        finally:env.close()


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(analysis, 'measured analysis implementation missing')

    def test_phase_boundaries_do_not_double_count_force_removal_sample(self):
        t=np.array([0.,1.99,2.,3.99,3.9999999999998,4.01,5.99,6.,10.,14.99])
        masks=[analysis.window_mask(t,lo,hi) for lo,hi in [(0,2),(2,4),(4,6),(6,10),(10,15)]]
        np.testing.assert_array_equal(sum(m.astype(int) for m in masks),np.ones(10))
        np.testing.assert_array_equal(np.flatnonzero(masks[2]),[4,5,6])

    def test_success_gates_require_position_and_speed_together(self):
        e=np.array([[.05,0,0],[.2,0,0],[.05,0,0]])
        v=np.array([[.2,0,0],[.1,0,0],[.1,0,0]])
        g=analysis.threshold_gates(e,v)
        np.testing.assert_array_equal(g['joint'],[False,False,True])
        np.testing.assert_array_equal(g['distance_only'],[True,False,False])
        np.testing.assert_array_equal(g['speed_only'],[False,True,False])

    def test_braking_direction_distinguishes_integral_assist_and_opposition(self):
        v=np.array([[1.,0,0],[-1.,0,0],[0,0,0]])
        p=np.array([[-2.,0,0],[2.,0,0],[0,0,0]])
        i=np.array([[.5,0,0],[.5,0,0],[0,0,0]])
        m=analysis.braking_masks(v,p,i,p+i)
        np.testing.assert_array_equal(m['p_braking_i_accelerating'],[True,False,False])
        np.testing.assert_array_equal(m['output_braking'],[True,True,False])

    def test_crossings_ignore_zero_plateau_and_return_observed_intervals(self):
        times=np.array([4.,4.1,4.2,4.3,4.4])
        intervals=analysis.zero_crossings(times,np.array([1.,0.,0.,-1.,1.]))
        self.assertEqual(intervals,[[4.,4.3],[4.3,4.4]])

    def test_quaternion_sign_equivalence_and_attitude_tracking_error(self):
        q=np.array([[1,0,0,0],[1,0,0,0]],float)
        desired=np.array([[-1,0,0,0],[np.cos(np.pi/12),0,np.sin(np.pi/12),0]])
        np.testing.assert_allclose(analysis.attitude_error_degrees(q,desired),[0,30],atol=1e-10)

    def test_phase_summary_uses_measured_integral_flags_not_magnitude_proxy(self):
        z=dict(control_time_s=np.array([4.,4.01]),control_position_m=np.zeros((2,3)),
          control_target_m=np.array([[.2,0,0],[.05,0,0]]),control_measured_velocity_m_s=np.array([[.1,0,0],[.2,0,0]]),
          control_commanded_velocity_m_s=np.zeros((2,3)),control_i_contribution_m_s2=np.array([[1.4,0,0],[1.4,0,0]]),
          control_p_contribution_m_s2_derived=np.array([[-.15,0,0],[-.3,0,0]]),
          control_output_after_limiting_m_s2=np.array([[1.25,0,0],[1.1,0,0]]),
          control_actual_quaternion_wxyz=np.tile([1.,0,0,0],(2,1)),
          control_desired_quaternion_wxyz=np.tile([1.,0,0,0],(2,1)))
        for f in ('integral_clamp_xy','integral_clamp_z','anti_windup_freeze_xy','anti_windup_freeze_z',
                  'output_limited_xy','output_limited_z','allocator_saturated'):
            z['control_'+f]=np.zeros(2,bool)
        r=analysis.phase_metrics(z,4,6)
        self.assertEqual(r['samples'],2)
        self.assertEqual(r['events']['integral_clamp_xy'],0)
        self.assertEqual(r['joint_threshold_samples'],0)
        self.assertAlmostEqual(r['speed_mean_m_s'],.15)

    def test_exact_state_restore_without_physics_or_controller_replacement(self):
        from uav_bc_external_disturbance import DisturbanceEnv,initial_snapshot
        from decision_fidelity_snapshot import capture,restore,fingerprint
        env=DisturbanceEnv()
        try:
            env.configure('gust_medium',[1,0,0])
            snap,_=initial_snapshot(env,dict(env_seed=950140000,target=[.5,0,1]))
            expected=fingerprint(snap)
            env.data.qpos[0]=.2;env.data.ctrl[:]=1;env.data.qacc_warmstart[:]=1
            env.velocity_controller._integral_error[:]=1;env.success_streak=3
            prior=restore(env,snap)
            self.assertEqual(fingerprint(capture(env,prior)),expected)
            env.configure('constant_medium',[1,0,0])
            other,_=initial_snapshot(env,dict(env_seed=950140000,target=[.5,0,1]))
            self.assertEqual(runner.physical_fingerprint(snap),runner.physical_fingerprint(other))
        finally:env.close()


if __name__ == '__main__':
    unittest.main()
