"""Offline contracts: no integrated flight episodes and no controller calls."""
import importlib
import importlib.util
from pathlib import Path
import unittest

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_attitude_thrust_response'), 'offline audit feature missing')
        self.api=importlib.import_module('uav_attitude_thrust_response')
        self.model=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco/models/mine_uav_dynamics_v2.xml'))

    def sample(self,**changes):
        args=dict(position=np.array([0.,0.,1.]),quat=np.array([1.,0.,0.,0.]),velocity=np.zeros(3),omega=np.zeros(3),
                  ctrl=np.full(4,340000.),force=np.zeros(3),time=4.)
        args.update(changes)
        return self.api.reconstruct(self.model,**args)

    def test_site_force_world_frame_and_scalar_actuator_units(self):
        # A +90degree rotation about worldY puts body+Z along world+X.
        r=self.sample(quat=np.array([2**-.5,0.,2**-.5,0.]))
        np.testing.assert_allclose(r['rotor_force_world_n'],[67.9695410274598,0,0],atol=1e-10,rtol=0)
        np.testing.assert_array_equal(r['actuator_scalar_output'],[340000.]*4)
        self.assertGreater(r['actuator_scalar_output'][0],100000)  # not a Newton-valued rotor thrust

    def test_com_acceleration_not_body_origin_when_rotating(self):
        r=self.sample(omega=np.array([.2,-.3,.4]),force=np.array([6.867,0,0]))
        np.testing.assert_allclose(r['com_acceleration_world_m_s2'],[.981,0,-.100065567505743],atol=1e-10,rtol=0)
        self.assertGreater(np.linalg.norm(r['origin_acceleration_world_m_s2']-r['com_acceleration_world_m_s2']),.001)
        self.assertLess(r['force_balance_max_abs_m_s2'],1e-10)

    def test_command_change_has_no_motor_dynamic_delay(self):
        a=self.sample(ctrl=np.full(4,100000.))
        b=self.sample(ctrl=np.full(4,200000.))
        np.testing.assert_allclose(b['rotor_force_world_n'],a['rotor_force_world_n']*2,atol=1e-12,rtol=0)
        self.assertEqual(a['time_s'],4.)
        self.assertEqual(b['time_s'],4.)

    def test_reconstruction_does_not_mutate_inputs_or_integrate_time(self):
        p=np.array([.2,-.1,1.]);q=np.array([1.,0,0,0]);v=np.array([.1,.2,.3]);w=np.array([.2,.3,.1]);u=np.full(4,340000.);f=np.array([6.867,0,0])
        snapshots=[x.copy() for x in (p,q,v,w,u,f)]
        a=self.api.reconstruct(self.model,p,q,v,w,u,f,3.14)
        b=self.api.reconstruct(self.model,p,q,v,w,u,f,3.14)
        for x,old in zip((p,q,v,w,u,f),snapshots):np.testing.assert_array_equal(x,old)
        for name in a:np.testing.assert_array_equal(a[name],b[name])
        self.assertEqual(a['time_s'],3.14)

    def test_dynamic_actuator_is_rejected(self):
        self.model.actuator_dyntype[0]=int(mujoco.mjtDyn.mjDYN_INTEGRATOR)
        with self.assertRaisesRegex(ValueError,'stateless'):self.sample()

    def test_delayed_actuator_is_rejected(self):
        self.model.actuator_delay[0]=.01
        with self.assertRaisesRegex(ValueError,'stateless'):self.sample()

    def test_global_control_callback_is_rejected(self):
        mujoco.set_mjcb_control(lambda m,d:None)
        try:
            with self.assertRaisesRegex(ValueError,'callback'):self.sample()
        finally:mujoco.set_mjcb_control(None)

    def test_contact_state_cannot_be_treated_as_freeflight(self):
        with self.assertRaisesRegex(ValueError,'contact|constraint'):self.sample(position=np.array([0.,0.,.02]))

    def test_nonunit_quaternion_and_nonfinite_input_are_rejected(self):
        with self.assertRaises(ValueError):self.sample(quat=np.array([2.,0,0,0]))
        with self.assertRaises(ValueError):self.sample(force=np.array([np.nan,0,0]))

    def test_saved_control_timestamp_force_and_command_alignment(self):
        with np.load(ROOT/'mujoco/reports/uav_pi_internal_telemetry_parts/gust_medium.npz') as z:
            small={k:z[k][:3].copy() for k in z.files if k.startswith('control_') and k!='control_update_counts'}
        r=self.api.reconstruct_series(self.model,small)
        np.testing.assert_array_equal(r['time_s'],small['control_time_s'])
        np.testing.assert_allclose(r['time_s'],[0.,.01,.02],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(r['actuator_scalar_output'],small['control_actual_ctrl_u'])
        np.testing.assert_array_equal(r['external_force_world_n'],np.zeros((3,3)))


class EventTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_attitude_thrust_response'),'offline audit feature missing')
        self.api=importlib.import_module('uav_attitude_thrust_response')

    def test_sustained_event_brackets_deadband_not_confirmation_time(self):
        t=np.arange(10)*.1;y=np.array([-1,-.02,-.001,0,.001,.1,.2,.3,-.1,-.2])
        e=self.api.crossing(t,y,0,1,epsilon=.01,hold=3,sign=1)
        np.testing.assert_allclose(e['bracket_s'],[.1,.5],atol=1e-12,rtol=0)
        self.assertAlmostEqual(e['confirmation_time_s'],.7)
        self.assertEqual(e['neutral_samples'],3)

    def test_single_sample_sign_glitch_not_an_event(self):
        self.assertIsNone(self.api.crossing(np.arange(7)*.01,[-1,1,-1,-1,1,-1,-1],0,1,epsilon=.01,hold=3,sign=1))

    def test_no_future_extrapolation_if_event_outside_window(self):
        self.assertIsNone(self.api.crossing(np.arange(8)*.1,[-1,-1,-1,-1,-1,1,1,1],0,.4,epsilon=.01,hold=3,sign=1))

    def test_reverse_direction_and_separation_interval(self):
        e=self.api.crossing(np.arange(6)*.1,[1,1,-1,-1,-1,-1],0,1,epsilon=.01,hold=3,sign=-1)
        np.testing.assert_allclose(e['bracket_s'],[.1,.2],atol=1e-12,rtol=0)
        np.testing.assert_allclose(self.api.interval_difference([5,5.01],[4,4.02]),[.98,1.01],atol=1e-12,rtol=0)

    def test_nonmonotonic_clock_and_invalid_hold_rejected(self):
        with self.assertRaises(ValueError):self.api.crossing([0,.1,.1],[-1,1,1],0,1,epsilon=0,hold=2,sign=1)
        with self.assertRaises(ValueError):self.api.crossing([0,.1,.2],[-1,1,1],0,1,epsilon=0,hold=0,sign=1)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.api=importlib.import_module('uav_attitude_thrust_response')
        self.assertTrue(hasattr(self.api,'force_error_components'),'analysis feature missing')

    def test_orientation_scalar_force_disturbance_and_com_offset_separated(self):
        z=dict(control_desired_thrust_n=np.array([70.]),control_desired_rotation_body_to_world=np.eye(3)[None],
               control_external_force_world_n=np.array([[7.,0.,0.]]),control_output_after_limiting_m_s2=np.array([[0.,0.,.19]]))
        d=dict(actual_body_z_world=np.array([[.6,0,.8]]),rotor_force_world_n=np.array([[39.9,0,53.2]]),
               origin_to_com_acceleration_world_m_s2=np.array([[.1,.2,.3]]),origin_acceleration_world_m_s2=np.array([[6.6,-.2,-2.51]]))
        c=self.api.force_error_components(z,d,7.)
        np.testing.assert_allclose(c['orientation_m_s2'],[[5.7,0,-1.9]],atol=1e-12,rtol=0)
        np.testing.assert_allclose(c['magnitude_m_s2'],[[0,0,-.5]],atol=1e-12,rtol=0)
        np.testing.assert_allclose(c['external_m_s2'],[[1,0,0]],atol=1e-12,rtol=0)
        self.assertLess(c['decomposition_max_abs_m_s2'],1e-12)

    def test_cached_acceleration_does_not_enter_current_physics(self):
        m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco/models/mine_uav_dynamics_v2.xml'))
        with np.load(ROOT/'mujoco/reports/uav_pi_internal_telemetry_parts/gust_medium.npz') as z:
            a={k:z[k][400:404].copy() for k in z.files if k.startswith('control_') and k!='control_update_counts'}
        before=self.api.reconstruct_series(m,a)
        a['control_cached_qacc_world_m_s2'][:]=9999
        after=self.api.reconstruct_series(m,a)
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_preselected_case_inputs_validate_without_flight(self):
        r=self.api.input_identity(ROOT)
        self.assertEqual(r['target_id'],'external-final-2026101201-000')
        self.assertEqual(set(r['raw']),{'constant_medium','gust_medium'})
        self.assertEqual(r['new_flight_executions'],0)
        self.assertTrue(r['baseline_arrays_bitwise_equal'])

    def test_changed_input_anchor_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'inputs.json'
            p.write_text('{"input":"first"}')
            with self.assertRaisesRegex(ValueError,'input identity changed'):
                self.api.preserve_identity(p,dict(input='second'))
            self.assertEqual(p.read_text(),'{"input":"first"}')

    def test_case_analysis_is_deterministic_and_keeps_saved_values(self):
        self.assertTrue(hasattr(self.api,'case_analysis'),'case analysis missing')
        m=mujoco.MjModel.from_xml_path(str(ROOT/'mujoco/models/mine_uav_dynamics_v2.xml'))
        with np.load(ROOT/'mujoco/reports/uav_pi_internal_telemetry_parts/gust_medium.npz') as z:
            a={k:z[k][:20].copy() for k in z.files if k.startswith('control_') and k!='control_update_counts'}
        d=self.api.reconstruct_series(m,a)
        old=a['control_actual_ctrl_u'].copy()
        first=self.api.case_analysis(a,d,[.9845354832399106,.1751852797513414,0.])
        second=self.api.case_analysis(a,d,[.9845354832399106,.1751852797513414,0.])
        self.assertEqual(first,second)
        self.assertEqual(first['control_samples'],20)
        self.assertIsNone(first['events_zero']['pi_output'])
        np.testing.assert_array_equal(a['control_actual_ctrl_u'],old)


class PublicationTests(unittest.TestCase):
    def test_missing_inputs_fail_before_publishing_or_simulating(self):
        import tempfile
        api=importlib.import_module('uav_attitude_thrust_response')
        self.assertTrue(hasattr(api,'publish'),'publication feature missing')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):api.publish(Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()),[])


if __name__=='__main__':unittest.main()
