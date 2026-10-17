"""Audit-clock/finite-difference/correlation contracts; no simulation or ML."""
import copy,importlib.util,math,unittest

class ControlResponseAuditTests(unittest.TestCase):
 def api(self):
  name='research_regression.authorized_rl.control_response_audit'
  self.assertIsNotNone(importlib.util.find_spec(name),'missing bounded controller-response audit')
  return __import__(name,fromlist=['response_samples'])
 def fixture(self,n=100):
  # Explicit synthetic arithmetic test only,not actual UAV evidence.
  rows=[]
  for i in range(n):
   t=.04*i;v=math.sin(i*.4)
   rows.append({'pre_time_s':t,'post_time_s':t+.04,'physics_steps':20,
    'pre_qvel':[v,0.,0.,0.,0.,0.],'post_qvel':[v+.08,0.,0.,0.,0.,0.],
    'post_qpos':[0.,0.,1.,1.,0.,0.,0.],'pi_integral_error_post':[1.,-2.,.5],
    'desired_acceleration_post_enu_m_s2':[v,0.,0.], 'velocity_command_enu_m_s':[.5,0.,0.],
    'distance_m':1.,'speed_m_s':abs(v+.08)})
  return rows
 def test_control_sample_and_interval_average_have_distinct_timestamps(self):
  s=self.api().response_samples(self.fixture(1))[0]
  self.assertAlmostEqual(s['last_control_time_s'],.03)
  self.assertAlmostEqual(s['interval_mid_time_s'],.02)
  self.assertAlmostEqual(s['average_acceleration_enu_m_s2'][0],2.)
  self.assertAlmostEqual(s['pi_integral_acceleration_enu_m_s2'][0],.5)
  self.assertAlmostEqual(s['pi_integral_acceleration_enu_m_s2'][1],-1.)
  self.assertAlmostEqual(s['pi_integral_acceleration_enu_m_s2'][2],.4)
 def test_positive_lag_has_correct_pairing_not_future_reference(self):
  m=self.api();s=m.response_samples(self.fixture(100))
  for i,v in enumerate(s):
   v['desired_acceleration_enu_m_s2'][0]=math.sin(i*.4)
   v['average_acceleration_enu_m_s2'][0]=math.sin((i-3)*.4)
  a=m.lag_alignment(s,3,axis=0,window=(.1,3.9))
  self.assertEqual(a['status'],'OBSERVED')
  self.assertAlmostEqual(a['reference_to_response_offset_s'],.11)
  self.assertAlmostEqual(a['pearson_r'],1.,places=12)
  self.assertTrue(all(p['reference_time_s']<=p['response_time_s'] for p in a['pairs']))
  with self.assertRaises(ValueError):m.lag_alignment(s,0)
 def test_constant_or_insufficient_signal_is_not_observable_not_zero(self):
  m=self.api();s=m.response_samples(self.fixture(10))
  a=m.lag_alignment(s,1,axis=1,window=(0.,1.))
  self.assertEqual(a['status'],'NOT_OBSERVABLE');self.assertIsNone(a['pearson_r'])
  self.assertEqual(m.lag_alignment(s,9,window=(0.,1.))['status'],'NOT_OBSERVABLE')
 def test_invalid_clock_shape_nan_and_quaternion_are_rejected(self):
  m=self.api()
  for field,value in (('post_time_s',.041),('physics_steps',19),('pre_qvel',[0.]),
   ('pi_integral_error_post',[float('nan'),0.,0.]),('post_qpos',[0.,0.,1.,0.,0.,0.,0.])):
   rows=self.fixture(1);rows[0][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):m.response_samples(rows)
  rows=self.fixture(2);rows[1]['pre_time_s']=0.
  with self.assertRaises(ValueError):m.response_samples(rows)
 def test_attitude_units_and_summary_do_not_claim_causality_or_motor_data(self):
  m=self.api();rows=self.fixture(100);angle=.2
  for v in rows:v['post_qpos'][3:7]=[math.cos(angle/2),math.sin(angle/2),0.,0.]
  s=m.response_samples(rows);self.assertAlmostEqual(s[0]['roll_rad'],.2)
  a=m.summarize_response(s)
  self.assertAlmostEqual(a['max_abs_roll_rad'],.2)
  self.assertEqual(a['motor_saturation'],'NOT_OBSERVABLE')
  self.assertEqual(a['causal_root_cause'],'UNVERIFIED')

if __name__=='__main__':unittest.main()
