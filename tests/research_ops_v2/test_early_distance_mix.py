"""Literal reset-only distribution tests; no NumPy/ML/physics imports."""
import importlib.util,math,unittest
class EarlyDistanceMixTests(unittest.TestCase):
 def api(self):
  name='research_regression.authorized_rl.early_distance_mix'
  self.assertIsNotNone(importlib.util.find_spec(name),'missing early-distance-mix distribution')
  return __import__(name,fromlist=['stage_at_reset'])
 def test_early_half_probability_boundary_is_literal_per_reset(self):
  f=self.api().stage_at_reset
  self.assertEqual([f(0,v) for v in (0.,.499999,.5,.999999)],['ORIGINAL','ORIGINAL','SHORT','SHORT'])
 def test_natural_reset_phase_boundary_does_not_need_future_draw(self):
  f=self.api().stage_at_reset
  self.assertEqual(f(32767,.7),'SHORT');self.assertEqual(f(32768,None),'ORIGINAL');self.assertEqual(f(65536,None),'ORIGINAL')
  with self.assertRaises(ValueError):f(32768,.7)
  with self.assertRaises(ValueError):f(32767,None)
 def test_early_base_layout_marginal_is_uniform_late_empty_half(self):
  f=self.api().base_weights;layouts=['empty']*3+['nonempty']*9
  self.assertEqual(f(0,layouts),tuple([1/12]*12));self.assertEqual(f(32767,layouts),tuple([1/12]*12))
  weights=f(32768,layouts)
  self.assertEqual(weights[:3],(1/6,)*3);self.assertEqual(weights[3:],(1/18,)*9)
  self.assertAlmostEqual(sum(weights),1.);self.assertAlmostEqual(sum(weights[:3]),.5)
 def test_distance_stream_seed_is_separate_and_reproducible(self):
  f=self.api().distance_seed
  self.assertEqual(f(0),330000);self.assertEqual(f(1),330001);self.assertEqual(f(0),f(0))
  for seed in (True,-1,1.5):
   with self.subTest(seed=seed),self.assertRaises(ValueError):f(seed)
 def test_invalid_draw_step_or_layout_is_fail_closed(self):
  m=self.api()
  for draw in (float('nan'),float('inf'),-1.,1.,True):
   with self.subTest(draw=draw),self.assertRaises(ValueError):m.stage_at_reset(0,draw)
  for step in (True,-1,1.5):
   with self.subTest(step=step),self.assertRaises(ValueError):m.stage_at_reset(step,.2)
  for layout in ([],['empty']*12,['empty']*3+['nonempty']*8,['empty']*3+[None]*9):
   with self.assertRaises(ValueError):m.base_weights(0,layout)
if __name__=='__main__':unittest.main()
