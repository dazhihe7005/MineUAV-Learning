"""R15 loads actual immutable R13/R4 observations; no model load or training."""
import importlib.util,unittest
import numpy as np
from research_ops.storage import OpsError


class CenteredFeedbackArrayTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.centered_feedback'),
                             'missing R15 immutable same-class reference loader')
        from . import centered_feedback
        return centered_feedback

    def test_same_class_historical_panel_matches_exact_sensor_and_cases(self):
        m=self.module();obs,seal,prior,scripted=m.load_background()
        self.assertEqual(obs['lidar'].shape,(3,12,24));self.assertEqual(obs['lidar'].dtype,np.float32)
        self.assertEqual(seal['observation_identity_sha256'],'453a7723fb4225d141af3c32f847d6b7df12aad9067d488fe343f82aac3702fc')
        self.assertEqual(prior['policy_sha256'],'20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001')
        self.assertEqual(len(prior['rows']),62);self.assertEqual(len(scripted),62)
        self.assertTrue(all(np.isfinite(a).all() for a in obs.values()))
        m.validate_background(obs)

    def test_mutated_historical_input_cannot_keep_reference_identity(self):
        m=self.module();obs,_,_,_=m.load_background();obs['state'][0]+=.01
        with self.assertRaises(OpsError):m.validate_background(obs)
        obs,_,_,_=m.load_background();obs['lidar'][0,0,0]=.123456
        with self.assertRaises(OpsError):m.validate_background(obs)


if __name__=='__main__':unittest.main()
