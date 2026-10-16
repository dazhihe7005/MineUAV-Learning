"""Actual R4 historical background loading, not a new simulation or inference."""
import importlib.util,unittest
import numpy as np
from .runtime import ROOT
from research_ops.storage import OpsError,sha256


class FollowupArrayTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.feedback_followup'),'missing R12 immutable input loader')
        from . import feedback_followup
        return feedback_followup

    def test_actual_historical_frame_identity_cases_and_prior_rows(self):
        m=self.module();obs,meta,baseline,scripted=m.load_background()
        self.assertEqual(meta['observation_identity_sha256'],'453a7723fb4225d141af3c32f847d6b7df12aad9067d488fe343f82aac3702fc')
        self.assertEqual(obs['lidar'].shape,(3,12,24));self.assertEqual(obs['lidar'].dtype,np.float32)
        self.assertEqual(len(baseline['rows']),62);self.assertEqual(len(scripted),62)
        self.assertEqual(baseline['policy_sha256'],'6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74')
        self.assertTrue(all(np.isfinite(v).all() for v in obs.values()))
        m.validate_background(obs)

    def test_mutated_actual_input_cannot_keep_old_source_identity(self):
        m=self.module();obs,_,_,_=m.load_background();obs['lidar'][0,0,0]=.123456
        with self.assertRaises(OpsError):m.validate_background(obs)
        obs,_,_,_=m.load_background();obs['state'][0]+=.01
        with self.assertRaises(OpsError):m.validate_background(obs)


if __name__=='__main__':unittest.main()
