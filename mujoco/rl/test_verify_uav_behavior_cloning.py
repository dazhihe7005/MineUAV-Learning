"""Independent experiment-contract checks, not just JSON smoke tests."""
import unittest
import numpy as np


class VerificationTests(unittest.TestCase):
    def test_record_indexing_and_label_correctness(self):
        from verify_uav_behavior_cloning import verify_arrays
        from test_env_scripted_policy import scripted_action
        x=np.array([[.6,0,0,0,0,0,.1],[.4,0,0,0,0,0,.1],[.2,0,0,0,0,0,.1]],np.float32)
        a=np.array([scripted_action(o) for o in x[:-1]])
        trace=dict(observations=x,actions=a,previous_actions=np.vstack([np.zeros(4),a[0]]),
            step_index=np.arange(2),target_id=np.array([7,7]),episode_id=np.array([33,33]),
            success=np.array([True,True]),failure=np.array([False,False]))
        row=dict(steps=2,target_id=7,episode_id=33,success=True,failure=False)
        verify_arrays(trace,row,expert=True)
        trace['actions']=a.copy();trace['actions'][0,0]+=.1
        with self.assertRaises(AssertionError):verify_arrays(trace,row,expert=True)

    def test_sample_weighting_not_episode_weighting(self):
        from verify_uav_behavior_cloning import mse_contract
        p=np.zeros((3,4));y=np.ones((3,4));y[0]*=2
        self.assertEqual(mse_contract(p,y,np.ones(4)),2.)


if __name__=='__main__':unittest.main()
