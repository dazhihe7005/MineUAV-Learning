"""Real restore tests catch hidden PI/yaw, solver or episode-state leakage."""
import importlib
import importlib.util
import unittest
import numpy as np
from ppo_pi_env import MineUAVPIEnv


class SnapshotTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('decision_fidelity_snapshot'),'exact snapshot missing')
        return importlib.import_module('decision_fidelity_snapshot')

    def test_nonzero_controller_and_full_simulator_replay_is_bit_exact(self):
        api=self.api(); env=MineUAVPIEnv(reward_version='v2')
        try:
            env.reset(seed=19,options={'target_position':[1.,-.8,1.2]})
            previous=np.array([.4,-.3,.2,.7],np.float32)
            for _ in range(3): env.step(previous)
            self.assertGreater(np.linalg.norm(env.velocity_controller.integral_error),0)
            self.assertNotEqual(env.velocity_controller.yaw_target,0)
            snap=api.capture(env,previous); digest=api.fingerprint(snap)
            actions=np.random.default_rng(12).uniform(-.3,.3,(10,4)).astype(np.float32)
            first=[]
            for a in actions: first.append(env.step(a)[0])
            end=api.fingerprint(api.capture(env,actions[-1]))
            restored=api.restore(env,snap); np.testing.assert_array_equal(restored,previous)
            self.assertEqual(env.episode_steps,3)
            self.assertEqual(api.fingerprint(api.capture(env,previous)),digest)
            second=[env.step(a)[0] for a in actions]
            np.testing.assert_array_equal(first,second)
            self.assertEqual(api.fingerprint(api.capture(env,actions[-1])),end)
            self.assertEqual(api.fingerprint(snap),digest,'stored snapshot mutated')
        finally: env.close()

    def test_target_rng_task_capture_and_cross_candidate_order_are_restored(self):
        api=self.api(); env=MineUAVPIEnv(reward_version='v2')
        try:
            env.begin_integral_capture(); env.reset(seed=37,options={'target_position':[.3,.4,.9]})
            snap=api.capture(env,np.zeros(4,np.float32))
            expected=env.np_random.normal(size=4)
            env.step(np.ones(4,np.float32)*.8); env.target_position[:]=[2,2,1.5]
            api.restore(env,snap)
            np.testing.assert_array_equal(env.np_random.normal(size=4),expected)
            np.testing.assert_array_equal(env.target_position,[.3,.4,.9])
            self.assertEqual(env._integral_active['steps'],[])
            self.assertFalse(env._done); self.assertEqual(env.success_streak,0)
            def run(a):
                api.restore(env,snap)
                return [env.step(a)[0] for _ in range(10)]
            zero=run(np.zeros(4,np.float32)); run(np.ones(4,np.float32)*.5)
            np.testing.assert_array_equal(run(np.zeros(4,np.float32)),zero)
        finally: env.close()


if __name__=='__main__': unittest.main()
