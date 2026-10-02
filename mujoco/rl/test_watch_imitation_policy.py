"""The viewer entry point must execute the saved actor, not the teacher."""

import unittest

import torch

from mine_uav_env import MineUAVEnv
from scripted_imitation_sanity import ActorMeanMLP, ImitationPolicy
from watch_imitation_policy import run_episode


class WatchImitationPolicyTests(unittest.TestCase):
    def test_episode_sends_actor_action_to_existing_velocity_controller(self):
        actor = ActorMeanMLP()
        with torch.no_grad():
            actor.action_net.weight.zero_()
            actor.action_net.bias[:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        env = MineUAVEnv(max_episode_seconds=0.08, reward_version="v2")
        try:
            result = run_episode(env, ImitationPolicy(actor), seed=20261001,
                                 status_sink=lambda message: None)
            self.assertEqual(result["policy_steps"], 2)
            self.assertEqual(result["reason"], "time_limit")
            self.assertAlmostEqual(result["last_command_m_s"][0], 1.5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
