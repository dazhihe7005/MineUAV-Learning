"""Real MuJoCo reset and SB3 step-zero snapshot tests, optimizer forbidden."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_preflight import make_ppo
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_formal_checkpoint import CheckpointManager
from lidar_integration.ppo_training_runner import _weights_sha
from research_regression.authorized_rl.runtime import ROOT
from research_regression.authorized_rl.curriculum import make_scheduler


class InitialSnapshotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_SNAPSHOT_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        self.env=make_scheduler(self.prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR','UNIFORM',0)
        self.addCleanup(self.env.close)
        self.model=make_ppo(self.env,seed=0)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.manager=CheckpointManager(Path(self.tmp.name),self.prereg,execution_protocol_sha256='a'*64)

    def snapshot(self):
        from research_regression.authorized_rl.prior_training import initialize_training_prior
        return initialize_training_prior(self.model,self.env,self.manager,identity='UNTRAINED_SNAPSHOT_TEST',seed=0)

    def test_initial_snapshot_is_zero_steps_and_setup_learn_does_not_reset_again(self):
        torch_rng=torch.get_rng_state().clone()
        info,event,probe=self.snapshot()
        self.assertTrue(torch.equal(torch_rng,torch.get_rng_state()))
        self.assertEqual(event['actual_step'],0)
        self.assertEqual(event['checkpoint']['training_timesteps'],0)
        self.assertEqual(len(self.env.selected_identities),1)
        self.assertEqual(self.env.completed_policy_steps,0)
        self.assertTrue(np.array_equal(self.model._last_episode_starts,[True]))
        state=copy.deepcopy(self.env.scheduler_state());obs={k:v.copy() for k,v in self.model._last_obs.items()}
        total,_=self.model._setup_learn(128,reset_num_timesteps=False)
        self.assertEqual(total,128)
        self.assertEqual(self.env.scheduler_state(),state)
        self.assertTrue(all(np.array_equal(obs[k],v) for k,v in self.model._last_obs.items()))
        self.assertEqual(float(self.env.active.unwrapped.data.time),0)
        reloaded=self.manager.load_verified(None,run_id=event['checkpoint']['run_id'])
        self.assertEqual(_weights_sha(reloaded.policy),info['assigned_policy_sha256'])
        self.assertEqual(reloaded.num_timesteps,0)
        self.assertFalse(reloaded.policy.optimizer.state)
        self.assertTrue(np.array_equal(probe[1],reloaded.predict(probe[0],deterministic=True)[0]))

    def test_already_reset_model_is_rejected_before_assignment(self):
        self.model._last_obs=self.model.get_env().reset()
        before=_weights_sha(self.model.policy);state=copy.deepcopy(self.env.scheduler_state())
        with self.assertRaises(ValueError):self.snapshot()
        self.assertEqual(_weights_sha(self.model.policy),before)
        self.assertEqual(self.env.scheduler_state(),state)

    def test_wrong_fresh_weights_are_rejected_without_reset(self):
        with torch.no_grad():self.model.policy.action_net.weight[0,0]+=.1  # FORCED_TEST_STATE
        before=_weights_sha(self.model.policy)
        with self.assertRaises(ValueError):self.snapshot()
        self.assertEqual(_weights_sha(self.model.policy),before)
        self.assertEqual(len(self.env.selected_identities),0)


if __name__=='__main__':unittest.main()
