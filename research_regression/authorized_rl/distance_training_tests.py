"""Distance training initialization/Validation interfaces: no optimizer allowed."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_formal_checkpoint import CheckpointManager
from lidar_integration.ppo_preflight import make_ppo
from lidar_integration.ppo_training_runner import _weights_sha
from .runtime import ROOT
from .short_goal_env import make_short_scheduler
from .short_goal import short_manifest


class DistanceTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_DISTANCE_INTERFACE_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        self.upstream=ROOT/'.research_local/dependencies/MuJoCo-LiDAR'

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.distance_training'),'missing distance training hooks')
        from . import distance_training
        return distance_training

    def test_fresh_std03_short_snapshot_is_same_control_weights_one_reset_and_no_update(self):
        m=self.module();env=make_short_scheduler(self.prereg,self.upstream,seed=0);self.addCleanup(env.close)
        model=make_ppo(env,seed=0)
        with tempfile.TemporaryDirectory() as d:
            manager=CheckpointManager(Path(d),self.prereg,execution_protocol_sha256='a'*64)
            initial,event,probe=m.initialize_distance_training(model,env,manager,identity='DISTANCE_NO_UPDATE_TEST',seed=0)
            self.assertEqual(_weights_sha(model.policy),'db33ba4df5f7aaf76f98b7915675cbfc80dbe52273045ed86811b5cc074dfa06')
            self.assertAlmostEqual(float(np.linalg.norm(model._last_obs['state'][0,:3])),.2,places=6)
            self.assertEqual(len(env.selected_tasks),1);self.assertEqual(model.num_timesteps,0)
            state=copy.deepcopy(env.scheduler_state());model._setup_learn(128,reset_num_timesteps=False)
            self.assertEqual(env.scheduler_state(),state)
            restored=manager.load_verified(None,run_id=event['checkpoint']['run_id'])
            self.assertEqual(_weights_sha(restored.policy),_weights_sha(model.policy));self.assertFalse(restored.policy.optimizer.state)
            self.assertTrue(np.array_equal(probe[1],restored.predict(probe[0],deterministic=True)[0]))

    def test_short_validation_is_independent_and_original_step0_is_explicit_old_evidence(self):
        m=self.module();env=make_short_scheduler(self.prereg,self.upstream,seed=0);self.addCleanup(env.close)
        model=make_ppo(env,seed=0)
        with tempfile.TemporaryDirectory() as d:
            manager=CheckpointManager(Path(d),self.prereg,execution_protocol_sha256='a'*64)
            m.initialize_distance_training(model,env,manager,identity='DISTANCE_VALIDATION_TEST',seed=0)
        before=_weights_sha(model.policy);state=copy.deepcopy(env.scheduler_state());env.close()
        evaluator=m.ShortValidationEvaluator(self.prereg,self.upstream)
        self.assertEqual([r['identity_sha256'] for r in evaluator.rows],[r['identity_sha256'] for r in short_manifest(self.prereg)['validation']])
        result=evaluator.evaluate(model,max_policy_steps=1)
        self.assertEqual(len(result['episodes']),6);self.assertTrue(result['probe_only'])
        self.assertEqual(_weights_sha(model.policy),before);self.assertFalse(model.policy.optimizer.state);self.assertEqual(model.num_timesteps,0)
        self.assertEqual(env.scheduler_state()['rng_state'],state['rng_state'])
        old=m.reuse_original_step0(before)
        self.assertTrue(old['reused_historical_evidence']);self.assertEqual(old['newly_executed_episodes'],0)
        self.assertEqual(old['overall']['total'],6)
        with self.assertRaises(ValueError):m.reuse_original_step0('0'*64)


if __name__=='__main__':unittest.main()
