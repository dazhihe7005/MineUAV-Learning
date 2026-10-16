"""Frozen checkpoint inference tests; no optimizer and no physics step."""
import importlib.util
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg,MultiScenarioTrainingEnv
from lidar_integration.ppo_formal_checkpoint import CheckpointManager
from lidar_integration.ppo_training_runner import _weights_sha
from .runtime import ROOT
from .feedback_spec import checkpoint_references,probe_cases


class FeedbackInferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_FEEDBACK_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        env=MultiScenarioTrainingEnv(self.prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
        try:
            self.base,_=env.reset(options={'scenario_identity_sha256':self.prereg['scenario_manifest']['train'][0]['identity_sha256']})
            self.assertEqual(float(env.active.unwrapped.data.time),0.)
            self.assertEqual(env.active.unwrapped.lidar_frame_count,1)
        finally:env.close()
        self.refs=checkpoint_references()

    def model(self,identity,step):
        ref=next(r for r in self.refs if r['experiment_id']==identity and r['actual_step']==step)
        return CheckpointManager(ROOT/ref['checkpoint_path'].rsplit('/',1)[0],self.prereg,
            execution_protocol_sha256=ref['protocol_sha256']).load_verified(None,run_id=ref['run_id'])

    def probe(self,model,base=None):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.feedback_probe'),'missing deterministic feedback probe')
        from .feedback_probe import infer_panel
        return infer_panel(model,self.base if base is None else base,probe_cases())

    def test_real_prior_checkpoint_matches_reference_without_updating_or_advancing_physics(self):
        from test_env_scripted_policy import scripted_action
        model=self.model('ARL_R2_PRIOR_SEED0',0)
        result=self.probe(model)
        self.assertEqual(len(result['rows']),62)
        for row in result['rows']:
            reference=scripted_action(np.array(row['state'],dtype=np.float32))
            self.assertLess(float(np.max(np.abs(np.array(row['normalized_action'])-reference))),.02)
        self.assertTrue(result['weights_unchanged'])
        self.assertTrue(result['optimizer_state_unchanged'])
        self.assertTrue(result['torch_rng_unchanged'])
        self.assertEqual(result['optimizer_updates'],0)

    def test_trained_checkpoint_optimizer_and_weights_unchanged_and_invalid_input_rejected(self):
        model=self.model('ARL_R3_RANDOM_STD03_SEED0',65536)
        self.assertTrue(model.policy.optimizer.state)
        before=_weights_sha(model.policy);r=self.probe(model)
        self.assertTrue(r['optimizer_state_unchanged']);self.assertEqual(before,_weights_sha(model.policy))
        for row in r['rows']:
            self.assertTrue(np.array_equal(np.clip(np.array(row['raw_mean'],dtype=np.float32),-1,1),np.array(row['normalized_action'],dtype=np.float32)))
        bad={k:v.copy() for k,v in self.base.items()};bad['lidar'][0,0,0]=np.nan
        with self.assertRaises(ValueError):self.probe(model,bad)
        bad={k:v.copy() for k,v in self.base.items()};bad['lidar'][1,0,0]=.5
        with self.assertRaises(ValueError):self.probe(model,bad)
        bad={k:v.copy() for k,v in self.base.items()};bad['state'][0]=101
        with self.assertRaises(ValueError):self.probe(model,bad)
        self.assertEqual(before,_weights_sha(model.policy))

    def test_saved_snapshot_is_bound_to_actual_input_and_replacement_rejected(self):
        from . import feedback_probe
        from research_ops.storage import atomic_json,OpsError
        self.assertTrue(callable(getattr(feedback_probe,'seal_snapshot',None)),'missing runtime snapshot binding')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            np.savez_compressed(p/'base_observation.npz',**self.base)
            np.savez_compressed(p/'sensor_frame.npz',test_only=np.array([1.]))
            atomic_json(p/'snapshot.json',{'kind':'FORCED_TEST_FIXTURE'},exclusive=True)
            seal=feedback_probe.seal_snapshot(p,self.base)
            feedback_probe.verify_snapshot(p,self.base,seal)
            changed={k:v.copy() for k,v in self.base.items()};changed['state'][0]+=.1
            with self.assertRaises(OpsError):feedback_probe.seal_snapshot(p,changed)
            with self.assertRaises(OpsError):feedback_probe.verify_snapshot(p,changed,seal)
            (p/'snapshot.json').write_text('{}')
            with self.assertRaises(OpsError):feedback_probe.verify_snapshot(p,self.base,seal)


if __name__=='__main__':unittest.main()
