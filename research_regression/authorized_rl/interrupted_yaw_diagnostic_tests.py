"""R26 historical no-update loader test; not current execution admission."""
import copy
import importlib
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from lidar_integration.ppo_training_runner import _weights_sha
from lidar_integration.ppo_formal_validation import _optimizer_states_equal
from research_ops.storage import OpsError
from .budget import BudgetStore
from .interrupted_yaw_diagnostic import references


class InterruptedYawLoaderTests(unittest.TestCase):
    def test_trusted_pair_readonly_load_forward_identity_and_no_optimizer(self):
        name='research_regression.authorized_rl.interrupted_yaw_diagnostic_run'
        self.assertIsNotNone(importlib.util.find_spec(name),'missing fixed R26 readonly artifact loader')
        module=importlib.import_module(name);root=Path(__file__).resolve().parents[2]
        prereg=load_l6_prereg(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        # Explicit test-only reader, shared with the stdlib historical tests.
        # Full frozen report identity is verified inside the context; actual
        # checkpoint/proof checks and optimizer prohibitions remain unchanged.
        fixture_spec=importlib.util.spec_from_file_location('_r34_historical_budget_fixture',
            root/'tests/research_ops_v2/_historical_budget_fixture.py')
        fixture=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture)
        original=BudgetStore.status;actual_before=BudgetStore(root/'.research_local/authorized_rl_v1').status()
        calls=[]
        def forbidden(*args,**kwargs):calls.append('FORBIDDEN');raise AssertionError('NO_LEARNING_IN_R26_TEST')
        with fixture.r28_budget_reader(root),patch.object(PPO,'learn',forbidden),patch.object(PPO,'train',forbidden),patch.object(torch.optim.Adam,'step',forbidden):
            refs=references(root)['models']
            for ref in refs:
                with self.assertRaises(OpsError):module.load_readonly_model({**ref,'checkpoint_sha256':'0'*64},prereg,root)
                model,meta=module.load_readonly_model(ref,prereg,root)
                before=_weights_sha(model.policy);optimizer=copy.deepcopy(model.policy.optimizer.state_dict())
                obs={k:np.zeros(space.shape,dtype=np.float32) for k,space in model.observation_space.spaces.items()}
                obs['heading'][:]=[0.,1.]
                with torch.inference_mode():
                    a,_=model.predict(obs,deterministic=True);b,_=model.predict(obs,deterministic=True)
                np.testing.assert_array_equal(a,b);self.assertEqual(a.shape,(4,));self.assertTrue(np.isfinite(a).all())
                self.assertEqual(before,_weights_sha(model.policy));self.assertEqual(model.num_timesteps,ref['step'])
                self.assertTrue(_optimizer_states_equal(optimizer,model.policy.optimizer.state_dict()))
                self.assertEqual(meta['legacy_loader_mode'],'WARM_START_NAME_DESERIALIZATION_ONLY_NO_TRAINING')
                self.assertEqual(meta['policy_sha256'],before);self.assertTrue(all(p.grad is None for p in model.policy.parameters()))
                del model
        self.assertEqual(calls,[])
        self.assertIs(BudgetStore.status,original)
        self.assertEqual(BudgetStore(root/'.research_local/authorized_rl_v1').status(),actual_before)
