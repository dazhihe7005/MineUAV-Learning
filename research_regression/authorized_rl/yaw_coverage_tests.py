"""R24 bounded real reset/physics/model contracts; no learning implementation called."""
import importlib
import importlib.util
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from research_ops.storage import read_json,digest,process_identity
from .runtime import ROOT


class YawCoverageTensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*_a,**_k):raise AssertionError('NO_OPTIMIZER_OR_ROLLOUT_R24_ADMISSION')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            p=patch.object(cls,name,forbidden);p.start();self.addCleanup(p.stop)
        from lidar_integration.ppo_formal_scheduler import load_l6_prereg
        self.prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')

    def module(self,name='yaw_coverage_env'):
        path='research_regression.authorized_rl.'+name
        self.assertIsNotNone(importlib.util.find_spec(path),'missing R24 scheduler or dispatch profile')
        return importlib.import_module(path)

    def scheduler(self,seed=0):
        env=self.module().make_yaw_scheduler(self.prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=seed)
        self.addCleanup(env.close);return env

    def test_all120_train_tasks_actual_pose_reset_lidar_pi_and_physics_tick_contract(self):
        env=self.scheduler();self.assertEqual(len(env.by_identity),120)
        for stage in ('SHORT','ORIGINAL'):
            for row in env.manifest[stage]:
                # FORCED_TEST_SCHEDULER_COUNTERS: finite initialization coverage,
                # never represented as completed policy training or navigation.
                env.completed_policy_steps=0 if stage=='SHORT' else 32768
                obs,info=env.reset(options={'scenario_identity_sha256':row['identity_sha256']})
                plant=env.active.unwrapped
                np.testing.assert_allclose(obs['heading'],[math.sin(row['start_yaw_rad']),math.cos(row['start_yaw_rad'])],atol=1e-6)
                np.testing.assert_array_equal(plant.data.qpos[:3],row['start_xyz'])
                np.testing.assert_array_equal(plant.target_position,row['target_xyz'])
                np.testing.assert_array_equal(plant.data.qvel,0.)
                np.testing.assert_array_equal(plant.velocity_controller.integral_error,0.)
                self.assertEqual((float(plant.data.time),plant.lidar_frame_count,env.active_environment_count),(0.,1,1))
                self.assertTrue(np.isfinite(obs['lidar']).all());self.assertEqual(obs['lidar'].shape,(3,12,24))
                self.assertEqual(info['scenario_identity_sha256'],row['identity_sha256'])
                self.assertEqual(info['task_start_yaw_rad'],row['start_yaw_rad'])
                _,_,t,tr,step=env.step(np.zeros(4,dtype=np.float32))
                self.assertFalse(t or tr);self.assertEqual((step['physics_steps'],step['controller_updates']),(20,4))
                self.assertAlmostEqual(float(plant.data.time),.04,places=9)
        env.close();self.assertEqual(env.active_environment_count,0)

    def test_yaw_rng_independent_base_draws_reproducible_and_invalid_reset_latches(self):
        def sample(seed):
            env=self.scheduler(seed);rng=np.random.default_rng(seed);pairs=[]
            for _ in range(12):
                obs,info=env.reset();expected=self.prereg['scenario_manifest']['train'][int(rng.integers(12))]['identity_sha256']
                self.assertEqual(info['base_scenario_identity_sha256'],expected)
                pairs.append((expected,info['yaw_option']))
            env.completed_policy_steps=32768
            for _ in range(8):
                _,info=env.reset();i=int(rng.choice(12,p=[1/6 if r['layout']=='empty' else 1/18 for r in self.prereg['scenario_manifest']['train']]))
                self.assertEqual(info['base_scenario_identity_sha256'],self.prereg['scenario_manifest']['train'][i]['identity_sha256'])
            with self.assertRaises(ValueError):env.reset(options={'scenario_identity_sha256':self.prereg['scenario_manifest']['validation'][0]['identity_sha256']})
            with self.assertRaises(RuntimeError):env.reset()
            env.close();return pairs
        self.assertEqual(sample(21),sample(21));self.assertNotEqual(sample(21),sample(22))

    def test_fresh_step0_same_policy_temporal_parameters_and_checkpoint_identity(self):
        scheduler=self.scheduler()
        from .state_centered_policy import make_centered_ppo,verify_centered_policy
        from .centered_noise_training import initialize_noise_training
        from .yaw_coverage import verify_yaw_temporal_configuration
        from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
        from lidar_integration.ppo_formal_checkpoint import CheckpointManager
        from lidar_integration.ppo_training_runner import _weights_sha
        with tempfile.TemporaryDirectory() as d:
            env=FormalHealthMetricsEnv(scheduler,diagnostics_dir=Path(d));model=make_centered_ppo(env,seed=0)
            manager=CheckpointManager(Path(d),self.prereg,execution_protocol_sha256='d'*64)
            init,event,probe=initialize_noise_training(model,scheduler,manager,identity='R24_NO_UPDATE_ONLY',seed=0)
            self.assertEqual(_weights_sha(model.policy),'650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4')
            verify_yaw_temporal_configuration(model);verify_centered_policy(model)
            restored=manager.load_verified(None,run_id='R24_NO_UPDATE_ONLY_STEP0')
            verify_yaw_temporal_configuration(restored)
            np.testing.assert_array_equal(restored.predict(probe[0],deterministic=True)[0],probe[1])
            self.assertEqual(restored.num_timesteps,0);self.assertFalse(restored.policy.optimizer.state)
            model._setup_learn(128,reset_num_timesteps=False)
            _,_,_,_,info=env.step(np.zeros(4,dtype=np.float32));self.assertEqual((env.policy_steps,env.physics_ticks),(1,20))
            self.assertIn(info['scenario_identity_sha256'],scheduler.by_identity)

    def test_actual_worker_uses_new_task_identity_before_any_learning_or_sampling(self):
        self.module();profile=self.module('yaw_coverage_profile')
        from . import worker
        from .budget import BudgetStore,grant
        from .yaw_coverage import verify_yaw_temporal_configuration
        class SamplingForbidden(Exception):pass
        with tempfile.TemporaryDirectory() as d:
            home=Path(d)/'grant';store=BudgetStore(home);store.initialize(expected_grant_sha256=digest(grant()))
            p=profile.yaw_experiment();p.update(registration_sha256='a'*64,source_identity_sha256='b'*64)
            try:store.reserve(p)
            except Exception as error:self.fail('R24 protocol not admitted to temporary test-only store: '+str(error))
            out=Path(d)/'out';out.mkdir();identity=p['experiment_id']
            permit={'experiment_id':identity,'parent_pid':os.getppid(),'parent_identity':process_identity(os.getppid()),'protocol_sha256':digest(p)}
            observed=[]
            def boundary(model,steps,**kwargs):
                verify_yaw_temporal_configuration(model)
                self.assertEqual((model.num_timesteps,model._n_updates,steps),(0,0,32768))
                self.assertFalse(model.policy.optimizer.state)
                record=read_json(out/f'{identity}_STEP0.json')['scheduler_state']
                self.assertEqual(record['yaw_task_count'],120)
                self.assertEqual(record['yaw_manifest_sha256'],p['yaw_manifest_sha256'])
                observed.append('STOP_BEFORE_SAMPLING');raise SamplingForbidden()
            with patch.object(worker,'HOME',home),patch.object(PPO,'learn',boundary):
                try:worker.train_worker(out,permit,p,{},identity)
                except Exception as error:self.assertIsInstance(error,SamplingForbidden,'wrong schema24 actual worker path: '+repr(error))
            self.assertEqual(observed,['STOP_BEFORE_SAMPLING'])
            self.assertFalse((out/'ppo_updates.jsonl').exists());self.assertFalse((out/'summary.json').exists())
