"""R17 real MuJoCo reset/step and RNG contracts, no optimizer updates."""
import copy,importlib.util,unittest
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from lidar_integration.ppo_formal_scheduler import load_l6_prereg
from .runtime import ROOT


class EmptyRehearsalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R17_TEST')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            g=patch.object(cls,name,forbidden);g.start();self.addCleanup(g.stop)
        self.p=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.empty_rehearsal'),
                             'missing R17 bounded reset adapter')
        from . import empty_rehearsal
        return empty_rehearsal

    def scheduler(self):
        s=self.module().make_rehearsal_scheduler(self.p,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
        self.addCleanup(s.close);return s

    def test_short_prefix_same_rng_obs_reward_and_trajectory(self):
        from .repaired_distance_training import make_repaired_scheduler
        results=[]
        for kind in ('old','new'):
            s=(make_repaired_scheduler(self.p,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0)
               if kind=='old' else self.scheduler())
            try:
                seq=[]
                for _ in range(3):
                    obs,i=s.reset();tr=s.step(np.array([.1,-.2,.1,.05],dtype=np.float32))
                    seq.append((i['scenario_identity_sha256'],obs,tr,s.active.unwrapped.data.qpos.copy()))
                results.append((seq,s.scheduler_state()['rng_state']))
            finally:s.close()
        self.assertEqual(results[0][1],results[1][1])
        for a,b in zip(results[0][0],results[1][0]):
            self.assertEqual(a[0],b[0]);self.assertEqual(a[2][1:4],b[2][1:4]);self.assertEqual(a[2][4]['reward_breakdown'],b[2][4]['reward_breakdown'])
            for k in a[1]:np.testing.assert_array_equal(a[1][k],b[1][k])
            np.testing.assert_array_equal(a[3],b[3])

    def test_frozen_weights_and_reproducible_sampling_not_uniform(self):
        mod=self.module();rows=self.p['scenario_manifest']['train'];w=mod.rehearsal_weights(self.p)
        self.assertEqual(len(w),12)
        for row,v in zip(rows,w):self.assertAlmostEqual(v,1/6 if row['layout']=='empty' else 1/18)
        sequences=[]
        for seed in (0,0,1):
            rng=np.random.default_rng(seed)
            sequences.append([mod.draw_rehearsal_identity(self.p,rng) for _ in range(10000)])
        self.assertEqual(sequences[0],sequences[1]);self.assertNotEqual(sequences[0],sequences[2])
        empty={r['identity_sha256'] for r in rows if r['layout']=='empty'}
        self.assertEqual(set(sequences[0]),{r['identity_sha256'] for r in rows})
        self.assertGreater(sum(i in empty for i in sequences[0]),4800);self.assertLess(sum(i in empty for i in sequences[0]),5200)
        bad=copy.deepcopy(self.p);bad['scenario_manifest']['train'][0]['target_xyz'][0]+=1
        with self.assertRaises(ValueError):mod.rehearsal_weights(bad)

    def test_natural_reset_boundary_all_train_clean_no_mid_episode_target_change(self):
        s=self.scheduler();s.reset();goal=s.active.unwrapped.target_position.copy()
        s.env.completed_policy_steps=32767  # FORCED_TEST_COUNTER, not actual training.
        s.step(np.zeros(4,dtype=np.float32));np.testing.assert_array_equal(s.active.unwrapped.target_position,goal)
        _,i=s.reset();self.assertEqual(i['distance_stage'],'ORIGINAL');self.assertEqual(s.phase_switch_step,32768)
        self.assertEqual(s.scheduler_state()['rehearsal_sample_counts']['WEIGHTED_ORIGINAL'],1)
        for row in self.p['scenario_manifest']['train']:
            rng=copy.deepcopy(s.env._rng.bit_generator.state)
            obs,i=s.reset(options={'scenario_identity_sha256':row['identity_sha256']});p=s.active.unwrapped
            self.assertEqual(i['scenario_identity_sha256'],row['identity_sha256']);self.assertEqual(s.env._rng.bit_generator.state,rng)
            np.testing.assert_array_equal(p.target_position,row['target_xyz']);np.testing.assert_array_equal(p.data.qvel,np.zeros(6))
            np.testing.assert_array_equal(p.velocity_controller.integral_error,np.zeros(3))
            self.assertEqual(float(p.data.time),0);self.assertEqual(p.lidar_frame_count,1);self.assertEqual(float(obs['lidar_age'][0]),0)
            s.step(np.zeros(4,dtype=np.float32));self.assertEqual(s.active_environment_count,1)

    def test_bad_identity_or_seed_latches_and_post_seed_is_reproducible(self):
        for bad in [self.p['scenario_manifest'][split][0]['identity_sha256'] for split in ('validation','test')]+['0'*64]:
            s=self.scheduler();s.env.completed_policy_steps=32768
            with self.assertRaises(ValueError):s.reset(options={'scenario_identity_sha256':bad})
            with self.assertRaises(RuntimeError):s.reset()
            s.close()
        s=self.scheduler();s.env.completed_policy_steps=32768
        with self.assertRaises(ValueError):s.reset(seed=True)
        with self.assertRaises(RuntimeError):s.reset()
        a=self.scheduler();b=self.scheduler()
        for s in (a,b):s.env.completed_policy_steps=32768
        self.assertEqual(a.reset(seed=7)[1]['scenario_identity_sha256'],b.reset(seed=7)[1]['scenario_identity_sha256'])
        self.assertEqual(a.scheduler_state()['rng_state'],b.scheduler_state()['rng_state'])

    def test_full_metrics_and_step0_checkpoint_keep_actual_weighted_scheduler_state(self):
        import tempfile
        from pathlib import Path
        from .state_centered_policy import make_centered_ppo,verify_centered_policy
        from .centered_noise_training import initialize_noise_training
        from lidar_integration.ppo_formal_checkpoint import CheckpointManager
        from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
        s=self.scheduler()
        with tempfile.TemporaryDirectory() as d:
            env=FormalHealthMetricsEnv(s,diagnostics_dir=Path(d));self.addCleanup(env.close)
            model=make_centered_ppo(env,seed=0);manager=CheckpointManager(Path(d),self.p,execution_protocol_sha256='a'*64)
            init,event,probe=initialize_noise_training(model,s,manager,identity='R17_NO_UPDATE_ONLY',seed=0)
            restored=manager.load_verified(None,run_id='R17_NO_UPDATE_ONLY_STEP0');verify_centered_policy(restored)
            self.assertFalse(restored.policy.optimizer.state)
            np.testing.assert_array_equal(restored.predict(probe[0],deterministic=True)[0],probe[1])
            self.assertEqual(event['scheduler']['rehearsal_sample_counts']['UNIFORM_SHORT'],1)
            env.step(np.zeros(4,dtype=np.float32));self.assertEqual((env.policy_steps,env.physics_ticks),(1,20))
            s.env.completed_policy_steps=32768  # FORCED_TEST_COUNTER, no optimization.
            env.reset();_,_,_,_,info=env.step(np.zeros(4,dtype=np.float32))
            self.assertEqual(info['distance_stage'],'ORIGINAL');self.assertEqual(env.physics_ticks,40)
