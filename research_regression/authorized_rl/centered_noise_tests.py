"""R16 real tensor equivalence/save/load/one-step chain, never optimization."""
import copy,importlib.util,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from stable_baselines3 import PPO
from .prior_tests import ObservationOnly
from .runtime import ROOT


class CenteredNoiseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        def forbidden(*a,**kw):raise AssertionError('NO_OPTIMIZER_R16_ADMISSION')
        for cls,name in ((PPO,'learn'),(PPO,'train'),(torch.optim.Adam,'step')):
            guard=patch.object(cls,name,forbidden);guard.start();self.addCleanup(guard.stop)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('research_regression.authorized_rl.centered_noise_training'),
                             'missing R16 fresh-noise initializer')
        from . import centered_noise_training
        return centered_noise_training

    def test_exact_same_62_initial_means_values_and_parameters_except_log_std(self):
        mod=self.module()
        from .state_centered_policy import make_centered_ppo,initialize_centered_prior
        from .centered_feedback import load_background
        from .feedback_spec import probe_cases
        old=make_centered_ppo(ObservationOnly(),seed=0);initialize_centered_prior(old)
        new=make_centered_ppo(ObservationOnly(),seed=0);init=mod.assign_noise_prior(new)
        old_sd=old.policy.state_dict();new_sd=new.policy.state_dict()
        self.assertEqual(set(old_sd),set(new_sd))
        self.assertEqual([k for k in old_sd if not torch.equal(old_sd[k],new_sd[k])],['log_std'])
        self.assertTrue(torch.equal(new.policy.log_std,torch.full((4,),np.log(.1),dtype=torch.float32)))
        obs,_,_,_=load_background();batch={k:np.repeat(v[None],62,axis=0) for k,v in obs.items()}
        batch['state']=np.asarray([c['state'] for c in probe_cases()],dtype=np.float32)
        t,_=new.policy.obs_to_tensor(batch)
        with torch.inference_mode():
            nd=new.policy.get_distribution(t).distribution;od=old.policy.get_distribution(t).distribution
            self.assertTrue(torch.equal(nd.mean,od.mean));self.assertTrue(torch.equal(new.policy.predict_values(t),old.policy.predict_values(t)))
            self.assertTrue(np.allclose(nd.stddev.numpy(),.1,atol=1e-8));self.assertTrue(np.allclose(od.stddev.numpy(),.3,atol=1e-8))
        self.assertFalse(new.policy.optimizer.state);self.assertTrue(new.policy.log_std.requires_grad)
        self.assertEqual(init['kind'],'FRESH_SEED0_HAND_PRIOR_CENTERED_INITIAL_STD01')
        before=copy.deepcopy(new.policy.state_dict())
        with self.assertRaises(ValueError):mod.assign_noise_prior(new)
        self.assertTrue(all(torch.equal(v,new.policy.state_dict()[k]) for k,v in before.items()))

    def test_actual_step0_checkpoint_reload_and_repaired_physics_hook(self):
        mod=self.module()
        from .state_centered_policy import make_centered_ppo,verify_centered_policy
        from .repaired_distance_training import make_repaired_scheduler
        from lidar_integration.ppo_formal_scheduler import load_l6_prereg
        from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv
        from lidar_integration.ppo_formal_checkpoint import CheckpointManager
        prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        scheduler=make_repaired_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR',seed=0);self.addCleanup(scheduler.close)
        with tempfile.TemporaryDirectory() as d:
            env=FormalHealthMetricsEnv(scheduler,diagnostics_dir=Path(d));self.addCleanup(env.close)
            model=make_centered_ppo(env,seed=0);manager=CheckpointManager(Path(d),prereg,execution_protocol_sha256='e'*64)
            init,event,probe=mod.initialize_noise_training(model,scheduler,manager,identity='R16_NO_UPDATE_ONLY',seed=0)
            restored=manager.load_verified(None,run_id='R16_NO_UPDATE_ONLY_STEP0');verify_centered_policy(restored)
            self.assertTrue(np.array_equal(restored.predict(probe[0],deterministic=True)[0],probe[1]))
            self.assertFalse(restored.policy.optimizer.state);self.assertEqual(restored.num_timesteps,0)
            self.assertTrue(np.allclose(restored.policy.log_std.exp().detach().numpy(),.1,atol=1e-8))
            labelled=mod.annotate_noise_validation({'training_timesteps_before_after':0,
                'policy_sha256_before_after':init['assigned_policy_sha256'],'model_status':'UNTRAINED_RANDOM_MEAN'})
            self.assertEqual(labelled['model_status'],'ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED');self.assertFalse(labelled['reused_historical_evidence'])
            with self.assertRaises(ValueError):mod.annotate_noise_validation({'training_timesteps_before_after':0,'policy_sha256_before_after':'20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001'})
            model._setup_learn(128,reset_num_timesteps=False);_,_,_,_,info=env.step(np.zeros(4,dtype=np.float32))
            self.assertEqual((env.policy_steps,env.physics_ticks),(1,20));self.assertEqual(info['distance_stage'],'SHORT')
            with self.assertRaises(ValueError):mod.initialize_noise_training(model,scheduler,manager,identity='AGAIN',seed=0)


if __name__=='__main__':unittest.main()
