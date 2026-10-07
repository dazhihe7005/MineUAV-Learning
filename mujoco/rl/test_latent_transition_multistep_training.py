"""Catches detach, teacher leakage, bad endpoints, changed normalization/selection."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from explicit_latent_models import ResidualTransition,ObservationDecoder,freeze_encoder,load_component
from run_latent_memory_ablation import parameter_hash
from test_latent_dynamics_multistep import episode,statistics

try:
    import latent_transition_multistep_training as training
    import latent_transition_multistep_evaluation as evaluation
except ImportError:
    training=evaluation=None


class ActionDelta(nn.Module):
    def forward(self,z,a):
        out=torch.zeros_like(z); out[:,0]=a[:,0]; return out


class StateDelta(nn.Module):
    def __init__(self):
        super().__init__(); self.scale=nn.Parameter(torch.tensor(.1))
    def forward(self,z,a): return z*self.scale


class CopyDecoder(nn.Module):
    def forward(self,z): return z[:,:7]


def latent_episode(n,target=1):
    ep=episode(n,target); ep['latent']=np.zeros((n+1,64),np.float32)
    ep['latent'][:,0]=np.arange(n+1)*.1
    return ep


def latent_stats(): return dict(mean=[0.]*64,std=[1.]*64)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(training,'multistep transition trainer missing')
        torch.set_num_threads(1); torch.manual_seed(0)

    def test_ten_step_action_recursion_reuses_latent_std_not_delta_std(self):
        latent=dict(mean=[10.]*64,std=[2.]*64); actions=torch.zeros((1,10,4),dtype=torch.float64)
        actions[0,:,0]=torch.arange(1,11,dtype=torch.float64)
        out=training.predict_latents(ActionDelta(),torch.ones((1,64),dtype=torch.float64)*10,actions,statistics(),latent)
        np.testing.assert_array_equal(out['predictions'][0,:,0].detach().numpy(),[12,16,22,30,40,52,66,82,100,120])

    def test_last_step_gradient_reaches_earliest_predicted_latent(self):
        model=StateDelta(); initial=torch.ones((1,64),dtype=torch.float64)
        out=training.predict_latents(model,initial,torch.zeros((1,10,4)),statistics(),latent_stats())
        for step in out['steps']: step.retain_grad()
        out['predictions'][:,-1].square().mean().backward()
        self.assertGreater(out['steps'][0].grad.abs().sum().item(),0)
        self.assertGreater(model.scale.grad.abs().item(),0)

    def test_future_teacher_targets_corrupted_or_removed_cannot_change_forward(self):
        ep=latent_episode(12); net=ResidualTransition(); stats=statistics(); latent=latent_stats()
        batch=training.window_batch([ep],10,[np.array([0])])
        initial=batch['initial'].clone(); action=batch['actions'].clone()
        a=training.predict_latents(net,initial,action,stats,latent)['predictions']
        altered=copy.deepcopy(ep); altered['latent'][1:]+=1000
        b=training.window_batch([altered],10,[np.array([0])])
        result=training.predict_latents(net,b['initial'],b['actions'],stats,latent)['predictions']
        torch.testing.assert_close(a,result,rtol=0,atol=0)
        self.assertFalse(torch.equal(batch['truth'],b['truth']))
        del b['truth']; del altered['latent']
        c=training.predict_latents(net,b['initial'],b['actions'],stats,latent)['predictions']
        torch.testing.assert_close(a,c,rtol=0,atol=0)

    def test_windows_preserve_last_pair_and_never_join_episodes(self):
        eps=[latent_episode(12),latent_episode(10,2),latent_episode(4,3)]
        batch=training.window_batch(eps,10)
        self.assertEqual(batch['initial'].shape,(4,64)); self.assertEqual(batch['truth'].shape,(4,10,64))
        self.assertAlmostEqual(batch['truth'][2,-1,0].item(),1.2,places=6)
        self.assertAlmostEqual(batch['truth'][3,-1,0].item(),1.,places=6)
        with self.assertRaisesRegex(ValueError,'boundary'):
            training.window_batch([eps[0]],10,[np.array([3])])
        with self.assertRaisesRegex(ValueError,'window'):
            training.window_batch([eps[2]],10)

    def test_uniform_loss_uses_all_steps_and_exact_latent_std(self):
        pred=torch.ones((1,10,64),dtype=torch.float64); truth=torch.zeros_like(pred)
        latent=dict(mean=[123.]*64,std=[2.]*64)
        self.assertEqual(training.latent_loss(pred,truth,latent).item(),.25)
        pred[:,0]*=2
        self.assertAlmostEqual(training.latent_loss(pred,truth,latent).item(),.325)

    def test_same_seed_init_val_best_reload_and_frozen_modules_unchanged(self):
        frozen=[freeze_encoder(nn.GRU(11,64,batch_first=True)),freeze_encoder(ObservationDecoder())]
        hashes=[parameter_hash(m) for m in frozen]
        torch.manual_seed(0); expected=parameter_hash(ResidualTransition())
        cfg=dict(seed=0,epochs=3,batch_size=2,learning_rate=.001,horizon=10)
        eps=[latent_episode(12),latent_episode(11,2)]; val=[latent_episode(10,3)]
        with tempfile.TemporaryDirectory() as td:
            a=training.train_transition(eps,val,statistics(),latent_stats(),cfg,Path(td)/'a.pt')
            b=training.train_transition(eps,val,statistics(),latent_stats(),cfg,Path(td)/'b.pt')
            net,saved,latent,meta=load_component(Path(td)/'a.pt')
            self.assertEqual(a['initial_parameter_sha256'],expected)
            self.assertEqual(a['history'],b['history']); self.assertEqual(a['parameter_count'],33600)
            self.assertEqual(a['best_epoch'],np.argmin([r['validation_loss'] for r in a['history']])+1)
            self.assertAlmostEqual(training.validation_loss(net,val,saved,latent,10,2),a['best_validation_loss'])
            self.assertEqual(meta['selection_metric'],'Validation uniform K10 autoregressive normalized latent MSE')
        self.assertEqual(hashes,[parameter_hash(m) for m in frozen])
        self.assertTrue(all(p.grad is None and not p.requires_grad for m in frozen for p in m.parameters()))


class FloorTests(unittest.TestCase):
    def test_floor_decodes_teacher_endpoint_with_exact_window_counts(self):
        self.assertIsNotNone(evaluation,'decoder floor missing')
        ep=latent_episode(3); ep['latent'][:,0]=[0,1,2,3]
        row=evaluation.decoder_floor(CopyDecoder(),[ep],statistics(),latent_stats(),[.1,.2],[1,2])
        # fixture next error_x=[2,3,4]; D teacher z_next=[1,2,3], constant -1 error.
        self.assertEqual(row['horizons']['1']['window_count'],3)
        self.assertEqual(row['horizons']['2']['window_count'],2)
        self.assertEqual(row['horizons']['2']['physical']['error_x']['rmse'],1.)
        self.assertEqual(row['common_max_horizon_windows']['1']['window_count'],2)
        self.assertAlmostEqual(row['horizons']['2']['normalized_observation_rmse'],1/np.sqrt(7))


if __name__=='__main__': unittest.main()
