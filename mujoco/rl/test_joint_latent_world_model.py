"""Literal raw-coordinate / causality / full-gradient / selection contracts."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from explicit_latent_models import ResidualTransition,ObservationDecoder,save_component
from latent_dynamics_models import HistoryLatentDynamics,save_model
from run_latent_memory_ablation import parameter_hash

try:
    import joint_latent_world_model as core
    import joint_latent_training as trainer
except ImportError:
    core=trainer=None


def fixture(root):
    torch.manual_seed(0)
    stats={k:dict(mean=np.linspace(-.1,.2,d).tolist(),std=np.linspace(.5,1.2,d).tolist())
           for k,d in [('obs',7),('action',4),('previous_action',4)]}
    latent=dict(mean=np.linspace(-.2,.3,64).tolist(),std=np.linspace(.12,.8,64).tolist())
    direct=HistoryLatentDynamics(); t=ResidualTransition(); d=ObservationDecoder()
    paths=[Path(root)/n for n in ['encoder.pt','transition.pt','decoder.pt']]
    save_model(paths[0],direct,'history',stats,{})
    save_component(paths[1],t,'transition',stats,latent,{})
    save_component(paths[2],d,'decoder',stats,latent,{})
    ep=dict(obs=np.arange(84,dtype=np.float64).reshape(12,7)/100,
            next_obs=np.arange(7,91,dtype=np.float64).reshape(12,7)/100,
            action=np.arange(48,dtype=np.float64).reshape(12,4)/100,
            previous_action=np.vstack([np.zeros(4),np.arange(44).reshape(11,4)/100]))
    return paths,stats,latent,direct,t,d,ep


class JointTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(core,'joint model core missing')
        self.assertIsNotNone(trainer,'joint trainer missing')
        torch.set_num_threads(1)

    def test_pretrained_raw_affine_conversion_preserves_functions_and_all_trainable(self):
        with tempfile.TemporaryDirectory() as td:
            paths,stats,ls,e,t,d,_=fixture(td)
            net,saved,meta=core.initialize_joint(*paths)
            self.assertEqual(saved,stats)
            self.assertEqual(parameter_hash(net.encoder),parameter_hash(e.encoder))
            z=torch.linspace(-.5,.5,192).reshape(3,64); a=torch.ones(3,4)*.13
            m=torch.tensor(ls['mean']); scale=torch.tensor(ls['std'])
            old=z+t((z-m)/scale,a)*scale; new=z+net.transition(z,a)
            torch.testing.assert_close(new,old,rtol=2e-6,atol=3e-7)
            torch.testing.assert_close(net.decoder(z),d((z-m)/scale),rtol=2e-6,atol=3e-7)
            self.assertTrue(all(p.requires_grad for p in net.parameters()))
            self.assertFalse(any('latent_mean' in k or 'latent_std' in k for k in net.state_dict()))
            self.assertEqual(sum(p.numel() for p in net.parameters()),74119)

    def test_literal_raw_residual_actions_and_uniform_current_plus_future_loss(self):
        class T(torch.nn.Module):
            def forward(self,z,a): return a[:,:1].expand_as(z)
        class D(torch.nn.Module):
            def forward(self,z): return z[:,:7]
        net=core.JointLatentWorldModel(torch.nn.GRU(11,64,batch_first=True),T(),D())
        stats={k:dict(mean=[0.]*d,std=[1.]*d) for k,d in [('obs',7),('action',4),('previous_action',4)]}
        z=torch.zeros(1,64); actions=torch.zeros(1,10,4); actions[0,:,0]=torch.arange(1,11)
        out=core.predict_latent(net,z,actions,stats)
        self.assertEqual(out['normalized_observations'].shape,(1,11,7))
        self.assertEqual(out['latents'].shape,(1,11,64))
        self.assertEqual(out['normalized_observations'][0,:,0].tolist(),[0.,1.,3.,6.,10.,15.,21.,28.,36.,45.,55.])
        self.assertAlmostEqual(trainer.observation_loss(torch.ones(1,11,7),torch.zeros(1,11,7),stats).item(),1.)
        current_only=torch.zeros(1,11,7); current_only[:,0]=1
        self.assertAlmostEqual(trainer.observation_loss(current_only,torch.zeros_like(current_only),stats).item(),1/11,places=7)

    def test_prefix_and_later_rollout_bptt_reaches_all_modules_without_detach(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td); net,_,_=core.initialize_joint(*paths)
            out=trainer.predict_windows(net,[ep],s,10,[np.array([2])])
            out['prefix_latents'].retain_grad(); out['steps'][1].retain_grad()
            loss=trainer.observation_loss(out['normalized_observations'][:,-1:],out['truth'][:,-1:],s)
            loss.backward()
            self.assertGreater(out['prefix_latents'].grad.norm().item(),0.)
            self.assertGreater(out['prefix_latents'].grad[0,2].norm().item(),0.)
            self.assertGreater(out['steps'][1].grad.norm().item(),0.)
            for component in (net.encoder,net.transition,net.decoder):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in component.parameters() if p.grad is not None),0.)
            self.assertTrue(out['latents'].requires_grad)

    def test_true_prefix_cache_causal_future_corruption_deletion_and_action_index(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td); net,_,_=core.initialize_joint(*paths)
            out=trainer.predict_windows(net,[ep],s,5,[np.array([2])])
            warm=core.encode_episode(net,ep['obs'][:3],ep['previous_action'][:3],s)[-1:]
            # GRU matrix shapes differ between full episode and short prefix;
            # allow FP32 roundoff here, NOT in identical-call leakage checks.
            torch.testing.assert_close(out['latents'][:,0],warm,rtol=1e-6,atol=5e-8)
            altered=copy.deepcopy(ep); altered['obs'][3:]+=100
            cached=trainer.predict_windows(net,[altered],s,5,[np.array([2])])
            self.assertTrue(torch.equal(out['normalized_observations'],cached['normalized_observations']))
            original=core.predict_window(net,ep,2,5,s)
            changed=copy.deepcopy(ep); changed['obs'][3:]+=100; changed['next_obs'][:]+=200
            corrupted=core.predict_window(net,changed,2,5,s)
            changed['obs']=changed['obs'][:3]; changed['previous_action']=changed['previous_action'][:3]; del changed['next_obs']
            deleted=core.predict_window(net,changed,2,5,s)
            for key in ['latents','normalized_observations']:
                self.assertTrue(torch.equal(original[key],corrupted[key]))
                self.assertTrue(torch.equal(original[key],deleted[key]))
            direct=core.predict_latent(net,warm,torch.tensor(ep['action'][None,2:7]),s)
            torch.testing.assert_close(original['latents'],direct['latents'],rtol=0,atol=0)

    def test_episode_reset_boundary_and_window_count_no_cross_episode(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,a=fixture(td); net,_,_=core.initialize_joint(*paths)
            b=copy.deepcopy(a); b['obs']=-b['obs']*3
            first=core.encode_episode(net,a['obs'],a['previous_action'],s)
            core.encode_episode(net,b['obs'],b['previous_action'],s)
            again=core.encode_episode(net,a['obs'],a['previous_action'],s)
            self.assertTrue(torch.equal(first,again))
            self.assertEqual(trainer.predict_windows(net,[a,b],s,10)['window_count'],6)
            with self.assertRaisesRegex(ValueError,'boundary'):
                trainer.predict_windows(net,[a],s,10,[np.array([3])])

    def test_selection_reload_deterministic_and_each_component_updates(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td); cfg=dict(seed=0,epochs=3,batch_size=2,learning_rate=.0003,horizon=10)
            outputs=[]
            for i in range(2):
                model,_,_=core.initialize_joint(*paths); p=Path(td)/f'joint{i}.pt'
                result=trainer.train_joint(model,[ep],[ep],s,cfg,p)
                loaded,ss,meta=core.load_joint(p); self.assertEqual(ss,s)
                self.assertEqual(meta['best_epoch'],int(np.argmin([r['validation_loss'] for r in result['history']]))+1)
                self.assertAlmostEqual(trainer.validation_loss(loaded,[ep],s,10,2),result['best_validation_loss'],places=10)
                for k in ['encoder','transition','decoder']:
                    self.assertNotEqual(result['parameter_hashes_before'][k],result['parameter_hashes_best'][k])
                    self.assertGreater(result['history'][0]['gradient_norms'][k]['max'],0.)
                outputs.append((parameter_hash(loaded),[r['validation_loss'] for r in result['history']]))
            self.assertEqual(outputs[0],outputs[1])

    def test_latent_variance_diagnostics_identify_constant_collapse(self):
        constant=np.ones((12,64))
        result=core.latent_summary(constant)
        self.assertEqual(result['near_zero_std_dimension_count'],64)
        self.assertEqual(result['per_dimension_std_min'],0.)
        self.assertEqual(result['effective_rank'],0.)
        varying=np.vstack([np.eye(64),-np.eye(64)])
        result=core.latent_summary(varying)
        self.assertEqual(result['near_zero_std_dimension_count'],0)
        self.assertAlmostEqual(result['effective_rank'],64.,places=7)


if __name__=='__main__': unittest.main()
