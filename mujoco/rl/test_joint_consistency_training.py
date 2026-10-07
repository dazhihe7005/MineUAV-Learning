"""Independent loss literals, gradient boundaries and unchanged v1 prediction."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_joint_latent_world_model import fixture
from joint_latent_world_model import initialize_joint,component_hashes,predict_window
from joint_latent_training import predict_windows,observation_loss

try:
    import joint_consistency_training as trainer
except ImportError:
    trainer=None


class ConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(trainer,'consistency trainer missing')
        torch.set_num_threads(1)

    def test_literal_normalized_loss_detaches_only_target_and_fixed_lambda(self):
        source=torch.ones(2,10,64,requires_grad=True)
        target=torch.zeros(2,10,64,requires_grad=True)
        loss=trainer.consistency_mse(source,target,[2.]*64)
        self.assertAlmostEqual(loss.item(),.25)
        loss.backward()
        self.assertGreater(source.grad.norm().item(),0.)
        self.assertIsNone(target.grad)
        self.assertAlmostEqual(trainer.total_loss(torch.tensor(2.),torch.tensor(3.)).item(),2.3,places=6)
        for scale in ([0.]*64,[float('nan')]*64,[1.]*63):
            with self.assertRaises(ValueError): trainer.consistency_mse(source,target,scale)

    def test_reference_source_and_transition_receive_gradient_but_target_and_decoder_do_not(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td); net,_,_=initialize_joint(*paths)
            row=trainer.local_loss(net,[ep],s,[.2]*64,10,[np.array([2])])
            row['source_latents'].retain_grad(); row['target_latents'].retain_grad()
            row['loss'].backward()
            self.assertGreater(row['source_latents'].grad.norm().item(),0.)
            self.assertIsNone(row['target_latents'].grad)
            for component in (net.encoder,net.transition):
                self.assertGreater(sum(p.grad.abs().sum().item() for p in component.parameters() if p.grad is not None),0.)
            self.assertTrue(all(p.grad is None for p in net.decoder.parameters()))
            self.assertEqual(row['prediction'].shape,(1,10,64))
            # Last legal pair is z[N-1] -> z[N], not an invented padded target.
            fullobs=np.vstack([ep['obs'],ep['next_obs'][-1]])
            prev=np.vstack([ep['previous_action'],ep['action'][-1]])
            from joint_latent_world_model import encode_episode
            z=encode_episode(net,fullobs,prev,s)
            torch.testing.assert_close(row['target_latents'][0,-1],z[12],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'boundary'):
                trainer.local_loss(net,[ep],s,[.2]*64,10,[np.array([3])])

    def test_action_index_loss_weights_prefix_causality_and_reference_branch_cannot_change_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td); net,_,_=initialize_joint(*paths)
            start=[np.array([2])]
            baseline=predict_windows(net,[ep],s,10,start)
            row=trainer.loss_batch(net,[ep],s,[.2]*64,10,start)
            self.assertTrue(torch.equal(row['normalized_observations'],baseline['normalized_observations']))
            self.assertEqual(row['window_count'],1)
            self.assertAlmostEqual(row['observation_loss'].item(),observation_loss(baseline['normalized_observations'],baseline['truth'],s).item(),places=12)
            action=(torch.tensor(ep['action'][2:12])-torch.tensor(s['action']['mean'],dtype=torch.float64))/torch.tensor(s['action']['std'],dtype=torch.float64)
            expected=row['source_latents']+net.transition(row['source_latents'].reshape(-1,64),action.float()).reshape(1,10,64)
            torch.testing.assert_close(row['local_prediction'],expected,rtol=0,atol=0)
            changed=copy.deepcopy(ep); changed['obs'][3:]+=100; changed['next_obs']+=200
            altered=trainer.loss_batch(net,[changed],s,[.2]*64,10,start)
            self.assertTrue(torch.equal(row['normalized_observations'],altered['normalized_observations']))
            self.assertNotEqual(row['consistency_loss'].item(),altered['consistency_loss'].item())
            original=predict_window(net,ep,2,10,s)
            changed['obs']=changed['obs'][:3]; changed['previous_action']=changed['previous_action'][:3]; del changed['next_obs']
            with patch.object(trainer,'local_loss',side_effect=AssertionError('reference branch forbidden at inference')):
                deleted=predict_window(net,changed,2,10,s)
            self.assertTrue(torch.equal(original['latents'],deleted['latents']))

    def test_initial_scale_train_only_literal_floor_and_does_not_mutate_initial_parameters(self):
        sequences=[np.vstack([np.zeros(64),np.ones(64)*2])]
        scale=trainer.fit_initial_scale(sequences)
        self.assertEqual(scale['std'],[1.]*64)
        constant=trainer.fit_initial_scale([np.ones((3,64))])
        self.assertEqual(constant['std'],[1e-6]*64)
        self.assertEqual(scale['frame_count'],2)
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td); net,_,_=initialize_joint(*paths)
            hashes=component_hashes(net)
            a=trainer.initial_scale(net,[ep],s)
            self.assertEqual(hashes,component_hashes(net))
            self.assertEqual(a,trainer.initial_scale(net,[ep],s))
            self.assertEqual(a['frame_count'],13)

    def test_total_backward_preserves_autonomous_bptt_and_all_module_gradients(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td); net,_,_=initialize_joint(*paths)
            row=trainer.loss_batch(net,[ep],s,[.2]*64,10,[np.array([2])])
            row['steps'][1].retain_grad()
            row['loss'].backward()
            self.assertGreater(row['steps'][1].grad.norm().item(),0.)
            for component in (net.encoder,net.transition,net.decoder):
                self.assertTrue(all(p.grad is not None for p in component.parameters()))
                self.assertGreater(sum(p.grad.square().sum().item() for p in component.parameters()),0.)

    def test_validation_selection_reload_and_reproduction_use_observation_not_total_loss(self):
        with tempfile.TemporaryDirectory() as td:
            paths,s,_,_,_,_,ep=fixture(td)
            results=[]; scale=[.2]*64
            cfg=dict(seed=0,epochs=3,batch_size=2,learning_rate=.0003,horizon=10)
            for i in range(2):
                net,_,_=initialize_joint(*paths); path=Path(td)/f'v2{i}.pt'
                report=trainer.train_consistency(net,[ep],[ep],s,scale,cfg,path)
                loaded,ss,meta=trainer.load_consistency(path)
                self.assertEqual(ss,s)
                self.assertEqual(meta['best_epoch'],int(np.argmin([r['validation']['observation'] for r in report['history']]))+1)
                self.assertEqual(meta['initial_latent_std'],scale)
                self.assertEqual(meta['lambda_consistency'],.1)
                self.assertEqual(report['parameter_count'],74119)
                check=trainer.validation_losses(loaded,[ep],s,scale,10,2)
                self.assertAlmostEqual(check['observation'],report['best_validation']['observation'],places=12)
                self.assertAlmostEqual(check['total'],check['observation']+.1*check['consistency'],places=12)
                for key in ('encoder','transition','decoder'):
                    self.assertNotEqual(report['parameter_hashes_before'][key],report['parameter_hashes_best'][key])
                self.assertEqual(report['history'][0]['objective_gradient_diagnostics']['decoder']['weighted_consistency_norm'],0.)
                results.append((component_hashes(loaded),report['history']))
            self.assertEqual(results[0],results[1])
            # Selector cannot favor a lower total with a worse observation score.
            self.assertEqual(trainer.selection_epoch([{'observation':1.,'total':2.},{'observation':2.,'total':.1}]),1)


if __name__=='__main__': unittest.main()
