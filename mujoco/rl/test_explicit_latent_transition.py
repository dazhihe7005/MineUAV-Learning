"""Breaks caught: encoder mutation, terminal-pair omission, leakage, recursion."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from latent_dynamics_models import make_model
from run_latent_memory_ablation import parameter_hash
from test_latent_dynamics_multistep import episode, statistics

try:
    import explicit_latent_models as models
    import explicit_latent_evaluation as evaluation
except ImportError:
    models = evaluation = None


class ActionDelta(nn.Module):
    def forward(self,z,action):
        out=torch.zeros_like(z); out[:,0]=action[:,0]; return out


class CopyDecoder(nn.Module):
    def forward(self,z): return z[:,:7]


class ExplicitTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(models,'explicit latent implementation missing')
        self.assertIsNotNone(evaluation,'pure latent evaluator missing')
        torch.set_num_threads(1); torch.manual_seed(0)

    def test_frozen_teacher_keeps_final_saved_frame_and_resets_a_b_a(self):
        encoder=make_model('history').encoder; models.freeze_encoder(encoder)
        ep=episode(6); original=parameter_hash(encoder)
        a=models.teacher_latents(encoder,ep,statistics())
        other=episode(3,2); other['obs']*=10
        models.teacher_latents(encoder,other,statistics())
        b=models.teacher_latents(encoder,ep,statistics())
        self.assertEqual(a.shape,(7,64)); np.testing.assert_array_equal(a,b)
        self.assertEqual(parameter_hash(encoder),original)
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))
        with torch.no_grad():
            inputs=np.concatenate([np.vstack([ep['obs'],ep['next_obs'][-1]]),
                np.vstack([ep['previous_action'],ep['action'][-1]])],axis=1)
            expected,_=encoder(torch.from_numpy(inputs)[None],None)
        np.testing.assert_array_equal(a,expected[0].numpy())

    def test_manifest_pairs_never_join_episodes_and_change_hash_if_latent_changes(self):
        encoder=make_model('history').encoder; models.freeze_encoder(encoder)
        eps=[episode(3),episode(2,2)]
        z=[models.teacher_latents(encoder,e,statistics()) for e in eps]
        manifest=models.transition_manifest(eps,z,'fixture')
        self.assertEqual(manifest['transition_count'],5)
        self.assertEqual(manifest['decoder_frame_count'],7)
        self.assertEqual([r['transition_count'] for r in manifest['episodes']],[3,2])
        z[1]=z[1].copy(); z[1][0,0]+=1
        self.assertNotEqual(manifest['manifest_sha256'],models.transition_manifest(eps,z,'fixture')['manifest_sha256'])
        with self.assertRaisesRegex(ValueError,'latent'):
            models.transition_manifest(eps,[z[0][:-1],z[1]],'fixture')

    def test_latent_statistics_only_given_train_frames_with_floor(self):
        train=[np.vstack([np.zeros(64),np.ones(64)*2])]
        stats=models.latent_statistics(train)
        np.testing.assert_array_equal(stats['mean'],np.ones(64))
        np.testing.assert_array_equal(stats['std'],np.ones(64))
        self.assertEqual(stats['provenance']['frames'],2)
        constant=models.latent_statistics([np.zeros((3,64))])
        np.testing.assert_array_equal(constant['std'],np.ones(64)*1e-6)

    def test_pure_recursion_uses_recorded_current_actions_and_inverse_normalization(self):
        latent=dict(mean=[10.]*64,std=[2.]*64)
        stats=statistics(); stats['obs']=dict(mean=[1.]*7,std=[3.]*7)
        # normalized z starts0, action sequence1,2,3: z'=12,16,22.
        action=np.zeros((1,3,4),np.float32); action[0,:,0]=[1,2,3]
        out=evaluation.latent_rollout(ActionDelta(),CopyDecoder(),np.ones((1,64))*10,action,stats,latent)
        np.testing.assert_array_equal(out['latents'][0,:,0],[12.,16.,22.])
        np.testing.assert_array_equal(out['observations'][0,:,0],[4.,10.,19.])

    def test_one_step_metrics_and_decoder_use_next_vs_current_teacher_indices(self):
        self.assertTrue(hasattr(evaluation,'one_step_metrics'),'one-step evaluator missing')
        ep=episode(3); z=np.zeros((4,64),np.float32)
        z[:,0]=[0,.0,.1,.3]; ep['latent']=z
        latent=dict(mean=[0.]*64,std=[1.]*64)
        row=evaluation.one_step_metrics(ActionDelta(),[ep],statistics(),latent)
        self.assertLess(row['normalized_latent_rmse'],1e-8)
        decoder=evaluation.decoder_metrics(CopyDecoder(),[ep],statistics(),latent)
        # Current obs error_x1,2,3,4 versus decoder output0,0,.1,.3.
        self.assertAlmostEqual(decoder['physical']['per_dimension']['error_x']['mae'],2.4,places=6)
        self.assertEqual(decoder['physical']['count'],4)

    def test_deleted_or_corrupted_future_observation_cannot_change_rollout(self):
        encoder=make_model('history').encoder; models.freeze_encoder(encoder)
        transition=models.ResidualTransition(); decoder=models.ObservationDecoder(); ep=episode(8)
        latent=dict(mean=[0.]*64,std=[1.]*64)
        a=evaluation.predict_window(encoder,transition,decoder,ep,2,5,statistics(),latent)
        changed=copy.deepcopy(ep); changed['obs'][3:]+=1e6; changed['next_obs'][:]=np.nan
        b=evaluation.predict_window(encoder,transition,decoder,changed,2,5,statistics(),latent)
        np.testing.assert_array_equal(a['observations'],b['observations'])
        del changed['next_obs']; changed['obs']=changed['obs'][:3]
        changed['previous_action']=changed['previous_action'][:3]
        c=evaluation.predict_window(encoder,transition,decoder,changed,2,5,statistics(),latent)
        np.testing.assert_array_equal(a['latents'],c['latents'])
        np.testing.assert_array_equal(a['observations'],c['observations'])
        with self.assertRaisesRegex(ValueError,'window'):
            evaluation.predict_window(encoder,transition,decoder,ep,6,3,statistics(),latent)

    def test_independent_trainers_reload_val_best_and_keep_encoder_frozen(self):
        encoder=make_model('history').encoder; models.freeze_encoder(encoder); before=parameter_hash(encoder)
        eps=[episode(6),episode(5,2)]; val=[episode(4,3)]; stats=statistics()
        train=models.attach_teacher(eps,encoder,stats); valid=models.attach_teacher(val,encoder,stats)
        latent=models.latent_statistics([e['latent'] for e in train])
        cfg=dict(seed=0,epochs=3,batch_size=2,learning_rate=.001)
        with tempfile.TemporaryDirectory() as td:
            for kind in ('transition','decoder'):
                path=Path(td)/f'{kind}.pt'
                result=models.train_component(kind,train,valid,stats,latent,cfg,path)
                net,saved,lat,meta=models.load_component(path)
                self.assertEqual(saved,stats); self.assertEqual(lat,latent)
                self.assertEqual(result['best_epoch'],np.argmin([r['validation_loss'] for r in result['history']])+1)
                self.assertAlmostEqual(models.validation_loss(net,kind,valid,stats,latent,2),result['best_validation_loss'])
                self.assertEqual(meta['config'],cfg)
                self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        self.assertEqual(parameter_hash(encoder),before)

    def test_repeated_initialization_training_is_deterministic(self):
        encoder=make_model('history').encoder; models.freeze_encoder(encoder); stats=statistics()
        train=models.attach_teacher([episode(5)],encoder,stats)
        valid=models.attach_teacher([episode(4,2)],encoder,stats)
        latent=models.latent_statistics([e['latent'] for e in train]); cfg=dict(seed=0,epochs=2,batch_size=1,learning_rate=.001)
        with tempfile.TemporaryDirectory() as td:
            a=models.train_component('transition',train,valid,stats,latent,cfg,Path(td)/'a.pt')
            b=models.train_component('transition',train,valid,stats,latent,cfg,Path(td)/'b.pt')
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(parameter_hash(models.load_component(Path(td)/'a.pt')[0]),parameter_hash(models.load_component(Path(td)/'b.pt')[0]))


if __name__=='__main__': unittest.main()
