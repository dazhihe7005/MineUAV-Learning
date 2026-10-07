"""Literal frozen diagnostic contracts; no optimizers or model training."""
import unittest
import numpy as np
import torch
from torch import nn
from joint_latent_world_model import JointLatentWorldModel, component_hashes

try:
    import joint_composition_core as core
except ImportError:
    core=None


class ActionResidual(nn.Module):
    def forward(self,z,a):
        out=torch.zeros_like(z); out[:,0]=a[:,0]; return out


class Readout(nn.Module):
    def forward(self,z): return z[:,:7]


def toy():
    model=JointLatentWorldModel(transition=ActionResidual(),decoder=Readout())
    stats={'action':{'mean':[0.]*4,'std':[1.]*4},'obs':{'mean':[0.]*7,'std':[1.]*7}}
    return model,stats


class CoreTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(core,'composition diagnostic core missing')

    def test_literal_action_recursion_and_correction_excludes_terminal_reference(self):
        model,stats=toy(); core.freeze_joint(model)
        z=torch.zeros(1,64); actions=torch.zeros(1,10,4); actions[0,:,0]=torch.arange(1,11)
        ref=torch.zeros(1,11,64); ref[0,:,0]=torch.arange(11)*10
        never=core.compose(model,z,actions,stats)
        self.assertEqual(never['latents'][0,-1,0].item(),55.)
        self.assertEqual(never['correction_steps'],[])
        for c,want,steps in [(1,100.,list(range(1,10))),(5,90.,[5]),(10,55.,[]),(50,55.,[])]:
            result=core.compose(model,z,actions,stats,c,ref)
            self.assertEqual(result['latents'][0,-1,0].item(),want)
            self.assertEqual(result['correction_steps'],steps)
            altered=ref.clone(); altered[:,-1]=1e6
            self.assertTrue(torch.equal(result['latents'],core.compose(model,z,actions,stats,c,altered)['latents']))
        altered=ref.clone(); altered[:,1:]=float('nan')
        self.assertTrue(torch.equal(never['latents'],core.compose(model,z,actions,stats,50,altered)['latents']))
        self.assertTrue(torch.equal(never['latents'],core.compose(model,z,actions,stats,None,None)['latents']))
        self.assertFalse(never['latents'].requires_grad)

    def test_freeze_all_parameters_and_guard_shapes_and_finite_predictions(self):
        model=JointLatentWorldModel(); before=component_hashes(model); core.freeze_joint(model)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.parameters()))
        self.assertTrue(all(not m.training for m in model.modules()))
        stats=toy()[1]; z=torch.zeros(1,64); actions=torch.zeros(1,3,4)
        core.compose(model,z,actions,stats)
        self.assertEqual(before,component_hashes(model))
        for c in (0,-1,1.5):
            with self.assertRaises(ValueError): core.compose(model,z,actions,stats,c,None)
        with self.assertRaises(ValueError): core.compose(model,z,actions,stats,1,torch.zeros(1,2,64))
        with self.assertRaises(ValueError): core.compose(model,z,actions[:,:,:3],stats)
        bad=actions.clone(); bad[:,1]=float('nan')
        with self.assertRaises(FloatingPointError): core.compose(model,z,bad,stats)

    def test_train_only_manifold_pca_statistics_literal_and_degenerate_covariance(self):
        z=np.tile(np.array([-2.,0.,2.])[:,None],(1,64))
        actions=np.zeros((3,4)); actions[:,0]=[0.,1.,2.]
        stats=core.fit_manifold([z],[actions])
        np.testing.assert_array_equal(stats['mean'],np.zeros(64))
        np.testing.assert_allclose(stats['std'],np.full(64,np.sqrt(8/3)))
        self.assertEqual(stats['pca_components'],1)
        self.assertEqual(stats['provenance']['split'],'train')
        np.testing.assert_allclose(stats['action_norm_tertiles'],[2/3,4/3])
        on=core.distribution_scores(np.ones((1,64)),stats)
        self.assertLess(on['pca_residual_l2'][0],1e-10)
        self.assertAlmostEqual(on['standardized_rms'][0],np.sqrt(3/8))
        off=np.zeros((1,64)); off[0,:2]=[1.,-1.]
        self.assertAlmostEqual(core.distribution_scores(off,stats)['pca_residual_l2'][0],np.sqrt(2))
        np.testing.assert_allclose(core.distribution_scores(off,stats)['pca_residual_fraction'],[1.])
        extreme_test=np.full((2,64),1e9)
        core.distribution_scores(extreme_test,stats)
        self.assertEqual(stats,core.fit_manifold([z],[actions]))
        const=core.fit_manifold([np.ones((3,64))],[actions])
        self.assertEqual(const['pca_components'],0)
        self.assertTrue(np.isfinite(core.distribution_scores(np.zeros((2,64)),const)['mahalanobis_rms']).all())

    def test_local_scale_metrics_and_sensitivity_ratio_have_hand_checked_units(self):
        p=np.zeros((2,64)); t=np.zeros_like(p); p[:,0]=[1.,3.]
        m=core.latent_metrics(p,t,np.ones(64))
        self.assertAlmostEqual(m['raw_mae'],4/128)
        self.assertAlmostEqual(m['normalized_rmse'],np.sqrt(10/128))
        self.assertEqual(m['mean_l2'],2.)
        self.assertEqual(m['mean_cosine'],0.)
        empty=core.latent_metrics(p[:0],t[:0],np.ones(64)); self.assertEqual(empty['count'],0)
        self.assertEqual(core.pearson(np.arange(4),np.arange(4)*2),1.)
        self.assertIsNone(core.pearson(np.ones(4),np.arange(4)))
        self.assertIsNone(core.pearson([],[]))


if __name__=='__main__': unittest.main()
