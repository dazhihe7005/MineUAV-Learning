"""Fixed-budget branch adaptation: actual loss graph and paired ordering."""
import copy, importlib.util, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from joint_latent_world_model import JointLatentWorldModel,component_hashes
from joint_autonomous_consistency_training import loss_batch,load_autonomous
from onpolicy_adaptation_data import BranchDataset

def fixture():
    rng=np.random.default_rng(4); obs=rng.normal(0,.1,(13,7)).astype(np.float32)
    a=rng.normal(0,.1,(12,4)).astype(np.float32)
    state=dict(prefix_observations=obs[:3],prefix_actions=a[:2],
        branch_actions=np.repeat(a[None,2:],8,0),branch_observations=np.repeat(obs[None,2:],8,0))
    stats={k:dict(mean=[0.]*n,std=[1.]*n) for k,n in [('obs',7),('action',4),('previous_action',4)]}
    return BranchDataset([state,state]),stats

class AdaptationTrainingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('onpolicy_adaptation_training'))
        import onpolicy_adaptation_training as api
        return api
    def test_budget_order_and_selection_observation_only(self):
        api=self.api(); a=list(api.batch_order(32,16,5,0));b=list(api.batch_order(32,16,5,0))
        self.assertEqual(len(a),5)
        for x,y in zip(a,b): np.testing.assert_array_equal(x,y);self.assertEqual(len(x),16)
        self.assertEqual(api.selected_step([dict(step=50,validation=dict(observation=2.,total=2.)),
            dict(step=100,validation=dict(observation=1.,total=3.))]),100)
    def test_full_graph_detached_reference_and_future_leakage(self):
        self.api();torch.set_num_threads(1);data,stats=fixture();torch.manual_seed(0);model=JointLatentWorldModel()
        ep,starts=data.batch([0]);out=loss_batch(model,ep,stats,[1.]*64,10,starts)
        other=copy.deepcopy(ep);other[0]['obs'][3:]+=10;other[0]['next_obs']+=10
        changed=loss_batch(model,other,stats,[1.]*64,10,starts)
        self.assertTrue(torch.equal(out['latents'],changed['latents']))
        self.assertNotEqual(out['consistency_loss'].item(),changed['consistency_loss'].item())
        self.assertFalse(out['reference_targets'].requires_grad);out['steps'][1].retain_grad()
        out['consistency_loss'].backward()
        self.assertGreater(out['steps'][1].grad.abs().sum().item(),0)
        for module in (model.encoder,model.transition): self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()),0)
        self.assertTrue(all(p.grad is None for p in model.decoder.parameters()))
    def test_paired_training_budget_reload_and_determinism(self):
        api=self.api();torch.set_num_threads(1); data,stats=fixture();torch.manual_seed(0);initial=JointLatentWorldModel()
        runs=[]
        with tempfile.TemporaryDirectory() as td:
            for i in range(2):
                model=copy.deepcopy(initial);path=Path(td)/f'{i}.pt'
                r=api.train_updates(model,data,data,stats,[1.]*64,path,updates=2,validation_interval=1)
                loaded,ss,meta=load_autonomous(path);self.assertEqual(ss,stats)
                self.assertEqual(r['final_step'],2);self.assertEqual(r['initial_hashes'],component_hashes(initial))
                self.assertEqual(meta['best_step'],api.selected_step(r['history']))
                self.assertEqual(meta['initial_latent_std'],[1.]*64)
                self.assertEqual(component_hashes(loaded),r['best_hashes'])
                runs.append((r['order_sha256'],r['best_hashes'],r['best_validation']))
        self.assertEqual(runs[0],runs[1])

if __name__=='__main__':unittest.main()
