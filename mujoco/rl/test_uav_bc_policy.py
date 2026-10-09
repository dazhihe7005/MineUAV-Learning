"""Actual actor scaling, selection, checkpoint and teacher-free behavior."""
import importlib,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
import torch

class BCPolicyTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_bc_policy'))
        return importlib.import_module('uav_bc_policy')
    def test_statistics_and_loss_use_train_action_std(self):
        a=self.api();x=np.arange(28,dtype=np.float32).reshape(4,7);y=np.array([[-1,0,0,0],[1,2,0,0]],np.float32)
        stats=a.fit_stats(x,y)
        np.testing.assert_array_equal(stats['action']['std'],[1,1,1e-6,1e-6])
        pred=torch.tensor([[2.,0.,0.,0.]]);target=torch.tensor([[0.,0.,0.,0.]])
        self.assertEqual(a.normalized_loss(pred,target).item(),1.)
        self.assertEqual(stats['provenance'],'train only')
    def test_reload_matches_and_policy_has_no_teacher_input(self):
        a=self.api();torch.manual_seed(0);actor=a.BCActor();stats=a.fit_stats(np.arange(28).reshape(4,7),np.ones((4,4)))
        self.assertEqual(sum(p.numel() for p in actor.parameters()),18052)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'best.pt';a.save(path,actor,stats,{'best_epoch':2})
            policy,meta=a.load(path);self.assertEqual(meta['best_epoch'],2)
            original=a.BCPolicy(actor,stats)
            np.testing.assert_array_equal(policy.predict(np.ones(7)),original.predict(np.ones(7)))
            self.assertTrue((np.abs(policy.predict(np.ones(7)))<=1).all())
            with self.assertRaises(ValueError):policy.predict(np.ones(11))
    def test_seed_reproduction_and_validation_only_selection(self):
        self.api();self.assertIsNotNone(importlib.util.find_spec('uav_bc_training'))
        from uav_bc_training import train,select_epoch
        self.assertEqual(select_epoch([{'epoch':1,'validation_loss':.2,'train_loss':0},
                                      {'epoch':2,'validation_loss':.1,'train_loss':2}]),2)
        x=np.linspace(-1,1,112,dtype=np.float32).reshape(16,7);y=x[:,:4]*.2
        with tempfile.TemporaryDirectory() as d:
            one=train(x,y,x[:8],y[:8],Path(d)/'one.pt',epochs=2,batch_size=8)
            two=train(x,y,x[:8],y[:8],Path(d)/'two.pt',epochs=2,batch_size=8)
            self.assertEqual(one['history'],two['history'])
            self.assertEqual(one['initial_hash'],two['initial_hash']);self.assertEqual(one['best_parameter_hash'],two['best_parameter_hash'])
            self.assertEqual(one['best_epoch'],select_epoch(one['history']))

if __name__=='__main__':unittest.main()
