"""Real tiny trainers catch unequal schedules, re-fit stats, Test selection and bad resume."""
import importlib,importlib.util,tempfile,unittest
from pathlib import Path
import numpy as np
from uav_bc_policy import load,parameter_hash

class YawTrainingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('uav_bc_yaw_training'),'matched trainer missing')
        return importlib.import_module('uav_bc_yaw_training')
    def fixture(self):
        rng=np.random.default_rng(7)
        def data(n):
            x=rng.normal(size=(n,7)).astype(np.float32)
            return dict(observations=x,actions=np.c_[x[:,:3]*.05,x[:,6]*.1].astype(np.float32),yaw_group=np.resize([0,10,30],n))
        stats=dict(obs=dict(mean=[0.]*7,std=[2.]*7),action=dict(mean=[0.]*4,std=[.5]*4),provenance='original Train only')
        return data(16),data(9),stats
    def test_same_initialization_schedule_budget_and_fixed_stats(self):
        a=self.api();tr,va,s=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);b=a.train_one(tr,va,s,p/'b.pt',p/'b_resume.pt',epochs=2,batch_size=8)
            different=dict(tr,actions=tr['actions']+.1)
            c=a.train_one(different,va,s,p/'c.pt',p/'c_resume.pt',epochs=2,batch_size=8)
            self.assertEqual(b['initial_parameter_hash'],c['initial_parameter_hash'])
            self.assertEqual(b['optimizer_updates'],4);self.assertEqual(c['optimizer_updates'],4)
            self.assertEqual([r['order_sha256'] for r in b['history']],[r['order_sha256'] for r in c['history']])
            self.assertEqual(b['normalization'],s);self.assertEqual(load(p/'c.pt')[0].stats,s)
            self.assertNotEqual(b['initial_parameter_hash'],b['best_parameter_hash'])
    def test_best_epoch_uses_validation_not_train(self):
        a=self.api()
        rows=[dict(epoch=1,train_loss=.01,validation_loss=.5),dict(epoch=2,train_loss=.9,validation_loss=.2)]
        self.assertEqual(a.select_best(rows),2)
    def test_atomic_epoch_resume_reproduces_uninterrupted_training(self):
        a=self.api();tr,va,s=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);full=a.train_one(tr,va,s,p/'full.pt',p/'full_resume.pt',epochs=2,batch_size=8)
            def interrupt(r):raise RuntimeError('simulated interruption after durable epoch')
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                a.train_one(tr,va,s,p/'resumed.pt',p/'resume.pt',epochs=2,batch_size=8,on_epoch=interrupt)
            restored=a.train_one(tr,va,s,p/'resumed.pt',p/'resume.pt',epochs=2,batch_size=8)
            self.assertEqual(full['history'],restored['history']);self.assertEqual(full['best_parameter_hash'],restored['best_parameter_hash'])
            cached=a.train_one(tr,va,s,p/'resumed.pt',p/'resume.pt',epochs=2,batch_size=8)
            self.assertEqual(restored['history'],cached['history']);self.assertEqual(cached['optimizer_updates'],4)
            with self.assertRaises(ValueError):a.train_one(tr,va,dict(s,provenance='different'),p/'resumed.pt',p/'resume.pt',epochs=2,batch_size=8)
    def test_checkpoint_reload_independent_and_deterministic(self):
        a=self.api();tr,va,s=self.fixture()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a.train_one(tr,va,s,p/'b.pt',p/'resume.pt',epochs=1,batch_size=8)
            first,_=load(p/'b.pt');second,_=load(p/'b.pt')
            np.testing.assert_array_equal(first.predict(va['observations'][0]),second.predict(va['observations'][0]))
            self.assertTrue(all(not q.requires_grad for q in first.actor.parameters()))

if __name__=='__main__':unittest.main()
