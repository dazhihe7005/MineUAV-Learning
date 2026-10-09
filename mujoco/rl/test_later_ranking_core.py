"""Observable stage/tail contracts; ties and missing stages are not hidden."""
import importlib
import importlib.util
import unittest
import numpy as np

class CoreTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('later_ranking_core'))
        return importlib.import_module('later_ranking_core')

    def test_nearest_distinct_legal_stages(self):
        api=self.api()
        self.assertEqual(api.stage_indices(101),dict(S0=0,S1=25,S2=50,S3=75))
        self.assertEqual(api.stage_indices(1),dict(S0=0))
        self.assertEqual(api.stage_indices(2),dict(S0=0,S2=1))
        with self.assertRaises(ValueError): api.stage_indices(0)

    def test_best_tail_exposes_reverse_order_in_otherwise_good_ranking(self):
        api=self.api(); true=np.arange(512,dtype=float); pred=true.copy(); pred[:26]=pred[:26][::-1]
        m=api.tail_metrics(pred,true)
        self.assertEqual(m['count'],26)
        self.assertAlmostEqual(m['spearman'],-1)
        self.assertEqual(m['ordering_accuracy'],0)
        self.assertEqual(m['true_cost_spread'],25)

    def test_tail_ties_are_disclosed_not_fabricated(self):
        m=self.api().tail_metrics(np.zeros(512),np.zeros(512))
        self.assertIsNone(m['spearman']); self.assertIsNone(m['ordering_accuracy'])
        self.assertEqual(m['tied_prediction_pairs'],325)

    def test_train_geometry_independent_of_test_outliers(self):
        api=self.api(); z=np.random.default_rng(1).normal(size=(90,64))
        obs=np.arange(630,dtype=float).reshape(90,7); act=np.ones((90,4))
        geometry=api.fit_geometry(z,obs,act,'train')
        with self.assertRaises(ValueError): api.fit_geometry(z,obs,act,'test')
        before=geometry.copy(); api.ood_scores(z[0]*100,obs[0]*100,act[0]*100,geometry)
        self.assertEqual(before,geometry)
        self.assertEqual(geometry['provenance']['split'],'train')

if __name__=='__main__': unittest.main()
