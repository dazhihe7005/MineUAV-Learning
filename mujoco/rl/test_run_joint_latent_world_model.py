"""Temporary real fixtures validate joint experiment orchestration/no leakage."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from latent_dynamics_data import file_hash
from test_run_latent_transition_multistep import source_fixture
from run_latent_transition_multistep import run_experiment as transition_fixture

try:
    import run_joint_latent_world_model as runner
except ImportError:
    runner=None


def reference(root):
    prior=source_fixture(root)
    report=transition_fixture(root,prior,dict(seed=0,epochs=2,batch_size=2,learning_rate=.001,horizon=10),make_figures=False)
    return Path(report['report_path'])


class RunnerTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(runner,'joint runner missing')

    def test_loader_reuses_manifest_normalization_and_does_not_read_test(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference(root); loader=np.load; names=[]
            def checked(path,*a,**kw):
                names.append(Path(path).name); self.assertNotEqual(Path(path).name,'test.npz')
                return loader(path,*a,**kw)
            with patch('numpy.load',side_effect=checked): context=runner.load_context(root,source)
            self.assertEqual(names,['train.npz','val.npz'])
            self.assertEqual(context['source']['normalization'],context['statistics'])
            self.assertTrue(all(p.requires_grad for p in context['joint'].parameters()))
            for split in ('train','val'):
                self.assertEqual(context['source']['train_validation_manifests'][split]['window_count'],6)

    def test_full_run_selection_reload_future_leakage_metrics_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference(root); old=json.loads(source.read_text())
            paths=[source,old['model']['path'],*[v['artifact']['path'] for v in old['frozen_components'].values()]]
            hashes={str(p):file_hash(p) for p in paths}
            cfg=dict(seed=0,epochs=2,batch_size=2,learning_rate=.0003,horizon=10)
            report=runner.run_experiment(root,source,cfg)
            self.assertEqual(len(report['figures']),9)
            self.assertEqual(report['windows'],old['windows'])
            self.assertEqual(report['training']['parameter_count'],74119)
            self.assertTrue(all(report['future_observation_leakage'].values()))
            for split in ('train','val'):
                self.assertEqual(report['window_manifests'][split],old['train_validation_manifests'][split])
            self.assertTrue(all(runner.verify_experiment(Path(report['report_path'])).values()))
            for p,h in hashes.items(): self.assertEqual(file_hash(p),h)
            with self.assertRaises(FileExistsError): runner.run_experiment(root,source,cfg)
            p=Path(report['report_path']); wrong=json.loads(p.read_text()); wrong['horizons_steps']=[1]
            p.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(AssertionError,'horizon'): runner.verify_experiment(p)

    def test_verifier_rejects_report_normalization_loss_and_before_hash_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference(root)
            report=runner.run_experiment(root,source,dict(seed=0,epochs=2,batch_size=2,learning_rate=.0003,horizon=10),make_figures=False)
            path=Path(report['report_path'])
            variants=[]
            for field in ('normalization','best_validation_loss','best_checkpoint_train_loss','parameter_hashes_before'):
                wrong=json.loads(json.dumps(report))
                if field=='normalization':
                    wrong[field]['obs']['std']=[999.]*7; wrong['normalization_sha256']='invalid'
                elif field=='parameter_hashes_before':
                    wrong['training'][field]=wrong['training']['parameter_hashes_best']
                else: wrong['training'][field]=-999.
                variants.append((field,wrong))
            for field,wrong in variants:
                with self.subTest(field=field):
                    path.write_text(json.dumps(wrong))
                    with self.assertRaises((AssertionError,ValueError)):
                        runner.verify_experiment(path)


if __name__=='__main__': unittest.main()
