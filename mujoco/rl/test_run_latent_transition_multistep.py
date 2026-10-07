"""Real temporary fixtures; immutable components, manifest reuse and selection."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from latent_dynamics_data import file_hash
from test_latent_objective_matched_control import reference_fixture
from run_explicit_latent_transition import run_experiment as explicit_fixture

try:
    import run_latent_transition_multistep as runner
except ImportError:
    runner=None


def source_fixture(root):
    source=reference_fixture(root)
    report=explicit_fixture(root,source,dict(seed=0,epochs=2,batch_size=2,learning_rate=.001),make_figures=False)
    return Path(report['report_path'])


class RunnerTests(unittest.TestCase):
    def setUp(self): self.assertIsNotNone(runner,'multistep transition runner missing')

    def test_loader_reuses_saved_statistics_without_opening_test(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=source_fixture(root); loader=np.load; names=[]
            def checked(path,*args,**kwargs):
                names.append(Path(path).name); self.assertNotEqual(Path(path).name,'test.npz')
                return loader(path,*args,**kwargs)
            with patch('numpy.load',side_effect=checked), patch('explicit_latent_models.latent_statistics',side_effect=AssertionError('must not refit latent stats')):
                context=runner.load_context(root,source)
            self.assertEqual(names,['train.npz','val.npz'])
            self.assertEqual(context['latent_statistics'],json.loads(source.read_text())['latent_statistics'])
            self.assertTrue(all(p.grad is None and not p.requires_grad for m in (context['encoder'],context['decoder']) for p in m.parameters()))

    def test_full_run_fixed_sources_initialization_floor_reload_and_verification(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=source_fixture(root)
            prior=json.loads(source.read_text()); immutable={p:file_hash(p) for p in [source,*[r['path'] for r in prior['models'].values()]]}
            cfg=dict(seed=0,epochs=2,batch_size=2,learning_rate=.001,horizon=10)
            report=runner.run_experiment(root,source,cfg)
            self.assertEqual(len(report['figures']),8)
            self.assertEqual(report['train_validation_manifests']['train']['window_count'],6)
            self.assertEqual(report['training']['initial_parameter_sha256'],prior['training']['transition']['initial_parameter_sha256'])
            self.assertEqual(report['windows'],prior['windows'])
            self.assertTrue(all(report['future_teacher_leakage'].values()))
            self.assertEqual(report['decoder_floor']['horizons']['1']['window_count'],24)
            self.assertTrue(all(runner.verify_experiment(Path(report['report_path'])).values()))
            for p,h in immutable.items(): self.assertEqual(file_hash(p),h)
            with self.assertRaises(FileExistsError): runner.run_experiment(root,source,cfg)
            p=Path(report['report_path']); content=p.read_text(); wrong=json.loads(content); wrong['horizons_steps']=[1]
            p.write_text(json.dumps(wrong))
            with self.assertRaisesRegex(AssertionError,'horizon'): runner.verify_experiment(p)


if __name__=='__main__': unittest.main()
