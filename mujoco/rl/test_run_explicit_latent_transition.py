"""Temporary untrained references; no simulation/data regeneration experiment."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from latent_dynamics_data import file_hash
from test_latent_objective_matched_control import reference_fixture

try:
    import run_explicit_latent_transition as runner
except ImportError:
    runner=None


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner,'explicit experiment runner missing')
        torch.set_num_threads(1); torch.manual_seed(0)

    def test_source_loader_never_reads_test_timestep_values(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference_fixture(root); load=np.load; names=[]
            def checked(path,*args,**kwargs):
                names.append(Path(path).name); self.assertNotEqual(Path(path).name,'test.npz')
                return load(path,*args,**kwargs)
            with patch('numpy.load',side_effect=checked): ctx=runner.load_context(root,source)
            self.assertEqual(names,['train.npz','val.npz'])
            self.assertEqual(ctx['latent_statistics']['provenance']['frames'],26)
            self.assertEqual(ctx['manifests']['train']['transition_count'],24)
            self.assertFalse(any(p.requires_grad for p in ctx['encoder'].parameters()))

    def test_full_run_reload_manifest_tamper_and_immutable_sources(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=reference_fixture(root)
            old={str(p):file_hash(p) for p in root.glob('*.pt')}
            cfg=dict(seed=0,epochs=2,batch_size=2,learning_rate=.001)
            result=runner.run_experiment(root,source,config=cfg)
            self.assertEqual(len(result['figures']),7)
            self.assertTrue(all(Path(p).is_file() for p in result['figures']))
            self.assertEqual(result['manifests']['test']['transition_count'],24)
            self.assertEqual(result['latent_rollout']['horizons']['1']['observation']['window_count'],24)
            self.assertTrue(all(runner.verify_experiment(Path(result['report_path'])).values()))
            for p,h in old.items(): self.assertEqual(file_hash(p),h)
            path=Path(result['manifests']['train']['path']); original=path.read_text(); path.write_text(original+' ')
            with self.assertRaisesRegex(ValueError,'manifest'): runner.verify_experiment(Path(result['report_path']))
            path.write_text(original)
            with self.assertRaises(FileExistsError): runner.run_experiment(root,source,config=cfg)


if __name__=='__main__': unittest.main()
