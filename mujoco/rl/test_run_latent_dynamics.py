import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from test_latent_dynamics_data import fixture_arrays

try:
    from run_latent_dynamics_one_step import run_experiment
except ImportError:
    run_experiment = None


class ExperimentTests(unittest.TestCase):
    def test_complete_offline_run_saves_reloadable_models_and_source_provenance(self):
        self.assertIsNotNone(run_experiment, 'offline experiment runner is not implemented')
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / 'source'; source.mkdir()
            hashes = {}
            for name, target in [('train', 1), ('val', 2), ('test', 3)]:
                path = source / f'{name}.npz'
                np.savez_compressed(path, **fixture_arrays(target))
                hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            (source / 'manifest.json').write_text(json.dumps(dict(dataset_sha256=hashes, seed=0)))
            report = run_experiment(source, root, epochs=1)
            self.assertEqual(report['dataset']['source_dataset_sha256'], hashes)
            self.assertFalse(report['dataset']['regenerated'])
            self.assertEqual(report['normalization']['provenance']['split'], 'train')
            for name in ('markov', 'history', 'oracle'):
                self.assertEqual(report['models'][name]['training']['final_epoch'], 1)
                self.assertEqual(report['models'][name]['test']['physical_delta']['count'], 3)
                self.assertEqual(report['models'][name]['reload_max_abs_error'], 0.)
                self.assertTrue(Path(report['models'][name]['artifact']['path']).exists())
            self.assertTrue(report['latent']['episode_A_B_A_reset_exact'])
            self.assertTrue(report['linear_probe']['encoder_unchanged'])
            self.assertEqual(report['linear_probe']['training_split'], 'train')
            for path in report['figures']:
                self.assertGreater(Path(path).stat().st_size, 1000)
            with self.assertRaises(FileExistsError):
                run_experiment(source, root, epochs=1)


if __name__ == '__main__':
    unittest.main()
