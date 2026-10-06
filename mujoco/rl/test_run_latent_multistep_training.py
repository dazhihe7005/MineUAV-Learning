"""Loader/runner contracts: immutable stats, split identities and validation-only selection."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash, json_hash
from latent_dynamics_models import make_model
from run_latent_memory_ablation import parameter_hash
from test_run_latent_multistep_eval import fixture
from run_latent_dynamics_multistep_eval import run_evaluation

try:
    import run_latent_multistep_training as runner
except ImportError:
    runner = None


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, 'multi-step training runner missing')
        torch.set_num_threads(1); torch.manual_seed(0)

    def source(self, root):
        memory = fixture(root)
        data = json.loads(memory.read_text())
        torch.manual_seed(0)
        data['architecture'] = dict(initial_parameter_sha256=parameter_hash(make_model('history')))
        memory.write_text(json.dumps(data))
        run_evaluation(root, memory, horizons=(1, 2), make_figures=False)
        return root/'mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json'

    def test_loader_uses_exact_train_val_and_saved_stats_without_test_arrays(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = self.source(root)
            (root/'source/test.npz').unlink()
            ctx = runner.load_training_inputs(root, source, horizon=2)
            self.assertEqual(set(ctx['splits']), {'train', 'val'})
            np.testing.assert_array_equal(ctx['splits']['train'][0]['next_obs'][:, 0], [11, 13])
            self.assertEqual(ctx['manifests']['train']['window_counts'], {'2': 1})
            self.assertEqual(ctx['statistics'], json.loads(source.read_text())['normalization'])
            self.assertEqual(ctx['statistics']['provenance']['split'], 'train')

    def test_changed_train_identity_or_normalization_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = self.source(root)
            path = root/'source/train.npz'
            with path.open('ab') as f: f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'dataset hash'):
                runner.load_training_inputs(root, source, 2)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = self.source(root)
            data = json.loads(source.read_text()); data['normalization']['obs']['mean'][0] += 1
            source.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'normalization'):
                runner.load_training_inputs(root, source, 2)

    def test_saved_window_manifest_hash_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = self.source(root)
            ctx = runner.load_training_inputs(root, source, 2)
            manifest = ctx['manifests']['train']
            expected = manifest.pop('manifest_sha256')
            self.assertEqual(json_hash(manifest), expected)
            manifest['window_counts']['2'] = 999
            self.assertNotEqual(json_hash(manifest), expected)

    def test_missing_initialization_hash_is_not_reported_as_a_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = self.source(root)
            frozen = json.loads(source.read_text()); memory = Path(frozen['source_report'])
            data = json.loads(memory.read_text()); data.pop('architecture')
            memory.write_text(json.dumps(data)); frozen['immutable_artifacts'][str(memory)] = file_hash(memory)
            source.write_text(json.dumps(frozen))
            with self.assertRaisesRegex(ValueError, 'initialization hash'):
                runner.load_training_inputs(root, source, 2)

    def test_pipeline_reload_validation_selection_frozen_metrics_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = self.source(root)
            cfg = dict(seed=0, epochs=2, batch_size=2, learning_rate=.001, horizon=1)
            report = runner.run_experiment(root, source, config=cfg, make_figures=True)
            self.assertEqual(len(report['figures']), 7)
            self.assertTrue(all(Path(p).is_file() for p in report['figures']))
            self.assertTrue(all(runner.verify_experiment(Path(report['report_path'])).values()))
            self.assertTrue(report['verification']['frozen_metrics_reproduced'])
            self.assertEqual(report['training']['final_epoch'], 2)
            self.assertEqual(report['training']['best_epoch'],
                np.argmin([r['validation_loss'] for r in report['training']['history']])+1)
            manifest_path = Path(report['test_start_manifest']['path']); original = manifest_path.read_text()
            manifest_path.write_text(original+' ')
            with self.assertRaisesRegex(ValueError, 'manifest.*hash'):
                runner.verify_experiment(Path(report['report_path']))
            manifest_path.write_text(original)
            with self.assertRaises(FileExistsError):
                runner.run_experiment(root, source, config=cfg, make_figures=False)


if __name__ == '__main__':
    unittest.main()
