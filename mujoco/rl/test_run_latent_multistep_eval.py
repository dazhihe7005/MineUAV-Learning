"""Offline fixture: never samples a simulator, fits weights or changes splits."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash, json_hash, load_dataset
from latent_dynamics_models import make_model, save_model, load_model
from test_run_latent_memory_ablation import baseline_fixture

try:
    import run_latent_dynamics_multistep_eval as runner
except ImportError:
    runner = None


def fixture(root):
    baseline = json.loads(baseline_fixture(root).read_text())
    stats = baseline['normalization']
    model = make_model('no_memory'); path = root / 'no_memory.pt'
    save_model(path, model, 'no_memory', stats, dict(epoch=0))
    # The fixture supplies real saved model one-step results, without training.
    from run_latent_memory_ablation import evaluate
    splits, _ = load_dataset(root / 'source')
    baseline['models']['no_memory'] = dict(artifact=dict(path=str(path), sha256=file_hash(path)),
        test=evaluate(model, 'no_memory', splits['test'], stats, [.1, .2]))
    report = root / 'memory.json'; report.write_text(json.dumps(baseline))
    return report


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, 'frozen multi-step runner is not implemented')
        torch.set_num_threads(1); torch.manual_seed(0)

    def test_test_only_run_reload_hash_and_report_reproduction(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = fixture(root)
            before = {str(p): file_hash(p) for p in root.glob('*.pt')}
            result = runner.run_evaluation(root, source, horizons=(1, 2))
            self.assertEqual(len(result['figures']), 6)
            self.assertTrue(all(Path(path).is_file() for path in result['figures']))
            self.assertEqual(result['windows']['window_counts'], {'1': 3, '2': 1})
            for p, h in before.items():
                self.assertEqual(file_hash(p), h)
            for key in ('markov', 'no_memory', 'history', 'pi_state'):
                self.assertEqual(result['models'][key]['horizons']['1']['window_count'], 3)
                self.assertTrue(result['models'][key]['teacher_forced']['previous_metrics_match'])
            self.assertTrue(all(runner.verify_report(root / 'mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json').values()))
            manifest_path = Path(result['start_manifest']['path'])
            original = manifest_path.read_text()
            tampered = json.loads(original)
            tampered['episodes'][0]['valid_start_count']['2'] = 999
            manifest_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(AssertionError, 'manifest'):
                runner.verify_report(root / 'mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json')
            manifest_path.write_text(original)
            with self.assertRaises(FileExistsError):
                runner.run_evaluation(root, source, horizons=(1, 2), make_figures=False)

    def test_checkpoint_corruption_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); report = fixture(root)
            with (root / 'history.pt').open('ab') as stream:
                stream.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash'):
                runner.load_inputs(root, report)

    def test_parent_immutable_controller_hash_is_inherited(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); report = fixture(root)
            controller = root / 'controller.py'; controller.write_text('frozen\n')
            data = json.loads(report.read_text())
            data['immutable_artifacts'] = {str(controller): file_hash(controller)}
            report.write_text(json.dumps(data))
            controller.write_text('changed\n')
            with self.assertRaisesRegex(ValueError, 'immutable'):
                runner.load_inputs(root, report)

    def test_normalization_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); report = fixture(root)
            data = json.loads(report.read_text()); data['normalization']['obs']['std'][0] *= 2
            report.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'normalization'):
                runner.load_inputs(root, report)

    def test_loader_preserves_recorded_final_next_observation_without_crossing(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); report = fixture(root)
            context = runner.load_inputs(root, report)
            self.assertEqual(len(context['episodes']), 2)
            np.testing.assert_array_equal(context['episodes'][0]['next_obs'][:, 0], [11, 13])
            np.testing.assert_array_equal(context['episodes'][1]['next_obs'][:, 0], [104])
            # Deliberately remove non-Test arrays: eval may hash them when present
            # but is required to read only Test timestep values.
            (root / 'source/train.npz').unlink(); (root / 'source/val.npz').unlink()
            self.assertEqual(len(runner.load_inputs(root, report)['episodes']), 2)


if __name__ == '__main__':
    unittest.main()
