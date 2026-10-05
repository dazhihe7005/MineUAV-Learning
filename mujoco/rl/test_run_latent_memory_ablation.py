import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import fit_statistics, flatten, json_hash, load_dataset, file_hash
from latent_dynamics_models import make_model, save_model
from latent_dynamics_evaluation import metrics, predict_episodes, grouped_metrics, distance_masks, magnitude_masks
from test_latent_dynamics_data import fixture_arrays

try:
    import run_latent_memory_ablation as ablation
except ImportError:
    ablation = None


def baseline_fixture(root):
    """Untrained baseline fixture; never retrains existing History/Markov/Oracle."""
    source = root / 'source'; source.mkdir(); hashes = {}
    for s, t in [('train', 1), ('val', 2), ('test', 3)]:
        p = source / f'{s}.npz'; np.savez_compressed(p, **fixture_arrays(t)); hashes[s] = file_hash(p)
    (source / 'manifest.json').write_text(json.dumps(dict(dataset_sha256=hashes, seed=0)))
    splits, manifest = load_dataset(source); stats = fit_statistics(splits['train'])
    report = dict(dataset=manifest, normalization=stats, normalization_sha256=json_hash(stats),
                  trainer=dict(seed=0, epochs=2, batch_size=16, learning_rate=.001),
                  integral_bins=dict(thresholds_m_s2=[.1, .2]), frozen_source_hashes={}, models={})
    truth = flatten(splits['test'], 'delta'); obs = flatten(splits['test'], 'obs')
    pi = flatten(splits['test'], 'pi'); std = np.asarray(stats['delta']['std'])
    for kind in ('markov', 'history', 'oracle'):
        model = make_model(kind); p = root / f'{kind}.pt'
        save_model(p, model, kind, stats, dict(epoch=0))
        pred, _ = predict_episodes(model, splits['test'], stats, history=(kind == 'history'))
        pred = np.concatenate(pred)
        report['models'][kind] = dict(artifact=dict(path=str(p), sha256=file_hash(p)),
            test=dict(physical_delta=metrics(pred, truth), standardized_delta=metrics(pred/std, truth/std),
                      distance_regions=grouped_metrics(pred, truth, distance_masks(np.linalg.norm(obs[:, :3], axis=1))),
                      integral_regions=grouped_metrics(pred, truth, magnitude_masks(np.linalg.norm(pi, axis=1), [.1, .2]))))
    path = root / 'baseline.json'; path.write_text(json.dumps(report)); return path


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ablation, 'memory ablation runner is not implemented')
        torch.set_num_threads(1)

    def test_only_new_model_is_trained_with_reused_statistics_and_frozen_baselines(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); baseline = baseline_fixture(root)
            before = file_hash(baseline)
            result = ablation.run_ablation(root, baseline)
            self.assertEqual(file_hash(baseline), before)
            self.assertEqual(result['normalization_sha256'], json_hash(result['normalization']))
            self.assertTrue(result['verification']['baseline_artifacts_unchanged'])
            self.assertTrue(result['verification']['same_initial_parameters'])
            self.assertEqual(result['models']['no_memory']['training']['final_epoch'], 2)
            self.assertEqual(result['models']['no_memory']['test']['physical_delta']['count'], 3)
            verified = ablation.verify_ablation(root / 'mujoco/reports/latent_memory_ablation_seed0.json')
            self.assertTrue(all(verified.values()))
            report_path = root / 'mujoco/reports/latent_memory_ablation_seed0.json'
            original = report_path.read_text()
            tampered = json.loads(original)
            training = tampered['models']['no_memory']['training']
            wrong_epoch = 1 if training['best_epoch'] != 1 else 2
            training['best_epoch'] = wrong_epoch
            training['history'][wrong_epoch - 1]['validation_loss'] = -1
            report_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(AssertionError, 'checkpoint'):
                ablation.verify_ablation(report_path)
            tampered = json.loads(original)
            training = tampered['models']['no_memory']['training']
            training['history'][training['best_epoch'] - 1]['validation_loss'] = -1
            training['best_validation_loss'] = -1
            report_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(AssertionError, 'validation'):
                ablation.verify_ablation(report_path)
            report_path.write_text(original)
            with self.assertRaises(FileExistsError):
                ablation.run_ablation(root, baseline)

    def test_corrupted_frozen_model_blocks_ablation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); baseline = baseline_fixture(root)
            with (root / 'history.pt').open('ab') as f:
                f.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                ablation.run_ablation(root, baseline)


if __name__ == '__main__':
    unittest.main()
