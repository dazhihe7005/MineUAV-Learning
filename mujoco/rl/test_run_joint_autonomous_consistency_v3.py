"""Untrained saved v1/v2 fixture baselines; only a tiny v3 run is optimized."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_run_joint_consistency_v2 import fixture as v1_fixture
from run_joint_consistency_v2 import load_context, BASELINE_FIELDS
from joint_consistency_training import save_consistency
from joint_consistency_evaluation import local_relative
from joint_latent_world_model import component_hashes
from joint_latent_evaluation import encoded_sequences
from joint_latent_world_model import latent_summary
from joint_composition_core import freeze_joint
from latent_dynamics_data import file_hash, json_hash

try:
    import run_joint_autonomous_consistency_v3 as runner
except ImportError:
    runner = None


def fixture(root):
    v1, audit = v1_fixture(root)
    c = load_context(root, v1, audit); base = c['baseline']
    output = root / 'mujoco/reports'
    scale_path = output / 'joint_latent_consistency_v2_initial_scale_seed0.json'
    scale_path.write_text(json.dumps(c['initial_scale']))
    model_path = root / 'mujoco/rl/models/untrained_v2_fixture.pt'
    save_consistency(model_path, c['model'], c['statistics'], {'fixture': 'untrained saved baseline, no optimizer'})
    from run_joint_consistency_v2 import load_split
    test = load_split(c['dataset'], 'test')
    freeze_joint(c['model'])
    relative = local_relative(c['model'], test, c['statistics'])
    variance = {split: latent_summary(np.concatenate(encoded_sequences(c['model'], episodes, c['statistics'])))
                for split, episodes in [('train', c['splits']['train']), ('test', test)]}
    v2 = dict(config=base['config'], dataset=base['dataset'], initialization=base['initialization'], normalization=base['normalization'],
        normalization_sha256=base['normalization_sha256'], window_manifests=base['window_manifests'], test_start_manifest=base['test_start_manifest'],
        initial_latent_scale=c['initial_scale'], initial_scale_file=dict(path=str(scale_path), sha256=file_hash(scale_path), semantic_sha256=json_hash(c['initial_scale'])),
        training=dict(parameter_hashes_before=base['training']['parameter_hashes_before']),
        model=dict(path=str(model_path), sha256=file_hash(model_path)),
        audit={k: c['baseline_audit'][k] for k in BASELINE_FIELDS},
        relative_local_residual={'v1': relative, 'v2': relative}, latent_variance=variance,
        fixture='saved untrained baseline for independent runner/verification tests')
    path = output / 'joint_latent_consistency_v2_seed0.json'; path.write_text(json.dumps(v2))
    return v1, audit, path


class AutonomousRunnerTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner, 'v3 runner missing')

    def test_context_reuses_scale_initialization_windows_and_does_not_load_test_values(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root); original = np.load; opened = []
            def checked(path, *a, **kw):
                opened.append(Path(path).name)
                self.assertNotEqual(Path(path).name, 'test.npz')
                return original(path, *a, **kw)
            with patch('numpy.load', side_effect=checked):
                c = runner.load_context(root, v1, audit, v2)
            self.assertEqual(opened, ['train.npz', 'val.npz'])
            self.assertEqual(c['initial_scale'], c['v2']['initial_latent_scale'])
            self.assertEqual(component_hashes(c['model']), c['baseline']['training']['parameter_hashes_before'])
            self.assertEqual(c['statistics'], c['v2']['normalization'])
            bad = json.loads(v2.read_text()); bad['training']['parameter_hashes_before']['transition'] = 'wrong'
            v2.write_text(json.dumps(bad))
            with self.assertRaises((AssertionError, ValueError)):
                runner.load_context(root, v1, audit, v2)

    def test_complete_run_independent_verification_no_baseline_training_or_source_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            before = {str(p): file_hash(p) for p in (v1, audit, v2)}
            import matplotlib.pyplot as plt
            original_save = plt.savefig
            def checked_save(path, *args, **kwargs):
                if Path(path).stem == 'v3_decoder_sensitivity':
                    fig = plt.gcf(); fig.canvas.draw()
                    for ax in fig.axes:
                        label = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
                        self.assertGreaterEqual(label.y0, 0.)
                        self.assertLessEqual(label.y1, fig.bbox.height)
                return original_save(path, *args, **kwargs)
            with patch('joint_consistency_training.train_consistency', side_effect=AssertionError('no v2 baseline retraining')), \
                 patch('matplotlib.pyplot.savefig', side_effect=checked_save):
                result = runner.run_experiment(root, v1, audit, v2)
            self.assertEqual(len(result['figures']), 13)
            self.assertTrue(all(Path(p).stat().st_size > 1000 for p in result['figures']))
            self.assertEqual(result['lambda_autonomous_consistency'], .1)
            self.assertFalse(result['local_consistency_enabled'])
            self.assertEqual(result['windows'], json.loads(v1.read_text())['windows'])
            self.assertTrue(all(runner.verify_experiment(result['report_path']).values()))
            for p, h in before.items():
                self.assertEqual(file_hash(p), h)
            with self.assertRaises(FileExistsError):
                runner.run_experiment(root, v1, audit, v2)
            report_path = Path(result['report_path'])
            for field in ('lambda', 'local', 'scale', 'baseline_empty', 'baseline_missing', 'epoch', 'total', 'metric', 'selected_epoch', 'gradient', 'selected_consistency_history'):
                bad = copy.deepcopy(result)
                if field == 'lambda': bad['lambda_autonomous_consistency'] = 1.
                elif field == 'local': bad['local_consistency_enabled'] = True
                elif field == 'scale': bad['initial_latent_scale']['std'][0] *= 2
                elif field == 'baseline_empty': bad['baselines'] = {}
                elif field == 'baseline_missing': del bad['baselines']['v2']['audit']['local_one_step']
                elif field == 'epoch': bad['training']['history'][0]['epoch'] = 99
                elif field == 'total': bad['training']['history'][0]['train']['total'] += 1
                elif field == 'metric': bad['audit']['autonomous']['50']['observation']['normalized_observation_rmse'] += 1
                elif field == 'selected_epoch': bad['training']['best_epoch'] = 99
                elif field == 'selected_consistency_history':
                    row = bad['training']['history'][bad['training']['best_epoch'] - 1]['validation']
                    row['consistency'] += 1.; row['weighted_consistency'] += .1; row['total'] += .1
                    if bad['training']['best_epoch'] == bad['training']['final_epoch']:
                        bad['training']['final_validation'] = copy.deepcopy(row)
                else: bad['training']['history'][0]['gradient_norms']['encoder']['mean'] = -1
                report_path.write_text(json.dumps(bad))
                with self.assertRaises((AssertionError, ValueError)):
                    runner.verify_experiment(report_path)

    def test_history_cannot_omit_modules_or_claim_direct_decoder_consistency_gradient(self):
        # Independent tiny v3 fixture history exercises the actual recorder schema.
        from joint_autonomous_consistency_training import train_autonomous
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            c = runner.load_context(root, v1, audit, v2)
            config = dict(c['baseline']['config']); config['epochs'] = 1
            t = train_autonomous(c['model'], c['splits']['train'], c['splits']['val'], c['statistics'],
                                 c['initial_scale']['std'], config, root/'history.pt')
            runner.verify_history(t, config)
            for change in ('empty', 'decoder', 'fraction', 'missing_transition'):
                bad = copy.deepcopy(t)
                row = bad['history'][0]
                if change == 'empty':
                    row['gradient_norms'] = {}; row['objective_gradient_diagnostics'] = {}
                elif change == 'decoder':
                    row['objective_gradient_diagnostics']['decoder']['weighted_consistency_norm'] = 1.
                elif change == 'fraction':
                    row['objective_gradient_diagnostics']['encoder']['fraction_batches_consistency_norm_exceeds_observation'] = 2.
                else:
                    del row['gradient_norms']['transition']
                with self.assertRaises(AssertionError):
                    runner.verify_history(bad, config)

    def test_baseline_context_rejects_empty_metric_section(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            bad = json.loads(v2.read_text()); bad['audit']['local_one_step'] = {}
            v2.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                runner.load_context(root, v1, audit, v2)


if __name__ == '__main__':
    unittest.main()
