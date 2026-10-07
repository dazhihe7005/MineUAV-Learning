"""Paired statistics keep every requested seed; runner isolates training and Test."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from test_run_joint_autonomous_consistency_v3 import fixture
from run_joint_autonomous_consistency_v3 import run_experiment as tiny_v3

try:
    import joint_multiseed_replication as replication
except ImportError:
    replication = None


class MultiSeedReplicationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(replication, 'paired replication runner missing')

    def test_literal_paired_statistics_and_missing_seed_not_discarded(self):
        # Reversed subtraction, unpaired sorting, ddof0 or dropping failure breaks.
        a = [.5, .4, .3, .2, .1]
        b = [.4, .3, .35, .15, .09]
        r = replication.paired_statistics(a, b)
        np.testing.assert_allclose(r['deltas'], [.1, .1, -.05, .05, .01], atol=1e-15)
        self.assertAlmostEqual(r['delta']['mean'], .042)
        self.assertAlmostEqual(r['delta']['std'], np.sqrt(.01628/4))
        self.assertAlmostEqual(r['delta']['median'], .05)
        self.assertEqual(r['v3_win_count'], 4)
        self.assertEqual(r['requested_pair_count'], 5)
        self.assertEqual(r['missing_pair_count'], 0)
        broken = replication.paired_statistics(a, b[:4]+[None])
        self.assertEqual(broken['requested_pair_count'], 5)
        self.assertEqual(broken['missing_pair_count'], 1)
        self.assertIsNone(broken['deltas'][4])
        self.assertEqual(broken['delta']['count'], 4)
        with self.assertRaises(ValueError):
            replication.paired_statistics(a[:4], b[:4])
        with self.assertRaises(ValueError):
            replication.paired_statistics(a, b[:4]+[float('nan')])

    def test_context_seed0_reuse_and_invalid_branch_cannot_train(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            old = tiny_v3(root, v1, audit, v2, make_figures=False)
            original = np.load
            def no_test(path, *a, **kw):
                self.assertNotEqual(Path(path).name, 'test.npz')
                return original(path, *a, **kw)
            with patch('numpy.load', side_effect=no_test):
                c = replication.load_context(root, v1, audit, old['report_path'])
            self.assertEqual(c['initial_scale'], old['initial_latent_scale'])
            self.assertEqual(c['baseline']['test_start_manifest'], old['test_start_manifest'])
            with patch.object(replication, 'train_joint', side_effect=AssertionError('seed0 retraining forbidden')), \
                 patch.object(replication, 'train_autonomous', side_effect=AssertionError('seed0 retraining forbidden')):
                for condition, seed in [('v1', 0), ('v3', 0), ('v2', 1), ('v3', 5)]:
                    with self.assertRaises(ValueError):
                        replication.run_branch(root, condition, seed, c)
            before = copy.deepcopy(c['baseline']['normalization'])
            bad = json.loads(Path(old['report_path']).read_text())
            bad['normalization']['obs']['std'][0] *= 2
            Path(old['report_path']).write_text(json.dumps(bad))
            with self.assertRaises((ValueError, AssertionError)):
                replication.load_context(root, v1, audit, old['report_path'])
            self.assertEqual(before, c['baseline']['normalization'])

    def test_actual_paired_branch_reload_test_after_selection_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            old = tiny_v3(root, v1, audit, v2, make_figures=False)
            c = replication.load_context(root, v1, audit, old['report_path'])
            rows = []
            for condition in ('v1', 'v3'):
                selected = []; original_load = np.load
                train_name = 'train_joint' if condition == 'v1' else 'train_autonomous'
                original_train = getattr(replication, train_name)
                def train(*args, **kwargs):
                    result = original_train(*args, **kwargs)
                    selected.append(result['best_epoch'])
                    return result
                def read(path, *args, **kwargs):
                    if Path(path).name == 'test.npz':
                        self.assertTrue(selected, 'Test opened before Validation selection finished')
                    return original_load(path, *args, **kwargs)
                with patch.object(replication, train_name, side_effect=train), patch('numpy.load', side_effect=read):
                    row = replication.run_branch(root, condition, 1, c)
                self.assertEqual(row['status'], 'completed')
                self.assertEqual(row['config']['seed'], 1)
                self.assertEqual(row['training']['final_epoch'], 2)
                self.assertTrue(all(replication.verify_branch(c, row).values()))
                self.assertEqual(row['evaluation']['periodic_correction']['never']['observation'],
                                 row['evaluation']['horizons']['50']['observation'])
                self.assertEqual(row['evaluation']['horizons']['50']['observation']['window_count'], 14)
                self.assertTrue(all(row['evaluation']['future_observation_leakage'].values()))
                rows.append(row)
                for defect in ('order', 'normalization', 'metric', 'selection', 'seed', 'negative_loss', 'gradient', 'final_loss'):
                    bad = copy.deepcopy(row)
                    if defect == 'order': bad['training']['history'][0]['episode_order_sha256'] = 'wrong'
                    elif defect == 'normalization': bad['normalization_sha256'] = 'wrong'
                    elif defect == 'metric': bad['evaluation']['horizons']['50']['observation']['normalized_observation_rmse'] += 1
                    elif defect == 'selection': bad['training']['best_epoch'] = 999
                    elif defect == 'seed': bad['config']['seed'] = 3
                    elif defect == 'negative_loss':
                        if condition == 'v1': bad['training']['history'][0]['train_loss'] = -999
                        else: bad['training']['history'][0]['train']['total'] = -999
                    elif defect == 'gradient': bad['training']['history'][0]['gradient_norms']['encoder']['mean'] = -999
                    elif condition == 'v1': bad['training']['final_train_loss'] = -999
                    else: bad['training']['final_train']['observation'] = -999
                    with self.assertRaises((AssertionError, ValueError)):
                        replication.verify_branch(c, bad)
                with self.assertRaises(FileExistsError):
                    replication.run_branch(root, condition, 1, c)
            self.assertEqual(rows[0]['episode_order_hashes'], rows[1]['episode_order_hashes'])
            self.assertEqual(rows[0]['initial_parameter_hashes'], rows[1]['initial_parameter_hashes'])

    def test_numerical_failure_is_saved_and_retained_without_replacing_seed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            old = tiny_v3(root, v1, audit, v2, make_figures=False)
            c = replication.load_context(root, v1, audit, old['report_path'])
            with patch.object(replication, 'train_joint', side_effect=FloatingPointError('nonfinite gradient')):
                failed = replication.run_branch(root, 'v1', 4, c)
            self.assertEqual(failed['status'], 'failed')
            self.assertEqual(failed['seed'], 4)
            self.assertEqual(failed['failure']['type'], 'FloatingPointError')
            self.assertTrue(Path(failed['part_path']).exists())
            self.assertEqual(json.loads(Path(failed['part_path']).read_text())['status'], 'failed')
            with patch.object(replication, 'evaluate_minimal', return_value={'stability': {'latent_norm_max': float('inf')}}):
                nonfinite = replication.run_branch(root, 'v1', 3, c)
            self.assertEqual(nonfinite['status'], 'failed')
            self.assertIn('latent_norm_max', nonfinite['failure']['message'])
            self.assertTrue(Path(nonfinite['part_path']).exists())
            self.assertNotIn('Infinity', Path(nonfinite['part_path']).read_text())

    def test_failed_seed_still_verifies_shared_configuration_and_available_orders(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            old = tiny_v3(root, v1, audit, v2, make_figures=False)
            c = replication.load_context(root, v1, audit, old['report_path'])
            with patch.object(replication, 'train_joint', side_effect=FloatingPointError('nonfinite gradient')):
                row = replication.run_branch(root, 'v1', 4, c)
            replication.verify_branch_contract(c, row)
            for defect in ('config', 'initial', 'normalization', 'orders', 'failure'):
                bad = copy.deepcopy(row)
                if defect == 'config': bad['config']['epochs'] = 120
                elif defect == 'initial': bad['initial_parameter_hashes']['encoder'] = 'wrong'
                elif defect == 'normalization': bad['normalization_sha256'] = 'wrong'
                elif defect == 'orders': bad['episode_order_hashes'] = ['wrong']
                else: bad.pop('failure')
                with self.assertRaises((ValueError, AssertionError)):
                    replication.verify_branch_contract(c, bad)

    def test_complete_report_seed0_reuse_readonly_verifier_and_contract_tampering(self):
        # Inference must not instantiate optimizer; report cannot misstate budget.
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); v1, audit, v2 = fixture(root)
            old = tiny_v3(root, v1, audit, v2, make_figures=False)
            c = replication.load_context(root, v1, audit, old['report_path'])
            original_hashes = {p: replication.file_hash(p) for p in c['immutable']}
            for seed in (1, 2, 3, 4):
                for condition in ('v1', 'v3'):
                    self.assertEqual(replication.run_branch(root, condition, seed, c)['status'], 'completed')
            with patch('torch.optim.Adam', side_effect=AssertionError('report/verify must not optimize')):
                result = replication.assemble_report(root, c, make_figures=False)
                self.assertEqual(len(result['branches']), 10)
                self.assertTrue(all(replication.verify_experiment(result['report_path']).values()))
            self.assertTrue(all(r['reused'] for r in result['branches'] if r['seed']==0))
            self.assertTrue(all(r['equal'] for r in result['paired_order_verification'].values()))
            for p, sha in original_hashes.items():
                self.assertEqual(replication.file_hash(p), sha)
            path = Path(result['report_path'])
            for defect in ('summary', 'budget', 'horizons', 'objective', 'missing_seed'):
                bad = copy.deepcopy(result)
                if defect == 'summary': bad['summary']['horizons']['50']['deltas'][0] += 1
                elif defect == 'budget': bad['common_config']['epochs'] = 120
                elif defect == 'horizons': bad['horizons_steps'] = [1, 50]
                elif defect == 'objective': bad['objectives']['v3'] = 'also includes local consistency'
                else: bad['branches'] = bad['branches'][:-1]
                path.write_text(json.dumps(bad))
                with self.assertRaises((ValueError, AssertionError)):
                    replication.verify_experiment(path)


if __name__ == '__main__':
    unittest.main()
