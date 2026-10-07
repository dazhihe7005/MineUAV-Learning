"""Same pretrained weights, paired episode permutations, unchanged seed0 math."""
import copy
import hashlib
import subprocess
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from test_joint_latent_world_model import fixture
from joint_latent_world_model import initialize_joint, component_hashes
from joint_latent_training import train_joint
from joint_autonomous_consistency_training import train_autonomous

BASE = 'fa357af3c1a9fc0f24e635e1b073b180bdca4b64'


class MultiSeedTrainingTests(unittest.TestCase):
    def test_paired_seeds_use_identical_recorded_epoch_orders(self):
        # A changed shuffle RNG, dropped episode, or v3 seed0-only guard fails.
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            episodes = []
            for i in range(4):
                row = copy.deepcopy(ep)
                row['obs'] += i * .01
                row['next_obs'] += i * .01
                episodes.append(row)
            for seed in (1, 2, 3, 4):
                config = dict(seed=seed, epochs=2, batch_size=2, learning_rate=.0003, horizon=10)
                v1, _, _ = initialize_joint(*paths)
                v3, _, _ = initialize_joint(*paths)
                self.assertEqual(component_hashes(v1), component_hashes(v3))
                a = train_joint(v1, episodes, [ep], stats, config, Path(td)/f'a{seed}.pt')
                b = train_autonomous(v3, episodes, [ep], stats, [.2]*64, config, Path(td)/f'b{seed}.pt')
                self.assertEqual([r['episode_order_sha256'] for r in a['history']],
                                 [r['episode_order_sha256'] for r in b['history']])
                self.assertTrue(all(r['windows_used'] == 12 for r in a['history'] + b['history']))
                if seed == 2:
                    # Hand-checked default_rng(2) first permutation of four episodes.
                    wanted = hashlib.sha256(np.array([3, 2, 0, 1], dtype='<i8').tobytes()).hexdigest()
                    self.assertEqual(a['history'][0]['episode_order_sha256'], wanted)

    def test_seed3_repeat_same_final_parameters_and_validation_only_selection(self):
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            config = dict(seed=3, epochs=2, batch_size=2, learning_rate=.0003, horizon=10)
            results = []
            for i in range(2):
                model, _, _ = initialize_joint(*paths)
                r = train_autonomous(model, [ep], [ep], stats, [.2]*64, config, Path(td)/f'{i}.pt')
                results.append((r['parameter_hashes_best'], r['history']))
                self.assertEqual(r['best_epoch'], np.argmin([a['validation']['observation'] for a in r['history']])+1)
            self.assertEqual(results[0], results[1])

    def test_seed0_update_is_bitwise_unchanged_from_saved_base_trainer(self):
        root = Path(__file__).resolve().parents[2]
        source = subprocess.check_output(['git', 'show', f'{BASE}:mujoco/rl/joint_autonomous_consistency_training.py'], cwd=root)
        archived = types.ModuleType('base_v3_trainer')
        exec(compile(source, f'git:{BASE}:v3_trainer', 'exec'), archived.__dict__)
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            config = dict(seed=0, epochs=2, batch_size=2, learning_rate=.0003, horizon=10)
            a, _, _ = initialize_joint(*paths)
            b, _, _ = initialize_joint(*paths)
            old = archived.train_autonomous(a, [ep], [ep], stats, [.2]*64, config, Path(td)/'old.pt')
            new = train_autonomous(b, [ep], [ep], stats, [.2]*64, config, Path(td)/'new.pt')
            self.assertEqual(old['parameter_hashes_best'], new['parameter_hashes_best'])
            self.assertEqual(old['best_validation'], new['best_validation'])

    def test_unrequested_seed_cannot_create_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            paths, stats, _, _, _, _, ep = fixture(td)
            model, _, _ = initialize_joint(*paths)
            path = Path(td)/'forbidden.pt'
            config = dict(seed=5, epochs=1, batch_size=2, learning_rate=.0003, horizon=10)
            with self.assertRaises(ValueError):
                train_autonomous(model, [ep], [ep], stats, [.2]*64, config, path)
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
