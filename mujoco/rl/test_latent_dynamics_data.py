"""Guard transition alignment, train-only statistics, split and hash integrity."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import latent_dynamics_data as data
except ImportError:
    data = None


def fixture_arrays(target=0):
    # Two episodes: three observations followed by two. Never pair rows 2→3.
    inputs = np.zeros((5, 11), np.float32)
    inputs[:, 0] = [10, 11, 13, 100, 104]
    inputs[1, 7] = 0.1
    inputs[2, 7] = 0.2
    inputs[4, 7] = 0.4
    actions = np.zeros((5, 4), np.float32)
    actions[:, 0] = [0.1, 0.2, 0.3, 0.4, 0.5]
    return dict(inputs=inputs, actions=actions, truth=np.zeros((5, 3), np.float32),
                offsets=np.array([0, 3, 5]), metadata=np.array(json.dumps([
                    dict(target_id=target, source='fixture'),
                    dict(target_id=target, source='fixture')])) )


class DataTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(data, 'transition loader is not implemented')

    def test_adjacent_transitions_do_not_cross_episode_and_action_is_aligned(self):
        episodes = data.episodes_from_arrays(fixture_arrays())
        self.assertEqual([len(e['obs']) for e in episodes], [2, 1])
        np.testing.assert_array_equal(episodes[0]['delta'][:, 0], [1, 2])
        np.testing.assert_array_equal(episodes[1]['delta'][:, 0], [4])
        np.testing.assert_allclose(episodes[0]['action'][:, 0], [.1, .2])
        np.testing.assert_allclose(episodes[0]['previous_action'][:, 0], [0, .1])

    def test_split_overlap_is_rejected(self):
        split = {s: data.episodes_from_arrays(fixture_arrays(t))
                 for s, t in [('train', 1), ('val', 2), ('test', 1)]}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            data.validate_splits(split)

    def test_normalization_is_train_only_and_zero_std_is_safe(self):
        train = data.episodes_from_arrays(fixture_arrays())
        stats = data.fit_statistics(train)
        np.testing.assert_allclose(stats['obs']['mean'][0], 121 / 3)
        self.assertEqual(stats['obs']['std'][1], 1e-6)
        held = data.episodes_from_arrays(fixture_arrays(8))
        held[0]['obs'][:] = 1e8
        before = json.dumps(stats, sort_keys=True)
        data.pad_episodes(held, stats)
        self.assertEqual(before, json.dumps(stats, sort_keys=True))

    def test_padding_mask_excludes_unavailable_rows(self):
        eps = data.episodes_from_arrays(fixture_arrays())
        b = data.pad_episodes(eps, data.fit_statistics(eps))
        np.testing.assert_array_equal(b['mask'].numpy(), [[True, True], [True, False]])
        self.assertEqual(tuple(b['obs'].shape), (2, 2, 7))

    def test_hash_manifest_is_verified_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); hashes = {}
            for s, t in [('train', 1), ('val', 2), ('test', 3)]:
                np.savez_compressed(root / f'{s}.npz', **fixture_arrays(t))
                hashes[s] = hashlib.sha256((root / f'{s}.npz').read_bytes()).hexdigest()
            (root / 'manifest.json').write_text(json.dumps(dict(dataset_sha256=hashes, seed=0)))
            splits, manifest = data.load_dataset(root)
            self.assertEqual(manifest['splits']['train']['transitions'], 3)
            self.assertEqual(len(manifest['split_sha256']), 64)
            with (root / 'train.npz').open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'hash'):
                data.load_dataset(root)


if __name__ == '__main__':
    unittest.main()
