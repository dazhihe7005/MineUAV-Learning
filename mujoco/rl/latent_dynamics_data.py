"""Immutable observer-dataset adapter; no simulator or historical audit imports.

Only intra-episode adjacent observations are paired. The original dataset lacks
terminal observations, so each episode's final command is deliberately excluded.
Previous/current actions are EXECUTED NORMALIZED [-1,1] actions, not physical units.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

SPLITS = ('train', 'val', 'test')
DIMENSIONS = ('error_x', 'error_y', 'error_z', 'vx', 'vy', 'vz', 'yaw_error')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def episodes_from_arrays(arrays):
    inputs, actions, truth = (np.asarray(arrays[k], dtype=np.float32)
                             for k in ('inputs', 'actions', 'truth'))
    offsets = np.asarray(arrays['offsets'])
    metadata = json.loads(str(arrays['metadata']))
    n = len(inputs)
    if inputs.shape != (n, 11) or actions.shape != (n, 4) or truth.shape != (n, 3):
        raise ValueError('invalid input/action/truth shapes')
    if (offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer)
            or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 2)
            or len(metadata) != len(offsets) - 1):
        raise ValueError('invalid episode offsets/metadata')
    if not all(np.isfinite(a).all() for a in (inputs, actions, truth)):
        raise ValueError('non-finite dataset')
    if np.abs(actions).max() > 1.00001:
        raise ValueError('actions are not executed normalized actions')
    episodes = []
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        if not np.allclose(inputs[start, 7:], 0, atol=1e-7):
            raise ValueError('previous action not reset')
        if not np.allclose(inputs[start + 1:stop, 7:], actions[start:stop - 1], atol=1e-7):
            raise ValueError('previous action alignment mismatch')
        if not np.allclose(truth[start], 0, atol=1e-7):
            raise ValueError('PI state not reset')
        obs = inputs[start:stop - 1, :7].copy()
        nxt = inputs[start + 1:stop, :7].copy()
        episodes.append(dict(obs=obs, previous_action=inputs[start:stop - 1, 7:].copy(),
                             action=actions[start:stop - 1].copy(),
                             pi=truth[start:stop - 1].copy(), delta=nxt - obs,
                             metadata=metadata[i], source_rows=[int(start), int(stop - 1)]))
    return episodes


def validate_splits(splits):
    ids = {s: {e['metadata']['target_id'] for e in splits[s]} for s in SPLITS}
    for a, b in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        if ids[a] & ids[b]:
            raise ValueError(f'target overlap: {a}/{b}')
    return {s: sorted(ids[s]) for s in SPLITS}


def load_dataset(root):
    root = Path(root)
    source = json.loads((root / 'manifest.json').read_text())
    splits, hashes = {}, {}
    for s in SPLITS:
        path = root / f'{s}.npz'
        hashes[s] = file_hash(path)
        if hashes[s] != source['dataset_sha256'][s]:
            raise ValueError(f'dataset hash mismatch: {s}')
        with np.load(path, allow_pickle=False) as arrays:
            splits[s] = episodes_from_arrays(arrays)
    target_ids = validate_splits(splits)
    split_identity = {s: [dict(target_id=e['metadata']['target_id'],
                              source=e['metadata']['source'], rows=e['source_rows'])
                          for e in splits[s]] for s in SPLITS}
    manifest = dict(source_directory=str(root), source_manifest_sha256=file_hash(root / 'manifest.json'),
                    source_dataset_sha256=hashes, generation_seed=source.get('seed'),
                    generation_config=source, regenerated=False,
                    transition_rule='o_t=inputs[t,:7], a_t=actions[t], next=inputs[t+1,:7]; within episode only',
                    previous_action_rule='inputs[t,7:11] = executed normalized action at t-1; reset=0',
                    omitted_terminal_transitions=sum(len(splits[s]) for s in SPLITS),
                    target_ids=target_ids, split_sha256=json_hash(split_identity),
                    split_identity=split_identity, splits={})
    for s in SPLITS:
        eps = splits[s]
        manifest['splits'][s] = dict(targets=len(target_ids[s]), episodes=len(eps),
                                     original_steps=sum(len(e['obs']) + 1 for e in eps),
                                     transitions=sum(len(e['obs']) for e in eps),
                                     yaw_wrap_jumps=sum(int(np.sum(np.abs(e['delta'][:, 6]) > np.pi)) for e in eps))
    manifest['derived_manifest_sha256'] = json_hash(manifest)
    return splits, manifest


def flatten(episodes, key):
    return np.concatenate([e[key] for e in episodes], axis=0)


def fit_statistics(train):
    """Call only with Train episodes; population std, shared for all models."""
    stats = {}
    for key in ('obs', 'action', 'delta', 'pi'):
        a = flatten(train, key).astype(np.float64)
        stats[key] = dict(mean=a.mean(0).tolist(), std=np.maximum(a.std(0), 1e-6).tolist())
    stats['provenance'] = dict(split='train', transitions=sum(len(e['obs']) for e in train),
                               std_floor=1e-6, previous_action_statistics='same as current train actions')
    return stats


def normalize(array, stats, key):
    stat = stats['action' if key == 'previous_action' else key]
    return ((array - np.asarray(stat['mean'])) / np.asarray(stat['std'])).astype(np.float32)


def pad_episodes(episodes, stats):
    lengths = [len(e['obs']) for e in episodes]
    max_len = max(lengths)
    batch = {}
    for key, dim in [('obs', 7), ('previous_action', 4), ('action', 4), ('delta', 7), ('pi', 3)]:
        array = np.zeros((len(episodes), max_len, dim), dtype=np.float32)
        for i, e in enumerate(episodes):
            array[i, :lengths[i]] = normalize(e[key], stats, key)
        batch[key] = torch.from_numpy(array)
    batch['mask'] = torch.arange(max_len)[None, :] < torch.tensor(lengths)[:, None]
    batch['lengths'] = torch.tensor(lengths)
    return batch
