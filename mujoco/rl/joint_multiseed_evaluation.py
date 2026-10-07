"""Frozen paired replication metrics, without covariance/PCA/sensitivity audits."""
import copy

import numpy as np
import torch

from joint_latent_world_model import component_hashes, predict_window
from joint_latent_evaluation import encoded_sequences, evaluate_joint, Consistency
from joint_composition_core import freeze_joint, compose
from latent_dynamics_multistep import ErrorAccumulator


@torch.inference_mode()
def periodic_correction(model, episodes, stats, train_latents, pure_h50):
    """Same H50 cohort; correction BEFORE next transition, not at endpoint."""
    scale = np.maximum(np.concatenate(train_latents).astype(np.float64).std(0), 1e-6)
    std, mean = np.asarray(stats['obs']['std']), np.asarray(stats['obs']['mean'])
    records = {c: (ErrorAccumulator(std), Consistency(scale)) for c in (1, 5, 10, 25)}
    sequences = encoded_sequences(model, episodes, stats)
    for ep, z in zip(episodes, sequences):
        count = max(0, len(ep['action']) - 50 + 1)
        for begin in range(0, count, 512):
            ids = np.arange(begin, min(count, begin + 512))
            actions = torch.from_numpy(ep['action'][ids[:, None] + np.arange(50)].astype(np.float64))
            reference = torch.from_numpy(z[ids[:, None] + np.arange(51)])
            initial = torch.from_numpy(z[ids])
            truth = ep['next_obs'][ids + 49]
            for c, (observations, latents) in records.items():
                out = compose(model, initial, actions, stats, c, reference)
                predicted = out['normalized_observations'][:, -1].numpy().astype(np.float64)*std + mean
                observations.add(predicted - truth)
                latents.add(out['latents'][:, -1].numpy(), z[ids + 50])
    result = {str(c): dict(observation=o.result(), latent_consistency=z.result()) for c, (o, z) in records.items()}
    result['never'] = dict(observation=pure_h50['observation'], latent_consistency=pure_h50['latent_consistency'])
    result['50'] = result['never']
    return result


@torch.inference_mode()
def evaluate_minimal(model, train, test, stats, baseline):
    before = component_hashes(model)
    freeze_joint(model)
    train_latents = encoded_sequences(model, train, stats)
    result = evaluate_joint(model, test, stats, baseline['pi_magnitude_bins']['thresholds_m_s2'],
                            baseline['horizons_steps'], train_latents)
    result['periodic_correction'] = periodic_correction(model, test, stats, train_latents, result['horizons']['50'])
    selected = baseline['selected_window']
    ep = test[selected['episode_id']]
    start, horizon = selected['start'], selected['horizon']
    original = predict_window(model, ep, start, horizon, stats)
    changed = copy.deepcopy(ep)
    changed['obs'][start + 1:] += 1000
    changed['next_obs'][:] = np.nan
    corrupted = predict_window(model, changed, start, horizon, stats)
    changed['obs'] = changed['obs'][:start + 1]
    changed['previous_action'] = changed['previous_action'][:start + 1]
    del changed['next_obs']
    deleted = predict_window(model, changed, start, horizon, stats)
    result['future_observation_leakage'] = {
        f'{name}_{field}_unchanged': torch.equal(original[field], out[field])
        for name, out in [('corrupted', corrupted), ('deleted', deleted)]
        for field in ('latents', 'normalized_observations')}
    if not all(result['future_observation_leakage'].values()) or component_hashes(model) != before:
        raise AssertionError('future input leakage or frozen parameter mutation')
    result['parameter_hashes_unchanged'] = True
    variance = result['latent_variance']
    # Explicit descriptive warning; retain flagged seeds, never drop or replace.
    result['potential_latent_collapse'] = any(
        row['near_zero_std_dimension_count'] >= 32 or row['total_variance'] < 1e-8
        for row in variance.values())
    result['collapse_warning_rule'] = '>=32/64 dimensions std<1e-6 OR total variance<1e-8; descriptive, not a sufficiency theorem'
    return result
