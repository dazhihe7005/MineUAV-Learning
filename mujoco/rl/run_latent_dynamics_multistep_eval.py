"""Frozen Test-only autoregressive evaluation; no optimizer or simulator.

Default runs all legal windows at 1/5/10/25/50 steps. --verify re-evaluates saved
metrics read-only. Models, dataset, splits and saved Train stats are immutable.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_dynamics_data import (episodes_from_arrays, file_hash, flatten, json_hash)
from latent_dynamics_evaluation import distance_masks, magnitude_masks, metrics, predict_episodes
from latent_dynamics_models import load_model
from latent_dynamics_multistep import (ErrorAccumulator, HORIZONS, KINDS, LatentAccumulator,
                                     rollout_windows, start_manifest, teacher_latents)

BASE_COMMIT = '9913006fe4f96a8be75a3bbc3a320be2744d0021'
SOURCE_KEYS = dict(markov='markov', no_memory='no_memory', history='history', pi_state='oracle')
LABELS = dict(markov='Markov MLP', no_memory='No-Memory GRU', history='Full-History GRU',
              pi_state='PI-State Baseline (TF PI)')
COLORS = dict(markov='#4477AA', no_memory='#CCBB44', history='#EE7733', pi_state='#228833')


def tensor_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode()); digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def compare(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError('metric keys differ')
        for key in expected:
            compare(actual[key], expected[key])
    elif expected is None or isinstance(expected, (str, bool)):
        if actual != expected:
            raise AssertionError('metric metadata mismatch')
    elif not np.isclose(actual, expected, rtol=1e-7, atol=1e-10):
        raise AssertionError(f'metric mismatch: {actual} vs {expected}')


def load_inputs(root, source_report):
    root, source_report = Path(root).resolve(), Path(source_report).resolve()
    source = json.loads(source_report.read_text()); stats = source['normalization']
    if json_hash(stats) != source['normalization_sha256']:
        raise ValueError('normalization hash mismatch')
    directory = Path(source['dataset']['source_directory'])
    manifest_path = directory / 'manifest.json'; test_path = directory / 'test.npz'
    immutable = {str(source_report): file_hash(source_report)}
    for previous_path, expected in source.get('immutable_artifacts', {}).items():
        if file_hash(previous_path) != expected:
            raise ValueError(f'parent immutable hash mismatch: {previous_path}')
        immutable[previous_path] = expected
    if file_hash(manifest_path) != source['dataset']['source_manifest_sha256']:
        raise ValueError('source manifest hash mismatch')
    if file_hash(test_path) != source['dataset']['source_dataset_sha256']['test']:
        raise ValueError('Test dataset hash mismatch')
    immutable[str(manifest_path)] = file_hash(manifest_path)
    immutable[str(test_path)] = file_hash(test_path)
    # No Train or Validation timestep values are loaded or used in evaluation.
    with np.load(test_path, allow_pickle=False) as arrays:
        episodes = episodes_from_arrays(arrays)
        for ep, start, stop in zip(episodes, arrays['offsets'][:-1], arrays['offsets'][1:]):
            ep['next_obs'] = arrays['inputs'][start+1:stop, :7].copy()
    identity = [dict(target_id=e['metadata']['target_id'], source=e['metadata']['source'], rows=e['source_rows'])
                for e in episodes]
    if identity != source['dataset']['split_identity']['test']:
        raise ValueError('Test split identity mismatch')
    loaded = {}; artifacts = {}
    for kind, previous_key in SOURCE_KEYS.items():
        row = source['models'][previous_key]; artifact = row['artifact']
        if file_hash(artifact['path']) != artifact['sha256']:
            raise ValueError(f'checkpoint hash mismatch: {kind}')
        net, saved, metadata = load_model(artifact['path'])
        if saved != stats:
            raise ValueError('model normalization mismatch')
        net.eval().requires_grad_(False); loaded[kind] = net
        immutable[artifact['path']] = artifact['sha256']
        artifacts[kind] = dict(artifact, label=LABELS[kind], architecture=repr(net),
                               parameter_count=sum(p.numel() for p in net.parameters()),
                               metadata=metadata, parameter_sha256=tensor_hash(net))
    for relative, expected in source.get('frozen_source_hashes', {}).items():
        if file_hash(root / relative) != expected:
            raise ValueError('frozen environment/controller hash mismatch')
        immutable[str(root / relative)] = expected
    return dict(source=source, source_report=str(source_report), statistics=stats,
                episodes=episodes, models=loaded, artifacts=artifacts, immutable=immutable,
                test_identity_sha256=json_hash(identity))


def teacher_forcing_check(model, kind, episodes, stats, expected):
    rows, _ = predict_episodes(model, episodes, stats, history=kind in ('history', 'no_memory'))
    pred, truth = np.concatenate(rows), flatten(episodes, 'delta')
    physical = metrics(pred, truth)
    standardized = metrics(pred/np.asarray(stats['delta']['std']), truth/np.asarray(stats['delta']['std']))
    compare(physical, expected['physical_delta'])
    compare(standardized, expected['standardized_delta'])
    return dict(physical_delta=physical, delta_standardized=standardized,
                observation_normalized_rmse=float(np.sqrt(np.mean(((pred-truth)/np.asarray(stats['obs']['std']))**2))),
                previous_metrics_match=True)


def evaluate_model(model, kind, episodes, stats, thresholds, horizons, verbose=False):
    std = stats['obs']['std']; overall = {h: ErrorAccumulator(std) for h in horizons}
    common = {h: ErrorAccumulator(std) for h in horizons}
    distance = {h: {name: ErrorAccumulator(std) for name in distance_masks(np.zeros(1))} for h in horizons}
    magnitude = {h: {name: ErrorAccumulator(std) for name in magnitude_masks(np.zeros(1), thresholds)} for h in horizons}
    drift = {h: LatentAccumulator() for h in horizons}
    stability = dict(nonfinite_predictions=0, nonfinite_latents=0, first_nonfinite=None,
                     predicted_observation_abs_max=0., latent_norm_max=0.)
    max_horizon = max(horizons)
    for episode_id, ep in enumerate(episodes):
        latent = teacher_latents(model, ep, stats) if kind == 'history' else None
        n = len(ep['obs'])
        for h in horizons:
            total_starts = max(0, n-h+1)
            for start in range(0, total_starts, 512):
                ids = np.arange(start, min(total_starts, start+512))
                out = rollout_windows(model, kind, ep, stats, ids, h, prefix_latents=latent)
                predictions = out['predictions']; error = predictions[:, -1] - ep['next_obs'][ids+h-1]
                overall[h].add(error)
                common[h].add(error[ids < max(0, n-max_horizon+1)])
                for name, mask in distance_masks(np.linalg.norm(ep['obs'][ids, :3], axis=1)).items():
                    distance[h][name].add(error[mask])
                for name, mask in magnitude_masks(np.linalg.norm(ep['pi'][ids], axis=1), thresholds).items():
                    magnitude[h][name].add(error[mask])
                finite = np.isfinite(predictions)
                stability['nonfinite_predictions'] += int((~finite).sum())
                if finite.any():
                    stability['predicted_observation_abs_max'] = max(stability['predicted_observation_abs_max'],
                                                                    float(np.abs(predictions[finite]).max()))
                if not finite.all() and stability['first_nonfinite'] is None:
                    row, step, _ = np.argwhere(~finite)[0]
                    stability['first_nonfinite'] = dict(episode_id=episode_id, target_id=ep['metadata']['target_id'],
                                                        start=int(ids[row]), step=int(step+1), evaluated_horizon=h)
                if kind == 'history':
                    auto = out['latent_autoregressive']; true = out['latent_teacher_forced']
                    drift[h].add(auto[:, -1], true[:, -1])
                    valid = np.isfinite(auto).all(-1)
                    stability['nonfinite_latents'] += int((~valid).sum())
                    if valid.any():
                        stability['latent_norm_max'] = max(stability['latent_norm_max'],
                                                          float(np.linalg.norm(auto[valid], axis=-1).max()))
        if verbose and (episode_id+1) % 25 == 0:
            print(f'{kind}: evaluated {episode_id+1}/{len(episodes)} Test episodes', flush=True)
    rows = {str(h): dict(overall[h].result(), seconds=h/25,
                        distance_regions={name: obj.result() for name,obj in distance[h].items()},
                        pi_magnitude_regions={name: obj.result() for name,obj in magnitude[h].items()})
            for h in horizons}
    return dict(horizons=rows, common_max_horizon_windows={str(h): common[h].result() for h in horizons},
                stability=stability,
                latent_drift={str(h): obj.result() for h,obj in drift.items()} if kind == 'history' else None)


def render_figures(report, context, directory):
    paths = []
    horizons = report['horizons_steps']; seconds = np.array(horizons)/25

    def save(fig, name):
        fig.tight_layout(); path = directory/name; fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, 5))
    for kind in KINDS:
        values = [report['models'][kind]['horizons'][str(h)]['normalized_observation_rmse'] for h in horizons]
        ax.plot(seconds, values, 'o-', label=LABELS[kind], color=COLORS[kind])
    ax.set(xlabel='Prediction horizon (s)', ylabel='Endpoint observation RMSE / Train observation std')
    ax.set_title('Frozen autoregressive prediction — all legal Test windows'); ax.legend(); ax.grid(alpha=.2)
    save(fig, 'multistep_normalized_rmse_vs_horizon.png')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, field, title in zip(axes, ('horizontal_velocity_rmse', 'vertical_velocity_rmse'), ('Horizontal vx/vy', 'Vertical vz')):
        for kind in KINDS:
            axis.plot(seconds, [report['models'][kind]['horizons'][str(h)][field] for h in horizons],
                      'o-', label=LABELS[kind], color=COLORS[kind])
        axis.set(xlabel='Horizon (s)', ylabel='Velocity RMSE (m/s)', title=title); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8); save(fig, 'multistep_velocity_rmse_vs_horizon.png')

    for field, filename in [('distance_regions', 'multistep_error_vs_target_distance.png'),
                            ('pi_magnitude_regions', 'multistep_error_vs_pi_magnitude.png')]:
        names = list(report['models']['markov']['horizons'][str(horizons[0])][field])
        fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 4.2), squeeze=False)
        for axis, name in zip(axes[0], names):
            for kind in KINDS:
                axis.plot(seconds, [report['models'][kind]['horizons'][str(h)][field][name].get('normalized_observation_rmse', np.nan)
                                    for h in horizons], 'o-', label=LABELS[kind], color=COLORS[kind])
            axis.set(xlabel='Horizon (s)', ylabel='Normalized endpoint RMSE', title=name); axis.grid(alpha=.2)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle('Grouped by TRUE starting '+('target distance (m)' if field=='distance_regions' else 'PI magnitude (Train tertiles)'))
        save(fig, filename)

    selected = report['selected_window']; ep = context['episodes'][selected['episode_id']]
    start, h = selected['start'], selected['horizon']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7)); clock = np.arange(h+1)/25
    truth = np.vstack([ep['obs'][start], ep['next_obs'][start:start+h]])
    for axis, dim, label in zip(axes.flat, range(6), ('error_x (m)','error_y (m)','error_z (m)','vx (m/s)','vy (m/s)','vz (m/s)')):
        axis.plot(clock, truth[:, dim], 'k--', lw=2, label='Recorded truth'); axis.set(xlabel='Time after start (s)', ylabel=label)
    for kind in KINDS:
        out = rollout_windows(context['models'][kind], kind, ep, context['statistics'], [start], h)
        trace = np.vstack([ep['obs'][start], out['predictions'][0]])
        for axis, dim in zip(axes.flat, range(6)):
            axis.plot(clock, trace[:, dim], label=LABELS[kind], color=COLORS[kind])
    axes[0,0].legend(fontsize=7); fig.suptitle('Deterministic example: first eligible episode, midpoint legal start; fixed recorded actions')
    save(fig, 'selected_autoregressive_rollout_prediction.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    drift = report['models']['history']['latent_drift']
    axes[0].plot(seconds, [drift[str(h)]['mean_l2'] for h in horizons], 'o-'); axes[0].set(ylabel='Mean latent L2 difference')
    axes[1].plot(seconds, [drift[str(h)]['mean_cosine'] for h in horizons], 'o-'); axes[1].set(ylabel='Mean latent cosine similarity')
    for axis in axes: axis.set(xlabel='Horizon (s)'); axis.grid(alpha=.2)
    fig.suptitle('Autoregressive vs teacher-forced encoder at endpoint-predicting latent')
    save(fig, 'latent_drift_vs_horizon.png')
    return paths


def run_evaluation(root, source_report, horizons=HORIZONS, make_figures=True):
    root = Path(root).resolve(); directory = root/'mujoco/reports'
    path = directory/'latent_dynamics_multistep_frozen_eval_seed0.json'
    manifest_path = directory/'latent_multistep_start_points_seed0.json'
    if path.exists() or manifest_path.exists():
        raise FileExistsError('multi-step outputs already exist; refusing to overwrite')
    started = time.monotonic(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    context = load_inputs(root, source_report); source=context['source']; stats=context['statistics']; episodes=context['episodes']
    windows = start_manifest(episodes, horizons, source['dataset']['source_dataset_sha256']['test'])
    report = dict(experiment='Frozen Multi-Step Autoregressive Rollout Evaluation', seed=0,
                  branch='feat/latent-multistep-eval', base_commit=BASE_COMMIT, horizons_steps=list(horizons),
                  policy_hz=25, source_report=context['source_report'], immutable_artifacts=context['immutable'],
                  normalization=stats, normalization_sha256=json_hash(stats), test_dataset=source['dataset']['splits']['test'],
                  test_identity_sha256=context['test_identity_sha256'], windows=windows,
                  pi_magnitude_bins=source['integral_bins'], models={},
                  protocol=dict(metric='endpoint error at t+H, normalized by saved Train OBSERVATION std (not delta std)',
                                action='fixed recorded executed normalized action sequence, not policy rollout',
                                warm_up='true prefix x_0..x_t -> z_t; first head uses z_t once, then predicted observation only',
                                pi_state='true recorded PI acceleration at each future step, but observation remains autoregressive',
                                yaw='raw observation/delta conventions unchanged; no wrapping or clipping predictions',
                                common_windows='auxiliary curve uses identical starts eligible for largest horizon',
                                latent_index='z_(t+H-1) used to predict o_(t+H)',
                                no_training=True, only_test_timesteps=True))
    for kind in KINDS:
        print(f'Evaluating frozen {LABELS[kind]}', flush=True)
        net = context['models'][kind]
        row = evaluate_model(net, kind, episodes, stats, source['integral_bins']['thresholds_m_s2'], horizons, verbose=True)
        row['artifact'] = context['artifacts'][kind]
        row['teacher_forced'] = teacher_forcing_check(net, kind, episodes, stats, source['models'][SOURCE_KEYS[kind]]['test'])
        if not np.isclose(row['horizons']['1']['normalized_observation_rmse'], row['teacher_forced']['observation_normalized_rmse'], rtol=1e-5, atol=1e-9):
            raise AssertionError('warmed one-step recursion differs from teacher forcing')
        if tensor_hash(net) != context['artifacts'][kind]['parameter_sha256']:
            raise AssertionError('frozen model parameters changed')
        report['models'][kind] = row
    report['comparison'] = {str(h): {
        'history_relative_reduction_vs_no_memory': 1-report['models']['history']['horizons'][str(h)]['normalized_observation_rmse']/report['models']['no_memory']['horizons'][str(h)]['normalized_observation_rmse'],
        'history_relative_reduction_vs_markov': 1-report['models']['history']['horizons'][str(h)]['normalized_observation_rmse']/report['models']['markov']['horizons'][str(h)]['normalized_observation_rmse']}
        for h in horizons if report['models']['markov']['horizons'][str(h)].get('normalized_observation_rmse', 0)>0}
    eligible = [(i,e) for i,e in enumerate(episodes) if len(e['obs'])>=max(horizons)]
    if eligible:
        i, ep = eligible[0]
        report['selected_window'] = dict(rule='first Test episode eligible for max horizon; midpoint legal start, no performance selection',
                                        episode_id=i, target_id=ep['metadata']['target_id'],
                                        start=(len(ep['obs'])-max(horizons))//2, horizon=max(horizons))
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(windows, indent=2, allow_nan=False)+'\n')
    report['start_manifest'] = dict(path=str(manifest_path), sha256=file_hash(manifest_path))
    report['figures'] = render_figures(report, context, directory) if make_figures else []
    report['verification'] = dict(immutable_hashes=all(file_hash(p)==h for p,h in context['immutable'].items()),
                                  no_model_training=True, models_frozen=True, test_only=True, teacher_forced_match=True)
    if not all(report['verification'].values()): raise AssertionError('immutable artifacts changed')
    report['elapsed_seconds'] = time.monotonic()-started
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Saved {path} ({report["elapsed_seconds"]:.1f}s)', flush=True)
    return report


def verify_report(path):
    path = Path(path).resolve(); report=json.loads(path.read_text()); torch.set_num_threads(1)
    for p,h in report['immutable_artifacts'].items():
        if file_hash(p)!=h: raise ValueError('immutable hash mismatch')
    context = load_inputs(path.parents[2], report['source_report'])
    windows = start_manifest(context['episodes'], report['horizons_steps'], context['source']['dataset']['source_dataset_sha256']['test'])
    if windows != report['windows'] or windows != json.loads(Path(report['start_manifest']['path']).read_text()):
        raise AssertionError('start manifest mismatch')
    if file_hash(report['start_manifest']['path']) != report['start_manifest']['sha256']:
        raise ValueError('start manifest hash mismatch')
    if context['statistics'] != report['normalization'] or json_hash(context['statistics']) != report['normalization_sha256']:
        raise AssertionError('normalization mismatch')
    for kind in KINDS:
        actual = evaluate_model(context['models'][kind], kind, context['episodes'], context['statistics'],
                                report['pi_magnitude_bins']['thresholds_m_s2'], report['horizons_steps'])
        for field in actual:
            compare(actual[field], report['models'][kind][field])
    return dict(immutable_hashes=True, start_manifest=True, saved_statistics=True, all_model_metrics=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true')
    args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    if args.verify:
        print(json.dumps(verify_report(root/'mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json'), indent=2))
    else:
        run_evaluation(root, root/'mujoco/reports/latent_memory_ablation_seed0.json')
