"""Offline one-step latent dynamics experiment. Never imports a policy or Env.

Run from project root: .venv/bin/python mujoco/rl/run_latent_dynamics_one_step.py
Raw data stay local; report embeds source manifest, hashes and split identities.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_dynamics_data import (DIMENSIONS, file_hash, fit_statistics, flatten,
                                  json_hash, load_dataset, pad_episodes)
from latent_dynamics_models import load_model, train_model
from latent_dynamics_evaluation import (apply_linear_probe, distance_masks, fit_linear_probe,
                                        gap_closure, grouped_metrics, magnitude_masks,
                                        metrics, predict_episodes)

MODEL_NAMES = dict(markov='markov_dynamics_mlp.pt', history='history_latent_gru.pt',
                   oracle='oracle_dynamics_mlp.pt')
COLORS = dict(markov='#4477AA', history='#EE7733', oracle='#228833')


def artifact(path):
    return dict(path=str(path), sha256=file_hash(path), bytes=path.stat().st_size)


def plot_results(report, episodes, predictions, true_pi, probe_pi, directory):
    paths = []

    def save(fig, name):
        path = directory / name
        fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
        paths.append(str(path))

    names = list(MODEL_NAMES)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, [report['models'][k]['test']['standardized_delta']['overall']['rmse'] for k in names],
           color=[COLORS[k] for k in names])
    ax.set(ylabel='Train-standardized delta RMSE', title='One-step delta prediction — shared Test split')
    save(fig, 'one_step_error_model_comparison.png')
    for field, filename, xlabel in [
        ('distance_regions', 'prediction_error_vs_target_distance.png', 'Distance to target (m)'),
        ('integral_regions', 'prediction_error_vs_integral_magnitude.png', 'True PI magnitude — Train tertiles')]:
        groups = list(report['models']['markov']['test'][field])
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(groups)); width = .24
        for j, k in enumerate(names):
            rows = report['models'][k]['test'][field]
            values = [rows[g]['overall']['rmse'] if rows[g]['overall'] else np.nan for g in groups]
            ax.bar(x + (j - 1) * width, values, width, label=k, color=COLORS[k])
        ax.set(xticks=x, xticklabels=groups, xlabel=xlabel,
               ylabel='Physical delta RMSE (mixed units)', title='Prediction error by region')
        ax.legend()
        save(fig, filename)

    # Fixed identity: first scripted episode in Test (otherwise first episode).
    selected = next((i for i, e in enumerate(episodes) if e['metadata']['source'] == 'scripted'), 0)
    episode = episodes[selected]
    offset = sum(len(e['obs']) for e in episodes[:selected]); n = len(episode['obs'])
    t = np.arange(n) / 25.
    fig, axes = plt.subplots(7, 1, figsize=(10, 15), sharex=True)
    for d, ax in enumerate(axes):
        ax.plot(t, episode['delta'][:, d], 'k', lw=1.5, label='true')
        for k in names:
            ax.plot(t, predictions[k][offset:offset + n, d], color=COLORS[k], alpha=.85, label=k)
        ax.set_ylabel(f'delta {DIMENSIONS[d]}')
    axes[0].legend(ncol=4); axes[0].set_title(f'Fixed Test episode: target {episode["metadata"]["target_id"]}, {episode["metadata"]["source"]}')
    axes[-1].set_xlabel('Episode time (s)')
    save(fig, 'selected_true_vs_predicted_delta_observation.png')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for d, ax in enumerate(axes):
        ax.plot(t, true_pi[offset:offset + n, d], 'k', label='true PI')
        ax.plot(t, probe_pi[offset:offset + n, d], color=COLORS['history'], label='frozen latent linear probe')
        ax.set_ylabel(f'PI {"xyz"[d]} (m/s²)')
    axes[0].legend(); axes[-1].set_xlabel('Episode time (s)')
    save(fig, 'latent_linear_probe_pi_state.png')
    return paths, dict(episode_index=selected, metadata=episode['metadata'],
                       selection_rule='first scripted Test episode, selected by identity, not prediction quality')


def run_experiment(dataset, root, epochs=60):
    root, dataset = Path(root).resolve(), Path(dataset).resolve()
    models_dir = root / 'mujoco/rl/models'; reports = root / 'mujoco/reports'
    report_path = reports / 'latent_dynamics_one_step_seed0.json'
    outputs = [models_dir / f for f in MODEL_NAMES.values()] + [report_path, models_dir / 'latent_pi_linear_probe.pt']
    if any(p.exists() for p in outputs):
        raise FileExistsError('experiment artifacts already exist; refusing to overwrite')
    models_dir.mkdir(parents=True, exist_ok=True); reports.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    splits, manifest = load_dataset(dataset)
    stats = fit_statistics(splits['train'])
    thresholds = np.quantile(np.linalg.norm(flatten(splits['train'], 'pi'), axis=1), [1/3, 2/3]).tolist()
    config = dict(seed=0, epochs=epochs, batch_size=16, learning_rate=.001,
                  optimizer='Adam', loss='normalized delta MSE, valid timestep/dimension mean',
                  sequence='whole ordered episodes, hidden=None at start, right padding masked out',
                  selection='minimum validation MSE; no early stopping; no Test-based selection')
    frozen_files = [root / p for p in ['mujoco/rl/mine_uav_env.py', 'mujoco/rl/ppo_pi_env.py',
                    'mujoco/control/velocity_command_controller.py', 'mujoco/control/hover_controller.py',
                    'mujoco/control/control_allocator.py', 'mujoco/models/mine_uav_dynamics_v2.xml']]
    frozen_hashes = {str(p.relative_to(root)): file_hash(p) for p in frozen_files if p.exists()}
    report = dict(experiment='One-Step Latent Dynamics', branch='feat/latent-dynamics-one-step',
                  seed=0, versions=dict(python=platform.python_version(), torch=str(torch.__version__),
                                       numpy=np.__version__, matplotlib=matplotlib.__version__),
                  dataset=manifest, normalization=stats, normalization_sha256=json_hash(stats),
                  trainer=config, prediction_target='raw o_(t+1)-o_t; 7D; no angular wrap applied',
                  metric_definitions=dict(overall_rmse='sqrt(mean squared error across all samples and dimensions)',
                    overall_r2='1 - sum(SSE per dimension) / sum(SST per dimension), dimension-centered',
                    physical_units='error xyz: m; velocity xyz: m/s; yaw_error: rad; overall mixes units',
                    standardized='divide physical residual by Train delta standard deviation; dimensionless',
                    next_observation_error='identical to delta error because same observed o_t is added'),
                  integral_bins=dict(rule='Train transition PI norm tertiles; small<q1, q1<=medium<q2, large>=q2',
                                     thresholds_m_s2=thresholds), models={}, frozen_source_hashes=frozen_hashes)
    predictions, networks = {}, {}
    truth = flatten(splits['test'], 'delta').astype(np.float64)
    obs = flatten(splits['test'], 'obs')
    pi = flatten(splits['test'], 'pi')
    delta_std = np.asarray(stats['delta']['std'])
    dm = distance_masks(np.linalg.norm(obs[:, :3], axis=1))
    im = magnitude_masks(np.linalg.norm(pi, axis=1), thresholds)
    for kind, filename in MODEL_NAMES.items():
        path = models_dir / filename
        training = train_model(kind, splits['train'], splits['val'], stats, config, path)
        model, loaded_stats, _ = load_model(path); networks[kind] = model
        pred_list, _ = predict_episodes(model, splits['test'], loaded_stats, history=(kind == 'history'))
        pred = np.concatenate(pred_list); predictions[kind] = pred
        second, _, _ = load_model(path)
        check, _ = predict_episodes(second, splits['test'][:1], stats, history=(kind == 'history'))
        report['models'][kind] = dict(training=training, artifact=artifact(path),
                architecture=repr(model), reload_max_abs_error=float(np.max(np.abs(check[0] - pred_list[0]))),
                test=dict(physical_delta=metrics(pred, truth), standardized_delta=metrics(pred / delta_std, truth / delta_std),
                          distance_regions=grouped_metrics(pred, truth, dm),
                          integral_regions=grouped_metrics(pred, truth, im)))

    latent_dir = root / 'mujoco/rl/datasets/latent_dynamics_one_step_seed0'
    latent_dir.mkdir(parents=True, exist_ok=True)
    latent = {}; latent_artifacts = {}
    history_model = networks['history']
    encoder_hash_before = file_hash(models_dir / MODEL_NAMES['history'])
    for split in ('train', 'val', 'test'):
        _, zlist = predict_episodes(history_model, splits[split], stats, history=True)
        latent[split] = np.concatenate(zlist)
        offsets = np.r_[0, np.cumsum([len(e['obs']) for e in splits[split]])]
        path = latent_dir / f'{split}_latent.npz'
        np.savez_compressed(path, latent=latent[split], offsets=offsets)
        latent_artifacts[split] = artifact(path)
    norms = np.linalg.norm(latent['test'], axis=1)
    # Same-instance A→B→A with real episodes, both complete-sequence and streamed API.
    a = pad_episodes([splits['test'][0]], stats); b = pad_episodes([splits['test'][1]], stats)
    with torch.no_grad():
        initial = history_model(a)
        history_model(b)
        reset_exact = bool(torch.equal(initial, history_model(a)))
        state = None; streamed = []
        for t in range(len(splits['test'][0]['obs'])):
            output, state = history_model.predict_step(torch.cat([a['obs'][:, t], a['previous_action'][:, t]], -1),
                                                       a['action'][:, t], state)
            streamed.append(output)
        streaming_error = float((torch.stack(streamed, 1) - initial).abs().max())
    report['latent'] = dict(dimension=64, test_norm_mean=float(norms.mean()), test_norm_std=float(norms.std()),
                            test_norm_max=float(norms.max()), finite=bool(np.isfinite(latent['test']).all()),
                            episode_A_B_A_reset_exact=reset_exact, streamed_vs_sequence_max_abs=streaming_error,
                            local_only_artifacts=latent_artifacts,
                            supervision='delta-observation normalized MSE only; no PI auxiliary signal')
    if not reset_exact or streaming_error > 1e-5 or not report['latent']['finite']:
        raise AssertionError('latent state stability/reset failed')
    probe = fit_linear_probe(latent['train'], flatten(splits['train'], 'pi'))
    probe_prediction = apply_linear_probe(latent['test'], probe)
    probe_path = models_dir / 'latent_pi_linear_probe.pt'
    torch.save(dict(coefficient=torch.from_numpy(probe['coefficient']), training_split='train',
                    encoder_sha256=encoder_hash_before, input_dim=64, output_dim=3), probe_path)
    report['linear_probe'] = dict(training_split='train', method='frozen encoder; ordinary least squares with intercept',
                                  encoder_unchanged=file_hash(models_dir / MODEL_NAMES['history']) == encoder_hash_before,
                                  rank=probe['rank'], artifact=artifact(probe_path),
                                  test=metrics(probe_prediction, pi, names=('pi_x', 'pi_y', 'pi_z')))
    report['gap_closure'] = dict(formula='(Markov_RMSE-History_RMSE)/(Markov_RMSE-Oracle_RMSE); only positive meaningful denominator',
            physical_delta=gap_closure(*[report['models'][k]['test']['physical_delta']['overall']['rmse'] for k in MODEL_NAMES]),
            standardized_delta=gap_closure(*[report['models'][k]['test']['standardized_delta']['overall']['rmse'] for k in MODEL_NAMES]))
    report['figures'], report['selected_plot_episode'] = plot_results(report, splits['test'], predictions, pi, probe_prediction, reports)
    report['limitations'] = [
        'Single seed, nominal simulation, one-step 25Hz supervised prediction only; no multi-step/robustness evidence.',
        'Oracle reveals PI state only; attitude, angular rates and other controller memory remain unobserved. It is not complete MuJoCo physical state.',
        'History model has recurrent capacity and more parameters; improvement cannot uniquely be attributed to PI recovery.',
        'One unavailable terminal next observation per episode excluded for every model.',
        'Physical overall error aggregates mixed observation units; standardized error and per-dimension metrics must also be considered.',
        'Post-hoc linear probe measures decodability, not causal semantics or physical meaning of individual latent dimensions.']
    report['elapsed_seconds'] = time.monotonic() - started
    report['verification'] = dict(frozen_source_unchanged=all(file_hash(root / p) == h for p, h in frozen_hashes.items()),
                                   model_reload_exact=all(report['models'][k]['reload_max_abs_error'] == 0 for k in MODEL_NAMES))
    if not all(report['verification'].values()):
        raise AssertionError('experiment verification failed')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    (latent_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Report saved: {report_path}', flush=True)
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=root / 'mujoco/rl/datasets/pi_hidden_state_seed0')
    args = parser.parse_args()
    run_experiment(args.dataset, root, epochs=60)


if __name__ == '__main__':
    main()
