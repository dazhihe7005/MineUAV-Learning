"""One seed0, K=10, 60-epoch dynamics run. --verify performs read-only checks."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import episodes_from_arrays, file_hash, json_hash
from latent_dynamics_models import load_model
from latent_dynamics_multistep import start_manifest
from latent_multistep_training import train_multistep, validation_loss
from run_latent_dynamics_multistep_eval import load_inputs, compare, evaluate_model, tensor_hash

BASE_COMMIT = '6cd1d3ab01a54ab7304ca52a5c6b92d4ac53ce35'
CONFIG = dict(seed=0, epochs=60, batch_size=16, learning_rate=.001, horizon=10)


def load_training_inputs(root, frozen_report, horizon=10):
    frozen_report = Path(frozen_report).resolve()
    frozen = json.loads(frozen_report.read_text())
    stats = frozen['normalization']
    if json_hash(stats) != frozen['normalization_sha256']:
        raise ValueError('saved normalization hash mismatch')
    memory_path = Path(frozen['source_report'])
    if file_hash(memory_path) != frozen['immutable_artifacts'][str(memory_path)]:
        raise ValueError('source report hash mismatch')
    memory = json.loads(memory_path.read_text()); dataset = memory['dataset']
    initial_hash = memory.get('architecture', {}).get('initial_parameter_sha256')
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise ValueError('missing original initialization hash')
    if memory['normalization'] != stats:
        raise ValueError('source normalization mismatch')
    directory = Path(dataset['source_directory'])
    if file_hash(directory/'manifest.json') != dataset['source_manifest_sha256']:
        raise ValueError('dataset manifest hash mismatch')
    ids = dataset['target_ids']
    if any(set(ids[a]) & set(ids[b]) for a,b in [('train','val'),('train','test'),('val','test')]):
        raise ValueError('target split overlap')
    splits, manifests = {}, {}
    # No Test timestep values are opened before training/selection completes.
    for split in ('train', 'val'):
        path = directory/f'{split}.npz'
        if file_hash(path) != dataset['source_dataset_sha256'][split]:
            raise ValueError(f'dataset hash mismatch: {split}')
        with np.load(path, allow_pickle=False) as arrays:
            episodes = episodes_from_arrays(arrays)
            for ep, start, stop in zip(episodes, arrays['offsets'][:-1], arrays['offsets'][1:]):
                ep['next_obs'] = arrays['inputs'][start+1:stop, :7].copy()
        identity = [dict(target_id=e['metadata']['target_id'], source=e['metadata']['source'], rows=e['source_rows'])
                    for e in episodes]
        if identity != dataset['split_identity'][split] or sorted({e['metadata']['target_id'] for e in episodes}) != ids[split]:
            raise ValueError(f'episode/target identity mismatch: {split}')
        splits[split] = episodes
        manifests[split] = start_manifest(episodes, [horizon], dataset['source_dataset_sha256'][split])
        # Keep the common manifest format lossless; identify non-Test provenance.
        manifests[split].pop('manifest_sha256')
        manifests[split]['split'] = split
        manifests[split]['dataset_split_sha256'] = manifests[split].pop('dataset_test_sha256')
        manifests[split]['manifest_sha256'] = json_hash(manifests[split])
    return dict(splits=splits, manifests=manifests, statistics=stats, dataset=dataset,
                frozen=frozen, memory=memory, source_report=str(frozen_report))


def comparisons(old, new, horizons):
    rows = {}
    for h in horizons:
        a = old['horizons'][str(h)]['normalized_observation_rmse']
        b = new['horizons'][str(h)]['normalized_observation_rmse']
        rows[str(h)] = dict(one_step_trained=a, multistep_trained=b,
            absolute_delta=b-a, percent_change=100*(b-a)/a,
            regime='in_training_horizon' if h <= 10 else 'out_of_training_horizon')
    growth = {}
    for name, model in [('one_step',old),('multistep',new)]:
        a = model['horizons'][str(horizons[0])]['normalized_observation_rmse']
        b = model['horizons'][str(horizons[-1])]['normalized_observation_rmse']
        growth[name] = dict(absolute_growth=b-a, relative_growth=(b-a)/a,
            endpoint_ratio=b/a, average_slope_per_step=(b-a)/(horizons[-1]-horizons[0]))
    return dict(horizons=rows, error_growth=growth)


def run_experiment(root, frozen_report, config=None, make_figures=True):
    root = Path(root).resolve(); cfg = dict(CONFIG if config is None else config)
    report_path = root/'mujoco/reports/latent_dynamics_multistep_training_seed0.json'
    model_path = root/'mujoco/rl/models/history_latent_gru_multistep10.pt'
    if report_path.exists() or model_path.exists():
        raise FileExistsError('experiment artifact already exists; never silently retrain/overwrite')
    started = time.monotonic(); context = load_training_inputs(root, frozen_report, cfg['horizon'])
    output = report_path.parent; output.mkdir(parents=True, exist_ok=True)
    manifests = {}
    for split, data in context['manifests'].items():
        path = output/f'latent_multistep10_{split}_windows_seed0.json'
        path.write_text(json.dumps(data, indent=2)+'\n')
        manifests[split] = dict(path=str(path), sha256=file_hash(path), semantic_sha256=data['manifest_sha256'],
                                window_count=data['window_counts'][str(cfg['horizon'])])
    training = train_multistep(context['splits']['train'], context['splits']['val'],
                              context['statistics'], cfg, model_path)
    expected_initial = context['memory']['architecture']['initial_parameter_sha256']
    if training['initial_parameter_sha256'] != expected_initial:
        raise AssertionError('random initialization does not match original History GRU')
    # Selection is now COMPLETE. Only now load Test timestep values/frozen models.
    selected_hash = file_hash(model_path)
    frozen = context['frozen']; evaluation = load_inputs(root, frozen['source_report'])
    windows = start_manifest(evaluation['episodes'], frozen['horizons_steps'],
                             context['dataset']['source_dataset_sha256']['test'])
    if windows != frozen['windows'] or windows != json.loads(Path(frozen['start_manifest']['path']).read_text()):
        raise AssertionError('fixed Test window manifest changed')
    if file_hash(frozen['start_manifest']['path']) != frozen['start_manifest']['sha256']:
        raise ValueError('Test manifest hash mismatch')
    thresholds = frozen['pi_magnitude_bins']['thresholds_m_s2']; horizons = frozen['horizons_steps']
    models = {}
    for kind, net in evaluation['models'].items():
        print(f'Reproducing frozen reference: {kind}', flush=True)
        actual = evaluate_model(net, kind, evaluation['episodes'], context['statistics'], thresholds, horizons)
        for field, value in actual.items():
            compare(value, frozen['models'][kind][field])
        models[kind] = frozen['models'][kind]
    net, stats, metadata = load_model(model_path); net.requires_grad_(False)
    new = evaluate_model(net, 'history', evaluation['episodes'], stats, thresholds, horizons, verbose=True)
    new['artifact'] = dict(path=str(model_path), sha256=selected_hash, bytes=model_path.stat().st_size,
                           parameter_sha256=tensor_hash(net))
    new['architecture'] = repr(net); new['metadata'] = metadata; models['multistep_history'] = new
    immutable = dict(evaluation['immutable']); immutable[str(Path(frozen_report).resolve())] = file_hash(frozen_report)
    for split in ('train', 'val'):
        p = Path(context['dataset']['source_directory'])/f'{split}.npz'
        immutable[str(p)] = context['dataset']['source_dataset_sha256'][split]
    report = dict(experiment='Multi-Step Latent Dynamics Training', branch='feat/latent-multistep-training',
        base_branch='feat/latent-multistep-eval', base_commit=BASE_COMMIT, seed=cfg['seed'],
        report_path=str(report_path), source_report=str(Path(frozen_report).resolve()),
        dataset=context['dataset'], dataset_regenerated=False, train_validation_manifests=manifests,
        test_start_manifest=frozen['start_manifest'], windows=windows,
        normalization=stats, normalization_sha256=json_hash(stats), config=cfg, training=training,
        training_horizon_seconds=cfg['horizon']/25, horizons_steps=horizons,
        objective='mean over all window/horizon/dimension ((predicted physical observation - recorded observation)/saved Train observation std)^2',
        protocol='true differentiable prefix including x_t once; horizon 100% autoregressive, no detach; recorded actions; no PI input or auxiliary loss',
        pi_magnitude_bins=frozen['pi_magnitude_bins'], models=models,
        comparison=comparisons(models['history'], new, horizons), selected_window=frozen['selected_window'],
        immutable_artifacts=immutable,
        verification=dict(frozen_metrics_reproduced=True, same_initial_parameters=training['initial_parameter_sha256']==expected_initial,
            fixed_test_windows=True, normalization_reused=True, test_loaded_after_validation_selection=True,
            selected_checkpoint_unchanged=file_hash(model_path)==selected_hash,
            immutable_artifacts_unchanged=all(file_hash(p)==h for p,h in immutable.items())),
        limitations=['Single seed/nominal recorded trajectories; correlated overlapping windows.',
            'Training .4s only; 1/2s results are out-of-training-horizon, not planning or policy evaluation.',
            'PI-State reference teacher-forces recorded PI at future steps; not a full-state Oracle.',
            'Loss changes from delta-normalized one-step MSE to observation-normalized K-step MSE; differences cannot isolate horizon length from this prescribed scaling.',
            'Different models have different latent coordinate systems; L2/norm drift is descriptive, not an invariant cross-model measure.'])
    if not all(report['verification'].values()):
        raise AssertionError('verification failed')
    if make_figures:
        from latent_multistep_training_plots import render_figures
        report['figures'] = render_figures(report, evaluation, net, output)
    else:
        report['figures'] = []
    report['elapsed_seconds'] = time.monotonic()-started
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(f'Saved {report_path}', flush=True)
    return report


def verify_experiment(report_path):
    path = Path(report_path).resolve(); report = json.loads(path.read_text())
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    context = load_training_inputs(path.parents[2], report['source_report'], report['config']['horizon'])
    for p, expected in report['immutable_artifacts'].items():
        if file_hash(p) != expected:
            raise ValueError(f'immutable artifact changed: {p}')
    for split, row in report['train_validation_manifests'].items():
        if file_hash(row['path']) != row['sha256'] or json.loads(Path(row['path']).read_text()) != context['manifests'][split]:
            raise ValueError('training window manifest mismatch')
    row = report['models']['multistep_history']; artifact = row['artifact']
    if file_hash(artifact['path']) != artifact['sha256']:
        raise ValueError('selected model hash mismatch')
    net, stats, metadata = load_model(artifact['path']); cfg = report['config']
    if stats != context['statistics'] or metadata['config'] != cfg:
        raise AssertionError('saved configuration/statistics mismatch')
    recorded = [r['validation_loss'] for r in report['training']['history']]
    if int(np.argmin(recorded))+1 != metadata['best_epoch'] or metadata['best_epoch'] != report['training']['best_epoch']:
        raise AssertionError('validation best checkpoint selection mismatch')
    actual_loss = validation_loss(net, context['splits']['val'], stats, cfg['horizon'], cfg['batch_size'])
    if not np.isclose(actual_loss, min(recorded), rtol=1e-7, atol=1e-10):
        raise AssertionError('saved checkpoint validation loss mismatch')
    frozen = context['frozen']
    if (report['horizons_steps'] != frozen['horizons_steps'] or report['windows'] != frozen['windows']
            or report['test_start_manifest'] != frozen['start_manifest']):
        raise AssertionError('canonical frozen Test manifest/horizons changed')
    if file_hash(frozen['start_manifest']['path']) != frozen['start_manifest']['sha256']:
        raise ValueError('Test manifest byte hash mismatch')
    if (metadata['initial_parameter_sha256'] != context['memory']['architecture']['initial_parameter_sha256']
            or report['training']['initial_parameter_sha256'] != metadata['initial_parameter_sha256']):
        raise AssertionError('initial parameter hash mismatch')
    ev = load_inputs(path.parents[2], frozen['source_report'])
    windows = start_manifest(ev['episodes'], report['horizons_steps'], report['dataset']['source_dataset_sha256']['test'])
    if windows != report['windows'] or windows != json.loads(Path(report['test_start_manifest']['path']).read_text()):
        raise AssertionError('Test window manifest mismatch')
    net.eval().requires_grad_(False)
    all_models = dict(ev['models'], multistep_history=net)
    for kind, model in all_models.items():
        actual = evaluate_model(model, 'history' if kind=='multistep_history' else kind, ev['episodes'], stats,
                                report['pi_magnitude_bins']['thresholds_m_s2'], report['horizons_steps'])
        for field, value in actual.items():
            compare(value, report['models'][kind][field])
    compare(comparisons(report['models']['history'], report['models']['multistep_history'], report['horizons_steps']), report['comparison'])
    return dict(immutable_hashes=True, same_dataset_and_manifests=True, saved_normalization=True,
        validation_best_selection=True, checkpoint_validation_recomputed=True, frozen_and_new_metrics_reproduced=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(); root = Path(__file__).resolve().parents[2]
    if args.verify:
        print(json.dumps(verify_experiment(root/'mujoco/reports/latent_dynamics_multistep_training_seed0.json'), indent=2))
    else:
        run_experiment(root, root/'mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json')
