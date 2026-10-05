"""One offline no-memory GRU training run; all three previous models frozen.

Default: train seed0 for the prior 60-epoch budget. --verify is read-only.
No simulator, policy, reward, controller or sequence dataset is regenerated.
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

from latent_dynamics_data import file_hash, fit_statistics, flatten, json_hash, load_dataset
from latent_dynamics_models import evaluate_loss, load_model, make_model, train_model
from latent_dynamics_evaluation import (distance_masks, grouped_metrics, magnitude_masks,
                                        metrics, predict_episodes)

ORDER = ('markov', 'no_memory', 'history', 'oracle')
LABELS = dict(markov='Markov MLP', no_memory='No-memory GRU', history='Full-history GRU', oracle='PI-state MLP')
COLORS = dict(markov='#4477AA', no_memory='#CCBB44', history='#EE7733', oracle='#228833')


def parameter_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        h.update(name.encode()); h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def compare_metrics(actual, expected):
    """Check every nested metric, including per-dimension, normalized and bins."""
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise AssertionError('metric fields mismatch')
        for key in expected:
            compare_metrics(actual[key], expected[key])
    elif expected is None:
        if actual is not None:
            raise AssertionError('undefined R2 mismatch')
    elif not np.isclose(actual, expected, rtol=1e-7, atol=1e-10):
        raise AssertionError(f'metric mismatch: {actual} vs {expected}')


def evaluate(model, kind, episodes, stats, thresholds):
    rows, _ = predict_episodes(model, episodes, stats, history=kind in ('history', 'no_memory'))
    pred = np.concatenate(rows); truth = flatten(episodes, 'delta')
    obs = flatten(episodes, 'obs'); pi = flatten(episodes, 'pi')
    std = np.asarray(stats['delta']['std'])
    return dict(physical_delta=metrics(pred, truth), standardized_delta=metrics(pred/std, truth/std),
                distance_regions=grouped_metrics(pred, truth, distance_masks(np.linalg.norm(obs[:, :3], axis=1))),
                integral_regions=grouped_metrics(pred, truth, magnitude_masks(np.linalg.norm(pi, axis=1), thresholds)))


def validate_baseline(root, path):
    baseline = json.loads(Path(path).read_text())
    splits, manifest = load_dataset(baseline['dataset']['source_directory'])
    if manifest != baseline['dataset']:
        raise ValueError('baseline dataset/split manifest mismatch')
    stats = baseline['normalization']  # reuse saved stats; recompute only to audit
    if json_hash(stats) != baseline['normalization_sha256'] or fit_statistics(splits['train']) != stats:
        raise ValueError('baseline normalization mismatch')
    immutable = {str(Path(path).resolve()): file_hash(path)}
    thresholds = baseline['integral_bins']['thresholds_m_s2']
    for kind in ('markov', 'history', 'oracle'):
        row = baseline['models'][kind]; model_path = row['artifact']['path']
        if file_hash(model_path) != row['artifact']['sha256']:
            raise ValueError(f'baseline model hash mismatch: {kind}')
        immutable[model_path] = row['artifact']['sha256']
        model, saved, _ = load_model(model_path)
        if saved != stats:
            raise ValueError('saved normalization mismatch')
        compare_metrics(evaluate(model, kind, splits['test'], stats, thresholds), row['test'])
    for p, h in baseline.get('frozen_source_hashes', {}).items():
        if file_hash(root / p) != h:
            raise ValueError(f'frozen environment/controller hash mismatch: {p}')
        immutable[str(root / p)] = h
    return baseline, splits, immutable


def figures(report, directory):
    paths = []

    def save(fig, filename):
        p = directory / filename; fig.tight_layout(); fig.savefig(p, dpi=160); plt.close(fig)
        paths.append(str(p))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, metric, label in zip(axes, ('physical_delta', 'standardized_delta'),
                                 ('Physical delta RMSE (mixed units)', 'Train-standardized delta RMSE')):
        ax.bar(np.arange(4), [report['models'][k]['test'][metric]['overall']['rmse'] for k in ORDER],
               color=[COLORS[k] for k in ORDER])
        ax.set(xticks=np.arange(4), xticklabels=[LABELS[k] for k in ORDER], ylabel=label)
        ax.tick_params(axis='x', rotation=15)
    fig.suptitle('Memory necessity ablation — identical Test transitions')
    save(fig, 'memory_ablation_model_comparison.png')
    for field, filename, title in [
        ('distance_regions', 'memory_ablation_vs_target_distance.png', 'Target distance (m)'),
        ('integral_regions', 'memory_ablation_vs_integral_magnitude.png', 'True PI magnitude — same Train tertiles')]:
        groups = list(report['models']['markov']['test'][field]); x = np.arange(len(groups))
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for j, k in enumerate(ORDER):
            values = [report['models'][k]['test'][field][g]['overall']['rmse']
                      if report['models'][k]['test'][field][g]['overall'] else np.nan for g in groups]
            ax.bar(x + (j - 1.5) * .2, values, .2, label=LABELS[k], color=COLORS[k])
        ax.set(xticks=x, xticklabels=groups, xlabel=title, ylabel='Physical delta RMSE (mixed units)')
        ax.legend(ncol=2)
        save(fig, filename)
    return paths


def run_ablation(root, baseline_path):
    root, baseline_path = Path(root).resolve(), Path(baseline_path).resolve()
    model_path = root / 'mujoco/rl/models/history_latent_gru_no_memory.pt'
    report_path = root / 'mujoco/reports/latent_memory_ablation_seed0.json'
    if model_path.exists() or report_path.exists():
        raise FileExistsError('memory-ablation artifacts already exist; refusing to overwrite')
    started = time.monotonic(); torch.set_num_threads(1)
    baseline, splits, immutable = validate_baseline(root, baseline_path)
    stats, config = baseline['normalization'], baseline['trainer']
    if config['seed'] != 0:
        raise ValueError('experiment requires original seed0 configuration')
    torch.manual_seed(0); full_initial = make_model('history')
    torch.manual_seed(0); no_initial = make_model('no_memory')
    initial_hash = parameter_hash(full_initial)
    same_initial = initial_hash == parameter_hash(no_initial)
    if not same_initial:
        raise AssertionError('GRU/head initial parameters differ')
    training = train_model('no_memory', splits['train'], splits['val'], stats, config, model_path)
    model, loaded_stats, _ = load_model(model_path)
    if loaded_stats != stats:
        raise AssertionError('reused statistics changed')
    test = evaluate(model, 'no_memory', splits['test'], stats, baseline['integral_bins']['thresholds_m_s2'])
    report = dict(experiment='No-Memory GRU Ablation', branch='feat/latent-memory-ablation',
                  base_checkpoint_commit='ad2a7a21f203fea3bfe4a010d58b58e2f69cee4c',
                  baseline_report=dict(path=str(baseline_path), sha256=immutable[str(baseline_path)]),
                  dataset=baseline['dataset'], dataset_regenerated=False,
                  normalization=stats, normalization_sha256=baseline['normalization_sha256'],
                  trainer=config, prediction_target='raw o_(t+1)-o_t, same 7D target',
                  intervention='each [o_t,previous executed normalized action] is an independent length-1 GRU lane with zero hidden; same head/current action',
                  architecture=dict(input=11, gru_hidden=64, layers=1, head=[68,64,64,7], activation='Tanh',
                                    parameter_count=23815, parameter_schema_identical=True,
                                    initial_parameter_sha256=initial_hash),
                  integral_bins=baseline['integral_bins'], immutable_artifacts=immutable,
                  models={k: baseline['models'][k] for k in ('markov', 'history', 'oracle')})
    report['models']['no_memory'] = dict(training=training, architecture=repr(model), test=test,
                    artifact=dict(path=str(model_path), sha256=file_hash(model_path), bytes=model_path.stat().st_size))
    improvements = {}
    for metric in ('physical_delta', 'standardized_delta'):
        values = {k: report['models'][k]['test'][metric]['overall']['rmse'] for k in ORDER}
        improvements[metric] = dict(
            full_history_relative_reduction_vs_no_memory=(values['no_memory'] - values['history']) / values['no_memory'],
            no_memory_relative_reduction_vs_markov=(values['markov'] - values['no_memory']) / values['markov'],
            rmse=values)
    report['comparison'] = improvements
    report['verification'] = dict(baseline_artifacts_unchanged=all(file_hash(p) == h for p,h in immutable.items()),
                                   same_initial_parameters=same_initial, reused_normalization=True,
                                   baseline_metrics_reproduced=True)
    if not all(report['verification'].values()):
        raise AssertionError('frozen-baseline verification failed')
    report['limitations'] = [
        'Single seed and nominal one-step 25Hz prediction only; no multi-step or physical generalization.',
        'No recurrent memory still retains previous executed action as a permitted current feature.',
        'Full-history/no-memory parameter tensors match, but zero hidden naturally disables temporal paths, including gradients through recurrent hidden-weight matrices.',
        'Markov vs no-memory additionally differs in input features/capacity; pure memory inference uses full-history vs no-memory.',
        'PI-state augmented MLP is not a full MuJoCo-state prediction lower bound.',
        'Physical overall metrics combine m,m/s,rad; consult normalized and per-dimension metrics.']
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report['figures'] = figures(report, report_path.parent)
    report['elapsed_seconds'] = time.monotonic() - started
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(f'Saved {report_path}', flush=True)
    return report


def verify_ablation(report_path):
    report_path = Path(report_path).resolve(); report = json.loads(report_path.read_text())
    torch.set_num_threads(1)
    for p,h in report['immutable_artifacts'].items():
        if file_hash(p) != h:
            raise ValueError(f'immutable hash mismatch: {p}')
    splits, manifest = load_dataset(report['dataset']['source_directory'])
    if manifest != report['dataset']:
        raise AssertionError('dataset manifest mismatch')
    if fit_statistics(splits['train']) != report['normalization']:
        raise AssertionError('Train normalization changed')
    for kind in ORDER:
        row = report['models'][kind]
        if file_hash(row['artifact']['path']) != row['artifact']['sha256']:
            raise ValueError(f'model hash mismatch: {kind}')
        net, stats, metadata = load_model(row['artifact']['path'])
        if stats != report['normalization']:
            raise AssertionError('model statistics mismatch')
        compare_metrics(evaluate(net, kind, splits['test'], stats, report['integral_bins']['thresholds_m_s2']), row['test'])
        if kind == 'no_memory':
            training = row['training']
            validation = [r['validation_loss'] for r in training['history']]
            best_epoch = int(np.argmin(validation)) + 1
            if best_epoch != training['best_epoch'] or metadata['best_epoch'] != best_epoch:
                raise AssertionError('checkpoint was not validation-best')
            if metadata['config'] != report['trainer']:
                raise AssertionError('checkpoint training config mismatch')
            validation_loss = evaluate_loss(net, splits['val'], stats, report['trainer']['batch_size'])
            for recorded in (min(validation), training['best_validation_loss'], metadata['validation_loss']):
                if not np.isclose(validation_loss, recorded, rtol=1e-7, atol=1e-10):
                    raise AssertionError('checkpoint validation loss mismatch')
    return dict(immutable_hashes=True, identical_dataset=True, train_statistics=True,
                all_four_models_all_metrics=True, best_validation_selection=True,
                checkpoint_validation_recomputed=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_ablation(root / 'mujoco/reports/latent_memory_ablation_seed0.json'), indent=2))
    else:
        run_ablation(root, root / 'mujoco/reports/latent_dynamics_one_step_seed0.json')
