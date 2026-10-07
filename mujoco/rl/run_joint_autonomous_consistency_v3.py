"""Single controlled v3 run. --verify recomputes saved-best metrics, never trains."""
import argparse
import json
import time
from pathlib import Path

from latent_dynamics_data import file_hash, json_hash
from joint_latent_world_model import component_hashes
from run_latent_memory_ablation import parameter_hash
from run_joint_latent_composition_audit import compare
from run_joint_consistency_v2 import load_context as load_v1_context, test_inputs, verify_history_summaries
from joint_autonomous_consistency_training import train_autonomous, load_autonomous, validation_losses, selection_epoch, LAMBDA
from joint_autonomous_consistency_evaluation import evaluate, comparison, baseline_records

BASE = 'ad3e13c95bbdbebdc97738c99747c81b89901169'
BRANCH = 'feat/joint-autonomous-consistency-v3'
REPORT = 'joint_latent_autonomous_consistency_v3_seed0.json'
MODEL = 'joint_latent_world_model_v3_autonomous_consistency.pt'
MANIFOLD = 'joint_latent_autonomous_consistency_v3_train_statistics_seed0.json'


def load_context(root, v1_report, v1_audit, v2_report):
    context = load_v1_context(root, v1_report, v1_audit)
    p = Path(v2_report).resolve(); v2 = json.loads(p.read_text()); b = context['baseline']
    for key in ('config', 'dataset', 'initialization', 'normalization', 'normalization_sha256', 'window_manifests', 'test_start_manifest'):
        compare(b[key], v2[key])
    compare(b['training']['parameter_hashes_before'], v2['training']['parameter_hashes_before'])
    compare(context['initial_scale'], v2['initial_latent_scale'])
    scale = v2['initial_scale_file']
    if file_hash(scale['path']) != scale['sha256'] or json_hash(context['initial_scale']) != scale['semantic_sha256']:
        raise ValueError('v2 fixed initial Train scale hash mismatch')
    compare(json.loads(Path(scale['path']).read_text()), context['initial_scale'])
    context['immutable'].update({str(p): file_hash(p), v2['model']['path']: v2['model']['sha256'], scale['path']: scale['sha256']})
    for path, sha in context['immutable'].items():
        if file_hash(path) != sha:
            raise ValueError(f'v1/v2 source changed: {path}')
    context.update(v2=v2, v2_path=str(p))
    # Mandatory baseline sections: missing or empty comparisons must fail.
    records = baseline_records(context)
    for name, row in records.items():
        for field, metrics in row['audit'].items():
            if not isinstance(metrics, dict) or not metrics:
                raise ValueError(f'empty baseline diagnostic: {name}.{field}')
        local = row['audit']['local_one_step']
        if not local.get('latent') or not local.get('observation'):
            raise ValueError(f'incomplete baseline local diagnostics: {name}')
        for h in b['horizons_steps']:
            endpoint = row['audit']['autonomous'].get(str(h), {})
            if not endpoint.get('latent') or not endpoint.get('observation'):
                raise ValueError(f'incomplete baseline horizon diagnostics: {name}.H{h}')
    return context


def run_experiment(root, v1_report, v1_audit, v2_report, make_figures=True):
    root = Path(root).resolve(); output = root/'mujoco/reports'
    path = output/REPORT; modelpath = root/'mujoco/rl/models'/MODEL; manifoldpath = output/MANIFOLD
    if any(p.exists() for p in (path, modelpath, manifoldpath)):
        raise FileExistsError('v3 artifacts exist; no overwrite/retraining')
    started = time.monotonic(); c = load_context(root, v1_report, v1_audit, v2_report)
    b = c['baseline']; cfg = dict(b['config']); scale = c['initial_scale']
    output.mkdir(parents=True, exist_ok=True)
    training = train_autonomous(c['model'], c['splits']['train'], c['splits']['val'], c['statistics'], scale['std'], cfg, modelpath)
    compare(training['parameter_hashes_before'], b['training']['parameter_hashes_before'])
    compare(training['initial_validation']['observation'], c['initial_observation_validation'])
    # Test values first opened after final observation-validation checkpoint selection.
    test, windows = test_inputs(c); actual, plot = evaluate(modelpath, c, test)
    manifoldpath.write_text(json.dumps(actual['manifold'], indent=2, allow_nan=False)+'\n')
    model, stats, _ = load_autonomous(modelpath); baselines = baseline_records(c)
    r = dict(experiment='Deterministic Latent World Model v3: Autonomous Multi-Step Latent Consistency',
        branch=BRANCH, base_branch='feat/joint-latent-consistency-v2', base_commit=BASE, seed=0, report_path=str(path),
        baseline_report=c['baseline_path'], baseline_audit_report=c['baseline_audit_path'], local_v2_report=c['v2_path'],
        config=cfg, lambda_autonomous_consistency=LAMBDA, local_consistency_enabled=False,
        dataset=c['dataset'], dataset_regenerated=False, initialization=c['initialization'], initial_latent_scale=scale,
        initial_scale_file=c['v2']['initial_scale_file'], initial_observation_validation=c['initial_observation_validation'],
        training=training, model=dict(path=str(modelpath), sha256=file_hash(modelpath), parameter_sha256=parameter_hash(model), bytes=modelpath.stat().st_size),
        normalization=stats, normalization_sha256=json_hash(stats), window_manifests=b['window_manifests'],
        test_start_manifest=b['test_start_manifest'], windows=windows, horizons_steps=b['horizons_steps'], selected_window=b['selected_window'],
        train_manifold=actual['manifold'], train_statistics_file=dict(path=str(manifoldpath), sha256=file_hash(manifoldpath), semantic_sha256=json_hash(actual['manifold'])),
        baselines=baselines, audit=actual['audit'], relative_local_residual=actual['relative'], latent_variance=actual['latent_variance'],
        comparison=comparison(baselines, actual['audit'], actual['relative'], b['horizons_steps']), immutable_artifacts=c['immutable'],
        verification=dict(actual['verification'], initial_hashes_match_v1_v2=True, original_windows_normalization_reused=True,
            test_loaded_after_observation_validation_selection=True,
            all_three_components_updated=all(training['parameter_hashes_before'][k] != training['parameter_hashes_best'][k] for k in ('encoder', 'transition', 'decoder')),
            baseline_and_sources_unchanged=all(file_hash(p) == h for p, h in c['immutable'].items())),
        objective='Lobs=mean k0..10 observation-std MSE; Lauto=mean k1..10 fixed initial Train latent-std MSE(z_pred_autonomous-stopgrad(E(real future history))); Ltotal=Lobs+0.1Lauto; no local term',
        loss_field_mapping='history.consistency denotes ONLY autonomous multi-step consistency; weighted_consistency=0.1*consistency',
        selection_metric='Validation Lobs only; not total/auto-consistency/Test',
        stop_gradient='All future real-history reference targets detached/no-grad; source prefix E and all composed T states keep gradients; D absent from direct consistency graph',
        protocol='Original v1 autonomous prediction path reused unmodified; real-history reference ONLY loss target, absent at inference. Complete K10 BPTT, no corrections or teacher input within horizon.',
        family_stopping_rule='No further consistency-family experiments in this task regardless of result; v1 remains baseline unless evidence supports a change.',
        limitations=['Single seed, nominal simulation, recorded actions, overlapping windows; fixedlambda.1, K10, moving online Encoder targets.',
                     'Own-model native latent geometry is not coordinate-invariant across models; relative residual epsilon raw-coordinate dependent.',
                     'Train covariance/PCA are distribution proxies; periodic correction diagnostic only, no causal module-root proof.',
                     'Historical gradients are recorded only; verifier recomputes saved-best checkpoint metrics without replaying optimization.',
                     'No reward/policy/planning/RSSM/Dreamer experiment.'])
    if not all(r['verification'].values()):
        raise AssertionError('v3 verification failed')
    if make_figures:
        from joint_autonomous_consistency_plots import render_figures
        r['figures'] = render_figures(r, plot, c, test, output)
    else:
        r['figures'] = []
    r['elapsed_seconds'] = time.monotonic()-started
    path.write_text(json.dumps(r, indent=2, allow_nan=False)+'\n'); print(f'Saved {path}', flush=True)
    return r


def verify_history(training, config):
    """Arithmetic plus required E/T/D diagnostics, not historical gradient replay."""
    verify_history_summaries(training, config)
    components = {'encoder', 'transition', 'decoder'}
    required = {'weighted_consistency_norm', 'observation_norm', 'mean_consistency_to_observation_ratio',
                'max_consistency_to_observation_ratio', 'mean_consistency_to_total_ratio',
                'fraction_batches_consistency_norm_exceeds_observation'}
    for row in training['history']:
        if set(row['gradient_norms']) != components or set(row['objective_gradient_diagnostics']) != components:
            raise AssertionError('required E/T/D gradient diagnostics missing')
        for module in components:
            norms = row['gradient_norms'][module]; g = row['objective_gradient_diagnostics'][module]
            if set(norms) != {'mean', 'max'} or set(g) != required:
                raise AssertionError('required gradient diagnostic fields missing')
            if not 0 <= g['fraction_batches_consistency_norm_exceeds_observation'] <= 1:
                raise AssertionError('gradient fraction outside [0,1]')
            if g['mean_consistency_to_observation_ratio'] > g['max_consistency_to_observation_ratio']:
                raise AssertionError('gradient ratio mean exceeds max')
            if module == 'decoder' and any(g[k] != 0 for k in required - {'observation_norm'}):
                raise AssertionError('Decoder cannot receive direct autonomous-consistency gradient')


def verify_experiment(report_path):
    p = Path(report_path).resolve(); r = json.loads(p.read_text()); root = p.parents[2]
    c = load_context(root, r['baseline_report'], r['baseline_audit_report'], r['local_v2_report']); b = c['baseline']
    for key, expected in [('config', b['config']), ('lambda_autonomous_consistency', .1), ('local_consistency_enabled', False),
        ('initialization', c['initialization']), ('initial_latent_scale', c['initial_scale']), ('initial_scale_file', c['v2']['initial_scale_file']),
        ('normalization', c['statistics']), ('dataset', c['dataset']), ('window_manifests', b['window_manifests']),
        ('test_start_manifest', b['test_start_manifest']), ('horizons_steps', b['horizons_steps']), ('selected_window', b['selected_window']),
        ('immutable_artifacts', c['immutable']), ('baselines', baseline_records(c))]:
        compare(expected, r[key])
    compare(c['initial_observation_validation'], r['initial_observation_validation'])
    compare(c['initial_observation_validation'], r['training']['initial_validation']['observation'])
    if json_hash(c['statistics']) != r['normalization_sha256']:
        raise ValueError('normalization hash changed')
    row = r['model']; model, stats, meta = load_autonomous(row['path']); cfg = r['config']; history = r['training']['history']
    verify_history(r['training'], cfg)
    if file_hash(row['path']) != row['sha256'] or parameter_hash(model) != row['parameter_sha256'] or stats != c['statistics'] or meta['config'] != cfg:
        raise ValueError('model/config/statistics changed')
    if (meta['initial_latent_std'] != c['initial_scale']['std'] or meta['lambda_autonomous_consistency'] != .1 or
        meta['local_consistency_enabled'] or meta['parameter_hashes_before'] != b['training']['parameter_hashes_before'] or
        r['training']['parameter_hashes_before'] != b['training']['parameter_hashes_before'] or
        r['training']['parameter_hashes_best'] != component_hashes(model)):
        raise AssertionError('initial/best parameter or objective contract changed')
    epoch = selection_epoch([a['validation'] for a in history])
    if meta['best_epoch'] != epoch or r['training']['best_epoch'] != epoch:
        raise AssertionError('observation-only validation selection mismatch')
    for split, field in [('train', 'best_checkpoint_train'), ('val', 'best_validation')]:
        compare(validation_losses(model, c['splits'][split], stats, c['initial_scale']['std'], 10, cfg['batch_size']), r['training'][field])
    compare(r['training']['best_validation'], history[epoch - 1]['validation'])
    compare(r['training']['best_validation']['observation'], meta['validation_observation_loss'])
    compare(r['training']['best_validation']['observation'], min(a['validation']['observation'] for a in history))
    if any(a['windows_used'] != b['window_manifests']['train']['window_count'] for a in history):
        raise AssertionError('training window counts changed')
    test, windows = test_inputs(c); ev, _ = evaluate(row['path'], c, test)
    for field, key in [('audit', 'audit'), ('manifold', 'train_manifold'), ('relative', 'relative_local_residual'), ('latent_variance', 'latent_variance')]:
        compare(ev[field], r[key])
    f = r['train_statistics_file']
    if file_hash(f['path']) != f['sha256'] or json_hash(ev['manifold']) != f['semantic_sha256']:
        raise ValueError('Train geometry hash changed')
    compare(json.loads(Path(f['path']).read_text()), ev['manifold']); compare(windows, r['windows'])
    compare(comparison(baseline_records(c), ev['audit'], ev['relative'], b['horizons_steps']), r['comparison'])
    if not all(ev['verification'].values()) or not all(file_hash(p) == h for p, h in c['immutable'].items()):
        raise AssertionError('inference/source contract failed')
    return dict(initial_hashes_match_v1_v2=True, fixed_initial_Train_scale_reused=True, no_local_consistency_term=True,
        observation_only_selection_reproduced=True, selected_checkpoint_and_evaluation_metrics_reproduced=True,
        recorded_history_internally_consistent=True, no_future_inference_reference=True,
        dataset_manifests_normalization_reused=True, baseline_sources_unchanged=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify', action='store_true'); args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.verify:
        print(json.dumps(verify_experiment(root/'mujoco/reports'/REPORT), indent=2))
    else:
        run_experiment(root, root/'mujoco/reports/joint_latent_world_model_v1_seed0.json',
                       root/'mujoco/reports/joint_latent_composition_audit_seed0.json', root/'mujoco/reports/joint_latent_consistency_v2_seed0.json')
