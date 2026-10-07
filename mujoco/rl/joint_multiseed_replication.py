"""Fixed upstream initialization; only joint-training stochasticity is varied."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from latent_dynamics_data import file_hash, json_hash
from joint_latent_world_model import initialize_joint, component_hashes, load_joint
from joint_latent_training import train_joint, validation_loss
from joint_autonomous_consistency_training import train_autonomous, load_autonomous, validation_losses
from joint_multiseed_evaluation import evaluate_minimal
from run_joint_consistency_v2 import load_context as initial_context, test_inputs
from run_joint_autonomous_consistency_v3 import verify_history as verify_autonomous_history
from run_joint_latent_composition_audit import compare
from run_latent_memory_ablation import parameter_hash

BASE = 'fa357af3c1a9fc0f24e635e1b073b180bdca4b64'
BRANCH = 'feat/joint-v1-v3-multiseed'
SEEDS = (0, 1, 2, 3, 4)
CONDITIONS = ('v1', 'v3')
REPORT = 'joint_v1_v3_multiseed_replication.json'
PARTS = 'joint_v1_v3_multiseed_parts'
OBJECTIVES = dict(v1='Ltotal=Lobs; uniform k0..10 observation-std normalized MSE',
                  v3='Ltotal=Lobs+0.1Lauto; uniform k1..10 composed latent vs detached online Encoder reference / fixed initial Train std; NO local term')


def load_context(root, v1_report=None, v1_audit=None, v3_report=None):
    root = Path(root).resolve(); directory = root/'mujoco/reports'
    v1_report = v1_report or directory/'joint_latent_world_model_v1_seed0.json'
    v1_audit = v1_audit or directory/'joint_latent_composition_audit_seed0.json'
    v3_report = Path(v3_report or directory/'joint_latent_autonomous_consistency_v3_seed0.json').resolve()
    c = initial_context(root, v1_report, v1_audit)
    b = c['baseline']; v3 = json.loads(v3_report.read_text())
    for key in ('config', 'dataset', 'initialization', 'normalization', 'normalization_sha256',
                'window_manifests', 'test_start_manifest', 'windows', 'horizons_steps'):
        compare(b[key], v3[key])
    compare(b['training']['parameter_hashes_before'], v3['training']['parameter_hashes_before'])
    compare(c['initial_scale'], v3['initial_latent_scale'])
    if v3['lambda_autonomous_consistency'] != .1 or v3['local_consistency_enabled']:
        raise ValueError('seed0 v3 objective mismatch')
    scale_file = v3['initial_scale_file']
    if file_hash(scale_file['path']) != scale_file['sha256'] or json_hash(c['initial_scale']) != scale_file['semantic_sha256']:
        raise ValueError('fixed initial Train latent scale changed')
    c['immutable'].update({str(v3_report): file_hash(v3_report), v3['model']['path']: v3['model']['sha256'],
                           scale_file['path']: scale_file['sha256']})
    for path, sha in c['immutable'].items():
        if file_hash(path) != sha:
            raise ValueError(f'immutable source changed: {path}')
    c.update(v3=v3, v3_path=str(v3_report), root=root)
    return c


def expected_orders(seed, episode_count, epochs):
    rng = np.random.default_rng(seed)
    return [hashlib.sha256(rng.permutation(episode_count).astype('<i8').tobytes()).hexdigest()
            for _ in range(epochs)]


def fresh_model(context):
    paths = context['baseline']['initialization']['checkpoints']
    model, stats, initialization = initialize_joint(*[paths[k]['path'] for k in ('encoder', 'transition', 'decoder')])
    compare(initialization, context['initialization'])
    compare(component_hashes(model), context['baseline']['training']['parameter_hashes_before'])
    compare(stats, context['statistics'])
    return model


def load_condition(path, condition):
    if condition not in CONDITIONS:
        raise ValueError('only v1 and v3')
    return (load_joint if condition == 'v1' else load_autonomous)(path)


def branch_paths(root, condition, seed):
    root = Path(root)
    name = f'joint_v1_seed{seed}.pt' if condition == 'v1' else f'joint_v3_autonomous_consistency_seed{seed}.pt'
    return root/'mujoco/rl/models'/name, root/'mujoco/reports'/PARTS/f'{condition}_seed{seed}.json'


def nonfinite_paths(value, prefix='record'):
    if isinstance(value, dict):
        return [p for key, item in value.items() for p in nonfinite_paths(item, f'{prefix}.{key}')]
    if isinstance(value, (list, tuple)):
        return [p for i, item in enumerate(value) for p in nonfinite_paths(item, f'{prefix}[{i}]')]
    return [prefix] if isinstance(value, (float, np.floating)) and not np.isfinite(value) else []


def sanitize_nonfinite(value):
    if isinstance(value, dict):
        return {k: sanitize_nonfinite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_nonfinite(v) for v in value]
    return None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value


def verify_branch_contract(context, row):
    """Failed seeds retain shared invariants; unavailable trajectories are not verified."""
    if row['condition'] not in CONDITIONS or type(row['seed']) is not int or row['seed'] not in (1, 2, 3, 4):
        raise ValueError('invalid new branch identity')
    if row['status'] not in ('completed', 'failed') or row['reused'] is not False:
        raise ValueError('invalid new branch status/provenance')
    compare(dict(context['baseline']['config'], seed=row['seed']), row['config'])
    compare(context['baseline']['training']['parameter_hashes_before'], row['initial_parameter_hashes'])
    compare(json_hash(context['statistics']), row['normalization_sha256'])
    if row['status'] == 'failed' and (not row.get('failure', {}).get('type') or not row['failure'].get('message')):
        raise ValueError('failed seed must retain explicit reason')
    if 'episode_order_hashes' in row:
        compare(expected_orders(row['seed'], len(context['splits']['train']), row['config']['epochs']), row['episode_order_hashes'])
    if nonfinite_paths(row):
        raise ValueError('nonfinite record must be explicit failed/null diagnostic')


def run_branch(root, condition, seed, context=None):
    if condition not in CONDITIONS or type(seed) is not int or seed not in (1, 2, 3, 4):
        raise ValueError('new training restricted to v1/v3 seeds1..4; seed0 is reused')
    model_path, part_path = branch_paths(root, condition, seed)
    if model_path.exists() or part_path.exists():
        raise FileExistsError('branch artifacts exist; no overwrite/retraining')
    c = load_context(root) if context is None else context
    config = dict(c['baseline']['config'], seed=seed)
    model = fresh_model(c)
    row = dict(condition=condition, seed=seed, config=config, reused=False, status='started',
               initial_parameter_hashes=component_hashes(model), normalization_sha256=json_hash(c['statistics']),
               part_path=str(part_path))
    started = time.monotonic()
    try:
        if condition == 'v1':
            training = train_joint(model, c['splits']['train'], c['splits']['val'], c['statistics'], config, model_path)
        else:
            training = train_autonomous(model, c['splits']['train'], c['splits']['val'], c['statistics'],
                                        c['initial_scale']['std'], config, model_path)
        orders = [a['episode_order_sha256'] for a in training['history']]
        if orders != expected_orders(seed, len(c['splits']['train']), config['epochs']):
            raise AssertionError('observed shuffle differs from prescribed paired rule')
        row.update(training=training, episode_order_hashes=orders, order_hash_provenance='recorded during every epoch')
        # Test timestep values first read only AFTER all training and Val selection.
        test, windows = test_inputs(c)
        selected, stats, _ = load_condition(model_path, condition)
        row.update(model=dict(path=str(model_path), sha256=file_hash(model_path), bytes=model_path.stat().st_size,
                              parameter_sha256=parameter_hash(selected)),
                   evaluation=evaluate_minimal(selected, c['splits']['train'], test, stats, c['baseline']),
                   test_windows=windows['window_counts'], status='completed')
        invalid = nonfinite_paths(row)
        if invalid:
            raise FloatingPointError('nonfinite returned diagnostics: '+', '.join(invalid))
        if any(file_hash(p) != h for p, h in c['immutable'].items()):
            raise ValueError('baseline/source mutation')
    except Exception as error:
        # A real numerical/training failure is part of robustness, never swap seeds.
        row.update(status='failed', failure=dict(type=type(error).__name__, message=str(error)))
        invalid = nonfinite_paths(row)
        if invalid:
            row['nonfinite_diagnostic_paths'] = invalid
            row = sanitize_nonfinite(row)
    row['elapsed_seconds'] = time.monotonic() - started
    part_path.parent.mkdir(parents=True, exist_ok=True)
    part_path.write_text(json.dumps(row, indent=2, allow_nan=False)+'\n')
    print(f'{condition} seed{seed}: {row["status"]}; saved {part_path}', flush=True)
    return row


def verify_branch(context, row):
    if row['status'] != 'completed' or row['condition'] not in CONDITIONS or row['seed'] not in (1, 2, 3, 4):
        raise ValueError('verification requires a completed new branch')
    verify_branch_contract(context, row)
    c = context; cfg = dict(c['baseline']['config'], seed=row['seed']); training = row['training']
    if row['condition'] == 'v3':
        verify_autonomous_history(training, cfg)
        compare(c['initial_observation_validation'], training['initial_validation']['observation'])
    else:
        def nonnegative(value):
            if isinstance(value, dict): return all(nonnegative(v) for v in value.values())
            return bool(np.isfinite(value) and value >= 0)
        for item in training['history']:
            for field in ('train_loss', 'validation_loss', 'gradient_norms', 'normalized_prediction_abs_max', 'latent_norm_max'):
                if not nonnegative(item[field]): raise AssertionError('invalid recorded loss/gradient/stability summary')
            if set(item['gradient_norms']) != {'encoder', 'transition', 'decoder'} or any(
                    v['mean'] > v['max'] for v in item['gradient_norms'].values()):
                raise AssertionError('invalid E/T/D gradient summaries')
        compare(training['history'][-1]['train_loss'], training['final_train_loss'])
        compare(training['history'][-1]['validation_loss'], training['final_validation_loss'])
        compare(c['initial_observation_validation'], training['initial_validation_loss'])
    compare(cfg, row['config']); compare(c['baseline']['training']['parameter_hashes_before'], row['initial_parameter_hashes'])
    compare(row['initial_parameter_hashes'], training['parameter_hashes_before'])
    if row['normalization_sha256'] != json_hash(c['statistics']):
        raise ValueError('normalization hash changed')
    record = row['model']; model, stats, metadata = load_condition(record['path'], row['condition'])
    if file_hash(record['path']) != record['sha256'] or parameter_hash(model) != record['parameter_sha256']:
        raise ValueError('checkpoint hash changed')
    compare(stats, c['statistics']); compare(metadata['config'], cfg)
    compare(component_hashes(model), training['parameter_hashes_best'])
    compare(metadata['parameter_hashes_before'], row['initial_parameter_hashes'])
    history = training['history']
    if len(history) != cfg['epochs'] or training['final_epoch'] != cfg['epochs']:
        raise AssertionError('epoch budget changed')
    expected = expected_orders(row['seed'], len(c['splits']['train']), cfg['epochs'])
    if row['episode_order_hashes'] != expected or [a['episode_order_sha256'] for a in history] != expected:
        raise AssertionError('paired epoch orders changed')
    if [a['epoch'] for a in history] != list(range(1, cfg['epochs']+1)) or any(
            a['windows_used'] != c['baseline']['window_manifests']['train']['window_count'] for a in history):
        raise AssertionError('epoch/window identities changed')
    losses = [a['validation_loss'] if row['condition'] == 'v1' else a['validation']['observation'] for a in history]
    best = int(np.argmin(losses))+1
    if best != training['best_epoch'] or best != metadata['best_epoch']:
        raise AssertionError('checkpoint not selected solely by Validation Lobs')
    if row['condition'] == 'v1':
        compare(validation_loss(model, c['splits']['val'], stats, 10, cfg['batch_size']), training['best_validation_loss'])
        compare(min(losses), metadata['validation_loss'])
        compare(validation_loss(model, c['splits']['train'], stats, 10, cfg['batch_size']), training['best_checkpoint_train_loss'])
    else:
        if (metadata['lambda_autonomous_consistency'] != .1 or metadata['local_consistency_enabled'] or
                metadata['initial_latent_std'] != c['initial_scale']['std']):
            raise ValueError('autonomous objective changed')
        for split, field in [('train', 'best_checkpoint_train'), ('val', 'best_validation')]:
            compare(validation_losses(model, c['splits'][split], stats, c['initial_scale']['std'], 10, cfg['batch_size']), training[field])
        compare(training['best_validation'], history[best-1]['validation'])
        compare(min(losses), metadata['validation_observation_loss'])
    test, windows = test_inputs(c)
    compare(windows['window_counts'], row['test_windows'])
    compare(evaluate_minimal(model, c['splits']['train'], test, stats, c['baseline']), row['evaluation'])
    if any(file_hash(p) != h for p, h in c['immutable'].items()):
        raise ValueError('source mutation')
    return dict(initialization_normalization_identical=True, paired_order_reconstructed=True,
                validation_observation_only_selection=True, metrics_reload_reproduced=True,
                frozen_evaluation_no_future_input=True, baseline_sources_unchanged=True)


def descriptive(values):
    a = np.asarray([v for v in values if v is not None], np.float64)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite metrics require explicit failed status, not silent filtering')
    if not len(a):
        return dict(count=0, mean=None, std=None, median=None, min=None, max=None)
    return dict(count=len(a), mean=float(a.mean()), std=float(a.std(ddof=1)) if len(a)>1 else None,
                median=float(np.median(a)), min=float(a.min()), max=float(a.max()))


def paired_statistics(v1, v3):
    if len(v1) != 5 or len(v3) != 5:
        raise ValueError('retain all five requested seed identities, including failures')
    a, b = descriptive(v1), descriptive(v3)
    deltas = [None if x is None or y is None else x-y for x, y in zip(v1, v3)]
    percent = [None if x is None or y is None or x<=0 else 100*(x-y)/x for x, y in zip(v1, v3)]
    return dict(v1=a, v3=b, v1_values=list(v1), v3_values=list(v3), delta=descriptive(deltas), deltas=deltas, improvement_percent=percent,
                improvement_percent_summary=descriptive(percent), requested_pair_count=5,
                missing_pair_count=sum(d is None for d in deltas),
                v3_win_count=sum(d is not None and d>0 for d in deltas),
                v3_loss_count=sum(d is not None and d<0 for d in deltas),
                exact_tie_count=sum(d == 0 for d in deltas if d is not None), std_ddof=1)


def summarize_branches(branches):
    index = {(r['condition'], r['seed']): r for r in branches}
    required = {(condition, seed) for condition in CONDITIONS for seed in SEEDS}
    if len(index) != len(branches) or set(index) != required:
        raise ValueError('exactly ten v1/v3 paired seed0..4 identities required; retain failures')
    def pair(getter):
        arrays = []
        for condition in CONDITIONS:
            arrays.append([None if index[condition, seed]['status'] == 'failed' else
                           getter(index[condition, seed]['evaluation']) for seed in SEEDS])
        return paired_statistics(*arrays)
    horizons = {str(h): pair(lambda e, key=str(h): e['horizons'][key]['observation']['normalized_observation_rmse'])
                for h in (1, 5, 10, 25, 50)}
    anomalies = []
    for row in branches:
        if row['status'] == 'failed':
            anomalies.append(dict(condition=row['condition'], seed=row['seed'], failure=row['failure']))
        elif row['evaluation']['potential_latent_collapse']:
            anomalies.append(dict(condition=row['condition'], seed=row['seed'], potential_latent_collapse=True))
    correction = {}
    for interval in (1, 10):
        correction[f'never_minus_C{interval}'] = pair(lambda e, key=str(interval):
            e['periodic_correction']['never']['observation']['normalized_observation_rmse'] -
            e['periodic_correction'][key]['observation']['normalized_observation_rmse'])
    return dict(horizons=horizons,
                current_reconstruction=pair(lambda e: e['current_reconstruction']['normalized_observation_rmse']),
                h50_horizontal_velocity=pair(lambda e: e['horizons']['50']['observation']['horizontal_velocity_rmse']),
                h50_yaw=pair(lambda e: e['horizons']['50']['observation']['physical']['yaw_error']['rmse']),
                correction_gaps=correction, anomalies=anomalies,
                statistics_rule='paired seed identities0..4; sample std ddof1; positive v1-v3 means v3 better; no p-values',
                failure_rule='retain every requested seed; failed metrics null and missing pairs explicit, never replace or silently discard')


def reuse_seed0(context, condition):
    if condition not in CONDITIONS:
        raise ValueError('only existing seed0 v1/v3')
    c = context; source = c['baseline'] if condition == 'v1' else c['v3']
    model, stats, _ = load_condition(source['model']['path'], condition)
    test, windows = test_inputs(c)
    measured = evaluate_minimal(model, c['splits']['train'], test, stats, c['baseline'])
    audit = c['baseline_audit'] if condition == 'v1' else source['audit']
    compare(measured['current_reconstruction'], audit['encoder_reconstruction'])
    for h in c['baseline']['horizons_steps']:
        key = str(h); endpoint = measured['horizons'][key]
        compare(endpoint['observation'], audit['autonomous'][key]['observation'])
        compare(endpoint['latent_consistency']['train_std_normalized_distance'], audit['autonomous'][key]['latent']['normalized_rmse'])
        compare(endpoint['latent_consistency']['mean_cosine'], audit['autonomous'][key]['latent']['mean_cosine'])
    for key in ('1', '5', '10', '25', 'never'):
        compare(measured['periodic_correction'][key]['observation'], audit['periodic_correction'][key]['observation'])
    return dict(condition=condition, seed=0, config=source['config'], reused=True, status='completed',
                source_report=c['baseline_path'] if condition == 'v1' else c['v3_path'],
                initial_parameter_hashes=source['training']['parameter_hashes_before'],
                normalization_sha256=source['normalization_sha256'], training=source['training'], model=source['model'],
                episode_order_hashes=expected_orders(0, len(c['splits']['train']), source['config']['epochs']),
                order_hash_provenance='reconstructed from original logged default_rng(seed0) rule; NOT retroactively recorded',
                test_windows=windows['window_counts'], evaluation=measured,
                seed0_checkpoint_unchanged_and_metrics_reproduced=True)


def assemble_report(root, context=None, make_figures=True):
    root = Path(root).resolve(); path = root/'mujoco/reports'/REPORT
    if path.exists():
        raise FileExistsError('replication report exists; verify instead of overwriting')
    c = load_context(root) if context is None else context
    branches = [reuse_seed0(c, condition) for condition in CONDITIONS]
    for seed in (1, 2, 3, 4):
        for condition in CONDITIONS:
            _, part = branch_paths(root, condition, seed)
            row = json.loads(part.read_text())
            if row['seed'] != seed or row['condition'] != condition:
                raise ValueError('saved branch identity mismatch')
            verify_branch_contract(c, row)
            if row['status'] != 'failed':
                if row['episode_order_hashes'] != expected_orders(seed, len(c['splits']['train']), row['config']['epochs']):
                    raise ValueError('branch order changed')
                if file_hash(row['model']['path']) != row['model']['sha256']:
                    raise ValueError('branch checkpoint changed')
            branches.append(row)
    summary = summarize_branches(branches)
    paired_order = {}
    for seed in SEEDS:
        pair = [next(r for r in branches if r['seed'] == seed and r['condition'] == k) for k in CONDITIONS]
        paired_order[str(seed)] = dict(
            equal=pair[0].get('episode_order_hashes') == pair[1].get('episode_order_hashes') if all(r['status'] == 'completed' for r in pair) else None,
            provenance=pair[0].get('order_hash_provenance', 'failed branch'),
            same_initial_hashes=pair[0]['initial_parameter_hashes'] == pair[1]['initial_parameter_hashes'])
        if paired_order[str(seed)]['equal'] is False:
            raise ValueError('paired batch realizations differ')
    b = c['baseline']
    report = dict(experiment='Paired Joint v1 vs Autonomous Consistency v3 training-seed replication',
        branch=BRANCH, base_branch='feat/joint-autonomous-consistency-v3', base_commit=BASE, seeds=list(SEEDS),
        replication_type='joint-training seed replication from ONE fixed pretrained initialization; NOT random initialization or end-to-end pipeline replication',
        report_path=str(path), seed0_reused=True, new_training_count=8,
        baseline_report=c['baseline_path'], baseline_audit=c['baseline_audit_path'], v3_seed0_report=c['v3_path'],
        common_config={k: v for k, v in b['config'].items() if k != 'seed'},
        initialization=c['initialization'], dataset=c['dataset'], dataset_regenerated=False,
        normalization=c['statistics'], normalization_sha256=json_hash(c['statistics']),
        initial_latent_scale=c['initial_scale'], initial_scale_file=c['v3']['initial_scale_file'],
        window_manifests=b['window_manifests'], test_start_manifest=b['test_start_manifest'], windows=b['windows'],
        horizons_steps=b['horizons_steps'], branches=branches, paired_order_verification=paired_order, summary=summary,
        immutable_artifacts=c['immutable'],
        objectives=dict(OBJECTIVES),
        selection_metric='Validation Lobs only for BOTH conditions; no Test/total/latent selection',
        pairing='same fixed E/T/D pretrained hashes; seed-specific default_rng complete-episode permutations, identical within each v1/v3 pair, full60epochs independent of selected best epoch',
        diagnostic_policy='all seeds minimal norm/std/effective-rank/nonfinite/own-model H50consistency and full periodic correction; no default covariance/PCA/sensitivity audits',
        limitations=['Five joint-training seeds; same upstream pretrained initialization, NOT full end-to-end replication.',
                     'Nominal simulation, one target-disjoint dataset, recorded future actions, overlapping Test windows.',
                     'Fixedlambda.1,K10,60epochcap; Validation-best budget unchanged, cap-best is not convergence.',
                     'Sample std ddof1, descriptive paired deltas/win counts; no statistical-significance or causal unique-root claim.',
                     'Own-model latent coordinates/scales differ; native latent discrepancy is not an invariant physical cross-model metric.',
                     'Periodic correction uses diagnostic real future histories, not deployable autonomous inference.',
                     'No upstream retraining, local-v2 experiment, new objectives, policy/planning/reward/RSSM/Dreamer work.'])
    if any(file_hash(p) != h for p, h in c['immutable'].items()):
        raise ValueError('immutable source changed')
    if make_figures:
        from joint_multiseed_plots import render_figures
        report['figures'] = render_figures(branches, summary, path.parent)
    else:
        report['figures'] = []
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def verify_experiment(report_path):
    report_path = Path(report_path).resolve(); root = report_path.parents[2]
    report = json.loads(report_path.read_text())
    c = load_context(root, report['baseline_report'], report['baseline_audit'], report['v3_seed0_report'])
    b = c['baseline']
    for key, expected in [('base_commit', BASE), ('branch', BRANCH), ('seeds', list(SEEDS)),
            ('seed0_reused', True), ('new_training_count', 8), ('objectives', OBJECTIVES),
            ('common_config', {k: v for k, v in b['config'].items() if k != 'seed'}),
            ('horizons_steps', b['horizons_steps']),
            ('normalization', c['statistics']), ('normalization_sha256', json_hash(c['statistics'])),
            ('initialization', c['initialization']), ('dataset', c['dataset']), ('initial_latent_scale', c['initial_scale']),
            ('window_manifests', b['window_manifests']), ('test_start_manifest', b['test_start_manifest']),
            ('windows', b['windows']), ('immutable_artifacts', c['immutable'])]:
        compare(expected, report[key])
    for row in report['branches']:
        if row['seed'] == 0:
            compare(reuse_seed0(c, row['condition']), row)
        elif row['status'] == 'completed':
            verify_branch(c, row)
        elif row['status'] != 'failed' or not row.get('failure'):
            raise ValueError('incomplete branch cannot masquerade as a failed/completed experiment')
        else:
            verify_branch_contract(c, row)
    compare(summarize_branches(report['branches']), report['summary'])
    expected = {}
    for seed in SEEDS:
        pair = [next(r for r in report['branches'] if r['seed'] == seed and r['condition'] == k) for k in CONDITIONS]
        expected[str(seed)] = dict(
            equal=pair[0].get('episode_order_hashes') == pair[1].get('episode_order_hashes') if all(r['status'] == 'completed' for r in pair) else None,
            provenance=pair[0].get('order_hash_provenance', 'failed branch'),
            same_initial_hashes=pair[0]['initial_parameter_hashes'] == pair[1]['initial_parameter_hashes'])
    compare(expected, report['paired_order_verification'])
    return dict(all_five_seed_identities_retained=True, seed0_reused_without_training=True,
                fixed_pretrained_initialization=True, available_recorded_orders_verified=True,
                completed_branch_validation_observation_selection=True, available_frozen_metrics_reproduced=True,
                paired_statistics_reproduced=True, no_future_autonomous_input=True)
