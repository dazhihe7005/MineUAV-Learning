"""Train ONLY observation-std matched K=1; immutable K10 is never retrained.

K1 uses exactly the saved K10 Train/Validation start identities. Slicing unused
episode tails preserves every used causal prefix, target and minibatch ordering;
it avoids adding nine new starts per episode as a second experimental change.
--verify is read-only (reloads/re-evaluates, never invokes an optimizer).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash, json_hash
from latent_dynamics_models import load_model, make_model
from latent_dynamics_multistep import start_manifest
from latent_multistep_training import train_multistep, validation_loss
from run_latent_dynamics_multistep_eval import compare, evaluate_model, load_inputs, tensor_hash
from run_latent_memory_ablation import parameter_hash
from run_latent_multistep_training import load_training_inputs

BASE_COMMIT = '63545a60396bbbc9421d87f1f28548104739cfbd'
BRANCH = 'feat/latent-objective-matched-control'
REPORT_NAME = 'latent_dynamics_objective_matched_control_seed0.json'
MODEL_NAME = 'history_latent_gru_matched_one_step.pt'
LABELS = dict(history='Historical delta-loss K1', matched_one_step='Matched obs-loss K1',
              multistep_history='Obs-loss K10')


def match_start_prefixes(episodes, manifest, reference_horizon=10):
    """Keep starts range(N-Kref+1), without changing any retained true prefix."""
    if len(episodes) != len(manifest['episodes']):
        raise ValueError('matched start episode count mismatch')
    result = []
    for index, (ep, row) in enumerate(zip(episodes, manifest['episodes'])):
        count = max(0, len(ep['obs'])-reference_horizon+1)
        if (count < 1 or row['episode_id'] != index
                or row['transition_count'] != len(ep['obs'])
                or row['target_id'] != ep['metadata']['target_id']
                or row['source'] != ep['metadata']['source']
                or row['source_rows'] != ep['source_rows']
                or row['valid_start_count'][str(reference_horizon)] != count):
            raise ValueError('matched start identity mismatch or no usable starts')
        # Copy dict only; slices are read-only views from the trainer's point of
        # view. No write to original arrays/dataset. Source rows stay original.
        matched = dict(ep)
        for key in ('obs', 'next_obs', 'previous_action', 'action', 'delta', 'pi'):
            matched[key] = ep[key][:count]
        result.append(matched)
    if sum(len(ep['obs']) for ep in result) != manifest['window_counts'][str(reference_horizon)]:
        raise ValueError('matched start total mismatch')
    return result


def load_reference(root, source_path):
    """Only Train/Val timestep values loaded here; Test remains unopened."""
    source_path = Path(source_path).resolve(); source = json.loads(source_path.read_text())
    for path, expected in source['immutable_artifacts'].items():
        if file_hash(path) != expected:
            raise ValueError(f'immutable reference hash mismatch: {path}')
    stats = source['normalization']
    if json_hash(stats) != source['normalization_sha256']:
        raise ValueError('reference normalization hash mismatch')
    cfg = dict(source['config'])
    if cfg['horizon'] != 10 or cfg['seed'] != 0:
        raise ValueError('reference must be the seed0 K10 experiment')
    artifact = source['models']['multistep_history']['artifact']
    if file_hash(artifact['path']) != artifact['sha256']:
        raise ValueError('K10 checkpoint hash mismatch')
    net, saved_stats, metadata = load_model(artifact['path'])
    if saved_stats != stats or metadata['config'] != cfg:
        raise ValueError('K10 saved configuration/normalization mismatch')
    expected = source['training']['initial_parameter_sha256']
    if metadata['initial_parameter_sha256'] != expected:
        raise ValueError('K10 initial parameter hash mismatch')
    torch.manual_seed(cfg['seed']); initialized = make_model('history')
    if (parameter_hash(initialized) != expected
            or [(n, tuple(p.shape)) for n,p in initialized.named_parameters()]
               != [(n, tuple(p.shape)) for n,p in net.named_parameters()]):
        raise ValueError('same architecture/initial parameter hash not reproduced')
    context = load_training_inputs(root, source['source_report'], horizon=10)
    if context['statistics'] != stats or context['dataset'] != source['dataset']:
        raise ValueError('K10 dataset/normalization mismatch')
    immutable = dict(source['immutable_artifacts'])
    immutable[str(source_path)] = file_hash(source_path)
    immutable[artifact['path']] = artifact['sha256']
    for split, row in source['train_validation_manifests'].items():
        manifest = context['manifests'][split]
        if (file_hash(row['path']) != row['sha256']
                or json.loads(Path(row['path']).read_text()) != manifest
                or row['semantic_sha256'] != manifest['manifest_sha256']
                or row['window_count'] != manifest['window_counts']['10']):
            raise ValueError('reference training start manifest hash/identity mismatch')
        immutable[row['path']] = row['sha256']
    cfg['horizon'] = 1
    matched = {s: match_start_prefixes(context['splits'][s], context['manifests'][s]) for s in ('train','val')}
    return dict(context, source=source, source_path=str(source_path), config=cfg,
                immutable=immutable, matched=matched, expected_initial=expected)


def matched_manifest(context, split):
    original = context['manifests'][split]
    data = dict(split=split, dataset_split_sha256=context['dataset']['source_dataset_sha256'][split],
        prediction_horizon=1, reference_start_horizon=10,
        protocol='exact saved K10 start identities range(N-10+1); same complete-episode order; unchanged causal prefixes',
        reference_manifest=context['source']['train_validation_manifests'][split],
        # Original episode length/provenance retained, no fabricated shortened trajectory.
        episodes=original['episodes'], window_count=original['window_counts']['10'])
    data['manifest_sha256'] = json_hash(data)
    return data


def yaw_audit(episodes):
    current = np.concatenate([e['obs'][:,6] for e in episodes])
    following = np.concatenate([e['next_obs'][:,6] for e in episodes])
    if not (np.isfinite(current).all() and np.isfinite(following).all()
            and np.abs(current).max() <= np.pi+1e-6 and np.abs(following).max() <= np.pi+1e-6):
        raise ValueError('stored yaw outside wrapped [-pi,pi] observation definition')
    return dict(transitions=len(current), current_min=float(current.min()), current_max=float(current.max()),
        adjacent_wrap_jumps=int(np.sum(np.abs(following-current)>np.pi)),
        definition='atan2(sin(target_yaw-actual_yaw), cos(target_yaw-actual_yaw))',
        prediction='physical raw coordinate recursion; no new prediction wrapping/clipping',
        metric='raw coordinate error, identical frozen evaluator; no new circular wrapping')


def comparison(models, horizons):
    result = {}
    for h in horizons:
        rows = {key: models[key]['horizons'][str(h)] for key in LABELS}
        fields = ['normalized_observation_rmse', 'horizontal_velocity_rmse', 'vertical_velocity_rmse']
        values = {name: {key: row[name] for key,row in rows.items()} for name in fields}
        for dim in ('vx','vy','vz','yaw_error'):
            values[dim+'_rmse'] = {key: row['physical'][dim]['rmse'] for key,row in rows.items()}
        record = {}
        for field, v in values.items():
            a, b, c = v['history'], v['matched_one_step'], v['multistep_history']
            record[field] = dict(values=v, k10_reduction_vs_matched_k1_percent=100*(b-c)/b,
                matched_k1_reduction_vs_historical_percent=100*(a-b)/a,
                historical_to_k10_absolute=a-c, historical_to_matched_k1_absolute=a-b,
                matched_k1_to_k10_absolute=b-c)
        result[str(h)] = record
    return result


def render_figures(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    horizons=report['horizons_steps']; clock=np.array(horizons)/25; paths=[]
    def save(fig, name):
        fig.tight_layout(); path=directory/name; fig.savefig(path,dpi=160); plt.close(fig); paths.append(str(path))
    fig, ax=plt.subplots(figsize=(8,4.8))
    for key,label in LABELS.items():
        ax.plot(clock,[report['models'][key]['horizons'][str(h)]['normalized_observation_rmse'] for h in horizons], 'o-',label=label)
    ax.set(xlabel='Recorded-action horizon (s)',ylabel='Endpoint observation RMSE / Train observation std',
           title='Frozen Test evaluation: objective-matched K1 vs K10')
    ax.grid(alpha=.2); ax.legend(); save(fig,'matched_k1_vs_k10_rmse_vs_horizon.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,field,title in zip(axes,('horizontal_velocity_rmse','vertical_velocity_rmse'),('Horizontal vx/vy','Vertical vz')):
        for key,label in LABELS.items():
            ax.plot(clock,[report['models'][key]['horizons'][str(h)][field] for h in horizons], 'o-',label=label)
        ax.set(xlabel='Horizon (s)',ylabel='Velocity RMSE (m/s)',title=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); save(fig,'matched_k1_vs_k10_velocity_rmse.png')
    fig,ax=plt.subplots(figsize=(8,4.8))
    for key,label in LABELS.items():
        ax.plot(clock,[report['models'][key]['horizons'][str(h)]['physical']['yaw_error']['rmse'] for h in horizons], 'o-',label=label)
    ax.set(xlabel='Horizon (s)',ylabel='Raw yaw_error coordinate RMSE (rad)',title='Yaw diagnostic — unchanged representation/wrap conventions')
    ax.grid(alpha=.2); ax.legend(); save(fig,'matched_k1_vs_k10_yaw_rmse.png')
    return paths


def evaluate_control(context, model_path, verbose=False):
    """Invoked only after validation selection; independently reproduce A and C."""
    frozen=context['frozen']; stats=context['statistics']
    ev=load_inputs(Path(model_path).parents[3], frozen['source_report'])
    windows=start_manifest(ev['episodes'], frozen['horizons_steps'], context['dataset']['source_dataset_sha256']['test'])
    test_manifest=frozen['start_manifest']
    if (windows != frozen['windows'] or windows != json.loads(Path(test_manifest['path']).read_text())
            or file_hash(test_manifest['path']) != test_manifest['sha256']):
        raise ValueError('frozen Test manifest hash/identity mismatch')
    thresholds=frozen['pi_magnitude_bins']['thresholds_m_s2']; horizons=frozen['horizons_steps']
    k10,_,_=load_model(context['source']['models']['multistep_history']['artifact']['path'])
    new,new_stats,metadata=load_model(model_path)
    if new_stats != stats: raise ValueError('new checkpoint normalization mismatch')
    models={}
    for key,net in [('history',ev['models']['history']),('multistep_history',k10),('matched_one_step',new)]:
        print(f'Frozen Test evaluation: {key}',flush=True)
        net.eval().requires_grad_(False)
        row=evaluate_model(net,'history',ev['episodes'],stats,thresholds,horizons,verbose=verbose)
        if key != 'matched_one_step':
            expected=context['source']['models'][key]
            for field,value in row.items(): compare(value,expected[field])
            row=expected
        else:
            row.update(artifact=dict(path=str(model_path),sha256=file_hash(model_path),
                       bytes=Path(model_path).stat().st_size,parameter_sha256=tensor_hash(net)),
                       architecture=repr(net),metadata=metadata)
        models[key]=row
    return models,windows,yaw_audit(ev['episodes'])


def run_control(root, source_path, make_figures=True):
    root=Path(root).resolve(); output=root/'mujoco/reports'
    report_path=output/REPORT_NAME; model_path=root/'mujoco/rl/models'/MODEL_NAME
    manifest_paths={s:output/f'latent_matched_k1_{s}_windows_seed0.json' for s in ('train','val')}
    if any(p.exists() for p in [report_path,model_path,*manifest_paths.values()]):
        raise FileExistsError('control artifacts already exist; never silently overwrite/retrain')
    started=time.monotonic(); context=load_reference(root,source_path)
    output.mkdir(parents=True,exist_ok=True); manifests={}
    for split,path in manifest_paths.items():
        manifest=matched_manifest(context,split); path.write_text(json.dumps(manifest,indent=2)+'\n')
        manifests[split]=dict(path=str(path),sha256=file_hash(path),semantic_sha256=manifest['manifest_sha256'],
                              window_count=manifest['window_count'])
    training=train_multistep(context['matched']['train'],context['matched']['val'],context['statistics'],context['config'],model_path)
    if training['initial_parameter_sha256'] != context['expected_initial']:
        raise AssertionError('K1 initialization changed from frozen K10')
    selected_hash=file_hash(model_path)
    models,windows,test_yaw=evaluate_control(context,model_path,verbose=True)
    report=dict(experiment='Objective-Matched One-Step Control',branch=BRANCH,
        base_branch='feat/latent-multistep-training',base_commit=BASE_COMMIT,seed=0,
        report_path=str(report_path),source_report=context['source_path'],
        dataset=context['dataset'],dataset_regenerated=False,matched_start_manifests=manifests,
        test_start_manifest=context['frozen']['start_manifest'],windows=windows,
        normalization=context['statistics'],normalization_sha256=json_hash(context['statistics']),
        config=context['config'],reference_config=context['source']['config'],training=training,
        reference_initial_parameter_sha256=context['expected_initial'],
        objective='mean ((predicted next physical observation - recorded next observation)/saved Train observation std)^2; K=1, same loss family/scale as K10',
        output_parameterization='same saved delta mean/std inverse transform at the head; NOT the loss weighting',
        start_matching='K1 uses exact original K10 starts; causal prefix slices end after last used start; source trajectories unchanged',
        protocol='current observation + previous executed normalized action -> GRU prefix; recorded current actions; no true PI input',
        horizons_steps=context['frozen']['horizons_steps'],pi_magnitude_bins=context['frozen']['pi_magnitude_bins'],
        models=models,comparison=comparison(models,context['frozen']['horizons_steps']),
        yaw_correctness=dict(train=yaw_audit(context['splits']['train']),val=yaw_audit(context['splits']['val']),test=test_yaw),
        immutable_artifacts=context['immutable'],
        verification=dict(same_initial_parameters=True,same_parameter_structure=True,loss_scaling_matched=True,
            train_validation_start_identities_matched=True,test_loaded_after_validation_selection=True,
            frozen_reference_metrics_reproduced=True,fixed_test_windows=True,
            selected_checkpoint_unchanged=file_hash(model_path)==selected_hash,
            immutable_artifacts_unchanged=all(file_hash(p)==h for p,h in context['immutable'].items())),
        versions=dict(python=__import__('platform').python_version(),torch=torch.__version__,numpy=np.__version__),
        limitations=['Single seed and nominal recorded-action trajectories; not policy/planning evaluation.',
            'Validation uses each condition\'s own prescribed horizon; no Test checkpoint selection.',
            'Historical delta-loss K1 additionally used terminal-tail starts: A-to-B is not a clean scaling-only intervention.',
            'K1-to-K10 holds start identities fixed, but future horizon labels and optimization trajectory differ by design.',
            'No additive causal percentage decomposition of the historical improvement; dimension-level tradeoffs require separate interpretation.',
            'Yaw remains the original wrapped observation coordinate; prediction/evaluation use unchanged raw coordinate differences.'])
    if not all(report['verification'].values()): raise AssertionError('control verification failed')
    report['figures']=render_figures(report,output) if make_figures else []
    report['elapsed_seconds']=time.monotonic()-started
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Saved {report_path}',flush=True); return report


def verify_control(path):
    path=Path(path).resolve(); report=json.loads(path.read_text()); root=path.parents[2]
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    context=load_reference(root,report['source_report'])
    if report['horizons_steps'] != context['frozen']['horizons_steps']:
        raise AssertionError('canonical frozen evaluation horizon metadata changed')
    for p,expected in report['immutable_artifacts'].items():
        if file_hash(p) != expected: raise ValueError(f'immutable hash mismatch: {p}')
    if (report['config'] != context['config'] or report['normalization'] != context['statistics']
            or report['normalization_sha256'] != json_hash(context['statistics'])):
        raise AssertionError('matched configuration/normalization changed')
    for split,row in report['matched_start_manifests'].items():
        if (file_hash(row['path']) != row['sha256']
                or json.loads(Path(row['path']).read_text()) != matched_manifest(context,split)):
            raise ValueError('matched training manifest hash/identity mismatch')
    artifact=report['models']['matched_one_step']['artifact']
    if file_hash(artifact['path']) != artifact['sha256']: raise ValueError('matched model hash mismatch')
    net,stats,metadata=load_model(artifact['path']); training=report['training']; cfg=report['config']
    if (stats != context['statistics'] or metadata['config'] != cfg
            or metadata['initial_parameter_sha256'] != context['expected_initial']
            or training['initial_parameter_sha256'] != context['expected_initial']):
        raise AssertionError('matched checkpoint configuration/initialization mismatch')
    losses=[r['validation_loss'] for r in training['history']]
    if (int(np.argmin(losses))+1 != metadata['best_epoch'] or metadata['best_epoch'] != training['best_epoch']):
        raise AssertionError('validation-best selection mismatch')
    actual=validation_loss(net,context['matched']['val'],stats,1,cfg['batch_size'])
    if not np.isclose(actual,min(losses),rtol=1e-7,atol=1e-10):
        raise AssertionError('selected validation loss does not reproduce')
    if any(r['windows_used'] != report['matched_start_manifests']['train']['window_count'] for r in training['history']):
        raise AssertionError('training start usage changed')
    models,windows,test_yaw=evaluate_control(context,artifact['path'])
    for key,row in models.items():
        for field in ('horizons','common_max_horizon_windows','latent_drift','stability'):
            compare(row[field],report['models'][key][field])
    if windows != report['windows'] or report['test_start_manifest'] != context['frozen']['start_manifest']:
        raise AssertionError('Test window identities changed')
    compare(comparison(models,report['horizons_steps']),report['comparison'])
    compare(test_yaw,report['yaw_correctness']['test'])
    return dict(immutable_hashes=True,matched_start_identities=True,same_initial_parameters=True,
                saved_normalization=True,validation_best_selection=True,selected_validation_recomputed=True,
                frozen_and_new_test_metrics_reproduced=True,yaw_conventions_unchanged=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',action='store_true')
    args=parser.parse_args(); root=Path(__file__).resolve().parents[2]
    if args.verify: print(json.dumps(verify_control(root/'mujoco/reports'/REPORT_NAME),indent=2))
    else: run_control(root,root/'mujoco/reports/latent_dynamics_multistep_training_seed0.json')
