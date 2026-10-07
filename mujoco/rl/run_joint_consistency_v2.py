"""One controlled v2 training; --verify reloads/recomputes but never trains."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import episodes_from_arrays,file_hash,json_hash
from latent_dynamics_multistep import start_manifest
from joint_latent_world_model import initialize_joint,component_hashes
from joint_latent_training import validation_loss
from run_joint_latent_composition_audit import compare
from joint_consistency_training import initial_scale,train_consistency,validation_losses,load_consistency,selection_epoch,LAMBDA
from joint_consistency_evaluation import evaluate,comparison
from run_latent_memory_ablation import parameter_hash

BASE='d13541be9a620a55303ab0079eb7a517d5cd1822'
BRANCH='feat/joint-latent-consistency-v2'
REPORT='joint_latent_consistency_v2_seed0.json'
MODEL='joint_latent_world_model_v2_consistency.pt'
SCALE='joint_latent_consistency_v2_initial_scale_seed0.json'
MANIFOLD='joint_latent_consistency_v2_train_statistics_seed0.json'
BASELINE_FIELDS=('encoder_reconstruction','local_one_step','autonomous','periodic_correction','stability')


def load_split(dataset,split):
    path=Path(dataset['source_directory'])/f'{split}.npz'
    if file_hash(path)!=dataset['source_dataset_sha256'][split]: raise ValueError('dataset split hash mismatch')
    with np.load(path,allow_pickle=False) as arrays:
        eps=episodes_from_arrays(arrays)
        for ep,start,stop in zip(eps,arrays['offsets'][:-1],arrays['offsets'][1:]): ep['next_obs']=arrays['inputs'][start+1:stop,:7].copy()
    identity=[dict(target_id=e['metadata']['target_id'],source=e['metadata']['source'],rows=e['source_rows']) for e in eps]
    if identity!=dataset['split_identity'][split] or sorted({e['metadata']['target_id'] for e in eps})!=dataset['target_ids'][split]:
        raise ValueError('dataset split/episode identity mismatch')
    return eps


def load_context(root,baseline_report,audit_report):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    paths=[Path(baseline_report).resolve(),Path(audit_report).resolve()]
    baseline,audited=[json.loads(p.read_text()) for p in paths]
    if audited['checkpoint']['sha256']!=baseline['model']['sha256'] or audited['source_report']!=str(paths[0]):
        raise ValueError('v1 audit/checkpoint provenance mismatch')
    immutable=dict(baseline['immutable_artifacts']); immutable.update({str(p):file_hash(p) for p in paths})
    immutable[baseline['model']['path']]=baseline['model']['sha256']
    checkpoints=baseline['initialization']['checkpoints']
    for row in checkpoints.values(): immutable[row['path']]=row['sha256']
    for row in [baseline['test_start_manifest'],*baseline['window_manifests'].values()]: immutable[row['path']]=row['sha256']
    dataset=baseline['dataset']; directory=Path(dataset['source_directory'])
    immutable[str(directory/'manifest.json')]=dataset['source_manifest_sha256']
    immutable.update({str(directory/f'{s}.npz'):h for s,h in dataset['source_dataset_sha256'].items()})
    for p,h in immutable.items():
        if file_hash(p)!=h: raise ValueError(f'immutable baseline/source hash mismatch: {p}')
    groups=dataset['target_ids']
    if any(set(groups[a])&set(groups[b]) for a,b in [('train','val'),('train','test'),('val','test')]): raise ValueError('target split overlap')
    model,stats,initialization=initialize_joint(*[checkpoints[k]['path'] for k in ('encoder','transition','decoder')])
    if component_hashes(model)!=baseline['training']['parameter_hashes_before'] or initialization!=baseline['initialization']:
        raise ValueError('pre-joint initial hashes/conversion differ from v1')
    if stats!=baseline['normalization'] or json_hash(stats)!=baseline['normalization_sha256']: raise ValueError('original normalization mismatch')
    splits={s:load_split(dataset,s) for s in ('train','val')}
    for split,row in baseline['window_manifests'].items():
        expected=start_manifest(splits[split],[10],dataset['source_dataset_sha256'][split]); expected.pop('manifest_sha256')
        expected['split']=split; expected['dataset_split_sha256']=expected.pop('dataset_test_sha256'); expected['manifest_sha256']=json_hash(expected)
        if expected!=json.loads(Path(row['path']).read_text()) or row['window_count']!=expected['window_counts']['10']:
            raise ValueError('original K10 window manifest mismatch')
    scale=initial_scale(model,splits['train'],stats)
    scale.update(encoder_initial_parameter_sha256=component_hashes(model)['encoder'],dataset_train_sha256=dataset['source_dataset_sha256']['train'])
    initial_obs=validation_loss(model,splits['val'],stats,10,baseline['config']['batch_size'])
    compare(initial_obs,baseline['training']['initial_validation_loss'])
    return dict(baseline=baseline,baseline_path=str(paths[0]),baseline_audit=audited,baseline_audit_path=str(paths[1]),
        model=model,statistics=stats,initialization=initialization,initial_scale=scale,dataset=dataset,splits=splits,immutable=immutable,
        initial_observation_validation=initial_obs)


def test_inputs(context):
    source=context['baseline']; test=load_split(context['dataset'],'test')
    windows=start_manifest(test,source['horizons_steps'],context['dataset']['source_dataset_sha256']['test'])
    if windows!=source['windows'] or windows!=json.loads(Path(source['test_start_manifest']['path']).read_text()):
        raise ValueError('original Test start manifest changed')
    return test,windows


def run_experiment(root,baseline_report,audit_report,make_figures=True):
    root=Path(root).resolve(); output=root/'mujoco/reports'; path=output/REPORT; modelpath=root/'mujoco/rl/models'/MODEL
    scalepath=output/SCALE; manifoldpath=output/MANIFOLD
    if any(p.exists() for p in [path,modelpath,scalepath,manifoldpath]): raise FileExistsError('v2 artifacts exist; no overwrite/retraining')
    started=time.monotonic(); context=load_context(root,baseline_report,audit_report); baseline=context['baseline']; cfg=dict(baseline['config'])
    if cfg['seed']!=0 or cfg['horizon']!=10 or cfg['learning_rate']!=.0003: raise ValueError('fixed seed0 K10 lr.0003')
    scale=context['initial_scale']; output.mkdir(parents=True,exist_ok=True)
    scalepath.write_text(json.dumps(scale,indent=2,allow_nan=False)+'\n')
    training=train_consistency(context['model'],context['splits']['train'],context['splits']['val'],context['statistics'],scale['std'],cfg,modelpath)
    if training['parameter_hashes_before']!=baseline['training']['parameter_hashes_before']: raise AssertionError('initial hashes changed')
    compare(training['initial_validation']['observation'],context['initial_observation_validation'])
    # Test values FIRST opened only after observation-Validation selection.
    test,windows=test_inputs(context); actual,plot=evaluate(modelpath,context,test)
    manifoldpath.write_text(json.dumps(actual['manifold'],indent=2,allow_nan=False)+'\n')
    model,stats,meta=load_consistency(modelpath)
    report=dict(experiment='Deterministic Latent World Model v2: Explicit Encoder–Transition Consistency',branch=BRANCH,
        base_branch='feat/joint-latent-composition-audit',base_commit=BASE,seed=0,report_path=str(path),
        baseline_report=context['baseline_path'],baseline_audit_report=context['baseline_audit_path'],config=cfg,lambda_consistency=LAMBDA,
        dataset=context['dataset'],dataset_regenerated=False,initialization=context['initialization'],initial_latent_scale=scale,
        initial_observation_validation=context['initial_observation_validation'],
        initial_scale_file=dict(path=str(scalepath),sha256=file_hash(scalepath),semantic_sha256=json_hash(scale)),
        training=training,model=dict(path=str(modelpath),sha256=file_hash(modelpath),parameter_sha256=parameter_hash(model),bytes=modelpath.stat().st_size),
        normalization=stats,normalization_sha256=json_hash(stats),window_manifests=baseline['window_manifests'],
        test_start_manifest=baseline['test_start_manifest'],windows=windows,horizons_steps=baseline['horizons_steps'],selected_window=baseline['selected_window'],
        train_manifold=actual['manifold'],train_statistics_file=dict(path=str(manifoldpath),sha256=file_hash(manifoldpath),semantic_sha256=json_hash(actual['manifold'])),
        baseline={k:context['baseline_audit'][k] for k in BASELINE_FIELDS},
        audit=actual['audit'],relative_local_residual=actual['relative'],latent_variance=actual['latent_variance'],
        comparison=comparison(context['baseline_audit'],actual['audit'],actual['relative'],baseline['horizons_steps']),
        immutable_artifacts=context['immutable'],verification=dict(actual['verification'],initial_hashes_match_v1=True,
            original_windows_normalization_reused=True,test_loaded_after_observation_validation_selection=True,
            all_three_components_updated=all(training['parameter_hashes_before'][k]!=training['parameter_hashes_best'][k] for k in ('encoder','transition','decoder')),
            baseline_and_sources_unchanged=all(file_hash(p)==h for p,h in context['immutable'].items())),
        objective='Lobs=uniform mean k0..10 observation-std MSE; Lcons=uniform local k0..9 fixedinitialTrainstd MSE(T(E(history_k),a_k)-stopgrad(E(history_k+1))); Ltotal=Lobs+0.1Lcons',
        selection_metric='Validation Lobs only; not total/consistency/Test',
        stop_gradient='next real-history Encoder output detached only on target side; source Encoder+T gradients enabled; D absent from consistency graph',
        protocol='same original observation forward; distinct training-only consistency real-history branch; inference is prefixE then T-only recursion/Dreadout, never future E/obs/latent input',
        limitations=['Single seed nominal simulation recorded actions overlapping windows; fixedlambda.1 and local-only consistency, no sweep.',
            'Moving online Encoder reference with targetstopgrad, no EMA/targetnetwork; one modified objective only.',
            'Native latent scales/geometry differ acrossmodels; absolute latent RMSE/Decoder ratio not physical coordinate-invariant comparisons.',
            'Relative residual epsilon is rawcoordinate-dependent; nearstationary denominators can dominate means.',
            'Periodic corrections inject realhistory diagnostically; PCA/covariance are distribution proxies, not proof of nonlinear manifold.',
            'No policy/planning/control/reward/RSSM/Dreamer experiment.'])
    if not all(report['verification'].values()): raise AssertionError('v2 verification failed')
    if make_figures:
        from joint_consistency_plots import render_figures
        report['figures']=render_figures(report,plot,context,output)
    else: report['figures']=[]
    report['elapsed_seconds']=time.monotonic()-started
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(f'Saved {path}',flush=True); return report


def verify_history_summaries(training,config):
    """Check recorded-history identities; this does not replay historical gradients."""
    history=training['history']
    if [a['epoch'] for a in history]!=list(range(1,config['epochs']+1)) or training['final_epoch']!=config['epochs']:
        raise AssertionError('epoch sequence/final epoch changed')
    def nonnegative_finite(value):
        if isinstance(value,dict): return all(nonnegative_finite(v) for v in value.values())
        return bool(np.isfinite(value) and value>=0)
    for row in history:
        for key in ('train','validation'):
            loss=row[key]
            if not nonnegative_finite(loss): raise AssertionError('non-finite/negative recorded loss')
            compare(LAMBDA*loss['consistency'],loss['weighted_consistency'])
            compare(loss['observation']+loss['weighted_consistency'],loss['total'])
        for key in ('gradient_norms','objective_gradient_diagnostics'):
            if not nonnegative_finite(row[key]): raise AssertionError('invalid recorded gradient diagnostics')
        if any(v['mean']>v['max'] for v in row['gradient_norms'].values()): raise AssertionError('gradient mean exceeds max')
    compare(history[-1]['train'],training['final_train'])
    compare(history[-1]['validation'],training['final_validation'])


def verify_experiment(report_path):
    p=Path(report_path).resolve(); root=p.parents[2]; r=json.loads(p.read_text())
    c=load_context(root,r['baseline_report'],r['baseline_audit_report']); b=c['baseline']
    for key,value in [('config',b['config']),('lambda_consistency',.1),('initialization',c['initialization']),('initial_latent_scale',c['initial_scale']),
        ('normalization',c['statistics']),('dataset',c['dataset']),('window_manifests',b['window_manifests']),('test_start_manifest',b['test_start_manifest']),
        ('horizons_steps',b['horizons_steps']),('selected_window',b['selected_window']),('immutable_artifacts',c['immutable'])]: compare(value,r[key])
    compare(c['initial_observation_validation'],r['initial_observation_validation'])
    compare(c['initial_observation_validation'],r['training']['initial_validation']['observation'])
    if r['normalization_sha256']!=json_hash(c['statistics']): raise ValueError('normalization hash changed')
    for key,expected in [('initial_scale_file',c['initial_scale'])]:
        row=r[key]
        if file_hash(row['path'])!=row['sha256'] or json_hash(expected)!=row['semantic_sha256']: raise ValueError('scale file/hash changed')
        compare(json.loads(Path(row['path']).read_text()),expected)
    row=r['model']; model,stats,meta=load_consistency(row['path']); history=r['training']['history']; cfg=r['config']
    verify_history_summaries(r['training'],cfg)
    if file_hash(row['path'])!=row['sha256'] or parameter_hash(model)!=row['parameter_sha256'] or stats!=c['statistics'] or meta['config']!=cfg:
        raise ValueError('model/config/stats hash mismatch')
    if (meta['initial_latent_std']!=c['initial_scale']['std'] or meta['lambda_consistency']!=.1 or
        meta['parameter_hashes_before']!=b['training']['parameter_hashes_before'] or
        r['training']['parameter_hashes_before']!=b['training']['parameter_hashes_before'] or
        r['training']['parameter_hashes_best']!=component_hashes(model)): raise AssertionError('initialization/best parameter hashes mismatch')
    epoch=selection_epoch([a['validation'] for a in history])
    if len(history)!=cfg['epochs'] or meta['best_epoch']!=epoch or r['training']['best_epoch']!=epoch:
        raise AssertionError('observation validation selection mismatch')
    for split,field in [('train','best_checkpoint_train'),('val','best_validation')]:
        loss=validation_losses(model,c['splits'][split],stats,c['initial_scale']['std'],10,cfg['batch_size'])
        compare(loss,r['training'][field])
    compare(r['training']['best_validation']['observation'],meta['validation_observation_loss'])
    compare(r['training']['best_validation']['observation'],min(a['validation']['observation'] for a in history))
    count=b['window_manifests']['train']['window_count']
    if any(a['windows_used']!=count for a in history): raise AssertionError('training window counts changed')
    test,windows=test_inputs(c); ev,_=evaluate(row['path'],c,test)
    for field,key in [('audit','audit'),('manifold','train_manifold'),('relative','relative_local_residual'),('latent_variance','latent_variance')]: compare(ev[field],r[key])
    statsfile=r['train_statistics_file']
    if file_hash(statsfile['path'])!=statsfile['sha256'] or json_hash(ev['manifold'])!=statsfile['semantic_sha256']: raise ValueError('Train geometry hash changed')
    compare(json.loads(Path(statsfile['path']).read_text()),ev['manifold']); compare(windows,r['windows'])
    compare({k:c['baseline_audit'][k] for k in BASELINE_FIELDS},r['baseline'])
    compare(comparison(c['baseline_audit'],ev['audit'],ev['relative'],b['horizons_steps']),r['comparison'])
    if not all(ev['verification'].values()) or not all(file_hash(p)==h for p,h in c['immutable'].items()): raise AssertionError('inference/source contract failed')
    return dict(same_pre_joint_initialization=True,fixed_initial_Train_scale=True,normalization_manifests_reused=True,
        observation_only_selection_reproduced=True,selected_checkpoint_and_evaluation_metrics_reproduced=True,
        recorded_history_internally_consistent=True,no_future_inference_reference=True,baseline_sources_unchanged=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.verify: print(json.dumps(verify_experiment(root/'mujoco/reports'/REPORT),indent=2))
    else: run_experiment(root,root/'mujoco/reports/joint_latent_world_model_v1_seed0.json',root/'mujoco/reports/joint_latent_composition_audit_seed0.json')
