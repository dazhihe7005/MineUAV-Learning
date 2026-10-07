"""Joint E/T/D v1 experiment only; --verify never trains or writes artifacts."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash,json_hash
from latent_dynamics_models import load_model
from explicit_latent_models import load_component,freeze_encoder
from explicit_latent_evaluation import evaluate_explicit
from run_latent_transition_multistep import load_context as load_prior
from run_explicit_latent_transition import test_context
from run_latent_dynamics_multistep_eval import evaluate_model,compare as compare_value
from joint_latent_world_model import initialize_joint,load_joint,component_hashes,predict_window
from joint_latent_training import train_joint,validation_loss
from joint_latent_evaluation import encoded_sequences,evaluate_joint
from run_latent_memory_ablation import parameter_hash

BASE='4fe448576b7df321778291e414c98788c13a8e99'
BRANCH='feat/joint-latent-world-model-v1'
REPORT='joint_latent_world_model_v1_seed0.json'
MODEL='joint_latent_world_model_v1.pt'
CONFIG=dict(seed=0,epochs=60,batch_size=16,learning_rate=.0003,horizon=10)


def compare(actual,expected):
    """Existing scalar metric tolerance, with explicit diagnostic-list handling."""
    if isinstance(expected,dict):
        if set(actual)!=set(expected): raise AssertionError('metric keys changed')
        for k in expected: compare(actual[k],expected[k])
    elif isinstance(expected,list):
        if len(actual)!=len(expected): raise AssertionError('metric list length changed')
        for a,b in zip(actual,expected): compare(a,b)
    else: compare_value(actual,expected)


def load_context(root,source_report):
    path=Path(source_report).resolve(); source=json.loads(path.read_text())
    for p,h in source['immutable_artifacts'].items():
        if file_hash(p)!=h: raise ValueError('immutable source hash mismatch')
    if file_hash(source['model']['path'])!=source['model']['sha256']: raise ValueError('source Transition hash mismatch')
    context=load_prior(root,source['source_report'])
    for split,row in source['train_validation_manifests'].items():
        if file_hash(row['path'])!=row['sha256'] or json.loads(Path(row['path']).read_text())!=context['manifests'][split]:
            raise ValueError('original K10 training window identity mismatch')
    paths=[source['frozen_components']['encoder']['artifact']['path'],source['model']['path'],source['frozen_components']['decoder']['artifact']['path']]
    joint,stats,init=initialize_joint(*paths)
    if stats!=context['statistics']: raise ValueError('joint Train normalization mismatch')
    immutable=dict(source['immutable_artifacts']); immutable[str(path)]=file_hash(path); immutable[source['model']['path']]=source['model']['sha256']
    return dict(context,source=source,source_path=str(path),joint=joint,initialization=init,immutable=immutable)


def evaluate_all(root,context,path):
    model,stats,meta=load_joint(path); model.requires_grad_(False)
    train_latents=encoded_sequences(model,context['splits']['train'],stats)
    _,test,windows=test_context(root,context)
    source=context['source']; horizons=source['horizons_steps']; thresholds=source['pi_magnitude_bins']['thresholds_m_s2']
    if windows!=source['windows']: raise ValueError('Test window manifest changed')
    baseline,_,_,_=load_component(source['model']['path']); freeze_encoder(baseline)
    frozen=evaluate_explicit(baseline,context['decoder'],test,stats,context['latent_statistics'],thresholds,horizons)
    compare(frozen,source['models']['multistep'])
    direct,_,_=load_model(source['frozen_components']['encoder']['artifact']['path']); freeze_encoder(direct)
    direct_metrics=evaluate_model(direct,'history',test,stats,thresholds,horizons); compare(direct_metrics,source['direct_k10'])
    result=evaluate_joint(model,test,stats,thresholds,horizons,train_latents)
    selected=source['selected_window']; ep=test[selected['episode_id']]; t=selected['start']; h=selected['horizon']
    with torch.inference_mode():
        original=predict_window(model,ep,t,h,stats)
        changed=copy.deepcopy(ep); changed['obs'][t+1:]+=1000; changed['next_obs'][:]=np.nan
        corrupted=predict_window(model,changed,t,h,stats)
        changed['obs']=changed['obs'][:t+1]; changed['previous_action']=changed['previous_action'][:t+1]
        del changed['next_obs']; del changed['latent']
        deleted=predict_window(model,changed,t,h,stats)
    checks={f'{name}_{field}_unchanged':torch.equal(original[field],out[field]) for name,out in [('corrupted',corrupted),('deleted',deleted)]
            for field in ['latents','normalized_observations']}
    if not all(checks.values()): raise AssertionError('future observation leakage')
    return dict(joint=result,frozen_explicit=frozen,direct_k10=direct_metrics,future_observation_leakage=checks,
                windows=windows,test=test,model=model,baseline=baseline,direct=direct)


def comparison(evaluation,horizons):
    out={}
    for h in horizons:
        j=evaluation['joint']['horizons'][str(h)]['observation']['normalized_observation_rmse']
        f=evaluation['frozen_explicit']['horizons'][str(h)]['observation']['normalized_observation_rmse']
        d=evaluation['direct_k10']['horizons'][str(h)]['normalized_observation_rmse']
        out[str(h)]=dict(joint=j,frozen_explicit=f,direct_k10=d,reduction_vs_frozen_percent=100*(f-j)/f,
            percent_change_vs_direct=100*(j-d)/d,gap_closed=(f-j)/(f-d) if f>d else None,
            regime='in_training_horizon' if h<=10 else 'out_of_training_horizon')
    return out


def run_experiment(root,source_report,config=None,make_figures=True):
    root=Path(root).resolve(); report_path=root/'mujoco/reports'/REPORT; model_path=root/'mujoco/rl/models'/MODEL
    if report_path.exists() or model_path.exists(): raise FileExistsError('joint artifacts exist; no overwrite or retraining')
    started=time.monotonic(); context=load_context(root,source_report); cfg=dict(CONFIG if config is None else config)
    if cfg['horizon']!=10 or cfg['learning_rate']!=.0003 or cfg['seed']!=0: raise ValueError('fixed joint K10 seed0 lr.0003 only')
    training=train_joint(context['joint'],context['splits']['train'],context['splits']['val'],context['statistics'],cfg,model_path)
    # Test timestep values first read after Validation-only checkpoint selection.
    evaluation=evaluate_all(root,context,model_path); source=context['source']; hs=source['horizons_steps']
    report=dict(experiment='Deterministic Latent World Model v1',branch=BRANCH,base_branch='feat/latent-transition-multistep',base_commit=BASE,
        seed=0,source_report=context['source_path'],report_path=str(report_path),dataset=context['dataset'],dataset_regenerated=False,
        config=cfg,training=training,initialization=context['initialization'],model=dict(path=str(model_path),sha256=file_hash(model_path),bytes=model_path.stat().st_size,
            parameter_sha256=parameter_hash(evaluation['model'])),normalization=context['statistics'],normalization_sha256=json_hash(context['statistics']),
        window_manifests=source['train_validation_manifests'],test_start_manifest=source['test_start_manifest'],windows=evaluation['windows'],horizons_steps=hs,
        pi_magnitude_bins=source['pi_magnitude_bins'],joint=evaluation['joint'],frozen_explicit=evaluation['frozen_explicit'],direct_k10=evaluation['direct_k10'],
        comparison=comparison(evaluation,hs),future_observation_leakage=evaluation['future_observation_leakage'],selected_window=source['selected_window'],
        immutable_artifacts=context['immutable'],protocol='true causal prefix -> current rawz; T-only recursion with recorded actions; D readout no feedback',
        objective='uniform mean normalized observation MSE over all windows, k0..10 and7dimensions; no latent target or auxiliary loss',
        interpretation='deterministic latent dynamics core only, not Dreamer/full MBRL or world-model agent',
        limitations=['Single seed nominal simulation recorded actions and overlapping windows; K10/.4s training, no reward/policy/planning.',
            'Joint fine-tuning changes all3components and loss; component-specific causal attribution is not isolated.',
            'Old teacher-latent Decoder Floor is historical, not a Joint floor; joint reconstruction and latent consistency are offline diagnostics.',
            'Post-training Train latent std is diagnostic only; no fixed old latent coordinate constraint, latent teacher or consistency training.',
            'Raw yaw coordinate error unchanged; horizon cohorts differ but identical across models at each horizon; commonH50 cohort also saved.'],
        verification=dict(test_loaded_after_validation_selection=True,original_windows_normalization_reused=True,
            all_three_components_updated=all(training['parameter_hashes_before'][k]!=training['parameter_hashes_best'][k] for k in ('encoder','transition','decoder')),
            original_baseline_metrics_reproduced=True,immutable_artifacts_unchanged=all(file_hash(p)==h for p,h in context['immutable'].items())))
    if not all(report['verification'].values()): raise AssertionError('joint verification failed')
    if make_figures:
        from joint_latent_plots import render_figures
        report['figures']=render_figures(report,evaluation,context,report_path.parent)
    else: report['figures']=[]
    report['elapsed_seconds']=time.monotonic()-started
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(f'Saved {report_path}',flush=True); return report


def verify_experiment(report_path):
    p=Path(report_path).resolve(); root=p.parents[2]; report=json.loads(p.read_text()); context=load_context(root,report['source_report'])
    if report['immutable_artifacts']!=context['immutable'] or report['dataset']!=context['dataset']:
        raise ValueError('source provenance changed')
    for path,h in report['immutable_artifacts'].items():
        if file_hash(path)!=h: raise ValueError('immutable source hash mismatch')
    source=context['source']
    if report['horizons_steps']!=source['horizons_steps']: raise AssertionError('horizon changed')
    if report['window_manifests']!=source['train_validation_manifests'] or report['test_start_manifest']!=source['test_start_manifest']:
        raise AssertionError('original manifest changed')
    row=report['model']; net,stats,meta=load_joint(row['path']); cfg=report['config']
    if file_hash(row['path'])!=row['sha256'] or stats!=context['statistics'] or cfg!=meta['config'] or parameter_hash(net)!=row['parameter_sha256']:
        raise ValueError('joint model/statistics/hash mismatch')
    if (report['normalization']!=stats or report['normalization_sha256']!=json_hash(stats)
            or json_hash(report['normalization'])!=report['normalization_sha256']):
        raise ValueError('reported normalization/hash mismatch')
    losses=[r['validation_loss'] for r in report['training']['history']]
    if len(losses)!=cfg['epochs'] or int(np.argmin(losses))+1!=meta['best_epoch'] or meta['best_epoch']!=report['training']['best_epoch']:
        raise AssertionError('validation selection mismatch')
    compare(context['initialization'],report['initialization'])
    before=component_hashes(context['joint']); after=component_hashes(net)
    if (before!=meta['parameter_hashes_before'] or before!=report['training']['parameter_hashes_before']
            or after!=report['training']['parameter_hashes_best'] or any(before[k]==after[k] for k in before)):
        raise AssertionError('joint initialization/component hashes mismatch')
    loss=validation_loss(net,context['splits']['val'],stats,10,cfg['batch_size'])
    for value in [min(losses),meta['validation_loss'],report['training']['best_validation_loss']]:
        if not np.isclose(loss,value,atol=1e-10,rtol=1e-7): raise AssertionError('best Validation loss not reproduced')
    train_loss=validation_loss(net,context['splits']['train'],stats,10,cfg['batch_size'])
    if not np.isclose(train_loss,report['training']['best_checkpoint_train_loss'],atol=1e-10,rtol=1e-7):
        raise AssertionError('best Train loss not reproduced')
    expected=context['manifests']['train']['window_counts']['10']
    if any(r['windows_used']!=expected or any(r['gradient_norms'][k]['max']<=0 for k in before) for r in report['training']['history']):
        raise AssertionError('training window counts or module-gradient record mismatch')
    ev=evaluate_all(root,context,row['path'])
    for field in ['joint','frozen_explicit','direct_k10','future_observation_leakage','windows']: compare(ev[field],report[field])
    compare(comparison(ev,report['horizons_steps']),report['comparison'])
    return dict(pretrained_initialization_reproduced=True,all_three_updated=True,raw_latent_runtime=True,
        normalization_manifests_reused=True,validation_selection_reload=True,all_metrics_reproduced=True,
        future_observation_leakage_absent=True,original_artifacts_unchanged=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.verify: print(json.dumps(verify_experiment(root/'mujoco/reports'/REPORT),indent=2))
    else: run_experiment(root,root/'mujoco/reports/latent_transition_multistep_training_seed0.json')
