"""Only new K10 Transition trained; E/D/oldT frozen. --verify is read-only."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash,json_hash
from latent_dynamics_models import load_model
from run_latent_objective_matched_control import load_reference
from run_latent_memory_ablation import parameter_hash
from run_latent_dynamics_multistep_eval import evaluate_model,compare
from run_explicit_latent_transition import test_context
from explicit_latent_models import freeze_encoder,attach_teacher,transition_manifest,load_component
from explicit_latent_evaluation import evaluate_explicit,predict_window,one_step_metrics
from latent_transition_multistep_training import train_transition,validation_loss
from latent_transition_multistep_evaluation import decoder_floor

BASE_COMMIT='126a3dbe20a01a2fd591c8e019560d64ef5e33a8'
BRANCH='feat/latent-transition-multistep'
REPORT='latent_transition_multistep_training_seed0.json'
MODEL='latent_transition_multistep10_mlp.pt'
CONFIG=dict(seed=0,epochs=60,batch_size=16,learning_rate=.001,horizon=10)


def check_teacher(context,split,episodes):
    prior=context['prior']; row=prior['manifests'][split]
    manifest=transition_manifest(episodes,[e['latent'] for e in episodes],context['dataset']['source_dataset_sha256'][split])
    if file_hash(row['path'])!=row['sha256'] or json.loads(Path(row['path']).read_text())!=manifest:
        raise ValueError('saved teacher manifest/sequence mismatch')


def check_frozen(context):
    for kind in ('encoder','decoder'):
        model=context[kind]
        if (parameter_hash(model)!=context['frozen_parameters'][kind]
                or any(p.requires_grad or p.grad is not None for p in model.parameters())):
            raise AssertionError(f'frozen {kind} changed or received gradient')


def load_context(root,source_report):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    source=Path(source_report).resolve(); prior=json.loads(source.read_text())
    for p,h in prior['immutable_artifacts'].items():
        if file_hash(p)!=h: raise ValueError('immutable source hash mismatch')
    context=load_reference(root,prior['source_report'])
    norm=prior['latent_normalization_file']; latent=json.loads(Path(norm['path']).read_text())
    if (file_hash(norm['path'])!=norm['sha256'] or latent!=prior['latent_statistics']
            or json_hash(latent)!=prior['latent_statistics_sha256']
            or context['statistics']!=prior['normalization']): raise ValueError('saved normalization mismatch')
    net,stats,_=load_model(prior['encoder']['checkpoint']['path']); encoder=freeze_encoder(net.encoder)
    decoder,saved,dlatent,_=load_component(prior['models']['decoder']['path']); freeze_encoder(decoder)
    old,saved_old,tlatent,_=load_component(prior['models']['transition']['path']); freeze_encoder(old)
    for kind in ('decoder','transition'):
        row=prior['models'][kind]
        if file_hash(row['path'])!=row['sha256']: raise ValueError('old component checkpoint hash mismatch')
    if (saved!=stats or saved_old!=stats or stats!=context['statistics']
            or dlatent!=latent or tlatent!=latent): raise ValueError('component normalization mismatch')
    hashes=dict(encoder=parameter_hash(encoder),decoder=parameter_hash(decoder))
    if hashes['encoder']!=prior['encoder']['parameter_sha256_before'] or hashes['decoder']!=prior['models']['decoder']['parameter_sha256']:
        raise ValueError('frozen parameter hash mismatch')
    teacher={s:attach_teacher(context['splits'][s],encoder,stats) for s in ('train','val')}
    immutable=dict(prior['immutable_artifacts']); immutable[str(source)]=file_hash(source)
    for row in [norm,prior['encoder']['checkpoint'],*prior['models'].values(),*prior['manifests'].values(),prior['test_start_manifest']]:
        immutable[row['path']]=row['sha256']
    context=dict(context,prior=prior,prior_path=str(source),encoder=encoder,decoder=decoder,old_transition=old,
                 teacher=teacher,latent_statistics=latent,frozen_parameters=hashes,immutable=immutable)
    for s in ('train','val'): check_teacher(context,s,teacher[s])
    check_frozen(context); return context


def evaluate_all(root,context,model_path):
    _,test,windows=test_context(root,context); check_teacher(context,'test',test)
    prior=context['prior']; stats=context['statistics']; latent=context['latent_statistics']
    horizons=prior['horizons_steps']; thresholds=prior['pi_magnitude_bins']['thresholds_m_s2']
    new,saved,nlatent,_=load_component(model_path); freeze_encoder(new)
    if saved!=stats or nlatent!=latent: raise ValueError('new normalization mismatch')
    results={}
    for kind,model in [('one_step',context['old_transition']),('multistep',new)]:
        results[kind]=evaluate_explicit(model,context['decoder'],test,stats,latent,thresholds,horizons)
    compare(results['one_step'],prior['latent_rollout'])
    one=one_step_metrics(new,test,stats,latent)
    compare(results['multistep']['horizons']['1']['latent'],{k:v for k,v in one.items() if k!='identity_transition_reference'})
    floor=decoder_floor(context['decoder'],test,stats,latent,thresholds,horizons)
    direct,_,_=load_model(prior['encoder']['checkpoint']['path']); freeze_encoder(direct)
    direct_metrics=evaluate_model(direct,'history',test,stats,thresholds,horizons)
    compare(direct_metrics,prior['direct_k10'])
    selected=prior['selected_window']; ep=test[selected['episode_id']]; t=selected['start']; h=selected['horizon']
    original=predict_window(context['encoder'],new,context['decoder'],ep,t,h,stats,latent)
    altered=copy.deepcopy(ep); altered['latent'][t+1:]+=1000
    corrupted=predict_window(context['encoder'],new,context['decoder'],altered,t,h,stats,latent)
    altered['latent']=altered['latent'][:t+1]; altered['obs']=altered['obs'][:t+1]
    altered['previous_action']=altered['previous_action'][:t+1]; del altered['next_obs']
    deleted=predict_window(context['encoder'],new,context['decoder'],altered,t,h,stats,latent)
    leakage=dict(corrupted_future_teacher_latent_unchanged=np.array_equal(original['latents'],corrupted['latents']),
                 deleted_future_teacher_and_observation_unchanged=np.array_equal(original['latents'],deleted['latents']),
                 deleted_future_decoded_observation_unchanged=np.array_equal(original['observations'],deleted['observations']))
    if not all(leakage.values()): raise AssertionError('future teacher leakage')
    check_frozen(context)
    return dict(results=results,decoder_floor=floor,direct_k10=direct_metrics,windows=windows,
                transition_one_step=one,future_teacher_leakage=leakage,test=test,new=new)


def comparisons(results,horizons):
    output={}
    for h in horizons:
        old=results['one_step']['horizons'][str(h)]; new=results['multistep']['horizons'][str(h)]
        row={}
        for part,field in [('latent','normalized_latent_rmse'),('observation','normalized_observation_rmse')]:
            a,b=old[part][field],new[part][field]
            row[part]=dict(one_step=a,multistep=b,absolute_delta=b-a,percent_change=100*(b-a)/a,reduction_percent=100*(a-b)/a)
        row['regime']='in_training_horizon' if h<=10 else 'out_of_training_horizon'; output[str(h)]=row
    return output


def run_experiment(root,source_report,config=None,make_figures=True):
    root=Path(root).resolve(); output=root/'mujoco/reports'; report_path=output/REPORT; model_path=root/'mujoco/rl/models'/MODEL
    manifest_paths={s:output/f'latent_transition10_{s}_windows_seed0.json' for s in ('train','val')}
    if any(p.exists() for p in [report_path,model_path,*manifest_paths.values()]): raise FileExistsError('artifacts exist; no overwrite/retraining')
    started=time.monotonic(); context=load_context(root,source_report); cfg=dict(CONFIG if config is None else config)
    if cfg['horizon']!=10: raise ValueError('only K10 experiment allowed')
    output.mkdir(parents=True,exist_ok=True); manifests={}
    for s,p in manifest_paths.items():
        data=context['manifests'][s]; p.write_text(json.dumps(data,indent=2)+'\n')
        manifests[s]=dict(path=str(p),sha256=file_hash(p),semantic_sha256=data['manifest_sha256'],window_count=data['window_counts']['10'])
    training=train_transition(context['teacher']['train'],context['teacher']['val'],context['statistics'],context['latent_statistics'],cfg,model_path)
    initial=context['prior']['training']['transition']['initial_parameter_sha256']
    if training['initial_parameter_sha256']!=initial: raise AssertionError('initial parameters differ from one-step Transition')
    check_frozen(context)
    # Test timestep values are first read after Validation-only checkpoint selection.
    evaluation=evaluate_all(root,context,model_path); horizons=context['prior']['horizons_steps']
    report=dict(experiment='Multi-Step Latent Transition Training',branch=BRANCH,base_branch='feat/explicit-latent-transition',
        base_commit=BASE_COMMIT,seed=0,source_report=context['prior_path'],report_path=str(report_path),
        dataset=context['dataset'],dataset_regenerated=False,config=cfg,training=training,
        model=dict(path=str(model_path),sha256=file_hash(model_path),bytes=model_path.stat().st_size,parameter_sha256=parameter_hash(evaluation['new'])),
        frozen_components={kind:dict(parameter_sha256_before=context['frozen_parameters'][kind],parameter_sha256_after=parameter_hash(context[kind]),requires_grad=False,
            artifact=context['prior']['encoder']['checkpoint'] if kind=='encoder' else context['prior']['models']['decoder']) for kind in ('encoder','decoder')},
        reference_initial_parameter_sha256=initial,normalization=context['statistics'],normalization_sha256=json_hash(context['statistics']),
        latent_statistics=context['latent_statistics'],latent_statistics_sha256=context['prior']['latent_statistics_sha256'],
        latent_normalization_file=context['prior']['latent_normalization_file'],teacher_manifests=context['prior']['manifests'],
        train_validation_manifests=manifests,test_start_manifest=context['prior']['test_start_manifest'],windows=evaluation['windows'],horizons_steps=horizons,
        pi_magnitude_bins=context['prior']['pi_magnitude_bins'],models=evaluation['results'],decoder_floor=evaluation['decoder_floor'],direct_k10=evaluation['direct_k10'],
        transition_one_step=evaluation['transition_one_step'],comparison=comparisons(evaluation['results'],horizons),
        future_teacher_leakage=evaluation['future_teacher_leakage'],selected_window=context['prior']['selected_window'],
        objective='uniform mean over all windows,10steps,64dimensions ((predicted latent-teacher latent)/unchanged Train latent std)^2',
        protocol='teacher z[t] initial only; fully differentiable T-only residual recursion with recorded actions; no future teacher input; frozen D readout never fed back',
        immutable_artifacts=context['immutable'],verification=dict(encoder_decoder_unchanged=True,same_initial_parameters=True,normalization_reused=True,
            test_loaded_after_validation_selection=True,old_transition_and_direct_metrics_reproduced=True,fixed_test_windows=True,
            immutable_artifacts_unchanged=all(file_hash(p)==h for p,h in context['immutable'].items())),
        limitations=['Single seed, nominal simulation, recorded future actions, overlapping windows.',
            'Only K10 latent objective; Encoder/Decoder fixed, no joint or observation loss, no policy/planning.',
            'Teacher latent is a frozen representation, not physical ground truth.',
            'Decoder Floor is an offline teacher readout reference, not a rigorous lower bound; nonlinear errors are not additive.',
            'K10 excludes last9starts per episode used by original K1; therefore comparison is prescribed objective/window-horizon intervention, not perfectly sample-matched.',
            'Checkpoint selection metrics are horizon-specific; short/long accuracy tradeoffs must both be reported.'])
    if not all(report['verification'].values()): raise AssertionError('immutable verification failed')
    if make_figures:
        from latent_transition_multistep_plots import render_figures
        report['figures']=render_figures(report,evaluation,context,output)
    else: report['figures']=[]
    report['elapsed_seconds']=time.monotonic()-started
    report_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); print(f'Saved {report_path}',flush=True); return report


def verify_experiment(report_path):
    p=Path(report_path).resolve(); root=p.parents[2]; report=json.loads(p.read_text()); context=load_context(root,report['source_report']); cfg=report['config']
    for path,h in report['immutable_artifacts'].items():
        if file_hash(path)!=h: raise ValueError('immutable artifact hash mismatch')
    if report['horizons_steps']!=context['prior']['horizons_steps']: raise AssertionError('evaluation horizon changed')
    for s,row in report['train_validation_manifests'].items():
        if file_hash(row['path'])!=row['sha256'] or json.loads(Path(row['path']).read_text())!=context['manifests'][s]: raise ValueError('window manifest mismatch')
    model=report['model']; net,stats,latent,meta=load_component(model['path'])
    if (file_hash(model['path'])!=model['sha256'] or stats!=context['statistics'] or latent!=context['latent_statistics']
            or meta['config']!=cfg or parameter_hash(net)!=model['parameter_sha256']): raise ValueError('model/config/normalization hash mismatch')
    history=report['training']['history']; losses=[r['validation_loss'] for r in history]
    if (len(history)!=cfg['epochs'] or int(np.argmin(losses))+1!=meta['best_epoch'] or meta['best_epoch']!=report['training']['best_epoch']
            or meta['initial_parameter_sha256']!=context['prior']['training']['transition']['initial_parameter_sha256']): raise AssertionError('initialization or validation selection mismatch')
    loss=validation_loss(net,context['teacher']['val'],stats,latent,cfg['horizon'],cfg['batch_size'])
    if not np.isclose(loss,min(losses),rtol=1e-7,atol=1e-10): raise AssertionError('selected checkpoint validation loss mismatch')
    ev=evaluate_all(root,context,model['path'])
    for field,key in [('results','models'),('decoder_floor','decoder_floor'),('direct_k10','direct_k10'),('transition_one_step','transition_one_step'),('future_teacher_leakage','future_teacher_leakage')]:
        compare(ev[field],report[key])
    compare(comparisons(ev['results'],report['horizons_steps']),report['comparison'])
    if (ev['windows']!=report['windows'] or report['test_start_manifest']!=context['prior']['test_start_manifest']
            or report['latent_statistics']!=context['latent_statistics'] or report['teacher_manifests']!=context['prior']['manifests']): raise AssertionError('saved source/windows/statistics mismatch')
    for kind,row in report['frozen_components'].items():
        if row['parameter_sha256_before']!=context['frozen_parameters'][kind] or row['parameter_sha256_after']!=context['frozen_parameters'][kind]:
            raise AssertionError('frozen component parameter hash changed')
    return dict(frozen_encoder_decoder=True,same_initialization=True,latent_statistics_reused=True,teacher_manifests=True,
        validation_selection_reload=True,fixed_test_windows=True,all_actual_metrics_reproduced=True,future_teacher_leakage_absent=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.verify: print(json.dumps(verify_experiment(root/'mujoco/reports'/REPORT),indent=2))
    else: run_experiment(root,root/'mujoco/reports/explicit_latent_transition_seed0.json')
