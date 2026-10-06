"""One frozen encoder, independently trained T/D; --verify is read-only."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import file_hash,json_hash
from latent_dynamics_models import load_model
from latent_dynamics_multistep import start_manifest
from run_latent_dynamics_multistep_eval import load_inputs,evaluate_model,compare
from run_latent_memory_ablation import parameter_hash
from run_latent_objective_matched_control import load_reference
from explicit_latent_models import (freeze_encoder,attach_teacher,latent_statistics,transition_manifest,
                                   train_component,load_component,validation_loss)
from explicit_latent_evaluation import (one_step_metrics,decoder_metrics,delta_magnitude,evaluate_explicit,predict_window)

BASE_COMMIT='259000a455956217318a242794f6e9e2e5481f22'
BRANCH='feat/explicit-latent-transition'
REPORT='explicit_latent_transition_seed0.json'
CONFIG=dict(seed=0,epochs=60,batch_size=16,learning_rate=.001)
MODEL_FILES=dict(transition='latent_transition_mlp.pt',decoder='latent_observation_decoder.pt')


def load_context(root,source_report):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    context=load_reference(root,source_report)
    artifact=context['source']['models']['multistep_history']['artifact']
    net,stats,_=load_model(artifact['path']); net.eval().requires_grad_(False)
    encoder=freeze_encoder(net.encoder); before=parameter_hash(encoder)
    teacher={s:attach_teacher(context['splits'][s],encoder,stats) for s in ('train','val')}
    lat_stats=latent_statistics([e['latent'] for e in teacher['train']])
    manifests={s:transition_manifest(teacher[s],[e['latent'] for e in teacher[s]],
                                    context['dataset']['source_dataset_sha256'][s]) for s in teacher}
    if parameter_hash(encoder)!=before: raise AssertionError('encoder changed during extraction')
    return dict(context,encoder=encoder,teacher=teacher,latent_statistics=lat_stats,manifests=manifests,
                encoder_artifact=artifact,encoder_parameter_sha256=before)


def test_context(root,context):
    frozen=context['frozen']; ev=load_inputs(root,frozen['source_report'])
    windows=start_manifest(ev['episodes'],frozen['horizons_steps'],context['dataset']['source_dataset_sha256']['test'])
    row=frozen['start_manifest']
    if (windows!=frozen['windows'] or windows!=json.loads(Path(row['path']).read_text())
            or file_hash(row['path'])!=row['sha256']): raise ValueError('Test start manifest hash/identity mismatch')
    teacher=attach_teacher(ev['episodes'],context['encoder'],context['statistics'])
    return ev,teacher,windows


def evaluate_all(root,context,model_paths):
    ev,test,windows=test_context(root,context); students={}
    for kind,path in model_paths.items():
        net,stats,latent,_=load_component(path)
        if stats!=context['statistics'] or latent!=context['latent_statistics']:
            raise ValueError('student normalization mismatch')
        students[kind]=net.eval().requires_grad_(False)
    # Reproduce existing direct K10 metrics, not a retrained comparison.
    direct,stats,_=load_model(context['encoder_artifact']['path']); direct.requires_grad_(False)
    frozen=context['frozen']; horizons=frozen['horizons_steps']; thresholds=frozen['pi_magnitude_bins']['thresholds_m_s2']
    direct_metrics=evaluate_model(direct,'history',ev['episodes'],stats,thresholds,horizons)
    for field,row in direct_metrics.items(): compare(row,context['source']['models']['multistep_history'][field])
    latent_stats=context['latent_statistics']; t=students['transition']; d=students['decoder']
    explicit=evaluate_explicit(t,d,test,stats,latent_stats,thresholds,horizons)
    one=one_step_metrics(t,test,stats,latent_stats); reconstruction=decoder_metrics(d,test,stats,latent_stats)
    # First latent step at every start must equal the dedicated one-step evaluator.
    compare(explicit['horizons']['1']['latent'],{k:v for k,v in one.items() if k!='identity_transition_reference'})
    selected=frozen['selected_window']; ep=test[selected['episode_id']]; start=selected['start']; h=selected['horizon']
    original=predict_window(context['encoder'],t,d,ep,start,h,stats,latent_stats)
    changed=copy.deepcopy(ep); changed['obs'][start+1:]+=1000; changed['next_obs'][:]=np.nan
    corrupted=predict_window(context['encoder'],t,d,changed,start,h,stats,latent_stats)
    changed['obs']=changed['obs'][:start+1]; changed['previous_action']=changed['previous_action'][:start+1]
    del changed['next_obs']; del changed['latent']
    deleted=predict_window(context['encoder'],t,d,changed,start,h,stats,latent_stats)
    leakage=dict(corrupted_future_observation_unchanged=np.array_equal(original['observations'],corrupted['observations']),
                 deleted_future_observation_unchanged=np.array_equal(original['observations'],deleted['observations']),
                 deleted_future_latent_unchanged=np.array_equal(original['latents'],deleted['latents']))
    if not all(leakage.values()): raise AssertionError('future-observation leakage')
    return dict(episodes=test,windows=windows,students=students,direct=direct,direct_metrics=direct_metrics,
        latent_rollout=explicit,transition_one_step=one,decoder_reconstruction=reconstruction,
        future_observation_leakage=leakage,selected_window=selected)


def run_experiment(root,source_report,config=None,make_figures=True):
    root=Path(root).resolve(); directory=root/'mujoco/reports'; path=directory/REPORT
    model_paths={k:root/'mujoco/rl/models'/p for k,p in MODEL_FILES.items()}
    manifest_paths={s:directory/f'explicit_latent_{s}_manifest_seed0.json' for s in ('train','val','test')}
    norm_path=directory/'explicit_latent_normalization_seed0.json'
    if any(p.exists() for p in [path,norm_path,*model_paths.values(),*manifest_paths.values()]):
        raise FileExistsError('explicit latent artifacts already exist; no overwrite/retraining')
    started=time.monotonic(); context=load_context(root,source_report); cfg=dict(CONFIG if config is None else config)
    directory.mkdir(parents=True,exist_ok=True); latent_stats=context['latent_statistics']; norm_path.write_text(json.dumps(latent_stats,indent=2)+'\n')
    training={}
    for kind in ('transition','decoder'):
        training[kind]=train_component(kind,context['teacher']['train'],context['teacher']['val'],
            context['statistics'],latent_stats,cfg,model_paths[kind])
        if (parameter_hash(context['encoder'])!=context['encoder_parameter_sha256']
                or any(p.grad is not None or p.requires_grad for p in context['encoder'].parameters())):
            raise AssertionError('encoder modified or unfrozen during independent training')
    # Both independent Validation selections are complete BEFORE reading Test values.
    evaluation=evaluate_all(root,context,model_paths)
    context['manifests']['test']=transition_manifest(evaluation['episodes'],[e['latent'] for e in evaluation['episodes']],
                                                   context['dataset']['source_dataset_sha256']['test'])
    manifests={}
    for split,manifest in context['manifests'].items():
        p=manifest_paths[split]; p.write_text(json.dumps(manifest,indent=2)+'\n')
        manifests[split]=dict(path=str(p),sha256=file_hash(p),semantic_sha256=manifest['manifest_sha256'],
            transition_count=manifest['transition_count'],decoder_frame_count=manifest['decoder_frame_count'])
    artifacts={k:dict(path=str(p),sha256=file_hash(p),bytes=p.stat().st_size,
                     parameter_sha256=parameter_hash(evaluation['students'][k])) for k,p in model_paths.items()}
    report=dict(experiment='Explicit Latent Transition',branch=BRANCH,base_branch='feat/latent-objective-matched-control',
        base_commit=BASE_COMMIT,seed=0,report_path=str(path),source_report=context['source_path'],
        dataset=context['dataset'],dataset_regenerated=False,config=cfg,training=training,models=artifacts,
        encoder=dict(checkpoint=context['encoder_artifact'],parameter_sha256_before=context['encoder_parameter_sha256'],
                     parameter_sha256_after=parameter_hash(context['encoder']),requires_grad=False,
                     teacher_rule='z[t] includes current observation and previous action, sequential frozen encoder from episode reset'),
        normalization=context['statistics'],normalization_sha256=json_hash(context['statistics']),
        latent_statistics=latent_stats,latent_statistics_sha256=json_hash(latent_stats),
        latent_normalization_file=dict(path=str(norm_path),sha256=file_hash(norm_path)),manifests=manifests,
        test_start_manifest=context['frozen']['start_manifest'],windows=evaluation['windows'],
        horizons_steps=context['frozen']['horizons_steps'],pi_magnitude_bins=context['frozen']['pi_magnitude_bins'],
        transition_objective='MSE(predicted Delta normalized latent, (teacher z[t+1]-teacher z[t])/Train latent std); no observation/PI/reward loss',
        decoder_objective='MSE(D(normalized teacher z[t]), (current recorded observation-Train obs mean)/Train obs std); independent of transition',
        residual_rule='z_hat_next = z_hat_current + Train latent std * T(normalized current latent,normalized recorded current action)',
        protocol='initial teacher z[t] from true prefix; thereafter T-only latent propagation, D readout; no future observation or GRU feedback',
        direct_k10=evaluation['direct_metrics'],latent_rollout=evaluation['latent_rollout'],
        transition_one_step=evaluation['transition_one_step'],decoder_reconstruction=evaluation['decoder_reconstruction'],
        delta_z_magnitude={s:delta_magnitude(eps) for s,eps in dict(context['teacher'],test=evaluation['episodes']).items()},
        future_observation_leakage=evaluation['future_observation_leakage'],selected_window=evaluation['selected_window'],
        immutable_artifacts=context['immutable'],
        verification=dict(encoder_parameters_unchanged=True,encoder_frozen_and_no_gradient=True,
            test_loaded_after_both_validation_selections=True,normalization_train_only=True,
            prior_test_windows_and_direct_metrics_reproduced=True,
            source_artifacts_unchanged=all(file_hash(p)==h for p,h in context['immutable'].items())),
        limitations=['Single seed/nominal recorded actions; no policy or planning evaluation.',
            'Teacher encoder latent is not physical ground-truth state; same frozen coordinates throughout.',
            'Independent one-step T and reconstruction D; no multi-step/joint optimization or encoder adaptation.',
            'Observation rollout error includes both transition and decoder errors; representation may omit dynamics variables.',
            'Raw wrapped-yaw observation convention unchanged; no new circular residual metric or clipping.'])
    if not all(report['verification'].values()): raise AssertionError('source artifact verification failed')
    if make_figures:
        from explicit_latent_plots import render_figures
        report['figures']=render_figures(report,evaluation,context,directory)
    else: report['figures']=[]
    report['elapsed_seconds']=time.monotonic()-started
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f'Saved {path}',flush=True); return report


def verify_experiment(report_path):
    path=Path(report_path).resolve(); root=path.parents[2]; report=json.loads(path.read_text())
    context=load_context(root,report['source_report']); cfg=report['config']
    for p,expected in report['immutable_artifacts'].items():
        if file_hash(p)!=expected: raise ValueError(f'immutable hash mismatch: {p}')
    if (context['latent_statistics']!=report['latent_statistics']
            or json_hash(context['latent_statistics'])!=report['latent_statistics_sha256']
            or context['statistics']!=report['normalization']): raise ValueError('normalization statistics mismatch')
    norm=report['latent_normalization_file']
    if file_hash(norm['path'])!=norm['sha256'] or json.loads(Path(norm['path']).read_text())!=context['latent_statistics']:
        raise ValueError('latent normalization file/hash mismatch')
    model_paths={}
    for kind,row in report['models'].items():
        if file_hash(row['path'])!=row['sha256']: raise ValueError('student checkpoint hash mismatch')
        net,stats,latent,meta=load_component(row['path']); model_paths[kind]=row['path']
        losses=[r['validation_loss'] for r in report['training'][kind]['history']]
        if (len(losses)!=cfg['epochs'] or int(np.argmin(losses))+1!=meta['best_epoch']
                or meta['best_epoch']!=report['training'][kind]['best_epoch'] or meta['config']!=cfg):
            raise AssertionError('independent validation best selection mismatch')
        loss=validation_loss(net,kind,context['teacher']['val'],stats,latent,cfg['batch_size'])
        if not np.isclose(loss,min(losses),rtol=1e-7,atol=1e-10): raise AssertionError('saved validation loss mismatch')
    if report['horizons_steps']!=context['frozen']['horizons_steps']: raise AssertionError('evaluation horizon changed')
    evaluation=evaluate_all(root,context,model_paths)
    context['manifests']['test']=transition_manifest(evaluation['episodes'],[e['latent'] for e in evaluation['episodes']],
        context['dataset']['source_dataset_sha256']['test'])
    for split,row in report['manifests'].items():
        if file_hash(row['path'])!=row['sha256'] or json.loads(Path(row['path']).read_text())!=context['manifests'][split]:
            raise ValueError('teacher transition manifest hash/identity mismatch')
    for field in ('direct_metrics','latent_rollout','transition_one_step','decoder_reconstruction','future_observation_leakage'):
        key='direct_k10' if field=='direct_metrics' else field
        compare(evaluation[field],report[key])
    if (evaluation['windows']!=report['windows'] or report['test_start_manifest']!=context['frozen']['start_manifest']
            or parameter_hash(context['encoder'])!=report['encoder']['parameter_sha256_before']
            or report['encoder']['parameter_sha256_after']!=report['encoder']['parameter_sha256_before']):
        raise AssertionError('encoder or canonical windows changed')
    return dict(encoder_immutable=True,train_only_latent_normalization=True,teacher_sequence_hashes=True,
        independent_validation_selection=True,checkpoint_reload=True,all_frozen_and_new_metrics_reproduced=True,
        fixed_test_windows=True,no_future_observation_leakage=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify',action='store_true'); args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    if args.verify: print(json.dumps(verify_experiment(root/'mujoco/reports'/REPORT),indent=2))
    else: run_experiment(root,root/'mujoco/reports/latent_dynamics_multistep_training_seed0.json')
