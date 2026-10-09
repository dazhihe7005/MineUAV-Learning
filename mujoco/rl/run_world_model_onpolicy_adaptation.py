"""Frozen shared-cohort offline evaluation; never runs adapted MPC policies."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from run_latent_random_shooting_mpc import write_json,execution_hardware
from run_onpolicy_adaptation_data import collection_context,MANIFEST,TARGETS,BASE,CHECKPOINT
from onpolicy_adaptation_final_data import FINAL_MANIFEST,require_completed_training
from onpolicy_adaptation_evaluation import evaluate_state,nominal_prediction
from onpolicy_adaptation_analysis import aggregate,ood_correlations,MODELS
from onpolicy_adaptation_plots import plot
from joint_autonomous_consistency_training import load_autonomous
from joint_latent_world_model import component_hashes,encode_episode,latent_summary
from latent_dynamics_data import file_hash,json_hash,load_dataset
from latent_dynamics_multistep import start_manifest
from later_ranking_core import ood_scores
from latent_mpc_support_evaluation import array_hash
from decision_fidelity_metrics import summary

REPORT='world_model_onpolicy_adaptation_seed0.json'

def nominal_run(root):
    """Independent nominal regression after training, before/alongside Final physics."""
    ctx=collection_context(root);root=ctx['root'];reports=root/'mujoco/reports';parts=reports/'world_model_onpolicy_adaptation_seed0_parts'
    training={s:json.loads((parts/f'training_{s}.json').read_text()) for s in ('replay','mpc_state')}
    require_completed_training(training,{s:file_hash(r['checkpoint']['path'])==r['checkpoint']['sha256'] for s,r in training.items()})
    original=json.loads((reports/'joint_latent_autonomous_consistency_v3_seed0.json').read_text())
    splits,dm=load_dataset(original['dataset']['source_directory'])
    for split in ('train','test'):
        with np.load(Path(original['dataset']['source_directory'])/f'{split}.npz',allow_pickle=False) as a:
            for ep in splits[split]:
                lo,hi=ep['source_rows'];ep['next_obs']=a['inputs'][lo+1:hi+1,:7].copy()
    manifest=start_manifest(splits['test'],[1,5,10,25,50],dm['source_dataset_sha256']['test'])
    if manifest!=original['windows']:raise AssertionError('nominal manifest changed')
    results={};models={'original':ctx['checkpoint'],**{s:r['checkpoint']['path'] for s,r in training.items()}}
    for name,path in models.items():
        model,stats,_=load_autonomous(path);model.eval().requires_grad_(False)
        before=component_hashes(model);prediction=nominal_prediction(model,splits['test'],stats)
        latent={}
        with torch.inference_mode():
            for split in ('train','test'):
                latent[split]=latent_summary(np.concatenate([encode_episode(model,e['obs'],e['previous_action'],stats).numpy() for e in splits[split]]))
        assert component_hashes(model)==before
        results[name]=dict(prediction=prediction,latent=latent,checkpoint_sha256=file_hash(path))
        print('Nominal',name,{h:r['normalized_observation_rmse'] for h,r in prediction.items()},flush=True)
    result=dict(models=results,test_manifest=original['test_start_manifest'],nominal_dataset=dm,
        source_sha256={p.name:file_hash(p) for p in (Path(__file__),Path(__file__).with_name('onpolicy_adaptation_evaluation.py'))})
    write_json(parts/'nominal_results.json',result);return result

def compact_training(r):
    out={k:v for k,v in r.items() if k!='steps'}
    out['gradient_norms']={k:summary([s['gradient_norms'][k] for s in r['steps']]) for k in ('encoder','transition','decoder')}
    out['consistency_gradient_ratios']={k:summary([s['consistency_to_observation_gradient_ratios'][k] for s in r['steps']]) for k in ('encoder','transition','decoder')}
    out['stability']={k:max(s[k] for s in r['steps']) for k in ('normalized_prediction_abs_max','latent_norm_max')}
    return out

def run(root):
    ctx=collection_context(root);root=ctx['root'];reports=root/'mujoco/reports';parts=reports/'world_model_onpolicy_adaptation_seed0_parts'
    data=json.loads((reports/MANIFEST).read_text());final=json.loads((reports/FINAL_MANIFEST).read_text())
    training={s:json.loads((parts/f'training_{s}.json').read_text()) for s in ('replay','mpc_state')}
    require_completed_training(training,{s:file_hash(r['checkpoint']['path'])==r['checkpoint']['sha256'] for s,r in training.items()})
    models={'original':ctx['checkpoint'],**{s:r['checkpoint']['path'] for s,r in training.items()}}
    if final['identity']['collection']!=ctx['identity']:raise AssertionError('Final data identity changed')
    geometry=json.loads((reports/'later_decision_train_distribution_seed0.json').read_text())
    if 'geometry' in geometry:geometry=geometry['geometry']
    rows=[];modelmeta={};nominal={};collapse={};started=time.monotonic()
    original=json.loads((reports/'joint_latent_autonomous_consistency_v3_seed0.json').read_text())
    splits,dm=load_dataset(original['dataset']['source_directory'])
    # Restore the actual next frame excluded from the original adapter input.
    for split in ('train','test'):
        with np.load(Path(original['dataset']['source_directory'])/f'{split}.npz',allow_pickle=False) as a:
            for ep in splits[split]:
                lo,hi=ep['source_rows'];ep['next_obs']=a['inputs'][lo+1:hi+1,:7].copy()
    manifest=start_manifest(splits['test'],[1,5,10,25,50],dm['source_dataset_sha256']['test'])
    if manifest!=original['windows'] or file_hash(original['test_start_manifest']['path'])!=original['test_start_manifest']['sha256']:
        raise AssertionError('nominal Test cohort changed')
    for name,path in models.items():
        model,stats,meta=load_autonomous(path);model.eval().requires_grad_(False);before=component_hashes(model)
        if stats!=ctx['stats'] or meta['initial_latent_std']!=load_autonomous(ctx['checkpoint'])[2]['initial_latent_std']:
            raise AssertionError('objective scales changed')
        modelmeta[name]=dict(path=str(path),sha256=file_hash(path),initial_parameter_hashes=training[name]['initial_hashes'] if name!='original' else before,
            parameter_hashes=before,parameter_count=sum(p.numel() for p in model.parameters()))
        for state in final['states']:
            if file_hash(state['path'])!=state['sha256']:raise ValueError('shared raw truth changed')
            with np.load(state['path'],allow_pickle=False) as raw:arrays={k:raw[k].copy() for k in raw.files}
            with torch.inference_mode():
                metrics,predcost=evaluate_state(model,stats,arrays,state)
                # State OOD is a common original-Train proxy, not cross-model raw latent comparisons.
                previous=np.vstack([np.zeros((1,4),np.float32),arrays['prefix_actions']])
                z=encode_episode(ctx['model'],arrays['prefix_observations'],previous,ctx['stats'])[-1].numpy()
            prev=previous[-1];ood=ood_scores(z,arrays['prefix_observations'][-1],prev,geometry)
            rows.append(dict(model=name,target_id=state['target_id'],stage=state['stage'],decision_index=state['decision_index'],
                metrics=metrics,state_ood=ood,predicted_cost_sha256=array_hash(predcost),candidate_sha256=state['candidate_sha256'],
                true_cost_sha256=state['true_cost_sha256'],state_array_sha256=state['sha256']))
        print(f'Evaluated {name}: {len(final["states"])} shared states',flush=True)
        nominal[name]=nominal_prediction(model,splits['test'],stats)
        latent={}
        with torch.inference_mode():
            for split in ('train','test'):
                latent[split]=latent_summary(np.concatenate([encode_episode(model,e['obs'],e['previous_action'],stats).numpy() for e in splits[split]]))
        collapse[name]=latent
        if component_hashes(model)!=before:raise AssertionError('evaluation mutated model')
    for h,value in [(1,.044057),(10,.104600),(25,.201413),(50,.308050)]:
        if not np.isclose(nominal['original'][str(h)]['normalized_observation_rmse'],value,atol=1e-6,rtol=0):
            raise AssertionError('original nominal regression did not reproduce')
    if file_hash(ctx['checkpoint'])!=CHECKPOINT or component_hashes(ctx['model'])!=ctx['before']:raise AssertionError('original mutated')
    result=dict(experiment='controlled on-policy visited-state + counterfactual candidate-action coverage adaptation',
        branch='feat/world-model-onpolicy-adaptation',base_commit=BASE,date=time.strftime('%Y-%m-%d'),models=modelmeta,
        normalization_sha256=json_hash(ctx['stats']),latent_consistency_scaling_sha256=training['replay']['scale_sha256'],
        setup=dict(train_targets=60,val_targets=20,final_targets=100,train_snapshots_per_source=800,val_snapshots_per_source=200,
            branches_per_snapshot=8,train_windows_per_source=6400,val_windows_per_source=1600,seed=0,K=10,
            updates=1000,batch_size=16,learning_rate=.0003,lambda_auto_cons=.1,N=512,H=10,
            selection='own independent Val L_obs only at fixed50..1000 steps; initial Val diagnostic only',
            initialization='same canonical v3 weights; fresh Adam states, not pre-joint or random initialization',
            loss='original v3: mean k0..10 normalized observation MSE +.1 mean k1..10 latent MSE / fixed initialEncoderTrainstd²'),
        provenance=dict(data_manifest=dict(path=str(reports/MANIFEST),sha256=file_hash(reports/MANIFEST)),
            targets=dict(path=str(reports/TARGETS),sha256=file_hash(reports/TARGETS)),final_manifest=dict(path=str(reports/FINAL_MANIFEST),sha256=file_hash(reports/FINAL_MANIFEST)),
            nominal_manifest=original['test_start_manifest'],nominal_dataset=dm,
            training_records={s:dict(path=str(parts/f'training_{s}.json'),sha256=file_hash(parts/f'training_{s}.json')) for s in training},
            train_distribution=dict(path=str(reports/'later_decision_train_distribution_seed0.json'),sha256=file_hash(reports/'later_decision_train_distribution_seed0.json')),
            source_sha256={p.name:file_hash(p) for p in Path(__file__).parent.glob('*onpolicy_adaptation*.py')}),
        training={s:compact_training(r) for s,r in training.items()},evaluation_rows=rows,aggregate=aggregate(rows),
        state_ood_correlations=ood_correlations(rows),nominal=nominal,latent_diagnostics=collapse,
        final_source_episodes=final['episodes'],adaptation_source_episodes=data['source_episodes'],elapsed_evaluation_seconds=time.monotonic()-started,
        hardware=execution_hardware(),closed_loop_adapted_model_evaluation=False,
        limitations=['single model/training seed','nominal simulation','800 finite-window filtered visited snapshots per source',
            'central first/later common original-MPC states, no adapted-policy closed-loop success test','fixed loss/cost/H/N',
            'counterfactual branches are not executed on-policy controller actions','separate source-specific Validation selection distributions'])
    result['conclusion']=dict(
        later_prediction_improvement_vs_replay_percent={s:100*(1-result['aggregate'][s]['models']['mpc_state']['prediction']['whole']['10']['mean']/
            result['aggregate'][s]['models']['replay']['prediction']['whole']['10']['mean']) for s in ('S2','S3')},
        nominal_h50_regression_vs_original_percent={m:100*(nominal[m]['50']['normalized_observation_rmse']/
            nominal['original']['50']['normalized_observation_rmse']-1) for m in ('replay','mpc_state')},
        accept_adapted_baseline=False,baseline='unchanged canonical v3',
        finding='MPC-state data improves later prediction relative to replay, but does not robustly repair decision ranking/best-tail utility; severe nominal forgetting violates retention goal.',
        interpretation='State coverage is relevant to prediction, but this fixed ordinary dynamics adaptation is not a sufficient remedy for decision-critical errors. Oracle H10 task progress remains limited in later states.',
        closed_loop_success='not evaluated for adapted models; no success improvement claim',
        next='One offline best-tail decision-cost supervision control, with nominal regression as an acceptance gate; do not execute from this experiment.')
    result['figures']=plot(result,reports);write_json(reports/REPORT,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);p.add_argument('--nominal-only',action='store_true')
    a=p.parse_args();nominal_run(a.root) if a.nominal_only else run(a.root)
