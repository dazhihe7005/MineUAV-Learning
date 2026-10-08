"""Exactly two frozen MPC conditions; support projection is the only intervention."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from run_latent_random_shooting_mpc import context,planner_seed,execution_hardware,condition_identity,write_json,direction_check
from latent_dynamics_data import file_hash,json_hash
from joint_latent_world_model import component_hashes
from latent_mpc_action_support import compute_support
from latent_mpc_support_evaluation import InstrumentedSupportMPC,ProjectionAccumulator,support_metrics,array_hash,window_manifest,verify_pairing,prediction_metrics
from latent_mpc_evaluation import evaluate_episode,summarize_episodes,realized_prefix_check,timing_summary,distribution
from latent_mpc_verification import verify_episode_cohort
from ppo_pi_env import MineUAVPIEnv

BASE='7d360778a0e401a091c77811a78968805e813154'
BRANCH='feat/latent-mpc-action-support'
REPORT='latent_mpc_train_supported_action_control_seed0.json'
CONDITIONS=('unconstrained','train_supported')
SPLITS=('benchmark','holdout')


def validate_cache(saved,identity,targets):
    if saved['identity']!=identity: raise ValueError('cached support/source/hardware identity differs')
    verify_episode_cohort(saved['result']['episodes'],targets)


def evaluate_condition(ctx,condition,split,support,parts):
    env=MineUAVPIEnv(reward_version='v2',render_mode=None)
    box=(support['lower'],support['upper']) if condition=='train_supported' else None
    planner=InstrumentedSupportMPC(ctx['model'],ctx['stats'],env.action_space.low,env.action_space.high,0,box)
    rows=[]; aggregates={k:[] for k in ('actions','planning_seconds','encoding_seconds','decision_seconds','costs','epsilon_hashes')}
    offsets=[0]; projection=ProjectionAccumulator(); representatives=[]; predicted=[]; actual=[]; windows=[]
    started=time.monotonic()
    try:
        for i,(seed,target) in enumerate(ctx['targets'][split]):
            row,trace=evaluate_episode(env,'mpc',seed,target,planner,planner_seed(split,i))
            row.update(target_index=i,first_observation_sha256=array_hash(trace['observations'][0]),**planner.first,
                epsilon_count=len(planner.epsilon_hashes),epsilon_stream_sha256=json_hash(planner.epsilon_hashes),
                projection=planner.projection.summary())
            action=row['first_action']
            row['first_action_standardized_norm']=float(np.linalg.norm((np.asarray(action)-ctx['stats']['action']['mean'])/ctx['stats']['action']['std']))
            row['first_action_support_departure']=bool(np.any(np.asarray(action)<np.asarray(support['lower'])-1e-7)|np.any(np.asarray(action)>np.asarray(support['upper'])+1e-7))
            rows.append(row); projection.merge(planner.projection)
            for key in aggregates:
                aggregates[key].append(np.asarray(planner.epsilon_hashes,dtype='U64') if key=='epsilon_hashes' else trace[key])
            offsets.append(offsets[-1]+row['episode_steps'])
            if i<3:
                check=realized_prefix_check(ctx['model'],ctx['stats'],trace)
                manifest=window_manifest(split,i,row['episode_steps'])
                if [w['decision_index'] for w in manifest]!=check['starts'].tolist(): raise AssertionError('fixed prefix windows changed')
                windows.extend(manifest); predicted.append(check['predicted']); actual.append(check['actual'])
                path=parts/f'{condition}_{split}_episode{i}.npz'
                np.savez_compressed(path,**trace,check_starts=check['starts'],check_predicted=check['predicted'],check_actual=check['actual'])
                representatives.append(dict(episode_index=i,path=str(path),sha256=file_hash(path),window_count=len(manifest)))
            if (i+1)%10==0:
                print(f'{condition} {split}: {i+1}/100 success={sum(r["success"] for r in rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally: env.close()
    arrays={k:np.concatenate(v) for k,v in aggregates.items()}
    path=parts/f'{condition}_{split}_aggregates.npz'
    np.savez_compressed(path,**arrays,offsets=np.asarray(offsets),projection_histogram=projection.hist)
    result=dict(summary=summarize_episodes(rows),episodes=rows,
        action_support=support_metrics(arrays['actions'],ctx['train_stats'],support),projection=projection.summary(),
        planning_time=timing_summary(arrays['planning_seconds']),total_decision_time=timing_summary(arrays['decision_seconds']),
        costs={k:distribution(arrays['costs'][:,j]) for j,k in enumerate(('selected_min','candidate_median','candidate_mean'))},
        prediction_sanity=prediction_metrics(np.concatenate(predicted),np.concatenate(actual),ctx['stats']),
        window_manifest=windows,window_manifest_sha256=json_hash(windows),representative_local_traces=representatives,
        aggregate=dict(path=str(path),sha256=file_hash(path)),elapsed_seconds=time.monotonic()-started)
    return result


def load_arrays(result):
    artifact=result['aggregate']
    if file_hash(artifact['path'])!=artifact['sha256']: raise AssertionError('local aggregate changed')
    with np.load(artifact['path'],allow_pickle=False) as raw: return {k:raw[k] for k in raw.files}


def paired_rows(result,arrays):
    rows=[]
    for i,row in enumerate(result['episodes']):
        hashes=arrays['epsilon_hashes'][arrays['offsets'][i]:arrays['offsets'][i+1]].tolist()
        if len(hashes)!=row['epsilon_count'] or json_hash(hashes)!=row['epsilon_stream_sha256']: raise AssertionError('epsilon record changed')
        rows.append(dict(row,epsilon_hashes=hashes))
    return rows


def sanity_by_key(result):
    values={}
    for row in result['representative_local_traces']:
        if file_hash(row['path'])!=row['sha256']: raise AssertionError('sanity trace changed')
        with np.load(row['path'],allow_pickle=False) as raw:
            for start,pred,actual in zip(raw['check_starts'],raw['check_predicted'],raw['check_actual']):
                values[(row['episode_index'],int(start))]=(pred.copy(),actual.copy())
    return values


def run(root,make_figures=True):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ctx=context(root); directory=ctx['root']/'mujoco/reports'
    parts=directory/'latent_mpc_train_supported_action_control_seed0_parts'; parts.mkdir(parents=True,exist_ok=True)
    train=ctx['root']/'mujoco/rl/datasets/pi_hidden_state_seed0/train.npz'
    support=compute_support(train,ctx['train_stats']['source_sha256'])
    write_json(directory/'train_action_support_central95.json',support)
    # Include all executable diagnostics/control helpers, frozen environment dependencies and original timing provenance.
    immutable=dict(ctx['immutable'])
    for name in ('latent_mpc_core.py','latent_mpc_evaluation.py','latent_mpc_action_support.py','latent_mpc_support_evaluation.py',
                 'run_latent_random_shooting_mpc.py','joint_latent_world_model.py','explicit_latent_models.py',
                 'joint_autonomous_consistency_training.py','latent_dynamics_data.py',Path(__file__).name):
        path=Path(__file__).with_name(name); immutable[str(path)]=file_hash(path)
    identity=condition_identity(dict(checkpoint_sha256=file_hash(ctx['checkpoint']),normalization_sha256=ctx['normalization_sha256'],
        support_sha256=support['sha256'],targets={k:v['target_sha256'] for k,v in ctx['target_meta'].items()},N=512,H=10,seed=0,
        experiment='Train central95% support-only paired control'),immutable,execution_hardware())
    results={condition:{} for condition in CONDITIONS}
    for split in SPLITS:
        for condition in CONDITIONS:
            path=parts/f'{condition}_{split}.json'
            if path.exists():
                saved=json.loads(path.read_text()); validate_cache(saved,identity,ctx['targets'][split]); result=saved['result']
                print(f'Reuse complete {condition} {split}',flush=True)
            else:
                result=evaluate_condition(ctx,condition,split,support,parts)
                write_json(path,dict(identity=identity,result=result))
            results[condition][split]=result
    paired={}; common={}; overall={}
    for split in SPLITS:
        u=results['unconstrained'][split]; s=results['train_supported'][split]
        paired[split]=verify_pairing(paired_rows(u,load_arrays(u)),paired_rows(s,load_arrays(s)))
        keys_u=sanity_by_key(u); keys_s=sanity_by_key(s); keys=sorted(set(keys_u)&set(keys_s))
        common[split]=dict(window_keys=[list(k) for k in keys],window_keys_sha256=json_hash(keys),
            caveat='same episode/decision identities, but each condition has its own closed-loop states/actions',
            **{c:prediction_metrics(np.array([v[k][0] for k in keys]),np.array([v[k][1] for k in keys]),ctx['stats'])
               for c,v in [('unconstrained',keys_u),('train_supported',keys_s)]})
    old=json.loads((directory/'latent_random_shooting_mpc_seed0.json').read_text())
    reproduced={split:results['unconstrained'][split]['summary']==old['results']['mpc'][split]['summary'] for split in SPLITS}
    if not all(reproduced.values()): raise AssertionError('UNCONSTRAINED changed original scientific summary; stop before interpretation')
    for condition in CONDITIONS:
        rows=[r for split in SPLITS for r in results[condition][split]['episodes']]
        arrays=[load_arrays(results[condition][split]) for split in SPLITS]
        overall[condition]=dict(summary=summarize_episodes(rows),
            action_support=support_metrics(np.concatenate([a['actions'] for a in arrays]),ctx['train_stats'],support),
            planning_time=timing_summary(np.concatenate([a['planning_seconds'] for a in arrays])),
            total_decision_time=timing_summary(np.concatenate([a['decision_seconds'] for a in arrays])),
            first_decision_target_response=direction_check(rows))
    after=component_hashes(ctx['model'])
    if after!=ctx['before'] or not all(file_hash(p)==h for p,h in immutable.items()): raise AssertionError('frozen/source parameter mutation')
    report=dict(experiment='Paired Train-supported action constraint control',branch=BRANCH,base_branch='feat/latent-random-shooting-mpc',base_commit=BASE,seed=0,
        model=dict(checkpoint=str(ctx['checkpoint']),checkpoint_sha256=file_hash(ctx['checkpoint']),components_before=ctx['before'],components_after=after,frozen=True),
        train_support=support,noise_normalization=ctx['train_stats'],normalization=ctx['stats'],normalization_sha256=ctx['normalization_sha256'],
        targets=ctx['target_meta'],hardware=identity['execution_hardware'],immutable_artifacts=immutable,evaluation_identity=identity,
        configuration=dict(N=512,H=10,noise_std=(.5*np.asarray(ctx['stats']['action']['std'])).tolist(),
            sampling='original recursive random walk; env clipping first; Supported additionally clips to fixed central95% Train box before next increment; anchors projected too',
            cost=old['configuration']['cost'],replanning=old['configuration']['replan'],
            pairing='original SeedSequence([0,split_index,episode_index]) episode seed; identical generator and one510x10x4draw everydecision; SHA256 actual scaled epsilon verifies matching decision index',
            projection_quantiles='fixed1e-4 histogram empirical-rank upper estimates, bounded error1e-4; exact mean/counts; no raw candidate arrays saved',
            metric_tolerance=1e-7,action_bounds=old['configuration']['action_bounds'],action_scale_physical=old['configuration']['action_scale_physical'],
            prediction_window_rule='first3 fixed episodes/split regardless of outcome; legal10step starts stride25; actions actually executed after replanning'),
        results=results,overall=overall,paired_rng_and_initial_state=paired,common_window_prediction_check=common,
        original_unconstrained_summary_exactly_reproduced=reproduced,
        verification=dict(all400episodes_retained=True,checkpoint_and_parameters_unchanged=True,no_training=True,
            no_PI_reward_task_change=True,paired_raw_epsilon=True),
        limitations=['Coordinate central95% box is a coarse marginal support proxy, NOT state-action-sequence support or strict OOD boundary.',
            'Later physical states differ; pairing is reset/noise source and common initial state, not identical trajectories.',
            'One fixed model/planner and nominal simulation, no uncertainty/cost/N/H/std tuning.',
            'Prediction sanity uses fixed first3episodes per split, actual executed prefixes; common-available windows also reported.',
            'Ordinary yaw RMSE uses unchanged wrapped representation; branch cuts may inflate it.',
            'Projection magnitude quantiles have explicit1e-4 histogram bound; exact mean/counts.',
            'Planning timer includes projection/hash audit; full decision includes encoder and diagnostic aggregation, no hard realtime guarantee.'])
    if make_figures:
        from latent_mpc_support_plots import render_figures
        report['figures']=render_figures(report,directory)
    write_json(directory/REPORT,report)
    print(f'Saved {directory/REPORT}',flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args(); run(args.root,not args.no_figures)
