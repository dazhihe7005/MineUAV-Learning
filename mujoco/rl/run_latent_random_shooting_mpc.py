"""Fixed-protocol evaluation only. No optimizer, tuning or learned policy."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch

from joint_autonomous_consistency_training import load_autonomous
from joint_latent_world_model import component_hashes
from latent_dynamics_data import file_hash,json_hash,episodes_from_arrays
from latent_mpc_core import RandomShootingMPC
from latent_mpc_evaluation import (load_targets,evaluate_episode,summarize_episodes,
    realized_prefix_check,action_ood,timing_summary,distribution)
from ppo_pi_env import MineUAVPIEnv

BASE='e3ce35a0c736b3332c70dddd9c7d03cc20627b0d'
BRANCH='feat/latent-random-shooting-mpc'
REPORT='latent_random_shooting_mpc_seed0.json'


def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def context(root):
    root=Path(root).resolve(); source=root/'mujoco/reports/joint_latent_autonomous_consistency_v3_seed0.json'
    previous=json.loads(source.read_text()); checkpoint=root/'mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt'
    if file_hash(checkpoint)!=previous['model']['sha256']: raise ValueError('seed0 reference checkpoint changed')
    model,stats,_=load_autonomous(checkpoint); model.eval().requires_grad_(False)
    if json_hash(stats)!=previous['normalization_sha256']: raise ValueError('normalization changed')
    dataset=previous['dataset']; train=Path(dataset['source_directory'])/'train.npz'
    if file_hash(train)!=dataset['source_dataset_sha256']['train']: raise ValueError('Train action source changed')
    manifest=train.with_name('manifest.json')
    if file_hash(manifest)!=dataset['source_manifest_sha256']: raise ValueError('original dataset manifest changed')
    with np.load(train,allow_pickle=False) as a:
        episodes=episodes_from_arrays(a)
    actions=np.concatenate([e['action'] for e in episodes]).astype(np.float64)
    # Use exactly the same intra-episode Train rows as original normalization.
    if not np.allclose(actions.std(0),stats['action']['std'],rtol=1e-10,atol=1e-12): raise ValueError('Train action std mismatch')
    train_stats=dict(stats['action'],q025=np.quantile(actions,.025,axis=0).tolist(),q975=np.quantile(actions,.975,axis=0).tolist(),
                     count=len(actions),split='train',source_sha256=file_hash(train),rule='central per-axis2.5..97.5 percentiles; last unpaired episode command excluded')
    targets,target_meta=load_targets(root)
    paths=[checkpoint,source,train,manifest,root/'mujoco/rl/mine_uav_env.py',root/'mujoco/rl/ppo_pi_env.py',
        root/'mujoco/rl/audit_velocity_pi.py',root/'mujoco/control/velocity_command_controller.py',
        root/'mujoco/control/hover_controller.py',root/'mujoco/control/control_allocator.py',
        root/'mujoco/control/position_controller.py',root/'mujoco/models/mine_uav_dynamics_v2.xml',
        root/'mujoco/models/rotor_actuators.xml',root/'mujoco/models/rotor_config.json',
        root/'mujoco/reports/dynamics_v2_report.json',root/'mujoco/rl/test_env_scripted_policy.py']
    immutable={str(p):file_hash(p) for p in paths}
    for value in target_meta.values():
        for k,h in [('source_report','source_report_sha256'),('trace_source','trace_sha256')]: immutable[value[k]]=value[h]
    return dict(root=root,model=model,stats=stats,checkpoint=checkpoint,immutable=immutable,
                before=component_hashes(model),target_meta=target_meta,targets=targets,train_stats=train_stats,
                normalization_sha256=json_hash(stats),dataset_manifest_sha256=dataset['source_manifest_sha256'])


def planner_seed(task,index):
    return int(np.random.SeedSequence([0,('benchmark','holdout').index(task),index]).generate_state(1)[0])


def execution_hardware():
    import mujoco
    import gymnasium
    return dict(device='CPU',processor=platform.processor(),platform=platform.platform(),python_version=platform.python_version(),
        numpy_version=np.__version__,mujoco_version=mujoco.__version__,gymnasium_version=gymnasium.__version__,
        cpu_description=Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0].strip(': \t') if Path('/proc/cpuinfo').exists() else platform.machine(),
        torch_version=str(torch.__version__),torch_threads=torch.get_num_threads(),
        inference='512 candidates batched; inference_mode; terminal decode only; incremental1-step GRU realhistory',
        timing_scope='planning includes sampling,T recursion,terminal decoding,cost,argmin/transfer; separate totaldecision includes causal encoder; excludes simulator stepping, disk writes, plotting; no sleep to enforce realtime')


def condition_identity(identity,immutable,hardware):
    # Timing provenance travels WITH the result; reject reuse on other runtime.
    return dict(identity,immutable_execution_dependencies=dict(immutable),execution_hardware=dict(hardware))


def evaluate_condition(ctx,controller,task,parts):
    env=MineUAVPIEnv(reward_version='v2',render_mode=None)
    planner=RandomShootingMPC(ctx['model'],ctx['stats'],env.action_space.low,env.action_space.high,0) if controller=='mpc' else None
    rows=[]; actions=[]; timing=[]; encoded=[]; decisions=[]; costs=[]; checks=[]; representative=[]
    started=time.monotonic()
    try:
        for i,(seed,target) in enumerate(ctx['targets'][task]):
            row,trace=evaluate_episode(env,controller,seed,target,planner,planner_seed(task,i))
            row['target_index']=i; rows.append(row); actions.append(trace['actions'])
            if controller=='mpc':
                timing.extend(trace['planning_seconds']); encoded.extend(trace['encoding_seconds']); decisions.extend(trace['decision_seconds']); costs.extend(trace['costs'])
                # Identity chosen before outcomes, no successful-only cherry-picking.
                if i<3:
                    check=realized_prefix_check(ctx['model'],ctx['stats'],trace)
                    checks.append(check)
                    raw=parts/f'mpc_{task}_episode{i}_trace.npz'
                    np.savez_compressed(raw,**trace,check_starts=check['starts'],check_predicted=check['predicted'],check_actual=check['actual'])
                    representative.append(dict(target_index=i,path=str(raw),sha256=file_hash(raw),steps=row['episode_steps'],
                        window_count=len(check['starts']),rule='first3 target identities, starts every25 policy steps with10 actual future steps remaining'))
            if (i+1)%10==0:
                print(f'{controller} {task}: {i+1}/100 successes={sum(r["success"] for r in rows)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    finally: env.close()
    result=dict(summary=summarize_episodes(rows),episodes=rows,action_distribution=action_ood(np.concatenate(actions),ctx['train_stats']),
        elapsed_seconds=time.monotonic()-started)
    if controller=='mpc':
        pred=np.concatenate([c['predicted'] for c in checks]); actual=np.concatenate([c['actual'] for c in checks])
        error=pred-actual; normal=error/ctx['stats']['obs']['std']
        sanity=dict(windows=len(pred),horizons=[1,10],normalized_observation_rmse=np.sqrt(np.mean(normal**2,axis=(0,2))).tolist(),
                    physical_per_dimension_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                    physical_per_dimension_mae=np.mean(np.abs(error),axis=0).tolist(),
                    protocol='offline pure T rollout from recorded decision latent under ACTUALLY EXECUTED subsequent action prefix; true future observation only target; not original selected candidate terminal accuracy')
        raw=parts/f'mpc_{task}_aggregates.npz'
        np.savez_compressed(raw,actions=np.concatenate(actions),planning_seconds=timing,encoding_seconds=encoded,decision_seconds=decisions,costs=costs)
        result.update(planning_time=timing_summary(timing),encoding_time=timing_summary(encoded),total_decision_time=timing_summary(decisions),
            costs={k:distribution(np.asarray(costs)[:,j]) for j,k in enumerate(('selected_min','candidate_median','candidate_mean'))},
            selected_zero_fraction=sum(r['selected_zero_count'] for r in rows)/len(timing),
            selected_repeat_previous_fraction=sum(r['selected_repeat_previous_count'] for r in rows)/len(timing),
            realized_prefix_sanity=sanity,representative_local_traces=representative,
            aggregates=dict(path=str(raw),sha256=file_hash(raw)))
    return result


def direction_check(rows):
    # Across-target variation, not a counterfactual model-use proof.
    target=np.array([r['target'] for r in rows]); target[:,2]-=1
    first=np.array([r['first_action'] for r in rows]); mean=np.array([r['mean_action'] for r in rows])
    by_axis={}
    for axis,name in enumerate(('x','y','z')):
        positive=target[:,axis]>0; negative=target[:,axis]<0
        by_axis[name]=dict(first_command_target_correlation=float(np.corrcoef(target[:,axis],first[:,axis])[0,1]) if np.std(first[:,axis])>0 else None,
                          first_command_sign_agreement_fraction=float(np.mean(np.sign(first[:,axis])==np.sign(target[:,axis]))),
                          positive_target_mean_first_action=float(first[positive,axis].mean()) if positive.any() else None,
                          negative_target_mean_first_action=float(first[negative,axis].mean()) if negative.any() else None)
    return dict(per_axis=by_axis,first_action_across_target_std=first.std(0).tolist(),mean_action_across_target_std=mean.std(0).tolist(),
                mean_within_episode_temporal_variance=np.mean([r['action_temporal_variance'] for r in rows],axis=0).tolist(),
                caveat='state/target responsiveness and nonconstant actions do not prove counterfactual world-model usefulness or superiority to a tuned heuristic')


def run(root,make_figures=True):
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    ctx=context(root); output=ctx['root']/'mujoco/reports'; parts=output/'latent_random_shooting_mpc_seed0_parts'; parts.mkdir(parents=True,exist_ok=True)
    hardware=execution_hardware()
    identity=condition_identity(dict(checkpoint_sha256=file_hash(ctx['checkpoint']),normalization_sha256=ctx['normalization_sha256'],target_hashes={k:v['target_sha256'] for k,v in ctx['target_meta'].items()},
        core_sha256=file_hash(Path(__file__).with_name('latent_mpc_core.py')),evaluation_sha256=file_hash(Path(__file__).with_name('latent_mpc_evaluation.py')),
        runner_sha256=file_hash(__file__),n=512,h=10,seed=0),ctx['immutable'],hardware)
    results={}
    for controller in ('zero','scripted','mpc'):
        results[controller]={}
        for task in ('benchmark','holdout'):
            part=parts/f'{controller}_{task}.json'
            if part.exists():
                saved=json.loads(part.read_text())
                if saved['identity']!=identity: raise ValueError(f'cached condition identity differs: {part}')
                actual=[(r['env_seed'],r['target']) for r in saved['result']['episodes']]
                if target_hash_for_validation(actual)!=ctx['target_meta'][task]['target_sha256']: raise ValueError('cached evaluation target order changed')
                result=saved['result']; print(f'Reuse complete {controller} {task}',flush=True)
            else:
                result=evaluate_condition(ctx,controller,task,parts)
                write_json(part,dict(identity=identity,result=result))
            results[controller][task]=result
    after=component_hashes(ctx['model'])
    if after!=ctx['before'] or not all(file_hash(p)==h for p,h in ctx['immutable'].items()): raise AssertionError('frozen model/environment/source mutation')
    allrows=[r for task in results['mpc'].values() for r in task['episodes']]
    arrays=[]
    for task in results['mpc'].values():
        aggregate=task['aggregates']
        if file_hash(aggregate['path'])!=aggregate['sha256']: raise ValueError('raw summary arrays changed')
        with np.load(aggregate['path'],allow_pickle=False) as a: arrays.append({k:a[k].copy() for k in a.files})
    report=dict(experiment='Frozen latent random-shooting MPC v1',branch=BRANCH,base_branch='feat/joint-v1-v3-multiseed',base_commit=BASE,seed=0,
        model=dict(checkpoint=str(ctx['checkpoint']),checkpoint_sha256=file_hash(ctx['checkpoint']),components_before=ctx['before'],components_after=after,
            reference='seed0 v3 fixed before planning; no Test-best seed selection',total_parameters=sum(p.numel() for p in ctx['model'].parameters()),frozen=True),
        targets=ctx['target_meta'],train_action_statistics=ctx['train_stats'],normalization=ctx['stats'],normalization_sha256=ctx['normalization_sha256'],dataset_manifest_sha256=ctx['dataset_manifest_sha256'],
        configuration=dict(candidates=512,horizon=10,policy_dt_s=.04,horizon_seconds=.4,environment='MineUAVPIEnv RewardV2 nominal',timeout_seconds=15,
            success='distance<0.10m AND speed<0.15m/s for5 consecutive policy steps',action_bounds=[[-1]*4,[1]*4],action_scale_physical=[1.5,1.5,1.,1.],
            sampling='candidate0=allzero; candidate1=repeat previous executed normalized action; other510 a[-1]=previous, a[k]=clip(a[k-1]+epsilon[k],envbounds); epsilon iid N(0,(0.5 Train action std)^2)',
            epsilon_std_normalized=(.5*np.asarray(ctx['stats']['action']['std'])).tolist(),
            cost='J=||error_xyz_H||^2 +0.5||velocity_xyz_H||^2 +0.1*yaw_error_H^2 +0.05*mean_k||normalized a[k]-a[k-1]||^2; a[-1]=previous executed; hand-designed NOT RL reward',
            replan='incremental true-history encoder with previous executed normalized action; pureT recursion; execute first action only',
            planner_seed_rule='NumPy SeedSequence([0,task_index,target_index]).generate_state(1)[0]; fresh RNG/GRU per episode'),
        hardware=hardware,
        results=results,overall_mpc=dict(summary=summarize_episodes(allrows),
            planning_time=timing_summary(np.concatenate([a['planning_seconds'] for a in arrays])),
            total_decision_time=timing_summary(np.concatenate([a['decision_seconds'] for a in arrays])),
            action_ood=action_ood(np.concatenate([a['actions'] for a in arrays]),ctx['train_stats']),
            target_direction_response=direction_check(allrows)),
        immutable_artifacts=ctx['immutable'],evaluation_identity=identity,
        verification=dict(checkpoint_unchanged=True,all_three_module_hashes_unchanged=True,environment_PI_reward_unchanged=True,
            all_targets_completed=all(v['summary']['episodes']==100 for group in results.values() for v in group.values()),
            no_model_training=True,no_policy_training=True,original_targets_reused=True),
        limitations=['Single fixed seed0 model/planner configuration; no objective/horizon/noise/weight sweep.',
            'Nominal simulation; model trained under recorded actions, planner changes action/state visitation.',
            'Hand-designed terminal cost and short0.4s horizon; no uncertainty or learned reward.',
            'Only selected-action OOD diagnostics; within per-axis action95% does not guarantee sequence/state in distribution.',
            'Realized-action-prefix prediction check uses fixed first3 episodes per task, not original selected future-sequence accuracy.',
            'Success alone does not prove useful model causality; no counterfactual model ablation and no learned policy/planning optimization.'])
    if not all(report['verification'].values()): raise AssertionError('evaluation verification failed')
    if make_figures:
        from latent_mpc_plots import render_figures
        report['figures']=render_figures(report,output)
    write_json(output/REPORT,report)
    print(f'Saved {output/REPORT}',flush=True)
    return report


def target_hash_for_validation(rows):
    from latent_mpc_evaluation import target_hash
    return target_hash(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args(); run(args.root,not args.no_figures)
