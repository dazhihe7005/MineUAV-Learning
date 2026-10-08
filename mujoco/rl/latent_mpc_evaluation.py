"""Unchanged PI task, archived targets and read-only planner diagnostics."""
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import torch

from latent_dynamics_data import file_hash
from joint_latent_world_model import predict_latent
from test_env_scripted_policy import scripted_action
from velocity_command_controller import ACTION_SCALE


def target_hash(rows):
    return hashlib.sha256(json.dumps(rows,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def load_targets(root):
    root=Path(root)
    source=root/'mujoco/reports/ppo_waypoint_pi_observable_lowstd_seed0.json'
    report=json.loads(source.read_text()); targets={}; meta={}
    for task in ('benchmark','holdout'):
        path=Path(report['milestones']['100k']['evaluation'][task]['integral_trace_path'])
        trace=json.loads(path.read_text())
        rows=list(zip(trace['waypoint_seeds'],trace['targets']))
        digest=target_hash(rows)
        if len(rows)!=100 or digest!=report['waypoint_sha256'][task]: raise ValueError('archived target hash/count changed')
        targets[task]=rows
        meta[task]=dict(source_report=str(source),source_report_sha256=file_hash(source),
                        trace_source=str(path),trace_sha256=file_hash(path),target_sha256=digest,
                        count=100,targets=[dict(env_seed=s,target=t) for s,t in rows])
    return targets,meta


def distribution(values):
    x=np.asarray(values,np.float64)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all(): raise ValueError('finite nonempty distribution required')
    return dict(mean=float(x.mean()),median=float(np.median(x)),std=float(x.std()),
                p95=float(np.quantile(x,.95)),min=float(x.min()),max=float(x.max()))


def timing_summary(seconds):
    x=np.asarray(seconds,np.float64)*1000
    out={f'{k}_ms':v for k,v in distribution(x).items()}
    return dict(out,decision_count=len(x),fraction_within_40ms=float(np.mean(x<=40)),
                p95_within_40ms=bool(np.quantile(x,.95)<=40),max_within_40ms=bool(x.max()<=40))


def action_ood(actions,train):
    a=np.asarray(actions,np.float64)
    if a.ndim!=2 or a.shape[1]!=4 or not len(a) or not np.isfinite(a).all(): raise ValueError('finite selected commands required')
    standardized=(a-np.asarray(train['mean']))/train['std']
    outside=(a<np.asarray(train['q025']))|(a>np.asarray(train['q975']))
    return dict(standardized_norm=distribution(np.linalg.norm(standardized,axis=1)),
                outside_train_95_any_dimension_fraction=float(outside.any(1).mean()),
                outside_train_95_element_fraction=float(outside.mean()),
                outside_train_95_per_dimension_fraction=outside.mean(0).tolist(),
                normalized_action_mean=a.mean(0).tolist(),normalized_action_std=a.std(0).tolist(),
                normalized_action_min=a.min(0).tolist(),normalized_action_max=a.max(0).tolist(),
                definition='coordinatewise central Train95% interval, NOT joint95% coverage; standardized L2 norm, not Mahalanobis')


def crossings(observations):
    obs=np.asarray(observations); distances=np.linalg.norm(obs[:,:3],axis=1)
    near=np.flatnonzero(distances<.2)
    if not len(near) or distances[near[0]]<=1e-9: return 0
    axis=obs[near[0],:3]/distances[near[0]]; side=1; count=0
    for error in obs[near[0]:,:3]:
        value=float(error@axis)
        if side>0 and value<=-.02: count+=1; side=-1
        elif side<0 and value>=.02: count+=1; side=1
    return count


def evaluate_episode(env,controller,env_seed,target,planner=None,planner_seed=0):
    if controller not in ('zero','scripted','mpc'): raise ValueError('unknown frozen controller')
    obs,info=env.reset(seed=env_seed,options={'target_position':target})
    if not np.array_equal(info['target_position_m'],target): raise ValueError('evaluation target changed')
    if controller=='mpc':
        if planner is None: raise ValueError('MPC requires frozen planner')
        planner.reset(planner_seed)
    previous=np.zeros(4,np.float32)
    observations=[obs.copy()]; actions=[]; priors=[]; latents=[]; times=[]; costs=[]; indices=[]; terminals=[]
    near_actual=[]; near_command=[]; rewards=[]; encoding_times=[]; decision_times=[]
    import time
    for _ in range(env.max_episode_steps):
        if controller=='mpc':
            begin=time.perf_counter(); z=planner.encode_current(obs,previous)
            encoded=time.perf_counter(); result=planner.plan(z,previous)
            encoding_times.append(encoded-begin); decision_times.append(time.perf_counter()-begin)
            action=result['action']  # NOT sequence execution: one real step only.
            latents.append(z.cpu().numpy()[0].copy())
            times.append(result['planning_seconds'])
            costs.append([result['selected_cost'],result['candidate_cost_median'],result['candidate_cost_mean']])
            indices.append(result['index']); terminals.append(result['terminal_observation'])
        elif controller=='scripted': action=scripted_action(obs)
        else: action=np.zeros(4,np.float32)
        action=np.asarray(action,np.float32)
        if action.shape!=(4,) or not np.isfinite(action).all() or np.any(action<env.action_space.low) or np.any(action>env.action_space.high):
            raise ValueError('invalid executed command')
        if np.linalg.norm(obs[:3])<.1:
            near_actual.append(float(np.linalg.norm(obs[3:6])))
            near_command.append(float(np.linalg.norm(action[:3]*ACTION_SCALE[:3])))
        priors.append(previous.copy()); actions.append(action.copy())
        obs,reward,terminated,truncated,info=env.step(action)
        observations.append(obs.copy()); rewards.append(float(reward)); previous=action.copy()
        if terminated or truncated: break
    else: raise AssertionError('unchanged task did not terminate by timeout')
    a=np.asarray(actions); count=len(a); reason=info['termination_reason']; success=reason=='success'
    row=dict(env_seed=int(env_seed),target=list(target),planner_seed=int(planner_seed) if controller=='mpc' else None,
        success=success,termination_reason=reason,episode_steps=count,episode_duration_s=float(env.data.time),
        final_distance_m=float(info['distance_m']),final_speed_m_s=float(info['speed_m_s']),
        completion_time_s=float(env.data.time) if success else None,
        near_target_steps=len(near_actual),near_actual_sum=float(sum(near_actual)),near_command_sum=float(sum(near_command)),
        crossing_count=crossings(observations),action_saturation_element_count=int((np.abs(a)>=.95).sum()),
        action_saturation_fraction=float((np.abs(a)>=.95).mean()),undiscounted_return=float(sum(rewards)),
        discounted_return=float(np.dot(rewards,.99**np.arange(count))),
        action_temporal_variance=a.var(0).tolist(),
        first_action=a[0].tolist(),mean_action=a.mean(0).tolist(),
        selected_zero_count=indices.count(0),selected_repeat_previous_count=indices.count(1))
    if controller=='mpc':
        row.update(planning_time=timing_summary(times),encoding_time=timing_summary(encoding_times),
                   total_decision_time=timing_summary(decision_times),
                   cost={k:distribution(np.asarray(costs)[:,j]) for j,k in enumerate(('selected_min','candidate_median','candidate_mean'))})
    trace=dict(observations=np.asarray(observations),actions=a,previous_actions=np.asarray(priors),
               latents=np.asarray(latents,np.float32),planning_seconds=np.asarray(times),
               encoding_seconds=np.asarray(encoding_times),decision_seconds=np.asarray(decision_times),
               costs=np.asarray(costs),selected_indices=np.asarray(indices),candidate_terminals=np.asarray(terminals))
    return row,trace


def summarize_episodes(rows):
    reasons=Counter(r['termination_reason'] for r in rows); n=len(rows)
    count=sum(r['episode_steps'] for r in rows); near=sum(r['near_target_steps'] for r in rows)
    times=[r['completion_time_s'] for r in rows if r['success']]
    return dict(episodes=n,successes=reasons['success'],success_rate=reasons['success']/n,
        failures=n-reasons['success']-reasons['time_limit'],timeouts=reasons['time_limit'],termination_reasons=dict(reasons),
        mean_final_distance_m=float(np.mean([r['final_distance_m'] for r in rows])),
        final_distance_distribution=distribution([r['final_distance_m'] for r in rows]),
        mean_completion_time_s=float(np.mean(times)) if times else None,
        mean_episode_duration_s=float(np.mean([r['episode_duration_s'] for r in rows])),
        mean_episode_steps=count/n,near_target_steps=near,
        near_target_actual_speed_m_s=sum(r['near_actual_sum'] for r in rows)/near if near else None,
        near_target_command_speed_m_s=sum(r['near_command_sum'] for r in rows)/near if near else None,
        crossing_rate=float(np.mean([r['crossing_count']>0 for r in rows])),
        mean_crossing_count=float(np.mean([r['crossing_count'] for r in rows])),
        action_saturation_fraction=sum(r['action_saturation_element_count'] for r in rows)/(4*count),
        mean_undiscounted_return=float(np.mean([r['undiscounted_return'] for r in rows])),
        mean_within_episode_action_temporal_variance=np.mean([r['action_temporal_variance'] for r in rows],axis=0).tolist())


@torch.inference_mode()
def realized_prefix_check(model,stats,trace,stride=25):
    """Only realized actions enter T. True future observations are targets only.

    This is an offline open-loop forecast under realized receding-horizon actions,
    NOT accuracy of the original selected future sequence after it was replaced.
    """
    actions=trace['actions']; starts=np.arange(0,max(0,len(actions)-10+1),stride,dtype=int)
    if not len(starts):
        return dict(starts=starts,predicted=np.empty((0,2,7)),actual=np.empty((0,2,7)))
    z=torch.from_numpy(trace['latents'][starts])
    a=torch.from_numpy(np.stack([actions[s:s+10] for s in starts]))
    out=predict_latent(model,z,a,stats)['normalized_observations'][:,[1,10]].cpu().numpy()
    pred=out*np.asarray(stats['obs']['std'])+stats['obs']['mean']
    actual=trace['observations'][starts[:,None]+np.array([1,10])]
    return dict(starts=starts,predicted=pred,actual=actual)
