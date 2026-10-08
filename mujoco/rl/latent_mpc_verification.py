"""Read-only independent recomputation from frozen checkpoint + local traces."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from joint_autonomous_consistency_training import load_autonomous
from joint_latent_world_model import component_hashes
from latent_dynamics_data import file_hash,json_hash
from latent_mpc_evaluation import load_targets,summarize_episodes,action_ood,timing_summary,realized_prefix_check


def close(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b): raise AssertionError('recorded metric fields differ')
        for k in a: close(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b): raise AssertionError('recorded metric lengths differ')
        for x,y in zip(a,b): close(x,y)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10)
    elif a!=b: raise AssertionError('recorded literal differs')


def verify_episode_cohort(rows,targets):
    if len(rows)!=len(targets): raise AssertionError('missing/duplicate target evaluation')
    for row,(seed,target) in zip(rows,targets):
        if row['env_seed']!=seed or row['target']!=target: raise AssertionError('evaluation target order/content changed')
        if not isinstance(row['episode_steps'],int) or not 1<=row['episode_steps']<=375: raise AssertionError('invalid fixed task step count')
        if row['success']!=(row['termination_reason']=='success'): raise AssertionError('success/termination mismatch')
        if row['success'] and row['episode_steps']<5: raise AssertionError('cannot meet five-step hold before step5')
        if row['termination_reason']=='time_limit' and row['episode_steps']!=375: raise AssertionError('changed nominal timeout')
        # Physical failure may interrupt the last20-substep policy step.
        expected=row['episode_steps']*.04
        if row['termination_reason'] in ('success','time_limit'):
            if abs(row['episode_duration_s']-expected)>1e-7: raise AssertionError('simulated episode duration changed')
        elif not expected-.04<=row['episode_duration_s']<=expected+1e-7:
            raise AssertionError('invalid physical failure time')


def verify(root,report=None):
    root=Path(root); path=root/'mujoco/reports/latent_random_shooting_mpc_seed0.json'
    r=json.loads(path.read_text()) if report is None else report
    if set(r['results'])!={'zero','scripted','mpc'}: raise AssertionError('missing/extra controller cohort')
    for group in r['results'].values():
        if set(group)!={'benchmark','holdout'}: raise AssertionError('missing/extra target split')
        if any(len(value['episodes'])!=100 for value in group.values()): raise AssertionError('each fixed cohort requires100 episodes')
    torch.set_num_threads(1)
    m,stats,_=load_autonomous(r['model']['checkpoint']); m.eval().requires_grad_(False)
    before=component_hashes(m)
    if before!=r['model']['components_before'] or before!=r['model']['components_after']: raise AssertionError('parameter hash mismatch')
    if file_hash(r['model']['checkpoint'])!=r['model']['checkpoint_sha256']: raise AssertionError('checkpoint mutated')
    if json_hash(stats)!=r['normalization_sha256'] or stats!=r['normalization']: raise AssertionError('normalization mismatch')
    for p,h in r['immutable_artifacts'].items():
        if file_hash(p)!=h: raise AssertionError(f'immutable changed {p}')
    targets,meta=load_targets(root)
    for task in targets:
        if meta[task]['target_sha256']!=r['targets'][task]['target_sha256']: raise AssertionError('target hash differs')
    checks=0; allarrays=[]; evaluated_count=0
    for controller,group in r['results'].items():
        for task,value in group.items():
            verify_episode_cohort(value['episodes'],targets[task])
            evaluated_count+=len(value['episodes'])
            close(summarize_episodes(value['episodes']),value['summary'])
            if controller!='mpc': continue
            row=value['aggregates']
            if file_hash(row['path'])!=row['sha256']: raise AssertionError('aggregate hash changed')
            with np.load(row['path'],allow_pickle=False) as data: arrays={k:data[k] for k in data.files}
            allarrays.append(arrays)
            count=sum(ep['episode_steps'] for ep in value['episodes'])
            if len(arrays['actions'])!=count or np.max(np.abs(arrays['actions']))>1: raise AssertionError('action count/bounds changed')
            if (arrays['costs'][:,0]>arrays['costs'][:,1]+1e-9).any() or (arrays['costs'][:,0]>arrays['costs'][:,2]+1e-9).any(): raise AssertionError('selected cost not minimal')
            close(action_ood(arrays['actions'],r['train_action_statistics']),value['action_distribution'])
            for raw,key in [('planning_seconds','planning_time'),('encoding_seconds','encoding_time'),('decision_seconds','total_decision_time')]:
                if not len(arrays[raw])==count or not np.isfinite(arrays[raw]).all() or (arrays[raw]<=0).any(): raise AssertionError('missing/nonfinite timing')
                close(timing_summary(arrays[raw]),value[key])
            pred=[]; actual=[]
            for row in value['representative_local_traces']:
                if file_hash(row['path'])!=row['sha256']: raise AssertionError('selected trace hash differs')
                with np.load(row['path'],allow_pickle=False) as data: trace={k:data[k] for k in data.files}
                if not len(trace['observations'])==len(trace['actions'])+1: raise AssertionError('trajectory observation indexing differs')
                np.testing.assert_array_equal(trace['previous_actions'][0],np.zeros(4))
                np.testing.assert_array_equal(trace['previous_actions'][1:],trace['actions'][:-1])
                result=realized_prefix_check(m,stats,trace)
                np.testing.assert_array_equal(result['predicted'],trace['check_predicted'])
                np.testing.assert_array_equal(result['actual'],trace['check_actual'])
                altered=dict(trace,observations=np.zeros_like(trace['observations']))
                changed=realized_prefix_check(m,stats,altered)
                np.testing.assert_array_equal(changed['predicted'],result['predicted'])
                pred.append(result['predicted']); actual.append(result['actual']); checks+=len(result['starts'])
            error=np.concatenate(pred)-np.concatenate(actual)
            close(np.sqrt(np.mean((error/stats['obs']['std'])**2,axis=(0,2))).tolist(),value['realized_prefix_sanity']['normalized_observation_rmse'])
    close(timing_summary(np.concatenate([a['planning_seconds'] for a in allarrays])),r['overall_mpc']['planning_time'])
    close(timing_summary(np.concatenate([a['decision_seconds'] for a in allarrays])),r['overall_mpc']['total_decision_time'])
    close(action_ood(np.concatenate([a['actions'] for a in allarrays]),r['train_action_statistics']),r['overall_mpc']['action_ood'])
    if before!=component_hashes(m): raise AssertionError('read-only verification mutated parameters')
    return dict(verified=True,fixed_target_evaluations=evaluated_count,realized_prefix_windows_recomputed=checks,
                future_observation_perturbation_predictions_bit_identical=True,all_three_parameter_hashes_unchanged=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); args=parser.parse_args()
    print(json.dumps(verify(args.root),indent=2))
