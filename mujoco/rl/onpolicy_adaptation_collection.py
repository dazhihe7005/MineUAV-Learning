"""Original controller visits plus counterfactual actions; no controller changes."""
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from decision_fidelity_snapshot import capture,fingerprint
from decision_fidelity_rollouts import true_rollout
from latent_mpc_core import RandomShootingMPC,sample_candidates
from test_env_scripted_policy import scripted_action
from latent_mpc_support_evaluation import array_hash
from latent_dynamics_data import file_hash,json_hash
from onpolicy_adaptation_data import GENERATION_SEED,branch_indices,valid_branch

def collect_visited(ctx,env,source,target,keep_all_candidates=False):
    if source not in ('replay','mpc_state'): raise ValueError('fixed state sources only')
    obs,info=env.reset(seed=target['env_seed'],options={'target_position':target['target']})
    np.testing.assert_array_equal(info['target_position_m'],target['target'])
    seed=int(np.random.SeedSequence([GENERATION_SEED,target['global_index']]).generate_state(1)[0])
    planner=RandomShootingMPC(ctx['model'],ctx['stats'],env.action_space.low,env.action_space.high,seed)
    rng=np.random.default_rng(seed); previous=np.zeros(4,np.float32)
    observations=[obs.copy()]; actions=[]; decisions=[]
    for t in range(env.max_episode_steps):
        snapshot=capture(env,previous)
        if source=='mpc_state':
            z=planner.encode_current(obs,previous); result=planner.plan(z,previous)
            candidates=result['candidates']; action=result['action']; selected=result['index']; costs=result['costs']
        else:
            candidates=sample_candidates(previous,ctx['stats']['action']['std'],env.action_space.low,env.action_space.high,rng)
            action=np.asarray(scripted_action(obs),np.float32); selected=None; costs=None
        ids=np.arange(512) if keep_all_candidates else branch_indices(target['global_index'],t)
        decisions.append(dict(snapshot=snapshot,snapshot_sha256=fingerprint(snapshot),decision_index=t,
            candidates=candidates[ids].copy(),candidate_indices=ids,candidate_set_sha256=array_hash(candidates),
            previous_action=previous.copy(),selected_index=selected,predicted_costs=costs))
        obs,_,terminated,truncated,info=env.step(action)
        actions.append(action.copy()); observations.append(obs.copy()); previous=action.copy()
        if terminated or truncated: break
    else: raise AssertionError('source episode did not terminate under unchanged timeout')
    return dict(observations=np.asarray(observations,np.float32),actions=np.asarray(actions,np.float32),decisions=decisions,
        episode=dict(target,source=source,planner_seed=seed,episode_steps=len(actions),termination_reason=info['termination_reason'],
            success=info['termination_reason']=='success',action_sha256=array_hash(np.asarray(actions)),
            observation_sha256=array_hash(np.asarray(observations)),final_snapshot_sha256=fingerprint(capture(env,previous))))

def collect_branches(ctx,env,source,target,quota,directory):
    trajectory=collect_visited(ctx,env,source,target)
    n=len(trajectory['actions']); rng=np.random.default_rng(np.random.SeedSequence([GENERATION_SEED,23,target['global_index']]))
    order=rng.permutation(n); selected=[]; attempts=[]; directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    for t in order:
        d=trajectory['decisions'][int(t)]; outs=[true_rollout(env,d['snapshot'],a) for a in d['candidates']]
        validity=[valid_branch(o) for o in outs]
        attempts.append(dict(decision_index=int(t),valid=all(validity),valid_branches=sum(validity),
            termination_reasons=dict(Counter(o['termination_reason'] or 'none' for o in outs)),
            executed_steps=[o['executed_steps'] for o in outs],invalid_count=sum(o['invalid_state'] for o in outs)))
        if not all(validity): continue
        # Full simulation/controller state repeat, not merely observation match.
        repeat=true_rollout(env,d['snapshot'],d['candidates'][0])
        np.testing.assert_array_equal(repeat['observations'],outs[0]['observations'])
        if repeat['final_snapshot_sha256']!=outs[0]['final_snapshot_sha256']: raise AssertionError('exact branch restore changed')
        path=directory/f'{source}_{target["split"]}_{target["global_index"]:03d}_state{len(selected):02d}.npz'
        np.savez_compressed(path,prefix_observations=trajectory['observations'][:t+1],prefix_actions=trajectory['actions'][:t],
            branch_actions=d['candidates'],branch_observations=np.stack([o['observations'] for o in outs]),
            executed_steps=np.asarray([o['executed_steps'] for o in outs]),invalid_state=np.asarray([o['invalid_state'] for o in outs]))
        selected.append(dict(target_id=target['target_id'],source=source,split=target['split'],decision_index=int(t),
            episode_steps=n,realized_progress=int(t)/max(n-1,1),snapshot_sha256=d['snapshot_sha256'],
            original_candidate_set_sha256=d['candidate_set_sha256'],candidate_indices=d['candidate_indices'].tolist(),
            branch_actions_sha256=array_hash(d['candidates']),branch_end_snapshot_sha256=[o['final_snapshot_sha256'] for o in outs],
            branch_termination_reasons=[o['termination_reason'] for o in outs],path=str(path),sha256=file_hash(path),branches=8))
        if len(selected)==quota: break
    if len(selected)!=quota:
        raise ValueError(f'unmatched valid snapshot quota {source}/{target["target_id"]}: {len(selected)}/{quota}; retain attempts, do not reduce budget')
    return dict(episode=trajectory['episode'],states=selected,attempts=attempts,requested_snapshot_count=quota,
        search_order_sha256=array_hash(order),valid_windows=len(selected)*8,invalid_windows=sum(8-r['valid_branches'] for r in attempts))
