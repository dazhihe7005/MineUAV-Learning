"""Recorded-command true rollout and frozen pure-latent prediction; no policy."""
import numpy as np
import torch
from latent_mpc_core import planning_cost
from test_env_scripted_policy import scripted_action
from decision_fidelity_snapshot import capture,restore,fingerprint


def _rollout(env,snapshot,actions=None):
    previous=restore(env,snapshot); initial=env._get_obs().copy()
    observations=[initial]; executed=[]; used=[]; done=False; reason=None; info=None
    for k in range(10):
        action=(scripted_action(observations[-1]) if actions is None else np.asarray(actions[k],np.float32))
        used.append(action.copy())
        if not done:
            obs,_,terminated,truncated,info=env.step(action)
            executed.append(action.copy()); previous=action.copy()
            done=terminated or truncated; reason=info['termination_reason'] if done else None
            observations.append(obs.copy())
        else:
            # Preserve termination, no subsequent physics/controller stepping.
            observations.append(observations[-1].copy())
    sequence=np.asarray(used,np.float32); obs=np.asarray(observations,np.float32)
    cost,_=planning_cost(obs[-1:],sequence[None],snapshot.previous_action)
    return dict(actions=sequence,executed_actions=np.asarray(executed,np.float32),observations=obs,cost=float(cost[0]),
        terminal_distance=float(info['distance_m']),terminal_speed=float(info['speed_m_s']),
        distance_reduction=float(snapshot.environment['previous_distance']-info['distance_m']),
        physical_failure=reason not in (None,'success','time_limit'),invalid_state=reason=='nonfinite_state',
        termination_reason=reason,executed_steps=len(executed),final_snapshot_sha256=fingerprint(capture(env,previous)))


def true_rollout(env,snapshot,actions):
    a=np.asarray(actions,np.float32)
    if a.shape!=(10,4) or not np.isfinite(a).all() or (np.abs(a)>1).any():
        raise ValueError('exactly10 legal recorded actions4 required')
    return _rollout(env,snapshot,a)


def scripted_reference(env,snapshot):
    return _rollout(env,snapshot,None)


@torch.inference_mode()
def frozen_prediction(model,stats,z,actions):
    a=np.asarray(actions,np.float32)
    if a.ndim!=3 or a.shape[1:]!=(10,4) or (np.abs(a)>1).any() or not np.isfinite(a).all():
        raise ValueError('fixedH10 legal candidate actions only')
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError('world model must be completely frozen/eval')
    z=torch.as_tensor(z,dtype=torch.float32)
    if z.shape!=(1,64): raise ValueError('one initial encoder latent required')
    action=torch.from_numpy(a)
    normalized=(action-torch.tensor(stats['action']['mean'],dtype=torch.float32))/torch.tensor(stats['action']['std'],dtype=torch.float32)
    current=z.expand(len(a),-1)
    for k in range(10): current=current+model.transition(current,normalized[:,k])
    return model.decoder(current).numpy()*np.asarray(stats['obs']['std'])+stats['obs']['mean']
