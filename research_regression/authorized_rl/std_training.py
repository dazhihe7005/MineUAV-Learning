"""Change only fixed initial log_std, with a declared fresh mean initialization."""
import math


def initialize_std_training(model,scheduler,manager,*,identity,seed,variant):
    import numpy as np
    import torch
    from lidar_integration.ppo_training_runner import _weights_sha
    from .std_profile import std_experiment,BASE_POLICY_SHA,PRIOR_POLICY_SHA
    from .actor_prior import initialize_actor_prior
    from .worker import save_checkpoint_probe
    p=std_experiment(variant)  # Reject unknown variant before any mutation.
    if (seed!=0 or model._last_obs is not None or scheduler.completed_policy_steps!=0
        or scheduler.selected_identities or model.get_vec_normalize_env() is not None
        or model.num_timesteps or model.policy.optimizer.state or model._n_updates):
        raise ValueError('std snapshot requires untouched seed0 model/environment')
    base=_weights_sha(model.policy)
    if base!=BASE_POLICY_SHA:raise ValueError('fresh seed0 policy differs from frozen controls')
    configuration=initialize_actor_prior(model) if variant=='PRIOR' else {'version':'FRESH_RANDOM'}
    mean_sha=_weights_sha(model.policy)
    if mean_sha!=(PRIOR_POLICY_SHA if variant=='PRIOR' else BASE_POLICY_SHA):
        raise ValueError('R3 mean initialization differs from frozen control')
    # No RNG draw, no clamp/optimizer/learning-rate change; std remains trainable.
    with torch.no_grad():model.policy.log_std.fill_(math.log(p['initial_std']))
    assigned=_weights_sha(model.policy)
    model._last_obs=model.get_env().reset()
    model._last_episode_starts=np.ones((model.n_envs,),dtype=bool)
    record,probe=save_checkpoint_probe(model,scheduler,manager,run_id=identity+'_STEP0',seed=seed)
    info={'kind':p['initialization'],'variant':variant,'base_random_policy_sha256':base,
        'base_mean_policy_sha256':mean_sha,'assigned_policy_sha256':assigned,'configuration':configuration,
        'initial_std':p['initial_std'],'actual_std':model.policy.log_std.exp().detach().cpu().tolist(),
        'std_remains_trainable':bool(model.policy.log_std.requires_grad),'optimizer_updates':0,
        'step0_usage':'EVALUATION_ONLY; never a resume or training initialization file'}
    event={'actual_step':0,'checkpoint':record,'weight_sha256':assigned,'scheduler':scheduler.scheduler_state()}
    return info,event,probe
