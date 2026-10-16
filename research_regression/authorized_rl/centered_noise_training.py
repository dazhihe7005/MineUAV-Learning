"""Fresh R16 initial-noise assignment; original Actor and algorithms unchanged."""
import copy,math
from .state_centered_policy import initialize_centered_prior,verify_centered_policy,canonical_policy_sha
from .centered_noise_profile import INITIAL_POLICY_SHA


def assign_noise_prior(model):
    import torch
    from lidar_integration.ppo_training_runner import _weights_sha
    # Original initializer enforces truly fresh seed0/zero steps/empty Adam.
    initialization=initialize_centered_prior(model)
    if _weights_sha(model.policy)!='20f5c226abc762e5aa4a9046ba400b2b0f7012b0da6715d14204b097cae06001':
        raise ValueError('R16 original deterministic mean/parameter initialization differs')
    with torch.no_grad():model.policy.log_std.fill_(math.log(.1))
    assigned=_weights_sha(model.policy)
    if assigned!=INITIAL_POLICY_SHA:raise ValueError('R16 fixed std01 initial identity differs')
    initialization.update(kind='FRESH_SEED0_HAND_PRIOR_CENTERED_INITIAL_STD01',assigned_policy_sha256=assigned,
        canonical_policy_sha256=canonical_policy_sha(model.policy),execution_profile='CENTERED_INITIAL_STD01_R16_V1',
        std_remains_trainable=bool(model.policy.log_std.requires_grad),actual_std=model.policy.log_std.exp().detach().cpu().tolist(),
        single_changed_parameter_from_R14_initial='log_std')
    return initialization


def initialize_noise_training(model,scheduler,manager,*,identity,seed):
    import numpy as np
    from .worker import save_checkpoint_probe
    if (seed!=0 or scheduler.completed_policy_steps or scheduler.selected_identities
        or model._last_obs is not None or model.get_vec_normalize_env() is not None):
        raise ValueError('R16 requires untouched fresh model/scheduler')
    initialization=assign_noise_prior(model)
    model._last_obs=model.get_env().reset();model._last_episode_starts=np.ones((model.n_envs,),dtype=bool)
    record,probe=save_checkpoint_probe(model,scheduler,manager,run_id=identity+'_STEP0',seed=seed)
    initialization['step0_usage']='NEW_EVALUATION_ONLY_NOT_TRAINING_INITIALIZATION_CHECKPOINT'
    event={'actual_step':0,'checkpoint':record,'weight_sha256':INITIAL_POLICY_SHA,'scheduler':scheduler.scheduler_state(),
           'actual_actor_parameterization':verify_centered_policy(model)}
    return initialization,event,probe


def annotate_noise_validation(result):
    actual=copy.deepcopy(result);step=actual['training_timesteps_before_after']
    if type(step) is not int or step<0:raise ValueError('invalid R16 evaluation step')
    if step==0:
        if actual['policy_sha256_before_after']!=INITIAL_POLICY_SHA:raise ValueError('R16 initial policy identity differs')
        actual['model_status']='ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED'
    actual.update(initialization='FRESH_SEED0_HAND_PRIOR_CENTERED_INITIAL_STD01',
        actor_parameterization='SAME_BACKGROUND_STATE_CENTERED_ACTOR_V1',
        newly_executed_episodes=len(actual.get('episodes',[])),reused_historical_evidence=False)
    return actual
