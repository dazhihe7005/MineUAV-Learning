"""R14 narrow hooks; original scheduler/checkpoint/PPO algorithms reused."""
import copy
from .state_centered_policy import initialize_centered_prior, verify_centered_policy
from .centered_training_profile import INITIAL_POLICY_SHA


def initialize_centered_training(model,scheduler,manager,*,identity,seed):
    import numpy as np
    from lidar_integration.ppo_training_runner import _weights_sha
    from .worker import save_checkpoint_probe
    if (seed != 0 or scheduler.completed_policy_steps or scheduler.selected_identities
            or model._last_obs is not None or model.get_vec_normalize_env() is not None):
        raise ValueError('centered training requires untouched fresh model/scheduler')
    initialization = initialize_centered_prior(model)
    assigned = _weights_sha(model.policy)
    if assigned != INITIAL_POLICY_SHA:
        raise ValueError('R14 new functional policy initialization differs')
    model._last_obs = model.get_env().reset()
    model._last_episode_starts = np.ones((model.n_envs,),dtype=bool)
    record,probe = save_checkpoint_probe(model,scheduler,manager,run_id=identity+'_STEP0',seed=seed)
    initialization.update(kind='FRESH_SEED0_HAND_PRIOR_CENTERED_FUNCTION',assigned_policy_sha256=assigned,
        execution_profile='CENTERED_ACTOR_R14_V1',std_remains_trainable=bool(model.policy.log_std.requires_grad),
        actual_std=model.policy.log_std.exp().detach().cpu().tolist(),
        step0_usage='NEW_EVALUATION_ONLY_NOT_TRAINING_INITIALIZATION_CHECKPOINT')
    event = {'actual_step':0,'checkpoint':record,'weight_sha256':assigned,'scheduler':scheduler.scheduler_state(),
             'actual_actor_parameterization':verify_centered_policy(model)}
    return initialization,event,probe


def annotate_centered_validation(result):
    actual = copy.deepcopy(result)
    step = actual['training_timesteps_before_after']
    if type(step) is not int or step < 0:
        raise ValueError('invalid centered evaluation step')
    if step == 0:
        if actual['policy_sha256_before_after'] != INITIAL_POLICY_SHA:
            raise ValueError('centered initial validation requires new full functional policy identity')
        actual['model_status'] = 'ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED'
    actual.update(initialization='FRESH_SEED0_HAND_PRIOR_CENTERED_FUNCTION',
                  actor_parameterization='SAME_BACKGROUND_STATE_CENTERED_ACTOR_V1',
                  newly_executed_episodes=len(actual.get('episodes',[])),reused_historical_evidence=False)
    return actual
