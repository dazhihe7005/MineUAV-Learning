"""One real initial reset and an evaluation-only step0 checkpoint, no learning."""


def initialize_training_prior(model,scheduler,manager,*,identity,seed):
    import numpy as np
    from lidar_integration.ppo_training_runner import _weights_sha
    from .actor_prior import initialize_actor_prior
    from .prior_profile import PRIOR_POLICY_SHA
    from .worker import save_checkpoint_probe
    if (seed!=0 or model._last_obs is not None or scheduler.completed_policy_steps!=0
        or scheduler.selected_identities or model.get_vec_normalize_env() is not None):
        raise ValueError('prior snapshot requires untouched seed0 environment/observation')
    base=_weights_sha(model.policy)
    if base!='bc1adbd197b52b4bfd0bb759a749c16e351092960557f2d341ba3321876b39ba':
        raise ValueError('fresh seed0 weights differ from R1 frozen control')
    configuration=initialize_actor_prior(model)
    assigned=_weights_sha(model.policy)
    if assigned!=PRIOR_POLICY_SHA:raise ValueError('assigned prior differs from physically tested prior')
    # Match SB3 2.9.0 BaseAlgorithm._setup_learn's initial reset contract.
    # learn(reset_num_timesteps=False) will reuse this exact current observation.
    model._last_obs=model.get_env().reset()
    model._last_episode_starts=np.ones((model.n_envs,),dtype=bool)
    record,probe=save_checkpoint_probe(model,scheduler,manager,run_id=identity+'_STEP0',seed=seed)
    info={'kind':configuration['version'],'base_random_policy_sha256':base,
          'assigned_policy_sha256':assigned,'configuration':configuration,'optimizer_updates':0,
          'step0_usage':'EVALUATION_ONLY; never a resume or training initialization file'}
    event={'actual_step':0,'checkpoint':record,'weight_sha256':assigned,'scheduler':scheduler.scheduler_state()}
    return info,event,probe
