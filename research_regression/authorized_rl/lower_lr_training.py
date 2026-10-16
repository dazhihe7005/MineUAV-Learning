"""R11 changes only global constant LR before initial snapshot; no optimizer calls."""
import math
from .lower_lr_profile import LEARNING_RATE


def verify_effective_lr(model):
    values=[model.learning_rate,*(model.lr_schedule(t) for t in (0.,.5,1.)),
            *(g['lr'] for g in model.policy.optimizer.param_groups)]
    if not model.policy.optimizer.param_groups or any(not math.isfinite(v) or v!=LEARNING_RATE for v in values):
        raise ValueError('R11 effective learning rate/schedule/optimizer identity changed')
    return {'learning_rate':LEARNING_RATE,'scope':'GLOBAL_ACTOR_AND_CRITIC','constant_schedule':True}


def initialize_lower_lr(model,scheduler,manager,*,identity,seed):
    if (seed!=0 or model.num_timesteps or model._n_updates or model.policy.optimizer.state
        or model._last_obs is not None or scheduler.selected_identities or scheduler.completed_policy_steps):
        raise ValueError('R11 LR override requires untouched fresh seed0 model and scheduler')
    if model.learning_rate!=3e-4 or any(g['lr']!=3e-4 for g in model.policy.optimizer.param_groups):
        raise ValueError('R11 base factory LR identity differs')
    model.learning_rate=LEARNING_RATE
    model._setup_lr_schedule()
    for group in model.policy.optimizer.param_groups:group['lr']=LEARNING_RATE
    verify_effective_lr(model)
    from .prior_distance_training import initialize_prior_distance
    initialization,event,probe=initialize_prior_distance(model,scheduler,manager,identity=identity,seed=seed)
    initialization.update(execution_profile='PRIOR_GLOBAL_LOWER_LR_R11_V1',effective_learning_rate=LEARNING_RATE,
        base_factory_learning_rate=3e-4,override_timing='BEFORE_PRIOR_ASSIGNMENT_RESET_AND_STEP0_SAVE; zero optimizer updates')
    event['effective_optimizer_configuration']=verify_effective_lr(model)
    return initialization,event,probe
