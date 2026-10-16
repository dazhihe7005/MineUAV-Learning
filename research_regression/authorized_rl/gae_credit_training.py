"""Configure only GAE metadata before first sampling; inherited SB3 algorithm."""
from research_ops.storage import OpsError


def configure_gae_credit(model):
    if (model.num_timesteps or model._n_updates or model.policy.optimizer.state
            or model.gamma != .99 or model.rollout_buffer.gamma != .99
            or model.gae_lambda != .95 or model.rollout_buffer.gae_lambda != .95
            or model.rollout_buffer.pos or model.rollout_buffer.full):
        raise OpsError('R22 requires untouched fresh constructor,not resume')
    model.gae_lambda = .99
    model.rollout_buffer.gae_lambda = .99
    return verify_gae_configuration(model)


def verify_gae_configuration(model):
    from .state_centered_policy import verify_centered_policy
    verify_centered_policy(model)
    if (model.gamma != .99 or model.rollout_buffer.gamma != .99
            or model.gae_lambda != .99 or model.rollout_buffer.gae_lambda != .99
            or model.n_steps != 128 or model.batch_size != 64 or model.n_epochs != 2 or model.ent_coef != 0.
            or model.vf_coef != .5 or model.max_grad_norm != .5 or model.n_envs != 1
            or not model.normalize_advantage or model.clip_range_vf is not None or model.use_sde
            or any(model.clip_range(v) != .2 for v in (0., .5, 1.))):
        raise OpsError('R22 actual PPO or buffer configuration differs')
    return {'gamma': model.gamma, 'rollout_buffer_gamma': model.rollout_buffer.gamma,
            'gae_lambda': model.gae_lambda, 'rollout_buffer_gae_lambda': model.rollout_buffer.gae_lambda,
            'n_steps': model.n_steps, 'batch_size': model.batch_size, 'n_epochs': model.n_epochs,
            'interpretation': 'actual SB3 GAE metadata;discount objective,Reward,network,actions,physics unchanged'}
