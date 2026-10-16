"""Explicit new Actor function: same-current-background zero-state anchoring.

Not a learned controller, not a safety filter, and not permission to train.
The CNN and Critic are unchanged. Existing policies and factories stay intact.
"""
import hashlib
import math
from types import SimpleNamespace

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
from lidar_integration.ppo_preflight import LidarStateExtractor

VERSION = 'SAME_BACKGROUND_STATE_CENTERED_ACTOR_V1'
FRESH_SHA = 'bc1adbd197b52b4bfd0bb759a749c16e351092960557f2d341ba3321876b39ba'
PRIOR_SHA = '6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74'


class CenteredMlp(torch.nn.Module):
    """Two shared Actor passes, one unchanged Critic pass; no new parameters."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.latent_dim_pi = 2 * base.latent_dim_pi
        self.latent_dim_vf = base.latent_dim_vf

    def forward_actor(self, features):
        anchor = features.clone()
        anchor[:, 64:71] = 0  # state7 only; current LiDAR64/heading2/age1 stay identical.
        return torch.cat((self.base.forward_actor(features), self.base.forward_actor(anchor)), dim=1)

    def forward_critic(self, features):
        return self.base.forward_critic(features)

    def forward(self, features):
        return self.forward_actor(features), self.forward_critic(features)


class CenteredHead(torch.nn.Module):
    """Both paths use the same actual Linear; head bias cancels structurally."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear

    def forward(self, paired_latent):
        current, anchor = paired_latent.split(64, dim=1)
        return self.linear(current) - self.linear(anchor)


class StateCenteredPolicy(MultiInputActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if (type(self.features_extractor) is not LidarStateExtractor or self.features_dim != 74
                or not self.share_features_extractor or self.net_arch != {'pi': [64, 64], 'vf': [64, 64]}
                or self.activation_fn is not torch.nn.Tanh
                or type(self.action_dist) is not DiagGaussianDistribution
                or type(self.action_net) is not torch.nn.Linear
                or (self.action_net.in_features, self.action_net.out_features) != (64, 4)):
            raise ValueError('unsupported frozen centered Actor structure')
        original_parameters = tuple(self.parameters())
        self.mlp_extractor = CenteredMlp(self.mlp_extractor)
        self.action_net = CenteredHead(self.action_net)
        if tuple(map(id, self.parameters())) != tuple(map(id, original_parameters)):
            raise ValueError('centering altered parameter objects/order')


def make_centered_ppo(env, *, seed=0):
    """Use frozen SB3 constructor configuration, with explicit new policy class/LR.

    PPO construction with StateCenteredPolicy is necessary for correct save/load
    class identity. No checkpoint initialization, reset, learn, or optimizer step.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('nonnegative integer seed required')
    # Constructor hyperparameters are compared to the frozen base factory in
    # no-update tests; PPO algorithms are inherited, not copied or reimplemented.
    return PPO(StateCenteredPolicy, env, learning_rate=3e-5, n_steps=128,
               batch_size=64, n_epochs=2, gamma=.99, gae_lambda=.95, clip_range=.2,
               ent_coef=0., vf_coef=.5, max_grad_norm=.5, seed=seed, device='cpu',
               policy_kwargs={'features_extractor_class': LidarStateExtractor,
                              'normalize_images': False,
                              'net_arch': {'pi': [64, 64], 'vf': [64, 64]}}, verbose=0)


def verify_centered_policy(model):
    p = model.policy
    if (type(p) is not StateCenteredPolicy or type(p.mlp_extractor) is not CenteredMlp
            or type(p.action_net) is not CenteredHead or type(p.features_extractor) is not LidarStateExtractor
            or not p.share_features_extractor or p.normalize_images
            or p.net_arch != {'pi': [64, 64], 'vf': [64, 64]}
            or p.features_dim != 74 or p.mlp_extractor.latent_dim_pi != 128
            or p.mlp_extractor.latent_dim_vf != 64
            or type(p.action_dist) is not DiagGaussianDistribution
            or type(p.action_net.linear) is not torch.nn.Linear
            or (p.action_net.linear.in_features, p.action_net.linear.out_features) != (64, 4)):
        raise ValueError('centered policy wrapper/feature identity differs')
    def linear(layer, inputs, outputs):
        return (type(layer) is torch.nn.Linear and layer.in_features == inputs
                and layer.out_features == outputs and layer.bias is not None)
    def trunk(net):
        return (type(net) is torch.nn.Sequential and len(net) == 4
                and linear(net[0], 74, 64) and type(net[1]) is torch.nn.Tanh
                and linear(net[2], 64, 64) and type(net[3]) is torch.nn.Tanh)
    def conv(layer, inputs, outputs, stride):
        return (type(layer) is torch.nn.Conv2d and layer.in_channels == inputs and layer.out_channels == outputs
                and layer.kernel_size == (3, 3) and layer.stride == (stride, stride)
                and layer.padding == (1, 1) and layer.dilation == (1, 1) and layer.groups == 1
                and layer.padding_mode == 'zeros' and layer.bias is not None)
    cnn = p.features_extractor.lidar_net
    base = p.mlp_extractor.base
    if (type(base) is not MlpExtractor or not trunk(base.policy_net) or not trunk(base.value_net)
            or base.latent_dim_pi != 64 or base.latent_dim_vf != 64
            or not linear(p.value_net, 64, 1)
            or p.pi_features_extractor is not p.features_extractor or p.vf_features_extractor is not p.features_extractor
            or type(cnn) is not torch.nn.Sequential or len(cnn) != 7
            or not conv(cnn[0], 3, 16, 1) or type(cnn[1]) is not torch.nn.ReLU or cnn[1].inplace
            or not conv(cnn[2], 16, 32, 2) or type(cnn[3]) is not torch.nn.ReLU or cnn[3].inplace
            or type(cnn[4]) is not torch.nn.Flatten or (cnn[4].start_dim, cnn[4].end_dim) != (1, -1)
            or not linear(cnn[5], 2304, 64) or type(cnn[6]) is not torch.nn.ReLU or cnn[6].inplace
            or p.log_std.shape != (4,) or p.device.type != 'cpu'):
        raise ValueError('actual centered Actor/Critic/CNN topology or shared feature aliases differ')
    parameters = [id(t) for t in p.parameters()]
    optimized = [id(t) for g in p.optimizer.param_groups for t in g['params']]
    if parameters != optimized or len(parameters) != len(set(parameters)):
        raise ValueError('centered parameter/optimizer objects differ')
    if not all(torch.isfinite(t).all() for t in p.state_dict().values()):
        raise ValueError('nonfinite centered policy parameters')
    if model.learning_rate != 3e-5 or any(model.lr_schedule(x) != 3e-5 for x in (0., .5, 1.)) or any(g['lr'] != 3e-5 for g in p.optimizer.param_groups):
        raise ValueError('centered global learning rate differs')
    return {'version': VERSION, 'parameters': sum(t.numel() for t in p.parameters()),
            'optimizer_parameter_objects_preserved': True, 'shared_cnn_passes': 1,
            'actor_mlp_passes': 2, 'zero_state_mean': 'STRUCTURAL_IDENTITY_NOT_LEARNED_SUCCESS'}


def canonical_policy_sha(policy):
    """Parameter-value comparison only; NOT the new policy's functional identity."""
    values = {}
    for key, tensor in policy.state_dict().items():
        canonical = key.replace('mlp_extractor.base.', 'mlp_extractor.').replace('action_net.linear.', 'action_net.')
        if canonical in values:
            raise ValueError('ambiguous canonical parameter key')
        values[canonical] = tensor
    result = hashlib.sha256()
    for key, tensor in sorted(values.items()):
        result.update(key.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def initialize_centered_prior(model):
    verify_centered_policy(model)
    p = model.policy
    if (model.seed != 0 or model.num_timesteps or model._n_updates or p.optimizer.state
            or model._last_obs is not None or canonical_policy_sha(p) != FRESH_SHA):
        raise ValueError('centered initialization requires untouched fresh seed0 model')
    from .actor_prior import initialize_actor_prior
    # Explicit view of actual underlying parameters, not a class monkeypatch or
    # fake runtime state. Reuse the audited assignment routine without changing it.
    view = SimpleNamespace(policy=SimpleNamespace(features_extractor=p.features_extractor,
        mlp_extractor=p.mlp_extractor.base, action_net=p.action_net.linear,
        optimizer=p.optimizer, log_std=p.log_std),
        num_timesteps=model.num_timesteps, _n_updates=model._n_updates)
    prior = initialize_actor_prior(view)
    with torch.no_grad():
        p.log_std.fill_(math.log(.3))
    canonical = canonical_policy_sha(p)
    if canonical != PRIOR_SHA:
        raise ValueError('centered initial parameter values differ from frozen R11 prior')
    return {**verify_centered_policy(model), 'prior': prior,
            'canonical_policy_sha256': canonical, 'canonical_comparison_only': True,
            'optimizer_updates': 0, 'initialization': 'FRESH_RANDOM_SEED0_PLUS_HAND_WRITTEN_PRIOR_NOT_CHECKPOINT'}
