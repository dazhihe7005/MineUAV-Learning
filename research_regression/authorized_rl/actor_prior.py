"""Analytic, approximate state-only mean initialization; zero optimizer calls.

Not a trained BC model, not an obstacle-aware controller. The unchanged
Gaussian exploration std remains 1; only deterministic mean is approximated.
"""
import math


def prior_configuration():
    return {'version':'ANALYTIC_STATE_MEAN_TANH_V1','alpha':.2,'beta':.5,
        'state_feature_indices':[64,65,66,70],'normalized_gains':[1/4.5,1/4.5,1/3,1.],
        'max_clipped_action_error':.02,'analytic_error_bound':1-10*math.tanh(.5*math.tanh(.2)),
        'kind':'HAND_WRITTEN_PRIOR_NOT_LEARNED_AVOIDANCE','optimizer_updates':0}


def initialize_actor_prior(model):
    import torch
    from lidar_integration.ppo_preflight import LidarStateExtractor
    p=model.policy;net=p.mlp_extractor.policy_net;head=p.action_net
    if (model.num_timesteps!=0 or p.optimizer.state or getattr(model,'_n_updates',0)!=0):
        raise ValueError('analytic prior is only for fresh, unoptimized initialization')
    if (not isinstance(p.features_extractor,LidarStateExtractor) or p.features_extractor.features_dim!=74
        or len(net)!=4 or type(net[0]) is not torch.nn.Linear or type(net[2]) is not torch.nn.Linear
        or type(net[1]) is not torch.nn.Tanh or type(net[3]) is not torch.nn.Tanh
        or (net[0].in_features,net[0].out_features)!=(74,64)
        or (net[2].in_features,net[2].out_features)!=(64,64)
        or type(head) is not torch.nn.Linear or (head.in_features,head.out_features)!=(64,4)
        or not torch.isfinite(p.log_std).all() or not torch.equal(p.log_std,torch.zeros_like(p.log_std))):
        raise ValueError('unsupported frozen Actor/extractor/std structure')
    configuration=prior_configuration()
    # Retain unused hidden neurons' original random weights; only four directed
    # paths drive the initial mean. No CNN/Critic/std tensor is assigned here.
    with torch.no_grad():
        net[0].weight[:4].zero_();net[0].bias[:4].zero_()
        for j,(index,gain) in enumerate(zip(configuration['state_feature_indices'],configuration['normalized_gains'])):
            net[0].weight[j,index]=configuration['alpha']*gain
        net[2].weight[:4].zero_();net[2].bias[:4].zero_()
        for j in range(4):net[2].weight[j,j]=configuration['beta']
        head.weight.zero_();head.bias.zero_()
        for j in range(4):head.weight[j,j]=1/(configuration['alpha']*configuration['beta'])
    return configuration
