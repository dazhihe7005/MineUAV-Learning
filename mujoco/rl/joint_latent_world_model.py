"""Deterministic latent dynamics core, not a policy or complete MBRL agent.

Old latent affine scales are folded into weights ONCE at initialization. Joint
forward has no latent statistics, teacher latent, observation feedback or memory
shared across episodes. Decoder output is normalized observation.
"""
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from latent_dynamics_data import normalize,file_hash
from latent_dynamics_models import load_model
from explicit_latent_models import ResidualTransition,ObservationDecoder,load_component
from run_latent_memory_ablation import parameter_hash


class JointLatentWorldModel(nn.Module):
    def __init__(self,encoder=None,transition=None,decoder=None):
        super().__init__()
        self.encoder=nn.GRU(11,64,num_layers=1,batch_first=True) if encoder is None else encoder
        self.transition=ResidualTransition() if transition is None else transition
        self.decoder=ObservationDecoder() if decoder is None else decoder


def component_hashes(model):
    return {k:parameter_hash(getattr(model,k)) for k in ('encoder','transition','decoder')}


@torch.no_grad()
def fold_latent_affines(transition,decoder,latent_stats):
    """Equivalent raw-z function; keep parameter shapes/activation unchanged."""
    mean=torch.tensor(latent_stats['mean'],dtype=torch.float64)
    std=torch.tensor(latent_stats['std'],dtype=torch.float64)
    if mean.shape!=(64,) or std.shape!=(64,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std<=0).any():
        raise ValueError('invalid pretrained latent affine')
    for model in (transition,decoder):
        first=model.head[0]; weight=first.weight.double().clone()
        first.bias.copy_((first.bias.double()-weight[:,:64]@(mean/std)).float())
        first.weight[:,:64].copy_((weight[:,:64]/std).float())
    last=transition.head[-1]
    last.weight.copy_((last.weight.double()*std[:,None]).float())
    last.bias.copy_((last.bias.double()*std).float())


def initialize_joint(encoder_path,transition_path,decoder_path):
    direct,stats,_=load_model(encoder_path)
    transition,ts,tl,_=load_component(transition_path)
    decoder,ds,dl,_=load_component(decoder_path)
    if ts!=stats or ds!=stats or tl!=dl: raise ValueError('pretrained normalization mismatch')
    model=JointLatentWorldModel(copy.deepcopy(direct.encoder),transition,decoder)
    original=component_hashes(model)
    with torch.no_grad():
        z=torch.linspace(-1.,1.,512).reshape(8,64); a=torch.linspace(-1.,1.,32).reshape(8,4)
        mean=torch.tensor(tl['mean']); std=torch.tensor(tl['std'])
        expected_z=z+model.transition((z-mean)/std,a)*std
        expected_o=model.decoder((z-mean)/std)
    fold_latent_affines(model.transition,model.decoder,tl)
    with torch.no_grad():
        dz=(z+model.transition(z,a)-expected_z).abs().max().item()
        do=(model.decoder(z)-expected_o).abs().max().item()
    if dz>3e-6 or do>3e-6: raise AssertionError('pretrained raw function conversion mismatch')
    model.train().requires_grad_(True)
    return model,stats,dict(source_parameter_hashes=original,
        raw_coordinate_parameter_hashes=component_hashes(model),
        checkpoints={k:dict(path=str(Path(p).resolve()),sha256=file_hash(p)) for k,p in
                     [('encoder',encoder_path),('transition',transition_path),('decoder',decoder_path)]},
        conversion='one-time affine folding into T/D weights; no runtime latent normalization',
        initial_function_equivalence=dict(latent_next_max_abs_error=dz,normalized_observation_max_abs_error=do,
            tolerance=3e-6,probe_rule='fixed linspace rawz[-1,1] 8x64 and normalized action[-1,1] 8x4; no Test selection'))


def encode_episode(model,observations,previous_actions,stats):
    if len(observations)==0 or len(observations)!=len(previous_actions): raise ValueError('invalid prefix')
    x=np.concatenate([normalize(observations,stats,'obs'),normalize(previous_actions,stats,'previous_action')],-1)
    z,_=model.encoder(torch.from_numpy(x)[None],None)
    return z[0]


def predict_latent(model,z0,actions,stats):
    """Pure raw-latent recursion; current plus all future decoded states."""
    if z0.ndim!=2 or z0.shape[1]!=64 or actions.ndim!=3 or actions.shape[0]!=len(z0) or actions.shape[2]!=4 or actions.shape[1]<1:
        raise ValueError('invalid pure latent rollout shape')
    mean=torch.as_tensor(stats['action']['mean'],dtype=actions.dtype)
    std=torch.as_tensor(stats['action']['std'],dtype=actions.dtype)
    current=z0.float(); steps=[current]; obs=[model.decoder(current)]
    for k in range(actions.shape[1]):
        action=((actions[:,k]-mean)/std).float()
        current=current+model.transition(current,action)
        steps.append(current); obs.append(model.decoder(current))
    return dict(latents=torch.stack(steps,1),normalized_observations=torch.stack(obs,1),steps=steps)


def predict_window(model,episode,start,horizon,stats):
    if (not isinstance(start,int) or not isinstance(horizon,int) or start<0 or horizon<1
            or start>=len(episode['obs']) or start+horizon>len(episode['action'])):
        raise ValueError('invalid prefix/action episode boundary')
    z=encode_episode(model,episode['obs'][:start+1],episode['previous_action'][:start+1],stats)[-1:]
    return predict_latent(model,z,torch.as_tensor(episode['action'][None,start:start+horizon]),stats)


def save_joint(path,model,stats,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind='joint_latent_world_model_v1',state_dict=model.state_dict(),statistics=stats,metadata=metadata),path)


def load_joint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['kind']!='joint_latent_world_model_v1': raise ValueError('invalid joint checkpoint')
    model=JointLatentWorldModel(); model.load_state_dict(saved['state_dict'])
    return model.eval(),saved['statistics'],saved['metadata']


def latent_summary(values):
    z=np.asarray(values,np.float64)
    if z.ndim!=2 or z.shape[1]!=64 or len(z)<2 or not np.isfinite(z).all(): raise ValueError('invalid latent diagnostic')
    norm=np.linalg.norm(z,axis=1); std=z.std(0); eig=np.maximum(np.linalg.eigvalsh(np.cov(z,rowvar=False,bias=True)),0.)
    trace=eig.sum(); effective=0.
    if trace>1e-20:
        p=eig/trace; positive=p[p>0]; effective=float(np.exp(-(positive*np.log(positive)).sum()))
    return dict(frame_count=len(z),mean_norm=float(norm.mean()),std_norm=float(norm.std()),max_norm=float(norm.max()),
        per_dimension_std=std.tolist(),per_dimension_std_min=float(std.min()),per_dimension_std_median=float(np.median(std)),
        per_dimension_std_max=float(std.max()),near_zero_std_dimension_count=int((std<1e-6).sum()),
        near_zero_std_threshold=1e-6,effective_rank=effective,total_variance=float(trace),
        rule='covariance entropy effective rank; diagnostic only, no variance loss or runtime latent normalization')
