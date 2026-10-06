"""Frozen teacher extraction and two independently supervised Tanh MLPs.

Teacher latents are encoder coordinates, not physical ground-truth states.
No optimizer owns the encoder, and neither learner receives true PI state.
"""
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from latent_dynamics_data import json_hash, normalize
from run_latent_memory_ablation import parameter_hash


def freeze_encoder(encoder):
    encoder.eval().requires_grad_(False)
    for parameter in encoder.parameters(): parameter.grad=None
    return encoder


def array_hash(array):
    a=np.ascontiguousarray(array)
    digest=hashlib.sha256(str((a.shape,str(a.dtype))).encode())
    digest.update(a.tobytes()); return digest.hexdigest()


def complete_observations(episode):
    return np.vstack([episode['obs'],episode['next_obs'][-1]])


@torch.no_grad()
def encode_prefix(encoder, observation, previous, stats):
    if any(p.requires_grad for p in encoder.parameters()):
        raise ValueError('encoder must be frozen')
    x=np.concatenate([normalize(observation,stats,'obs'),normalize(previous,stats,'previous_action')],axis=1)
    if len(x)==0 or not np.isfinite(x).all(): raise ValueError('invalid encoder prefix')
    z,_=encoder(torch.from_numpy(x)[None],None)
    if not torch.isfinite(z).all(): raise FloatingPointError('non-finite teacher latent')
    return z[0].numpy().copy()


def teacher_latents(encoder, episode, stats):
    # The original adapter has N transitions but N+1 saved observations. The
    # last saved input is NOT an invented terminal observation; include it.
    return encode_prefix(encoder,complete_observations(episode),
                         np.vstack([episode['previous_action'],episode['action'][-1]]),stats)


def attach_teacher(episodes, encoder, stats):
    return [dict(ep,latent=teacher_latents(encoder,ep,stats)) for ep in episodes]


def latent_statistics(train_sequences):
    z=np.concatenate(train_sequences).astype(np.float64)
    if z.ndim!=2 or z.shape[1]!=64 or not np.isfinite(z).all(): raise ValueError('invalid Train latent')
    return dict(mean=z.mean(0).tolist(),std=np.maximum(z.std(0),1e-6).tolist(),
                provenance=dict(split='train',frames=len(z),std_floor=1e-6,
                    frame_rule='all saved observation frames; includes final saved frame of each episode'))


def latent_normalize(z, stats):
    return ((z-np.asarray(stats['mean']))/np.asarray(stats['std'])).astype(np.float32)


def transition_manifest(episodes, sequences, dataset_hash):
    if len(episodes)!=len(sequences): raise ValueError('latent/episode count mismatch')
    rows=[]
    for index,(ep,z) in enumerate(zip(episodes,sequences)):
        n=len(ep['obs'])
        if z.shape!=(n+1,64) or not np.isfinite(z).all(): raise ValueError('latent length/finite mismatch')
        rows.append(dict(episode_id=index,target_id=ep['metadata']['target_id'],source=ep['metadata']['source'],
            source_rows=ep['source_rows'],transition_count=n,decoder_frame_count=n+1,
            teacher_latent_sha256=array_hash(z),executed_actions_sha256=array_hash(ep['action'])))
    result=dict(dataset_split_sha256=dataset_hash,episodes=rows,
        transition_count=sum(r['transition_count'] for r in rows),
        decoder_frame_count=sum(r['decoder_frame_count'] for r in rows),
        pair_rule='z[t], action[t], z[t+1]; t=0..N-1 inside same original episode',
        teacher_rule='frozen GRU, normalized [recorded observation,previous executed action]; hidden=None at each episode')
    result['manifest_sha256']=json_hash(result); return result


def head(input_size,output_size):
    return nn.Sequential(nn.Linear(input_size,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,output_size))


class ResidualTransition(nn.Module):
    """Predict normalized Delta-z; physical z'=z + Train latent std*output."""
    def __init__(self):
        super().__init__(); self.head=head(68,64)
    def forward(self,z,action): return self.head(torch.cat([z,action],axis=-1))


class ObservationDecoder(nn.Module):
    """Normalized latent -> normalized current observation; no action input."""
    def __init__(self):
        super().__init__(); self.head=head(64,7)
    def forward(self,z): return self.head(z)


def training_batch(kind,episodes,stats,latent_stats):
    if kind=='transition':
        z=np.concatenate([e['latent'][:-1] for e in episodes])
        action=normalize(np.concatenate([e['action'] for e in episodes]),stats,'action')
        delta=np.concatenate([np.diff(e['latent'].astype(np.float64),axis=0) for e in episodes])
        target=(delta/np.asarray(latent_stats['std'])).astype(np.float32)
        args=(torch.from_numpy(latent_normalize(z,latent_stats)),torch.from_numpy(action))
    elif kind=='decoder':
        z=np.concatenate([e['latent'] for e in episodes])
        target=normalize(np.concatenate([complete_observations(e) for e in episodes]),stats,'obs')
        args=(torch.from_numpy(latent_normalize(z,latent_stats)),)
    else: raise ValueError('invalid component kind')
    return args,torch.from_numpy(target)


def save_component(path,model,kind,stats,latent_stats,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind=kind,state_dict=model.state_dict(),statistics=stats,
                    latent_statistics=latent_stats,metadata=metadata),path)


def load_component(path):
    ckpt=torch.load(path,map_location='cpu',weights_only=True)
    kind=ckpt['kind']
    if kind not in ('transition','decoder'): raise ValueError('invalid saved component kind')
    model=ResidualTransition() if kind=='transition' else ObservationDecoder()
    model.load_state_dict(ckpt['state_dict'])
    return model.eval(),ckpt['statistics'],ckpt['latent_statistics'],ckpt['metadata']


@torch.no_grad()
def validation_loss(model,kind,episodes,stats,latent_stats,batch_size=16):
    model.eval(); total=count=0
    for start in range(0,len(episodes),batch_size):
        args,target=training_batch(kind,episodes[start:start+batch_size],stats,latent_stats)
        loss=(model(*args)-target).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('non-finite validation loss')
        total+=loss.item()*len(target); count+=len(target)
    return total/count


def train_component(kind,train,val,stats,latent_stats,config,path):
    """Two independent calls/optimizers/seeds; Test is intentionally not accepted."""
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model=ResidualTransition() if kind=='transition' else ObservationDecoder()
    initial=parameter_hash(model); optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    rng=np.random.default_rng(seed); history=[]; best=float('inf'); best_epoch=None; started=time.monotonic()
    for epoch in range(1,config['epochs']+1):
        model.train(); total=count=0; norms=[]; output_max=0.
        order=rng.permutation(len(train))
        for start in range(0,len(train),config['batch_size']):
            args,target=training_batch(kind,[train[i] for i in order[start:start+config['batch_size']]],stats,latent_stats)
            optimizer.zero_grad(set_to_none=True); prediction=model(*args)
            loss=(prediction-target).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('non-finite training loss; no clipping')
            loss.backward(); norm=torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters()))
            if not torch.isfinite(norm): raise FloatingPointError('non-finite gradient; no new clipping')
            norms.append(norm.item()); output_max=max(output_max,prediction.detach().abs().max().item())
            optimizer.step(); total+=loss.item()*len(target); count+=len(target)
        validation=validation_loss(model,kind,val,stats,latent_stats,config['batch_size'])
        history.append(dict(epoch=epoch,train_loss=total/count,validation_loss=validation,
            gradient_norm_max=max(norms),gradient_norm_mean=float(np.mean(norms)),samples_used=count,normalized_output_abs_max=output_max))
        if validation<best:
            best,best_epoch=validation,epoch
            save_component(path,model,kind,stats,latent_stats,dict(config=config,best_epoch=epoch,
                validation_loss=best,initial_parameter_sha256=initial,
                selection_metric=f'Validation normalized {"latent delta" if kind=="transition" else "observation reconstruction"} MSE'))
        if epoch==1 or epoch%10==0:
            print(f'{kind}: epoch{epoch:02d}/{config["epochs"]} train={total/count:.8f} val={validation:.8f}',flush=True)
    selected,_,_,_=load_component(path)
    return dict(best_epoch=best_epoch,final_epoch=config['epochs'],best_validation_loss=best,
        best_checkpoint_train_loss=validation_loss(selected,kind,train,stats,latent_stats,config['batch_size']),
        initial_parameter_sha256=initial,parameter_count=sum(p.numel() for p in model.parameters()),
        architecture=repr(model),history=history,elapsed_seconds=time.monotonic()-started,
        optimizer='independent Adam defaults lr=.001, betas=.9/.999 eps=1e-8; no gradient clipping',
        order_rule='default_rng(0).permutation(complete Train episode count), reset for each learner')
