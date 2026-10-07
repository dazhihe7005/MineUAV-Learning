"""Joint E/T/D supervised training: equal k0..10 decoded observation loss."""
import random
import time

import numpy as np
import torch

from latent_dynamics_data import normalize
from joint_latent_world_model import predict_latent,save_joint,load_joint,component_hashes


def predict_windows(model,episodes,stats,horizon,starts=None):
    if not episodes or not isinstance(horizon,int) or horizon<1: raise ValueError('invalid window horizon')
    if starts is None: starts=[np.arange(max(0,len(e['action'])-horizon+1)) for e in episodes]
    if len(starts)!=len(episodes): raise ValueError('invalid window lists')
    selected=[]
    for ep,ids in zip(episodes,starts):
        ids=np.asarray(ids); n=len(ep['action'])
        if len(ep['obs'])!=n or len(ep['next_obs'])!=n or len(ep['previous_action'])!=n: raise ValueError('episode boundary lengths mismatch')
        if len(ids)==0: continue
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or (ids<0).any() or (ids+horizon>n).any():
            raise ValueError('window crosses episode boundary')
        selected.append((ep,ids))
    if not selected: raise ValueError('no legal windows')
    length=max(len(e['obs']) for e,_ in selected)
    x=np.zeros((len(selected),length,11),np.float32)
    for i,(ep,_) in enumerate(selected):
        x[i,:len(ep['obs'])]=np.concatenate([normalize(ep['obs'],stats,'obs'),normalize(ep['previous_action'],stats,'previous_action')],-1)
    # Each independent lane starts hidden=None. GRU is causal even with padded
    # suffix; no suffix/padding output enters selected starting states.
    prefix,_=model.encoder(torch.from_numpy(x),None)
    z0=torch.cat([prefix[i,ids] for i,(_,ids) in enumerate(selected)])
    actions=np.concatenate([e['action'][ids[:,None]+np.arange(horizon)] for e,ids in selected])
    current=np.concatenate([e['obs'][ids,None] for e,ids in selected])
    future=np.concatenate([e['next_obs'][ids[:,None]+np.arange(horizon)] for e,ids in selected])
    truth=torch.from_numpy(np.concatenate([current,future],axis=1).astype(np.float64))
    out=predict_latent(model,z0,torch.from_numpy(actions.astype(np.float64)),stats)
    return dict(out,truth=truth,prefix_latents=prefix,window_count=len(z0))


def observation_loss(normalized_prediction,physical_truth,stats):
    mean=torch.as_tensor(stats['obs']['mean'],dtype=torch.float64)
    std=torch.as_tensor(stats['obs']['std'],dtype=torch.float64)
    return (normalized_prediction.double()-(physical_truth.double()-mean)/std).square().mean()


def diagnostics(out):
    p=out['normalized_observations'].detach(); z=out['latents'].detach()
    if not torch.isfinite(p).all() or not torch.isfinite(z).all(): raise FloatingPointError('non-finite joint prediction/latent; stop, no clipping')
    return dict(normalized_prediction_abs_max=p.abs().max().item(),latent_norm_max=z.norm(dim=-1).max().item())


@torch.no_grad()
def validation_loss(model,episodes,stats,horizon=10,batch_size=16):
    model.eval(); total=count=0
    for i in range(0,len(episodes),batch_size):
        selected=episodes[i:i+batch_size]
        if not any(len(e['action'])>=horizon for e in selected): continue
        out=predict_windows(model,selected,stats,horizon); diagnostics(out)
        loss=observation_loss(out['normalized_observations'],out['truth'],stats)
        if not torch.isfinite(loss): raise FloatingPointError('non-finite joint validation loss')
        total+=loss.item()*out['window_count']; count+=out['window_count']
    if not count: raise ValueError('no validation windows')
    return total/count


def train_joint(model,train,val,stats,config,path):
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model.requires_grad_(True); before=component_hashes(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    rng=np.random.default_rng(seed); history=[]; best=float('inf'); started=time.monotonic()
    initial=validation_loss(model,val,stats,config['horizon'],config['batch_size'])
    for epoch in range(1,config['epochs']+1):
        model.train(); total=count=0; norms={k:[] for k in before}; maxima=dict(normalized_prediction_abs_max=0.,latent_norm_max=0.)
        order=rng.permutation(len(train))
        for i in range(0,len(train),config['batch_size']):
            selected=[train[j] for j in order[i:i+config['batch_size']]]
            if not any(len(e['action'])>=config['horizon'] for e in selected): continue
            optimizer.zero_grad(set_to_none=True)
            out=predict_windows(model,selected,stats,config['horizon'])
            for k,v in diagnostics(out).items(): maxima[k]=max(maxima[k],v)
            loss=observation_loss(out['normalized_observations'],out['truth'],stats)
            if not torch.isfinite(loss): raise FloatingPointError('non-finite joint training loss')
            loss.backward()
            for k in before:
                parameters=list(getattr(model,k).parameters())
                if any(p.grad is None for p in parameters): raise AssertionError(f'{k} missing gradient')
                norm=torch.sqrt(sum(p.grad.detach().square().sum() for p in parameters))
                if not torch.isfinite(norm): raise FloatingPointError(f'{k} non-finite gradient; stop, no clipping')
                norms[k].append(norm.item())
            optimizer.step(); total+=loss.item()*out['window_count']; count+=out['window_count']
        if not count: raise ValueError('no training windows')
        validation=validation_loss(model,val,stats,config['horizon'],config['batch_size'])
        row=dict(epoch=epoch,train_loss=total/count,validation_loss=validation,windows_used=count,
                 gradient_norms={k:dict(mean=float(np.mean(v)),max=max(v)) for k,v in norms.items()},**maxima)
        history.append(row)
        if validation<best:
            best=validation; save_joint(path,model,stats,dict(config=config,best_epoch=epoch,validation_loss=best,
                parameter_hashes_before=before,selection_metric='Validation uniform all-window k0..10 observation-normalized MSE'))
        print(f'Joint epoch{epoch:02d}/{config["epochs"]} train={total/count:.8f} val={validation:.8f} '
              f'grad_max E/T/D={"/".join(f"{max(norms[k]):.5f}" for k in before)} latent_max={maxima["latent_norm_max"]:.3f}',flush=True)
    selected,_,meta=load_joint(path); after=component_hashes(selected)
    if any(before[k]==after[k] for k in before): raise AssertionError('a component did not update')
    return dict(best_epoch=meta['best_epoch'],final_epoch=config['epochs'],best_validation_loss=best,initial_validation_loss=initial,
        best_checkpoint_train_loss=validation_loss(selected,train,stats,config['horizon'],config['batch_size']),
        final_train_loss=history[-1]['train_loss'],final_validation_loss=history[-1]['validation_loss'],history=history,
        parameter_hashes_before=before,parameter_hashes_best=after,parameter_count=sum(p.numel() for p in model.parameters()),
        architecture={k:repr(getattr(model,k)) for k in before},elapsed_seconds=time.monotonic()-started,
        optimizer='joint Adam defaults lr=.0003 betas=.9/.999 eps=1e-8; no clipping',
        loss='uniform mean allwindows,k0..10,7dimensions normalized observation MSE; no latent teacher/auxiliary loss',
        order_rule='default_rng(seed0).permutation(Train complete episodes); all legal K10 starts per16episodebatch',
        backpropagation='full causal prefix and10step latent recursion; no detach, no future input or decoder feedback')
