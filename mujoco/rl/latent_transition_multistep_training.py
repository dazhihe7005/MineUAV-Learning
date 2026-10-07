"""K10 T-only BPTT; frozen teachers are initial states or loss targets only."""
import random
import time

import numpy as np
import torch

from explicit_latent_models import ResidualTransition,save_component,load_component
from run_latent_memory_ablation import parameter_hash


def window_batch(episodes,horizon,starts=None):
    if not isinstance(horizon,int) or horizon<1 or not episodes:
        raise ValueError('invalid window horizon/episodes')
    if starts is None: starts=[np.arange(max(0,len(e['action'])-horizon+1)) for e in episodes]
    if len(starts)!=len(episodes): raise ValueError('invalid window start lists')
    initial=[]; actions=[]; truth=[]
    for ep,ids in zip(episodes,starts):
        ids=np.asarray(ids); n=len(ep['action'])
        if ep['latent'].shape!=(n+1,64): raise ValueError('teacher/action episode boundary mismatch')
        if len(ids)==0: continue
        if (ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer)
                or np.any(ids<0) or np.any(ids+horizon>n)):
            raise ValueError('window crosses episode boundary')
        initial.append(ep['latent'][ids]); actions.append(ep['action'][ids[:,None]+np.arange(horizon)])
        truth.append(ep['latent'][ids[:,None]+np.arange(1,horizon+1)])
    if not initial: raise ValueError('no legal windows')
    return dict(initial=torch.from_numpy(np.concatenate(initial).astype(np.float64)),
                actions=torch.from_numpy(np.concatenate(actions).astype(np.float64)),
                truth=torch.from_numpy(np.concatenate(truth).astype(np.float64)))


def predict_latents(model,initial,actions,stats,latent_stats):
    """Physical latent carry; no teacher target, encoder or decoder in forward."""
    if (initial.ndim!=2 or initial.shape[1]!=64 or actions.ndim!=3
            or actions.shape[0]!=len(initial) or actions.shape[2]!=4 or actions.shape[1]<1):
        raise ValueError('invalid latent rollout shape')
    mean=torch.as_tensor(latent_stats['mean'],dtype=torch.float64)
    std=torch.as_tensor(latent_stats['std'],dtype=torch.float64)
    action_mean=torch.as_tensor(stats['action']['mean'],dtype=torch.float64)
    action_std=torch.as_tensor(stats['action']['std'],dtype=torch.float64)
    current=initial.double(); steps=[]
    for k in range(actions.shape[1]):
        normalized=(current-mean)/std
        action=(actions[:,k].double()-action_mean)/action_std
        delta=model(normalized.float(),action.float())
        current=current+delta.double()*std
        steps.append(current)
    return dict(predictions=torch.stack(steps,1),steps=steps)


def latent_loss(prediction,truth,latent_stats):
    std=torch.as_tensor(latent_stats['std'],dtype=prediction.dtype)
    return ((prediction-truth)/std).square().mean()


def diagnostics(out):
    prediction=out['predictions'].detach()
    if not torch.isfinite(prediction).all(): raise FloatingPointError('non-finite latent rollout; stop, no clipping/detach')
    return dict(latent_norm_max=prediction.norm(dim=-1).max().item(),latent_abs_max=prediction.abs().max().item())


@torch.no_grad()
def validation_loss(model,episodes,stats,latent_stats,horizon=10,batch_size=16):
    model.eval(); total=count=0
    for start in range(0,len(episodes),batch_size):
        selected=episodes[start:start+batch_size]
        if not any(len(e['action'])>=horizon for e in selected): continue
        batch=window_batch(selected,horizon)
        out=predict_latents(model,batch['initial'],batch['actions'],stats,latent_stats); diagnostics(out)
        loss=latent_loss(out['predictions'],batch['truth'],latent_stats)
        if not torch.isfinite(loss): raise FloatingPointError('non-finite validation loss')
        total+=loss.item()*len(batch['initial']); count+=len(batch['initial'])
    if not count: raise ValueError('no validation windows')
    return total/count


def train_transition(train,val,stats,latent_stats,config,path):
    """Fresh same-seed architecture; only T optimizer, no Test argument."""
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model=ResidualTransition(); initial=parameter_hash(model)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    rng=np.random.default_rng(seed); history=[]; best=float('inf'); best_epoch=None; started=time.monotonic()
    initial_validation=validation_loss(model,val,stats,latent_stats,config['horizon'],config['batch_size'])
    for epoch in range(1,config['epochs']+1):
        model.train(); total=count=0; norms=[]; maxima=dict(latent_norm_max=0.,latent_abs_max=0.)
        order=rng.permutation(len(train))
        for start in range(0,len(train),config['batch_size']):
            selected=[train[i] for i in order[start:start+config['batch_size']]]
            if not any(len(e['action'])>=config['horizon'] for e in selected): continue
            batch=window_batch(selected,config['horizon']); optimizer.zero_grad(set_to_none=True)
            out=predict_latents(model,batch['initial'],batch['actions'],stats,latent_stats)
            for k,v in diagnostics(out).items(): maxima[k]=max(maxima[k],v)
            loss=latent_loss(out['predictions'],batch['truth'],latent_stats)
            if not torch.isfinite(loss): raise FloatingPointError('non-finite training loss; stop')
            loss.backward(); norm=torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters()))
            if not torch.isfinite(norm): raise FloatingPointError('non-finite gradient; no added clipping')
            norms.append(norm.item()); optimizer.step(); total+=loss.item()*len(batch['initial']); count+=len(batch['initial'])
        if not count: raise ValueError('no training windows')
        validation=validation_loss(model,val,stats,latent_stats,config['horizon'],config['batch_size'])
        history.append(dict(epoch=epoch,train_loss=total/count,validation_loss=validation,windows_used=count,
                            gradient_norm_mean=float(np.mean(norms)),gradient_norm_max=max(norms),**maxima))
        if validation<best:
            best,best_epoch=validation,epoch
            save_component(path,model,'transition',stats,latent_stats,dict(config=config,best_epoch=epoch,
                validation_loss=best,initial_parameter_sha256=initial,
                selection_metric=f'Validation uniform K{config["horizon"]} autoregressive normalized latent MSE'))
        print(f'T K{config["horizon"]}: epoch{epoch:02d}/{config["epochs"]} train={total/count:.8f} '
              f'val={validation:.8f} grad_max={max(norms):.5f}',flush=True)
    selected,_,_,_=load_component(path)
    return dict(best_epoch=best_epoch,final_epoch=config['epochs'],best_validation_loss=best,
        initial_validation_loss=initial_validation,initial_parameter_sha256=initial,
        best_checkpoint_train_loss=validation_loss(selected,train,stats,latent_stats,config['horizon'],config['batch_size']),
        final_train_loss=history[-1]['train_loss'],final_validation_loss=history[-1]['validation_loss'],
        parameter_count=sum(p.numel() for p in model.parameters()),architecture=repr(model),history=history,
        elapsed_seconds=time.monotonic()-started,optimizer='Adam defaults lr=.001, betas=.9/.999 eps=1e-8; no gradient clipping',
        order_rule='default_rng(seed=0).permutation(Train complete episodes); all legal K10 windows per episode minibatch',
        backpropagation='All K latent states connected; no detach and no horizon teacher input; E/D outside graph')
