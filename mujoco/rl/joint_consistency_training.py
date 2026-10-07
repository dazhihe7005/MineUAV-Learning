"""v1 observation path unchanged; only lambda=.1 local consistency is added."""
import random
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import normalize
from joint_latent_world_model import JointLatentWorldModel,component_hashes
from joint_latent_evaluation import encoded_sequences
from joint_latent_training import predict_windows,observation_loss,diagnostics

LAMBDA=.1


def fit_initial_scale(train_sequences):
    z=np.concatenate(train_sequences).astype(np.float64)
    if z.ndim!=2 or z.shape[1]!=64 or not np.isfinite(z).all(): raise ValueError('invalid Train initial latent')
    return dict(std=np.maximum(z.std(0),1e-6).tolist(),std_floor=1e-6,frame_count=len(z),
        split='train',rule='initial Encoder real-history population std; fixed throughout training, difference loss has no mean subtraction')


def initial_scale(model,train,stats):
    return fit_initial_scale(encoded_sequences(model,train,stats))


def consistency_mse(prediction,target,scale):
    std=torch.as_tensor(scale,dtype=torch.float64)
    if std.shape!=(64,) or not torch.isfinite(std).all() or (std<=0).any(): raise ValueError('invalid fixed initial latent scale')
    return ((prediction.double()-target.detach().double())/std).square().mean()


def total_loss(obs,cons): return obs+LAMBDA*cons


def local_loss(model,episodes,stats,scale,horizon=10,starts=None):
    if horizon!=10: raise ValueError('fixed local horizon10 only')
    if starts is None: starts=[np.arange(max(0,len(e['action'])-horizon+1)) for e in episodes]
    if len(starts)!=len(episodes): raise ValueError('window episode mismatch')
    selected=[]
    for ep,ids in zip(episodes,starts):
        n=len(ep['action']); ids=np.asarray(ids)
        if not (len(ep['obs'])==len(ep['previous_action'])==len(ep['next_obs'])==n): raise ValueError('episode boundary length mismatch')
        if not len(ids): continue
        if ids.ndim!=1 or not np.issubdtype(ids.dtype,np.integer) or (ids<0).any() or (ids+horizon>n).any():
            raise ValueError('window crosses episode boundary')
        selected.append((ep,ids))
    if not selected: raise ValueError('no legal consistency windows')
    nmax=max(len(e['obs'])+1 for e,_ in selected); x=np.zeros((len(selected),nmax,11),np.float32)
    for i,(ep,_) in enumerate(selected):
        obs=np.vstack([ep['obs'],ep['next_obs'][-1]])
        previous=np.vstack([ep['previous_action'],ep['action'][-1]])
        x[i,:len(obs)]=np.concatenate([normalize(obs,stats,'obs'),normalize(previous,stats,'previous_action')],-1)
    # Separate real-history loss branch; never feeds the autonomous state.
    reference,_=model.encoder(torch.from_numpy(x),None)
    source=torch.cat([reference[i,ids[:,None]+np.arange(horizon)] for i,(_,ids) in enumerate(selected)])
    target=torch.cat([reference[i,ids[:,None]+np.arange(1,horizon+1)] for i,(_,ids) in enumerate(selected)])
    actions=np.concatenate([ep['action'][ids[:,None]+np.arange(horizon)] for ep,ids in selected])
    action=torch.from_numpy(normalize(actions,stats,'action'))
    predicted=source+model.transition(source.reshape(-1,64),action.reshape(-1,4)).reshape_as(source)
    return dict(loss=consistency_mse(predicted,target,scale),prediction=predicted,source_latents=source,target_latents=target)


def loss_batch(model,episodes,stats,scale,horizon=10,starts=None):
    out=predict_windows(model,episodes,stats,horizon,starts)
    obs=observation_loss(out['normalized_observations'],out['truth'],stats)
    local=local_loss(model,episodes,stats,scale,horizon,starts)
    return dict(out,loss=total_loss(obs,local['loss']),observation_loss=obs,consistency_loss=local['loss'],
        source_latents=local['source_latents'],target_latents=local['target_latents'],local_prediction=local['prediction'])


@torch.no_grad()
def validation_losses(model,episodes,stats,scale,horizon=10,batch_size=16):
    model.eval(); totals=np.zeros(3); count=0
    for begin in range(0,len(episodes),batch_size):
        selected=episodes[begin:begin+batch_size]
        if not any(len(e['action'])>=horizon for e in selected): continue
        row=loss_batch(model,selected,stats,scale,horizon); diagnostics(row)
        values=np.array([row['observation_loss'].item(),row['consistency_loss'].item(),row['loss'].item()])
        if not np.isfinite(values).all(): raise FloatingPointError('non-finite v2 validation objective')
        totals+=values*row['window_count']; count+=row['window_count']
    if not count: raise ValueError('no validation windows')
    a=totals/count
    return dict(observation=float(a[0]),consistency=float(a[1]),weighted_consistency=float(LAMBDA*a[1]),total=float(a[2]))


def selection_epoch(rows): return int(np.argmin([r['observation'] for r in rows]))+1


def save_consistency(path,model,stats,metadata):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(kind='joint_latent_world_model_v2_consistency',state_dict=model.state_dict(),statistics=stats,metadata=metadata),path)


def load_consistency(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['kind']!='joint_latent_world_model_v2_consistency': raise ValueError('invalid consistency checkpoint')
    model=JointLatentWorldModel(); model.load_state_dict(saved['state_dict'])
    return model.eval(),saved['statistics'],saved['metadata']


def train_consistency(model,train,val,stats,scale,config,path):
    if config['horizon']!=10 or config['seed']!=0 or config['learning_rate']!=.0003:
        raise ValueError('fixed K10 seed0 lr.0003 only')
    seed=config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model.requires_grad_(True); before=component_hashes(model); initial=validation_losses(model,val,stats,scale,10,config['batch_size'])
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']); rng=np.random.default_rng(seed)
    history=[]; best=float('inf'); started=time.monotonic(); parameters=list(model.parameters())
    slices={}; offset=0
    for k in before:
        length=len(list(getattr(model,k).parameters())); slices[k]=slice(offset,offset+length); offset+=length
    for epoch in range(1,config['epochs']+1):
        model.train(); total=np.zeros(3); count=0; norms={k:[] for k in before}
        objective={k:[] for k in before}; maxima=dict(normalized_prediction_abs_max=0.,latent_norm_max=0.)
        order=rng.permutation(len(train))
        for begin in range(0,len(train),config['batch_size']):
            selected=[train[j] for j in order[begin:begin+config['batch_size']]]
            if not any(len(e['action'])>=10 for e in selected): continue
            optimizer.zero_grad(set_to_none=True)
            row=loss_batch(model,selected,stats,scale,10)
            for k,v in diagnostics(row).items(): maxima[k]=max(maxima[k],v)
            if not torch.isfinite(row['loss']): raise FloatingPointError('non-finite consistency training loss; no clipping')
            # Read-only gradient measurement, no RNG use or parameter modification.
            cg=torch.autograd.grad(LAMBDA*row['consistency_loss'],parameters,retain_graph=True,allow_unused=True)
            row['loss'].backward()
            for k in before:
                ps=parameters[slices[k]]; cs=cg[slices[k]]
                if any(p.grad is None for p in ps): raise AssertionError(f'{k} missing total gradient')
                cn=torch.sqrt(sum(c.square().sum() for c in cs if c is not None)) if any(c is not None for c in cs) else torch.tensor(0.)
                tn=torch.sqrt(sum(p.grad.square().sum() for p in ps))
                # Total minus weighted-consistency gives observation gradient by
                # linearity, subject to floating-point subtraction roundoff.
                on=torch.sqrt(sum((p.grad-(c if c is not None else 0)).square().sum() for p,c in zip(ps,cs)))
                if not torch.isfinite(torch.stack([cn,tn,on])).all(): raise FloatingPointError('non-finite module gradient; no clipping')
                norms[k].append(tn.item()); objective[k].append((cn.item(),on.item(),cn.item()/max(on.item(),1e-12),cn.item()/max(tn.item(),1e-12)))
            optimizer.step()
            total+=np.array([row['observation_loss'].item(),row['consistency_loss'].item(),row['loss'].item()])*row['window_count']; count+=row['window_count']
        if not count: raise ValueError('no training windows')
        valid=validation_losses(model,val,stats,scale,10,config['batch_size']); a=total/count
        row=dict(epoch=epoch,train=dict(observation=float(a[0]),consistency=float(a[1]),weighted_consistency=float(LAMBDA*a[1]),total=float(a[2])),
            validation=valid,windows_used=count,gradient_norms={k:dict(mean=float(np.mean(v)),max=max(v)) for k,v in norms.items()},
            objective_gradient_diagnostics={k:dict(weighted_consistency_norm=float(np.mean([a[0] for a in v])),
                observation_norm=float(np.mean([a[1] for a in v])),mean_consistency_to_observation_ratio=float(np.mean([a[2] for a in v])),
                max_consistency_to_observation_ratio=float(max(a[2] for a in v)),mean_consistency_to_total_ratio=float(np.mean([a[3] for a in v])),
                fraction_batches_consistency_norm_exceeds_observation=float(np.mean([a[0]>a[1] for a in v]))) for k,v in objective.items()},**maxima)
        history.append(row)
        if valid['observation']<best:
            best=valid['observation']; save_consistency(path,model,stats,dict(config=config,best_epoch=epoch,validation_observation_loss=best,
                parameter_hashes_before=before,initial_latent_std=list(scale),lambda_consistency=LAMBDA,
                selection_metric='Validation observation objective only, uniform all-window k0..10 normalized MSE'))
        print(f'Consistency epoch{epoch:02d}/{config["epochs"]} train_obs={a[0]:.8f} train_cons={a[1]:.8f} '
              f'val_obs={valid["observation"]:.8f} val_cons={valid["consistency"]:.8f} '
              f'grad_max E/T/D={"/".join(f"{max(norms[k]):.5f}" for k in before)} latent_max={maxima["latent_norm_max"]:.3f}',flush=True)
    selected,_,meta=load_consistency(path); after=component_hashes(selected)
    if any(before[k]==after[k] for k in before): raise AssertionError('component did not update')
    return dict(best_epoch=meta['best_epoch'],final_epoch=config['epochs'],initial_validation=initial,
        best_validation=validation_losses(selected,val,stats,scale,10,config['batch_size']),
        best_checkpoint_train=validation_losses(selected,train,stats,scale,10,config['batch_size']),
        final_train=history[-1]['train'],final_validation=history[-1]['validation'],history=history,
        parameter_hashes_before=before,parameter_hashes_best=after,parameter_count=sum(p.numel() for p in model.parameters()),
        architecture={k:repr(getattr(model,k)) for k in before},elapsed_seconds=time.monotonic()-started,
        optimizer='same v1 Adam defaults lr=.0003 betas=.9/.999 eps1e-8; no clipping',
        order_rule='same default_rng(seed0) complete-episode permutations/all legalK10 starts/16episodes per batch',
        objective_gradient_rule='all optimizer minibatches: weighted consistency via autograd.grad; total via loss.backward; observation=total-consistency (FP roundoff); norm ratios describe dominance, no updates changed')
