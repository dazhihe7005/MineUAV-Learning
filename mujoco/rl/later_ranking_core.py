"""Fixed-stage, true-best-tail and Train-only distribution diagnostics."""
import numpy as np
import torch
from decision_fidelity_metrics import correlation
from joint_composition_core import fit_manifold,distribution_scores

STAGES=('S0','S1','S2','S3')

def stage_indices(length):
    if not isinstance(length,(int,np.integer)) or length<1:
        raise ValueError('positive episode decision length required')
    out={}; used=set()
    for s,p in zip(STAGES,(0,.25,.5,.75)):
        i=int(np.floor(p*(length-1)+.5))
        if i not in used: out[s]=i; used.add(i)
    return out

def tail_metrics(predicted,true):
    p=np.asarray(predicted,float); t=np.asarray(true,float)
    if p.shape!=(512,) or t.shape!=(512,) or not np.isfinite(np.r_[p,t]).all():
        raise ValueError('all original512 finite costs required')
    ids=np.argsort(t,kind='stable')[:26]; x=p[ids]; y=t[ids]
    i,j=np.triu_indices(26,1); sx=np.sign(x[i]-x[j]); sy=np.sign(y[i]-y[j]); comparable=sy!=0
    # A prediction tie earns half credit; true ties do not require an ordering.
    accuracy=np.mean((sx[comparable]*sy[comparable]+1)/2) if comparable.any() else None
    return dict(count=26,spearman=correlation(x,y),kendall=correlation(x,y,'kendall'),
        pearson=correlation(x,y,'pearson'),cost_mae=float(np.abs(x-y).mean()),
        ordering_accuracy=float(accuracy) if accuracy is not None else None,
        comparable_pairs=int(comparable.sum()),tied_prediction_pairs=int((sx==0).sum()),
        true_cost_spread=float(np.ptp(y)),indices=ids.tolist())

def fit_geometry(latents,observations,actions,split):
    if split!='train': raise ValueError('only Train statistics allowed')
    z,o,a=[np.asarray(x,np.float64) for x in (latents,observations,actions)]
    if o.shape!=(len(z),7) or a.shape!=(len(z),4) or not np.isfinite(np.r_[o.ravel(),a.ravel()]).all():
        raise ValueError('aligned finite Train observations/actions required')
    geometry=fit_manifold([z],[a])
    geometry['observation']=dict(mean=o.mean(0).tolist(),std=np.maximum(o.std(0),1e-6).tolist())
    geometry['action']=dict(mean=a.mean(0).tolist(),std=np.maximum(a.std(0),1e-6).tolist())
    return geometry

def ood_scores(z,observation,action,geometry):
    out={k:float(v[0]) for k,v in distribution_scores(np.asarray(z).reshape(1,64),geometry).items()}
    for name,value in [('observation',observation),('action',action)]:
        stats=geometry[name]; scaled=(np.asarray(value)-stats['mean'])/stats['std']
        out[name+'_standardized_rms']=float(np.sqrt(np.mean(scaled**2)))
    return out

@torch.inference_mode()
def prediction_trajectory(model,stats,z,actions):
    """No future-observation argument: only initial latent and original actions."""
    a=np.asarray(actions,np.float32)
    if a.ndim!=3 or a.shape[1:]!=(10,4) or not np.isfinite(a).all() or (np.abs(a)>1).any():
        raise ValueError('original legal N×10×4 sequences required')
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('frozen model required')
    current=torch.as_tensor(z,dtype=torch.float32)
    if current.shape!=(1,64): raise ValueError('one starting latent required')
    current=current.expand(len(a),-1)
    normalized=(torch.from_numpy(a)-torch.tensor(stats['action']['mean'],dtype=torch.float32))/torch.tensor(stats['action']['std'],dtype=torch.float32)
    outputs=[]
    for k in range(10):
        current=current+model.transition(current,normalized[:,k])
        outputs.append(model.decoder(current).numpy()*np.asarray(stats['obs']['std'])+stats['obs']['mean'])
    return np.stack(outputs,1)
