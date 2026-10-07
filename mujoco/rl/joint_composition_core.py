"""Frozen latent composition and Train-only distribution proxies; no optimizer."""
import numpy as np
import torch
from joint_latent_world_model import predict_latent


def freeze_joint(model):
    model.eval().requires_grad_(False)
    for p in model.parameters(): p.grad=None
    return model


@torch.inference_mode()
def compose(model,z0,actions,stats,interval=None,reference=None):
    """Correct input state BEFORE step k+1; terminal prediction is never corrected.

    reference[:,k] is encoder on real history through t+k, diagnostic only.
    Interval>=horizon ignores all future reference, exactly like pure prediction.
    """
    if interval is not None and (not isinstance(interval,(int,np.integer)) or interval<1):
        raise ValueError('invalid correction interval')
    if (z0.ndim!=2 or z0.shape[1]!=64 or actions.ndim!=3 or actions.shape[0]!=len(z0)
            or actions.shape[2]!=4 or actions.shape[1]<1): raise ValueError('invalid composition shape')
    if any(p.requires_grad for p in model.parameters()) or model.training:
        raise ValueError('composition requires frozen eval model')
    h=actions.shape[1]
    if not torch.isfinite(z0).all() or not torch.isfinite(actions).all():
        raise FloatingPointError('nonfinite composition input')
    if interval is None or interval>=h:
        out=predict_latent(model,z0,actions,stats); corrected=[]
    else:
        if reference is None or reference.ndim!=3 or reference.shape[0]!=len(z0) or reference.shape[1]<h or reference.shape[2]!=64:
            raise ValueError('invalid correction reference shape')
        current=z0.float(); zs=[current]; os=[model.decoder(current)]; corrected=[]
        for k in range(h):
            if k>0 and k%interval==0:
                current=reference[:,k].float(); corrected.append(k)
            step=predict_latent(model,current,actions[:,k:k+1],stats)
            current=step['latents'][:,-1]; zs.append(current); os.append(step['normalized_observations'][:,-1])
        out=dict(latents=torch.stack(zs,1),normalized_observations=torch.stack(os,1))
    if not torch.isfinite(out['latents']).all() or not torch.isfinite(out['normalized_observations']).all():
        raise FloatingPointError('nonfinite composition output; no prediction clipping')
    return dict(latents=out['latents'],normalized_observations=out['normalized_observations'],correction_steps=corrected)


def fit_manifold(train_sequences,train_actions):
    """Descriptive Train-only geometry; no learned model or Test thresholds."""
    z=np.concatenate(train_sequences).astype(np.float64); actions=np.concatenate(train_actions).astype(np.float64)
    if z.ndim!=2 or z.shape[1]!=64 or len(z)<2 or actions.ndim!=2 or actions.shape[1]!=4 or not len(actions):
        raise ValueError('invalid Train manifold input')
    if not np.isfinite(z).all() or not np.isfinite(actions).all(): raise ValueError('nonfinite Train statistics')
    mean=z.mean(0); centered=z-mean; covariance=centered.T@centered/len(z)
    eig,basis=np.linalg.eigh(covariance); eig=np.maximum(eig[::-1],0); basis=basis[:,::-1]
    trace=float(eig.sum()); components=int(np.searchsorted(np.cumsum(eig)/trace,.95)+1) if trace>1e-20 else 0
    regularization=max(1e-12,1e-4*trace/64)
    return dict(mean=mean.tolist(),std=np.maximum(z.std(0),1e-6).tolist(),covariance=covariance.tolist(),
        covariance_regularization=regularization,regularization_rule='max(1e-12,1e-4*Train covariance trace/64)',
        pca_eigenvalues=eig.tolist(),pca_basis=basis[:,:components].tolist(),pca_components=components,
        pca_retained_variance_fraction=float(eig[:components].sum()/trace) if trace>1e-20 else 0.,
        pca_rule='smallest Train PCA subspace explaining>=95% variance',
        action_norm_tertiles=np.quantile(np.linalg.norm(actions,axis=1),[1/3,2/3]).tolist(),
        action_norm_rule='executed normalized4D action L2, Train-only tertiles',
        provenance=dict(split='train',latent_frames=len(z),action_transitions=len(actions),std_floor=1e-6))


def distribution_scores(z,statistics):
    z=np.asarray(z,np.float64); mean=np.asarray(statistics['mean']); std=np.asarray(statistics['std'])
    if z.ndim!=2 or z.shape[1]!=64 or not np.isfinite(z).all(): raise ValueError('invalid latent distribution query')
    centered=z-mean; cov=np.asarray(statistics['covariance']); reg=statistics['covariance_regularization']
    inverse=np.linalg.inv(cov+reg*np.eye(64)); basis=np.asarray(statistics['pca_basis'])
    residual=centered-centered@basis@basis.T if statistics['pca_components'] else centered
    norm=np.linalg.norm(centered,axis=1); rn=np.linalg.norm(residual,axis=1)
    return dict(standardized_rms=np.sqrt(np.mean((centered/std)**2,axis=1)),
        mahalanobis_rms=np.sqrt(np.maximum(np.einsum('ni,ij,nj->n',centered,inverse,centered,optimize=True)/64,0)),
        pca_residual_l2=rn,pca_residual_fraction=rn/np.maximum(norm,1e-12),latent_norm=np.linalg.norm(z,axis=1))


def describe(values):
    a=np.asarray(values,np.float64)
    if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError('nonfinite/invalid descriptive values')
    if not len(a): return dict(count=0)
    return dict(count=len(a),mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),
        q05=float(np.quantile(a,.05)),median=float(np.median(a)),q95=float(np.quantile(a,.95)))


def latent_metrics(predicted,reference,scale):
    p=np.asarray(predicted,np.float64); r=np.asarray(reference,np.float64); scale=np.asarray(scale,np.float64)
    if p.shape!=r.shape or p.ndim!=2 or p.shape[1]!=64 or scale.shape!=(64,) or (scale<=0).any():
        raise ValueError('invalid latent metric shape/scale')
    if not np.isfinite(p).all() or not np.isfinite(r).all(): raise FloatingPointError('nonfinite latent metric')
    if not len(p): return dict(count=0)
    error=p-r; pn=np.linalg.norm(p,axis=1); rn=np.linalg.norm(r,axis=1)
    cosine=(p*r).sum(1)/np.maximum(pn*rn,1e-12); cosine[(pn<1e-12)&(rn<1e-12)]=1.
    return dict(count=len(p),normalized_rmse=float(np.sqrt(np.mean((error/scale)**2))),
        normalized_mae=float(np.abs(error/scale).mean()),raw_rmse=float(np.sqrt(np.mean(error**2))),raw_mae=float(np.abs(error).mean()),
        mean_l2=float(np.linalg.norm(error,axis=1).mean()),mean_cosine=float(cosine.mean()),
        predicted_mean_norm=float(pn.mean()),reference_mean_norm=float(rn.mean()),predicted_max_norm=float(pn.max()))


def pearson(x,y):
    x=np.asarray(x,np.float64); y=np.asarray(y,np.float64)
    if x.shape!=y.shape: raise ValueError('correlation shape mismatch')
    good=np.isfinite(x)&np.isfinite(y); x=x[good]; y=y[good]
    if len(x)<2 or x.std()<1e-15 or y.std()<1e-15: return None
    return float(np.clip(np.corrcoef(x,y)[0,1],-1,1))
