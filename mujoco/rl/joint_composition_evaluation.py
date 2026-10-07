"""Read-only A–F diagnostics; future reference is target or explicit correction."""
import numpy as np
import torch

from joint_latent_evaluation import encoded_sequences,reconstruction
from joint_composition_core import compose,latent_metrics,distribution_scores,describe,pearson
from latent_dynamics_multistep import ErrorAccumulator
from latent_dynamics_evaluation import distance_masks,magnitude_masks


class Records:
    def __init__(self): self.rows=[]
    def add(self,**row): self.rows.append(row)
    def arrays(self): return {k:np.concatenate([r[k] for r in self.rows]) for k in self.rows[0]}


def summary(rows,scale,obs_std,mask=None):
    mask=np.ones(len(rows['p']),bool) if mask is None else mask
    error=ErrorAccumulator(obs_std); error.add(rows['error'][mask])
    return dict(latent=latent_metrics(rows['p'][mask],rows['ref'][mask],scale),observation=error.result())


def grouped(rows,scale,obs_std,pi_thresholds,action_thresholds):
    bins=[('distance_regions',distance_masks(rows['distance'])),
          ('pi_magnitude_regions',magnitude_masks(rows['pi'],pi_thresholds)),
          ('action_magnitude_regions',magnitude_masks(rows['action'],action_thresholds))]
    return {field:{name:summary(rows,scale,obs_std,mask) for name,mask in masks.items()} for field,masks in bins}


def additional(rows,manifold,obs_std):
    p,r=rows['p'].astype(np.float64),rows['ref'].astype(np.float64)
    latent_l2=np.linalg.norm(p-r,axis=1); decoded_l2=np.linalg.norm(rows['decoded_difference'],axis=1)
    pred,ref=distribution_scores(p,manifold),distribution_scores(r,manifold)
    excess={k:pred[k]-ref[k] for k in pred}
    norm_obs=np.sqrt(np.mean((rows['error']/obs_std)**2,axis=1))
    horizontal=np.sqrt(np.mean(rows['error'][:,3:5]**2,axis=1))
    correlations={k:dict(predicted=pearson(v,horizontal),reference=pearson(ref[k],horizontal),
                         paired_excess=pearson(excess[k],horizontal)) for k,v in pred.items()}
    velocity_bins={}
    # Descriptive partitions use Train-distribution PCA-residual quantiles, not Test fitting.
    cuts=manifold['train_score_quantiles']['pca_residual_l2']
    for name,mask in magnitude_masks(pred['pca_residual_l2'],cuts).items():
        e=rows['error'][mask]
        velocity_bins[name]=dict(count=int(mask.sum()),vx_rmse=float(np.sqrt(np.mean(e[:,3]**2))) if len(e) else None,
            vy_rmse=float(np.sqrt(np.mean(e[:,4]**2))) if len(e) else None,
            horizontal_velocity_rmse=float(np.sqrt(np.mean(e[:,3:5]**2))) if len(e) else None)
    return dict(distribution=dict(predicted={k:describe(v) for k,v in pred.items()},
        reference={k:describe(v) for k,v in ref.items()},paired_excess={k:describe(v) for k,v in excess.items()}),
        decoder_sensitivity=dict(latent_l2=describe(latent_l2),decoded_normalized_l2=describe(decoded_l2),
            amplification_ratio=describe(decoded_l2/(latent_l2+1e-12)),epsilon=1e-12,
            rule='per-sample normalized7D decoded L2/raw64D latent L2; empirical ratio, NOT Jacobian spectral norm'),
        local_residual_future_correlations=dict(count=len(p),latent_l2=pearson(rows['local_residual'],latent_l2),
            normalized_observation_error=pearson(rows['local_residual'],norm_obs),horizontal_velocity_error=pearson(rows['local_residual'],horizontal)),
        horizontal_error_distribution_correlations=correlations,velocity_by_train_pca_residual_bins=velocity_bins,
        correlation_interpretation='Descriptive Pearson only; compare reference and paired-excess proxies; overlapping windows not independent evidence.')


@torch.inference_mode()
def audit(model,train,test,stats,pi_thresholds,horizons,selected,manifold):
    sequences=encoded_sequences(model,test,stats); scale=np.asarray(manifold['std']); obs_std=np.asarray(stats['obs']['std']); obs_mean=np.asarray(stats['obs']['mean'])
    local=Records(); autonomous={h:Records() for h in horizons}; corrected={c:Records() for c in (1,5,10,25)}
    local_movement=[]; local_identity_p=[]; local_identity_r=[]
    stability=dict(nonfinite_predictions=0,nonfinite_latents=0,prediction_abs_max=0.,latent_norm_max=0.)
    def record(out,z,ep,ids,h,decoded_ref,residual):
        p=out['latents'][:,-1].numpy(); pn=out['normalized_observations'][:,-1].numpy().astype(np.float64)
        physical=pn*obs_std+obs_mean
        stability['prediction_abs_max']=max(stability['prediction_abs_max'],float(np.abs(out['normalized_observations'].numpy()*obs_std+obs_mean).max()))
        stability['latent_norm_max']=max(stability['latent_norm_max'],float(np.linalg.norm(out['latents'].numpy(),axis=-1).max()))
        return dict(p=p,ref=z[ids+h],error=physical-ep['next_obs'][ids+h-1],
            decoded_difference=pn-decoded_ref[ids+h],distance=np.linalg.norm(ep['obs'][ids,:3],axis=1),
            pi=np.linalg.norm(ep['pi'][ids],axis=1),action=np.linalg.norm(ep['action'][ids],axis=1),local_residual=residual[ids])
    for ei,(ep,z) in enumerate(zip(test,sequences)):
        n=len(ep['action']); decoded_ref=model.decoder(torch.from_numpy(z)).numpy().astype(np.float64)
        # Every local transition independently starts on encoder-reference history.
        one=compose(model,torch.from_numpy(z[:-1]),torch.from_numpy(ep['action'][:,None].astype(np.float64)),stats)
        residual=np.linalg.norm(one['latents'][:,-1].numpy().astype(np.float64)-z[1:],axis=1)
        local.add(**record(one,z,ep,np.arange(n),1,decoded_ref,residual))
        local_movement.append(np.linalg.norm(np.diff(z.astype(np.float64),axis=0),axis=1))
        local_identity_p.append(z[:-1]); local_identity_r.append(z[1:])
        for h in horizons:
            for begin in range(0,max(0,n-h+1),512):
                ids=np.arange(begin,min(n-h+1,begin+512)); indices=ids[:,None]+np.arange(h)
                actions=torch.from_numpy(ep['action'][indices].astype(np.float64)); z0=torch.from_numpy(z[ids])
                out=compose(model,z0,actions,stats)
                autonomous[h].add(**record(out,z,ep,ids,h,decoded_ref,residual))
                if h==50:
                    reference=torch.from_numpy(z[ids[:,None]+np.arange(h+1)])
                    for c in corrected:
                        co=compose(model,z0,actions,stats,c,reference)
                        corrected[c].add(**record(co,z,ep,ids,h,decoded_ref,residual))
        if (ei+1)%25==0: print(f'Composition audit {ei+1}/{len(test)} Test episodes',flush=True)
    localrows=local.arrays(); groups=grouped(localrows,scale,obs_std,pi_thresholds,manifold['action_norm_tertiles'])
    local_result=dict(summary(localrows,scale,obs_std),**groups,
        identity_transition_reference=latent_metrics(np.concatenate(local_identity_p),np.concatenate(local_identity_r),scale),
        encoder_reference_delta_norm=describe(np.concatenate(local_movement)),local_residual_norm=describe(localrows['local_residual']))
    results={}; plot={}
    for h,acc in autonomous.items():
        rows=acc.arrays(); results[str(h)]=dict(summary(rows,scale,obs_std),**grouped(rows,scale,obs_std,pi_thresholds,manifold['action_norm_tertiles']),
            **additional(rows,manifold,obs_std),seconds=h/25)
        if h==50:
            plot['scatter']=dict(local_residual=rows['local_residual'][::max(1,len(rows['p'])//1000)].copy(),
                future_latent_l2=np.linalg.norm(rows['p'].astype(np.float64)-rows['ref'],axis=1)[::max(1,len(rows['p'])//1000)])
    periodic={str(c):dict(summary(r.arrays(),scale,obs_std),**grouped(r.arrays(),scale,obs_std,pi_thresholds,manifold['action_norm_tertiles'])) for c,r in corrected.items()}
    # No actual reset at endpoint: C50/never uses EXACT same already-evaluated rows.
    rows50=autonomous[50].arrays(); pure=dict(summary(rows50,scale,obs_std),**grouped(rows50,scale,obs_std,pi_thresholds,manifold['action_norm_tertiles']))
    periodic['50']=pure; periodic['never']=pure
    ei=selected['episode_id']; start=selected['start']; h=selected['horizon']; ep=test[ei]; z=sequences[ei]
    actions=torch.from_numpy(ep['action'][None,start:start+h].astype(np.float64)); ref=torch.from_numpy(z[None,start:start+h+1]); z0=ref[:,0]
    traces={str(c) if c is not None else 'never':compose(model,z0,actions,stats,c,ref) for c in (1,5,10,25,None)}
    plot['selected']=dict(truth=np.vstack([ep['obs'][start],ep['next_obs'][start:start+h]]),
        reference=z[start:start+h+1],observations={c:o['normalized_observations'][0].numpy()*obs_std+obs_mean for c,o in traces.items()},
        latent_l2={c:np.linalg.norm(o['latents'][0].numpy()-z[start:start+h+1],axis=1) for c,o in traces.items()})
    return dict(encoder_reconstruction=reconstruction(model,test,stats,sequences),local_one_step=local_result,
        autonomous=results,periodic_correction=periodic,stability=stability),plot
