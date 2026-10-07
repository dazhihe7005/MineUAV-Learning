"""Read-only pure latent prediction; future encoder consistency is offline only."""
import numpy as np
import torch

from explicit_latent_models import complete_observations
from latent_dynamics_evaluation import metrics,distance_masks,magnitude_masks
from latent_dynamics_multistep import ErrorAccumulator
from joint_latent_world_model import encode_episode,predict_latent,latent_summary


@torch.inference_mode()
def encoded_sequences(model,episodes,stats):
    model.eval(); result=[]
    for ep in episodes:
        obs=complete_observations(ep)
        previous=np.vstack([ep['previous_action'],ep['action'][-1]])
        result.append(encode_episode(model,obs,previous,stats).numpy().copy())
    return result


@torch.inference_mode()
def reconstruction(model,episodes,stats,sequences=None):
    seq=encoded_sequences(model,episodes,stats) if sequences is None else sequences
    mean=np.asarray(stats['obs']['mean']); std=np.asarray(stats['obs']['std']); pred=[]; truth=[]
    for ep,z in zip(episodes,seq):
        pred.append(model.decoder(torch.from_numpy(z)).numpy().astype(np.float64)*std+mean)
        truth.append(complete_observations(ep))
    p,t=np.concatenate(pred),np.concatenate(truth)
    return dict(normalized_observation_rmse=float(np.sqrt(np.mean(((p-t)/std)**2))),physical=metrics(p,t),
                frame_count=len(p),rule='D(E(real current history)); all saved N+1frames, offline current reconstruction, NOT rollout floor')


class Consistency:
    def __init__(self,scale): self.scale=np.asarray(scale); self.count=0; self.sum=np.zeros(5); self.max_norm=0.
    def add(self,p,t):
        if not len(p): return
        p=np.asarray(p,np.float64); t=np.asarray(t,np.float64)
        if not np.isfinite(p).all() or not np.isfinite(t).all(): raise FloatingPointError('non-finite joint latent consistency')
        norm=np.linalg.norm(p,axis=1); tn=np.linalg.norm(t,axis=1); cos=(p*t).sum(1)/np.maximum(norm*tn,1e-12)
        cos[(norm<1e-12)&(tn<1e-12)]=1.
        self.sum+=np.array([np.linalg.norm(p-t,axis=1).sum(),(((p-t)/self.scale)**2).sum()/64,cos.sum(),norm.sum(),tn.sum()])
        self.count+=len(p); self.max_norm=max(self.max_norm,float(norm.max()))
    def result(self):
        if not self.count: return dict(window_count=0)
        a=self.sum/self.count
        return dict(window_count=self.count,mean_l2=float(a[0]),train_std_normalized_distance=float(np.sqrt(a[1])),
            mean_cosine=float(a[2]),predicted_mean_norm=float(a[3]),encoded_mean_norm=float(a[4]),predicted_max_norm=self.max_norm,
            interpretation='offline same-Joint-Encoder consistency; Train post-training std diagnostic only, never a loss or input scale')


@torch.inference_mode()
def evaluate_joint(model,episodes,stats,thresholds,horizons,train_latents):
    model.eval(); sequences=encoded_sequences(model,episodes,stats)
    train_array=np.concatenate(train_latents).astype(np.float64)
    scale=np.maximum(train_array.std(0),1e-6)
    def pair(): return ErrorAccumulator(stats['obs']['std']),Consistency(scale)
    overall={h:pair() for h in horizons}; common={h:pair() for h in horizons}
    regions={h:{field:{name:pair() for name in masks} for field,masks in [
        ('distance_regions',distance_masks(np.zeros(1))),('pi_magnitude_regions',magnitude_masks(np.zeros(1),thresholds))]} for h in horizons}
    stability=dict(nonfinite_predictions=0,nonfinite_latents=0,first_nonfinite=None,predicted_observation_abs_max=0.,latent_norm_max=0.)
    obs_mean=np.asarray(stats['obs']['mean']); obs_std=np.asarray(stats['obs']['std'])
    for ei,(ep,z) in enumerate(zip(episodes,sequences)):
        n=len(ep['action'])
        for h in horizons:
            for begin in range(0,max(0,n-h+1),512):
                ids=np.arange(begin,min(n-h+1,begin+512))
                actions=torch.from_numpy(ep['action'][ids[:,None]+np.arange(h)].astype(np.float64))
                # Future z enters ONLY diagnostic below, not predict_latent.
                out=predict_latent(model,torch.from_numpy(z[ids]),actions,stats)
                lat=out['latents'].numpy(); obs=out['normalized_observations'].numpy().astype(np.float64)*obs_std+obs_mean
                if not np.isfinite(lat).all() or not np.isfinite(obs).all():
                    raise FloatingPointError(f'joint numerical explosion episode{ei} H{h} start{begin}; no prediction clipping')
                p=lat[:,-1]; ref=z[ids+h]; error=obs[:,-1]-ep['next_obs'][ids+h-1]
                def add(acc,mask): acc[0].add(error[mask]); acc[1].add(p[mask],ref[mask])
                add(overall[h],np.ones(len(ids),bool)); add(common[h],ids<max(0,n-max(horizons)+1))
                for field,masks in [('distance_regions',distance_masks(np.linalg.norm(ep['obs'][ids,:3],axis=1))),
                                    ('pi_magnitude_regions',magnitude_masks(np.linalg.norm(ep['pi'][ids],axis=1),thresholds))]:
                    for name,mask in masks.items(): add(regions[h][field][name],mask)
                stability['predicted_observation_abs_max']=max(stability['predicted_observation_abs_max'],float(np.abs(obs).max()))
                stability['latent_norm_max']=max(stability['latent_norm_max'],float(np.linalg.norm(lat,axis=2).max()))
        if (ei+1)%25==0: print(f'Joint evaluation {ei+1}/{len(episodes)} episodes',flush=True)
    def result(acc): return dict(observation=acc[0].result(),latent_consistency=acc[1].result())
    return dict(horizons={str(h):dict(result(overall[h]),seconds=h/25,
        **{field:{name:result(acc) for name,acc in group.items()} for field,group in regions[h].items()}) for h in horizons},
        common_max_horizon_windows={str(h):result(common[h]) for h in horizons},stability=stability,
        current_reconstruction=reconstruction(model,episodes,stats,sequences),
        latent_variance=dict(train=latent_summary(train_array),test=latent_summary(np.concatenate(sequences))))
