"""Pure latent recursion: no observation/encoder input after initial state."""
import numpy as np
import torch

from latent_dynamics_data import normalize
from latent_dynamics_evaluation import metrics,distance_masks,magnitude_masks
from latent_dynamics_multistep import ErrorAccumulator
from explicit_latent_models import complete_observations,encode_prefix,latent_normalize


@torch.inference_mode()
def latent_rollout(transition,decoder,z0,actions,stats,latent_stats):
    z=np.asarray(z0,np.float64).copy(); actions=np.asarray(actions)
    if (z.ndim!=2 or z.shape[1]!=64 or actions.ndim!=3 or actions.shape[0]!=len(z)
            or actions.shape[2]!=4 or actions.shape[1]<1): raise ValueError('invalid latent rollout shape')
    transition.eval(); decoder.eval(); latents=[]; observations=[]
    scale=np.asarray(latent_stats['std']); obs_std=np.asarray(stats['obs']['std']); obs_mean=np.asarray(stats['obs']['mean'])
    for k in range(actions.shape[1]):
        delta=transition(torch.from_numpy(latent_normalize(z,latent_stats)),
                         torch.from_numpy(normalize(actions[:,k],stats,'action'))).numpy()
        z=z+delta.astype(np.float64)*scale
        normalized=decoder(torch.from_numpy(latent_normalize(z,latent_stats))).numpy()
        latents.append(z.copy()); observations.append(normalized.astype(np.float64)*obs_std+obs_mean)
    return dict(latents=np.stack(latents,axis=1),observations=np.stack(observations,axis=1))


def predict_window(encoder,transition,decoder,episode,start,horizon,stats,latent_stats):
    if (not isinstance(start,int) or not isinstance(horizon,int) or start<0 or horizon<1
            or start>=len(episode['obs']) or start+horizon>len(episode['action'])):
        raise ValueError('invalid prediction window/action boundary')
    # This function remains valid with every future observation removed.
    z=encode_prefix(encoder,episode['obs'][:start+1],episode['previous_action'][:start+1],stats)[-1:]
    return latent_rollout(transition,decoder,z,episode['action'][None,start:start+horizon],stats,latent_stats)


class LatentAccumulator:
    """Same frozen coordinates; endpoint and incremental Delta-z errors."""
    def __init__(self,std):
        self.std=np.asarray(std); self.count=0; self.sum=np.zeros(6); self.max_norm=0.; self.nonfinite=0

    def add(self,prediction,teacher,delta_prediction,delta_teacher):
        good=np.isfinite(prediction).all(1)&np.isfinite(delta_prediction).all(1)
        self.nonfinite+=int((~good).sum())
        p,t,dp,dt=[a[good].astype(np.float64) for a in (prediction,teacher,delta_prediction,delta_teacher)]
        if not len(p): return
        norm=np.linalg.norm(p,axis=1); teacher_norm=np.linalg.norm(t,axis=1)
        cosine=np.sum(p*t,axis=1)/np.maximum(norm*teacher_norm,1e-12)
        cosine[(norm<1e-12)&(teacher_norm<1e-12)]=1.
        error=p-t; delta_error=dp-dt; self.count+=len(p)
        self.sum+=np.array([np.sum((error/self.std)**2)/64,np.sum(error**2)/64,
            np.abs(error).sum()/64,cosine.sum(),norm.sum(),np.sum(delta_error**2)/64])
        self.max_norm=max(self.max_norm,float(norm.max()))

    def result(self):
        if not self.count: return dict(window_count=0,nonfinite_windows=self.nonfinite)
        average=self.sum/self.count
        return dict(window_count=self.count,nonfinite_windows=self.nonfinite,
            normalized_latent_rmse=float(np.sqrt(average[0])),latent_rmse=float(np.sqrt(average[1])),
            latent_mae=float(average[2]),mean_cosine=float(average[3]),mean_predicted_norm=float(average[4]),
            max_predicted_norm=self.max_norm,increment_delta_z_rmse=float(np.sqrt(average[5])),
            delta_error_rule='(pred_z[k]-pred_z[k-1])-(teacher_z[k]-teacher_z[k-1]); endpoint increment, not displacement')


@torch.inference_mode()
def one_step_metrics(transition,episodes,stats,latent_stats):
    acc=LatentAccumulator(latent_stats['std']); identity=LatentAccumulator(latent_stats['std'])
    transition.eval()
    for ep in episodes:
        z=ep['latent'].astype(np.float64); delta=transition(torch.from_numpy(latent_normalize(z[:-1],latent_stats)),
            torch.from_numpy(normalize(ep['action'],stats,'action'))).numpy().astype(np.float64)*np.asarray(latent_stats['std'])
        acc.add(z[:-1]+delta,z[1:],delta,np.diff(z,axis=0))
        identity.add(z[:-1],z[1:],np.zeros_like(delta),np.diff(z,axis=0))
    row=acc.result(); row['identity_transition_reference']=identity.result(); return row


@torch.inference_mode()
def decoder_metrics(decoder,episodes,stats,latent_stats):
    pred=[]; truth=[]; decoder.eval()
    for ep in episodes:
        output=decoder(torch.from_numpy(latent_normalize(ep['latent'],latent_stats))).numpy()
        pred.append(output.astype(np.float64)*np.asarray(stats['obs']['std'])+np.asarray(stats['obs']['mean']))
        truth.append(complete_observations(ep))
    p,t=np.concatenate(pred),np.concatenate(truth)
    return dict(normalized_observation_rmse=float(np.sqrt(np.mean(((p-t)/np.asarray(stats['obs']['std']))**2))),
                physical=metrics(p,t),frame_rule='D(teacher z[t]) -> current recorded observation[t], all saved frames')


def delta_magnitude(episodes):
    values=np.concatenate([np.linalg.norm(np.diff(e['latent'].astype(np.float64),axis=0),axis=1) for e in episodes])
    return dict(count=len(values),mean=float(values.mean()),std=float(values.std()),max=float(values.max()),
                quantiles=dict(zip(['q01','q25','q50','q75','q99'],np.quantile(values,[.01,.25,.5,.75,.99]).tolist())))


def evaluate_explicit(transition,decoder,episodes,stats,latent_stats,thresholds,horizons):
    def pair(): return (ErrorAccumulator(stats['obs']['std']),LatentAccumulator(latent_stats['std']))
    overall={h:pair() for h in horizons}; common={h:pair() for h in horizons}
    regions={h:{field:{name:pair() for name in masks} for field,masks in [
        ('distance_regions',distance_masks(np.zeros(1))),('pi_magnitude_regions',magnitude_masks(np.zeros(1),thresholds))]} for h in horizons}
    stability=dict(nonfinite_predictions=0,nonfinite_latents=0,first_nonfinite=None,
                   predicted_observation_abs_max=0.,latent_norm_max=0.)
    for episode_id,ep in enumerate(episodes):
        n=len(ep['obs']); teacher=ep['latent'].astype(np.float64)
        for h in horizons:
            for start in range(0,max(0,n-h+1),512):
                ids=np.arange(start,min(n-h+1,start+512))
                actions=ep['action'][ids[:,None]+np.arange(h)[None,:]]
                out=latent_rollout(transition,decoder,teacher[ids],actions,stats,latent_stats)
                p=out['latents'][:,-1]; t=teacher[ids+h]
                dp=p-(out['latents'][:,-2] if h>1 else teacher[ids]); dt=t-teacher[ids+h-1]
                error=out['observations'][:,-1]-ep['next_obs'][ids+h-1]
                def add(acc,mask):
                    acc[0].add(error[mask]); acc[1].add(p[mask],t[mask],dp[mask],dt[mask])
                add(overall[h],np.ones(len(ids),bool)); add(common[h],ids<max(0,n-max(horizons)+1))
                for field,masks in [('distance_regions',distance_masks(np.linalg.norm(ep['obs'][ids,:3],axis=1))),
                                    ('pi_magnitude_regions',magnitude_masks(np.linalg.norm(ep['pi'][ids],axis=1),thresholds))]:
                    for name,mask in masks.items(): add(regions[h][field][name],mask)
                for array,key in [(out['observations'],'nonfinite_predictions'),(out['latents'],'nonfinite_latents')]:
                    finite=np.isfinite(array); stability[key]+=int((~finite).sum())
                    if not finite.all() and stability['first_nonfinite'] is None:
                        row,k,_=np.argwhere(~finite)[0]
                        stability['first_nonfinite']=dict(episode_id=episode_id,start=int(ids[row]),step=int(k+1),array=key)
                if np.isfinite(out['observations']).any():
                    stability['predicted_observation_abs_max']=max(stability['predicted_observation_abs_max'],float(np.nanmax(np.abs(out['observations']))))
                if np.isfinite(out['latents']).all():
                    stability['latent_norm_max']=max(stability['latent_norm_max'],float(np.linalg.norm(out['latents'],axis=2).max()))
        if (episode_id+1)%25==0: print(f'Pure latent evaluation: {episode_id+1}/{len(episodes)} episodes',flush=True)
    def result(acc): return dict(observation=acc[0].result(),latent=acc[1].result())
    return dict(horizons={str(h):dict(result(overall[h]),seconds=h/25,
            **{field:{name:result(acc) for name,acc in groups.items()} for field,groups in regions[h].items()}) for h in horizons},
        common_max_horizon_windows={str(h):result(common[h]) for h in horizons},stability=stability)
