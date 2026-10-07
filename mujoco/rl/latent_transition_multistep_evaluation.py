"""Offline teacher-latent decoder reference, not a deployable model or bound."""
import numpy as np
import torch

from explicit_latent_models import latent_normalize
from latent_dynamics_multistep import ErrorAccumulator
from latent_dynamics_evaluation import distance_masks,magnitude_masks


@torch.inference_mode()
def decoder_floor(decoder,episodes,stats,latent_stats,thresholds,horizons):
    def acc(): return ErrorAccumulator(stats['obs']['std'])
    overall={h:acc() for h in horizons}; common={h:acc() for h in horizons}
    groups={h:{field:{name:acc() for name in masks} for field,masks in [
        ('distance_regions',distance_masks(np.zeros(1))),('pi_magnitude_regions',magnitude_masks(np.zeros(1),thresholds))]} for h in horizons}
    decoder.eval()
    for ep in episodes:
        n=len(ep['action'])
        if ep['latent'].shape!=(n+1,64): raise ValueError('teacher decoder episode boundary mismatch')
        decoded=decoder(torch.from_numpy(latent_normalize(ep['latent'],latent_stats))).numpy().astype(np.float64)
        decoded=decoded*np.asarray(stats['obs']['std'])+np.asarray(stats['obs']['mean'])
        for h in horizons:
            ids=np.arange(max(0,n-h+1)); error=decoded[ids+h]-ep['next_obs'][ids+h-1]
            overall[h].add(error); common[h].add(error[ids<max(0,n-max(horizons)+1)])
            for field,masks in [('distance_regions',distance_masks(np.linalg.norm(ep['obs'][ids,:3],axis=1))),
                                ('pi_magnitude_regions',magnitude_masks(np.linalg.norm(ep['pi'][ids],axis=1),thresholds))]:
                for name,mask in masks.items(): groups[h][field][name].add(error[mask])
    return dict(horizons={str(h):dict(overall[h].result(),seconds=h/25,
        **{field:{name:a.result() for name,a in rows.items()} for field,rows in groups[h].items()}) for h in horizons},
        common_max_horizon_windows={str(h):common[h].result() for h in horizons},
        protocol='D(teacher z[t+H]) vs recorded o[t+H] at exact shared Test starts; offline reconstruction reference, no T',
        interpretation='Called Decoder Floor by convention; not a mathematical lower bound or additive decomposition of nonlinear T/D error')
