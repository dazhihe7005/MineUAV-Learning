"""Frozen v3 random shooting: normalized commands, physical terminal cost.

No training, simulated future observations, correction branch or observation
feedback exists in this planner. The only persistent state is real-history E.
"""
import time
import numpy as np
import torch
from latent_dynamics_data import normalize

N_CANDIDATES = 512
HORIZON = 10


def sample_candidates(previous, train_std, low, high, rng):
    previous, train_std, low, high = [np.asarray(x, dtype=np.float64) for x in (previous, train_std, low, high)]
    if any(x.shape != (4,) or not np.isfinite(x).all() for x in (previous, train_std, low, high)):
        raise ValueError('finite four-dimensional action statistics required')
    if (train_std <= 0).any() or (low > 0).any() or (high < 0).any() or (low >= high).any() or (previous < low).any() or (previous > high).any():
        raise ValueError('invalid command bounds/statistics/previous action')
    noise = rng.normal(size=(N_CANDIDATES-2,HORIZON,4)) * (.5*train_std)
    candidates = np.empty((N_CANDIDATES,HORIZON,4),np.float32)
    candidates[0] = 0; candidates[1] = previous
    current = np.broadcast_to(previous,(N_CANDIDATES-2,4)).copy()
    for k in range(HORIZON):
        current = np.clip(current+noise[:,k],low,high)
        candidates[2:,k] = current
    return candidates


def planning_cost(terminal_observation, candidates, previous):
    obs, actions, previous = [np.asarray(x,np.float64) for x in (terminal_observation,candidates,previous)]
    if obs.shape != (len(actions),7) or actions.shape[1:] != (HORIZON,4) or previous.shape != (4,):
        raise ValueError('invalid terminal/candidate cost shape')
    if not all(np.isfinite(x).all() for x in (obs,actions,previous)):
        raise FloatingPointError('nonfinite planning cost inputs; no hidden clipping/fallback')
    differences=np.diff(np.concatenate([np.broadcast_to(previous,(len(actions),1,4)),actions],1),axis=1)
    parts=dict(position=np.square(obs[:,:3]).sum(1),velocity=np.square(obs[:,3:6]).sum(1),
               yaw=np.square(obs[:,6]),smoothness=np.square(differences).sum(2).mean(1))
    return parts['position']+.5*parts['velocity']+.1*parts['yaw']+.05*parts['smoothness'],parts


def select_lowest(costs):
    if np.ndim(costs)!=1 or len(costs)==0 or not np.isfinite(costs).all():
        raise FloatingPointError('invalid/nonfinite shooting costs')
    return int(np.argmin(costs))


class RandomShootingMPC:
    def __init__(self,model,statistics,low,high,seed=0):
        self.model=model.eval().requires_grad_(False)
        self.statistics=statistics
        self.low=np.asarray(low,np.float32); self.high=np.asarray(high,np.float32)
        self.device=next(model.parameters()).device
        self.action_mean=torch.tensor(statistics['action']['mean'],device=self.device,dtype=torch.float32)
        self.action_std=torch.tensor(statistics['action']['std'],device=self.device,dtype=torch.float32)
        self.obs_mean=np.asarray(statistics['obs']['mean'])
        self.obs_std=np.asarray(statistics['obs']['std'])
        self.reset(seed)

    def reset(self,seed):
        self.hidden=None
        self.rng=np.random.default_rng(seed)
        self.encoding_count=0
        self.planning_count=0

    @torch.inference_mode()
    def encode_current(self,observation,previous_action):
        observation=np.asarray(observation,np.float32); previous_action=np.asarray(previous_action,np.float32)
        if observation.shape!=(7,) or previous_action.shape!=(4,) or not np.isfinite(np.r_[observation,previous_action]).all():
            raise ValueError('one real observation7 and previous executed action4 required')
        x=np.r_[normalize(observation,self.statistics,'obs'),normalize(previous_action,self.statistics,'previous_action')]
        z,self.hidden=self.model.encoder(torch.from_numpy(x).to(self.device)[None,None],self.hidden)
        self.encoding_count+=1
        return z[:,0]

    @torch.inference_mode()
    def plan(self,z,previous_action):
        if z.shape!=(1,64) or not torch.isfinite(z).all(): raise ValueError('finite current inferred latent1x64 required')
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        started=time.perf_counter()
        candidates=self.make_candidates(previous_action)
        actions=torch.from_numpy(candidates).to(self.device)
        current=z.expand(N_CANDIDATES,-1)
        normalized=(actions-self.action_mean)/self.action_std
        for k in range(HORIZON):
            current=current+self.model.transition(current,normalized[:,k])
        terminal=self.model.decoder(current).cpu().numpy()*self.obs_std+self.obs_mean
        costs,parts=planning_cost(terminal,candidates,previous_action)
        index=select_lowest(costs)
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        elapsed=time.perf_counter()-started
        self.planning_count+=1
        return dict(action=candidates[index,0].copy(),index=index,candidates=candidates,costs=costs,
                    terminal_observation=terminal[index].copy(),cost_parts={k:float(v[index]) for k,v in parts.items()},
                    selected_cost=float(costs[index]),candidate_cost_min=float(costs.min()),
                    candidate_cost_median=float(np.median(costs)),candidate_cost_mean=float(costs.mean()),
                    planning_seconds=elapsed)

    def make_candidates(self,previous_action):
        # Default remains the original sampler; interventions override ONLY this.
        return sample_candidates(previous_action,self.statistics['action']['std'],self.low,self.high,self.rng)
