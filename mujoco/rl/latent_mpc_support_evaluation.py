"""Paired support-only diagnostic instrumentation; no environment/model changes."""
import hashlib
import numpy as np
import torch
from latent_mpc_action_support import ActionSupportMPC,TOLERANCE
from latent_mpc_evaluation import action_ood,distribution

PROJECTION_BIN_WIDTH=1e-4


def array_hash(a):
    return hashlib.sha256(np.asarray(a,np.float32).tobytes()).hexdigest()


def support_metrics(actions,train,support):
    a=np.asarray(actions,np.float64)
    bounds=dict(train,q025=support['lower'],q975=support['upper'])
    result=action_ood(a,bounds)
    outside=(a<np.asarray(support['lower'])-TOLERANCE)|(a>np.asarray(support['upper'])+TOLERANCE)
    result.update(outside_train_95_any_dimension_fraction=float(outside.any(1).mean()),
        outside_train_95_element_fraction=float(outside.mean()),
        outside_train_95_per_dimension_fraction=outside.mean(0).tolist(),tolerance=TOLERANCE,
        on_support_boundary_element_fraction=float((np.isclose(a,support['lower'],atol=TOLERANCE,rtol=0)|np.isclose(a,support['upper'],atol=TOLERANCE,rtol=0)).mean()),
        quantile_bounds_source='all executed Train commands, not only paired dynamics rows')
    return result


class ProjectionAccumulator:
    """Exact mean/counts; fixed1e-4 histogram quantiles to avoid raw candidate arrays."""
    def __init__(self):
        self.hist=np.zeros(40001,np.int64)
        self.count=0; self.changed=0; self.changed_elements=0; self.total=0.; self.max=0.
        self.anchors=0; self.calls=0; self.boundary_elements=0; self.unique_fraction_sum=0.

    def update(self,audit,candidates,support):
        mag=np.asarray(audit['projection_magnitudes'],np.float64)
        if mag.shape!=(512,10) or not np.isfinite(mag).all() or (mag<0).any() or (mag>4+1e-9).any():
            raise ValueError('invalid projection magnitudes')
        self.hist+=np.bincount(np.minimum(np.ceil(mag.ravel()/PROJECTION_BIN_WIDTH).astype(int),40000),minlength=40001)
        self.count+=mag.size; self.changed+=int((mag>TOLERANCE).sum())
        self.changed_elements+=int(audit['projected_elements'].sum()); self.total+=float(mag.sum()); self.max=max(self.max,float(mag.max()))
        self.anchors+=audit['anchor_projected_candidates']; self.calls+=1
        if support is not None:
            self.boundary_elements+=int((np.isclose(candidates,support[0],atol=TOLERANCE,rtol=0)|np.isclose(candidates,support[1],atol=TOLERANCE,rtol=0)).sum())
        self.unique_fraction_sum+=len(np.unique(candidates[:,0],axis=0))/512

    def summary(self):
        if not self.count: raise ValueError('no decisions')
        cumulative=np.cumsum(self.hist)
        def quantile(q,positive=False):
            counts=cumulative-self.hist[0] if positive else cumulative
            n=self.count-self.hist[0] if positive else self.count
            return float(np.searchsorted(counts,max(1,int(np.ceil(q*n))))*PROJECTION_BIN_WIDTH) if n else None
        return dict(decisions=self.calls,candidate_timesteps=self.count,
            candidate_timestep_projection_fraction=self.changed/self.count,
            candidate_element_projection_fraction=self.changed_elements/(4*self.count),
            anchor_projection_count=self.anchors,anchor_projection_opportunities=2*self.calls,
            projection_magnitude_all=dict(mean=self.total/self.count,median=quantile(.5),p95=quantile(.95),max=self.max),
            projection_magnitude_projected=dict(mean=self.total/self.changed if self.changed else None,median=quantile(.5,True),p95=quantile(.95,True)),
            quantile_rule='empirical rank rounded UP into fixed1e-4 normalized-L2 bins; error <=1e-4; mean exact',
            candidate_boundary_element_fraction=self.boundary_elements/(4*self.count),
            mean_unique_first_action_fraction=self.unique_fraction_sum/self.calls)

    def merge(self,other):
        self.hist+=other.hist
        for name in ('count','changed','changed_elements','total','anchors','calls','boundary_elements','unique_fraction_sum'):
            setattr(self,name,getattr(self,name)+getattr(other,name))
        self.max=max(self.max,other.max)


class InstrumentedSupportMPC(ActionSupportMPC):
    def reset(self,seed):
        super().reset(seed)
        self.projection=ProjectionAccumulator(); self.epsilon_hashes=[]; self.first=None

    @torch.inference_mode()
    def plan(self,z,previous_action):
        result=super().plan(z,previous_action)
        self.projection.update(self.candidate_audit,result['candidates'],self.support)
        self.epsilon_hashes.append(self.candidate_audit['epsilon_sha256'])
        if self.first is None:
            self.first=dict(first_latent_sha256=array_hash(z.cpu().numpy()),
                first_selected_cost=result['selected_cost'],first_candidate_median_cost=result['candidate_cost_median'],
                first_candidate_mean_cost=result['candidate_cost_mean'],first_selected_candidate_index=result['index'],
                first_predicted_terminal_observation=result['terminal_observation'].tolist(),
                first_epsilon_sha256=self.epsilon_hashes[0])
        return result


def window_manifest(split,episode_index,steps):
    if episode_index>=3: return []
    return [dict(split=split,episode_index=episode_index,decision_index=int(k),horizon=10)
            for k in range(0,max(0,steps-10+1),25)]


def verify_pairing(unconstrained,supported):
    if len(unconstrained)!=len(supported): raise AssertionError('missing paired episode')
    matched=0
    for u,s in zip(unconstrained,supported):
        for key in ('env_seed','target','target_index','first_observation_sha256','first_latent_sha256'):
            if u[key]!=s[key]: raise AssertionError(f'initial paired state/cohort differs: {key}')
        count=min(len(u['epsilon_hashes']),len(s['epsilon_hashes']))
        if not count or u['epsilon_hashes'][:count]!=s['epsilon_hashes'][:count]: raise AssertionError('paired epsilon differs')
        matched+=count
    return dict(episodes=len(unconstrained),matched_noise_decisions=matched,initial_state_latent_and_raw_epsilon_identical=True,
        caveat='later physical states/previous actions differ; pairing is stochastic source, NOT identical-state matching after decision0')


def prediction_metrics(predicted,actual,stats):
    error=np.asarray(predicted)-np.asarray(actual)
    if not len(error): return dict(windows=0,horizons=[1,10],normalized_observation_rmse=None)
    return dict(windows=len(error),horizons=[1,10],normalized_observation_rmse=np.sqrt(np.mean((error/stats['obs']['std'])**2,axis=(0,2))).tolist(),
        physical_per_dimension_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),physical_per_dimension_mae=np.mean(np.abs(error),axis=0).tolist())
