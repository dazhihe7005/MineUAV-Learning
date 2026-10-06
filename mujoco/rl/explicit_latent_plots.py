"""Seven small scientific plots; no raw traces are written."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from explicit_latent_models import latent_normalize,complete_observations
from explicit_latent_evaluation import latent_rollout
from latent_dynamics_multistep import rollout_windows
from latent_dynamics_data import DIMENSIONS


def render_figures(report,evaluation,context,directory):
    paths=[]; hs=report['horizons_steps']; clock=np.array(hs)/25; rows=report['latent_rollout']['horizons']; direct=report['direct_k10']['horizons']
    def save(fig,name):
        fig.tight_layout(); p=directory/name; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    fig,ax=plt.subplots(figsize=(6.5,4.5)); one=report['transition_one_step']
    ax.bar(['Identity transition','Learned residual T'],[one['identity_transition_reference']['normalized_latent_rmse'],one['normalized_latent_rmse']])
    ax.set(ylabel='One-step latent RMSE / Train latent std',title='Same frozen teacher coordinates — all Test transitions')
    save(fig,'latent_transition_one_step_error.png')
    fig,ax=plt.subplots(figsize=(8,4.7))
    ax.plot(clock,[direct[str(h)]['normalized_observation_rmse'] for h in hs],'o-',label='Frozen Direct K10 (obs feedback)')
    ax.plot(clock,[rows[str(h)]['observation']['normalized_observation_rmse'] for h in hs],'o-',label='Explicit latent T + Decoder')
    ax.set(xlabel='Recorded-action horizon (s)',ylabel='Observation RMSE / Train obs std'); ax.legend(); ax.grid(alpha=.2)
    save(fig,'explicit_latent_vs_direct_rmse_horizon.png')
    fig,ax=plt.subplots(figsize=(8,4.7)); ax.plot(clock,[rows[str(h)]['latent']['normalized_latent_rmse'] for h in hs],'o-')
    ax.set(xlabel='Horizon (s)',ylabel='Latent RMSE / Train latent std',title='Pure latent rollout vs frozen teacher latent'); ax.grid(alpha=.2)
    save(fig,'latent_rollout_error_vs_horizon.png')
    fig,ax=plt.subplots(figsize=(8,4.7)); ax.plot(clock,[rows[str(h)]['latent']['mean_cosine'] for h in hs],'o-')
    ax.set(xlabel='Horizon (s)',ylabel='Mean latent cosine',title='Predicted vs teacher: same frozen coordinates'); ax.grid(alpha=.2)
    save(fig,'latent_cosine_vs_horizon.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,field,label in zip(axes,['horizontal_velocity_rmse','vertical_velocity_rmse'],['Horizontal vx/vy','Vertical vz']):
        ax.plot(clock,[direct[str(h)][field] for h in hs],'o-',label='Direct K10')
        ax.plot(clock,[rows[str(h)]['observation'][field] for h in hs],'o-',label='Explicit latent')
        ax.set(xlabel='Horizon (s)',ylabel='RMSE (m/s)',title=label); ax.grid(alpha=.2)
    axes[0].legend(); save(fig,'explicit_latent_velocity_rmse.png')
    selected=report['selected_window']; ep=evaluation['episodes'][selected['episode_id']]; start=selected['start']; h=selected['horizon']
    actions=ep['action'][None,start:start+h]
    out=latent_rollout(evaluation['students']['transition'],evaluation['students']['decoder'],ep['latent'][start:start+1],actions,
        context['statistics'],context['latent_statistics'])
    old=rollout_windows(evaluation['direct'],'history',ep,context['statistics'],[start],h)
    trace_clock=np.arange(h+1)/25; actual=np.vstack([ep['obs'][start],ep['next_obs'][start:start+h]])
    new=np.vstack([ep['obs'][start],out['observations'][0]]); previous=np.vstack([ep['obs'][start],old['predictions'][0]])
    fig,axes=plt.subplots(4,2,figsize=(12,11))
    for dim,ax in enumerate(axes.flat):
        if dim==7: ax.axis('off'); continue
        ax.plot(trace_clock,actual[:,dim],'k--',label='Recorded'); ax.plot(trace_clock,previous[:,dim],label='Direct K10'); ax.plot(trace_clock,new[:,dim],label='Explicit latent')
        ax.set(xlabel='Time after start (s)',ylabel=DIMENSIONS[dim]); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); fig.suptitle('Prior fixed example window; initial obs true, no future obs input')
    save(fig,'selected_explicit_latent_rollout.png')
    decoder=report['decoder_reconstruction']; fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    std=np.asarray(context['statistics']['obs']['std'])
    axes[0].bar(DIMENSIONS,[decoder['physical']['per_dimension'][d]['rmse']/std[i] for i,d in enumerate(DIMENSIONS)])
    axes[1].bar(DIMENSIONS,[decoder['physical']['per_dimension'][d]['r2'] or 0 for d in DIMENSIONS])
    axes[0].set(ylabel='Reconstruction RMSE / Train obs std'); axes[1].set(ylabel='Per-dimension R²')
    for ax in axes: ax.tick_params(axis='x',rotation=45)
    fig.suptitle('D(teacher latent) -> current observation — no transition error')
    save(fig,'decoder_reconstruction_results.png'); return paths
