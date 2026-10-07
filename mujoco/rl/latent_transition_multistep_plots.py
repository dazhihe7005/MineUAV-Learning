"""Eight compact plots, fixed prior example; no raw traces written."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from explicit_latent_evaluation import latent_rollout
from explicit_latent_models import latent_normalize
from latent_dynamics_data import DIMENSIONS


def render_figures(report,evaluation,context,directory):
    paths=[]; hs=report['horizons_steps']; clock=np.array(hs)/25
    labels=dict(one_step='One-step T',multistep='Multi-step K10 T')
    models=report['models']; floor=report['decoder_floor']['horizons']
    def save(fig,name):
        fig.tight_layout(); p=directory/name; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    for field,ylabel,filename in [('normalized_latent_rmse','Latent RMSE / Train latent std','latent_transition_rmse_vs_horizon.png'),
                                  ('mean_cosine','Mean latent cosine','latent_transition_cosine_vs_horizon.png')]:
        fig,ax=plt.subplots(figsize=(8,4.6))
        for kind,label in labels.items(): ax.plot(clock,[models[kind]['horizons'][str(h)]['latent'][field] for h in hs],'o-',label=label)
        ax.set(xlabel='Recorded-action horizon (s)',ylabel=ylabel); ax.grid(alpha=.2); ax.legend(); save(fig,filename)
    fig,ax=plt.subplots(figsize=(8,4.6))
    for kind,label in labels.items(): ax.plot(clock,[models[kind]['horizons'][str(h)]['observation']['normalized_observation_rmse'] for h in hs],'o-',label=label+' + frozen D')
    ax.plot(clock,[floor[str(h)]['normalized_observation_rmse'] for h in hs],'o--',label='Teacher-latent Decoder Floor (offline)')
    ax.plot(clock,[report['direct_k10']['horizons'][str(h)]['normalized_observation_rmse'] for h in hs],'o:',label='Frozen Direct K10')
    ax.set(xlabel='Horizon (s)',ylabel='Observation RMSE / Train obs std'); ax.grid(alpha=.2); ax.legend(fontsize=9)
    save(fig,'transition_observation_rmse_vs_horizon.png')
    fig,ax=plt.subplots(figsize=(8,4.6)); width=.24; ids=np.arange(len(hs))
    ax.bar(ids-width,[floor[str(h)]['normalized_observation_rmse'] for h in hs],width,label='Teacher D (offline)')
    for offset,kind in zip([0,width],labels): ax.bar(ids+offset,[models[kind]['horizons'][str(h)]['observation']['normalized_observation_rmse'] for h in hs],width,label=labels[kind]+' + D')
    ax.set_xticks(ids,[str(h) for h in hs]); ax.set(xlabel='Horizon (steps)',ylabel='Normalized observation RMSE',title='Same decoder; no additive error decomposition')
    ax.legend(); save(fig,'decoder_floor_vs_transition_rollout.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,field,title in zip(axes,['horizontal_velocity_rmse','vertical_velocity_rmse'],['Horizontal vx/vy','Vertical vz']):
        for kind,label in labels.items(): ax.plot(clock,[models[kind]['horizons'][str(h)]['observation'][field] for h in hs],'o-',label=label)
        ax.plot(clock,[floor[str(h)][field] for h in hs],'o--',label='Teacher D')
        ax.set(xlabel='Horizon (s)',ylabel='RMSE (m/s)',title=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9); save(fig,'latent_transition_velocity_rmse.png')
    for field,filename,title in [('distance_regions','latent_transition_error_vs_distance.png','True start distance (m)'),
                                  ('pi_magnitude_regions','latent_transition_error_vs_pi_magnitude.png','True PI magnitude (offline grouping)')]:
        h=str(max(hs)); names=list(models['one_step']['horizons'][h][field]); ids=np.arange(len(names)); fig,axes=plt.subplots(1,2,figsize=(11,4.6))
        for ax,part,metric in zip(axes,['latent','observation'],['normalized_latent_rmse','normalized_observation_rmse']):
            for offset,kind in zip([-.16,.16],labels):
                values=[models[kind]['horizons'][h][field][n][part].get(metric,np.nan) for n in names]
                ax.bar(ids+offset,values,.32,label=labels[kind])
                for i,value in enumerate(values):
                    if not np.isfinite(value): ax.text(i+offset,0,'N/A',ha='center',va='bottom',fontsize=8)
            ax.set_xticks(ids,names); ax.set(xlabel=title,ylabel='Normalized '+part+' RMSE',title='H50 / 2s'); ax.grid(axis='y',alpha=.2)
        axes[0].legend(); save(fig,filename)
    selected=report['selected_window']; ep=evaluation['test'][selected['episode_id']]; t=selected['start']; h=selected['horizon']
    stats=context['statistics']; latent=context['latent_statistics']; actions=ep['action'][None,t:t+h]
    with torch.inference_mode():
        z0=ep['latent'][t:t+1]; traces={}
        for kind,model in [('one_step',context['old_transition']),('multistep',evaluation['new'])]:
            traces[kind]=latent_rollout(model,context['decoder'],z0,actions,stats,latent)['observations'][0]
        teacher=context['decoder'](torch.from_numpy(latent_normalize(ep['latent'][t+1:t+h+1],latent))).numpy()
        teacher=teacher*np.asarray(stats['obs']['std'])+np.asarray(stats['obs']['mean'])
    times=np.arange(h+1)/25; actual=np.vstack([ep['obs'][t],ep['next_obs'][t:t+h]])
    fig,axes=plt.subplots(4,2,figsize=(12,11))
    for dim,ax in enumerate(axes.flat):
        if dim==7: ax.axis('off'); continue
        ax.plot(times,actual[:,dim],'k--',label='Recorded')
        ax.plot(times,np.vstack([ep['obs'][t],teacher])[:,dim],':',label='Teacher D (offline)')
        for kind,label in labels.items(): ax.plot(times,np.vstack([ep['obs'][t],traces[kind]])[:,dim],label=label+' + D')
        ax.set(xlabel='Time after fixed start (s)',ylabel=DIMENSIONS[dim]); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); fig.suptitle('Prior fixed Test window; only initial teacher latent enters T')
    save(fig,'selected_latent_transition_rollout.png'); return paths
