"""Nine scientific figures from fixed Test protocol; no raw traces saved."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from latent_dynamics_data import DIMENSIONS
from explicit_latent_models import latent_normalize
from explicit_latent_evaluation import latent_rollout
from latent_dynamics_multistep import rollout_windows
from joint_latent_world_model import predict_window


def render_figures(report,evaluation,context,directory):
    paths=[]; hs=report['horizons_steps']; clock=np.array(hs)/25
    curves=dict(joint=('Joint latent v1',report['joint']),frozen=('Frozen explicit K10',report['frozen_explicit']),direct=('Direct K10',report['direct_k10']))
    def metric(kind,h):
        row=curves[kind][1]['horizons'][str(h)]
        return row if kind=='direct' else row['observation']
    def save(fig,name):
        fig.tight_layout(); p=directory/name; fig.savefig(p,dpi=160); plt.close(fig); paths.append(str(p))
    rows=report['training']['history']; fig,ax=plt.subplots(figsize=(8,4.6))
    ax.plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label='Train online average')
    ax.plot([r['epoch'] for r in rows],[r['validation_loss'] for r in rows],label='Validation k0..10 MSE')
    ax.axvline(report['training']['best_epoch'],ls=':',color='k',label='Validation best')
    ax.set(xlabel='Epoch',ylabel='Uniform normalized observation MSE',yscale='log'); ax.legend(); ax.grid(alpha=.2)
    save(fig,'joint_world_model_training_curve.png')
    fig,ax=plt.subplots(figsize=(8,4.6))
    for kind,(label,_) in curves.items(): ax.plot(clock,[metric(kind,h)['normalized_observation_rmse'] for h in hs],'o-',label=label)
    ax.set(xlabel='Recorded-action horizon (s)',ylabel='Observation RMSE / Train obs std'); ax.legend(); ax.grid(alpha=.2)
    save(fig,'joint_vs_explicit_vs_direct_rmse_horizon.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,field,title in zip(axes,['horizontal_velocity_rmse','vertical_velocity_rmse'],['Horizontal vx/vy','Vertical vz']):
        for kind,(label,_) in curves.items(): ax.plot(clock,[metric(kind,h)[field] for h in hs],'o-',label=label)
        ax.set(xlabel='Horizon (s)',ylabel='RMSE (m/s)',title=title); ax.grid(alpha=.2)
    axes[0].legend(fontsize=9); save(fig,'joint_world_model_velocity_rmse.png')
    maxh=max(hs); ids=np.arange(7); fig,ax=plt.subplots(figsize=(10,4.7))
    for offset,kind in zip([-.24,0,.24],curves):
        values=[metric(kind,maxh)['physical'][n]['rmse']/report['normalization']['obs']['std'][i] for i,n in enumerate(DIMENSIONS)]
        ax.bar(ids+offset,values,.24,label=curves[kind][0])
    ax.set_xticks(ids,DIMENSIONS); ax.set(ylabel='Per-dimension RMSE / Train obs std',title=f'H{maxh} endpoint'); ax.legend(); ax.grid(axis='y',alpha=.2)
    save(fig,'joint_world_model_per_dimension_rmse.png')
    for field,filename,title in [('distance_regions','joint_world_model_error_vs_distance.png','True start distance (m)'),
                                  ('pi_magnitude_regions','joint_world_model_error_vs_pi.png','True start PI magnitude (offline)')]:
        groups=report['joint']['horizons'][str(maxh)][field]; names=list(groups); ids=np.arange(len(names)); fig,ax=plt.subplots(figsize=(9,4.6))
        for offset,kind in zip([-.24,0,.24],curves):
            values=[]
            for name in names:
                row=curves[kind][1]['horizons'][str(maxh)][field][name]
                if kind!='direct': row=row['observation']
                values.append(row.get('normalized_observation_rmse',np.nan))
            ax.bar(ids+offset,values,.24,label=curves[kind][0])
            for i,value in enumerate(values):
                if not np.isfinite(value): ax.text(i+offset,0,'N/A',ha='center',fontsize=8)
        ax.set_xticks(ids,names); ax.set(xlabel=title,ylabel='Normalized observation RMSE',title=f'H{maxh}'); ax.legend(); ax.grid(axis='y',alpha=.2)
        save(fig,filename)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    values=report['joint']['horizons']
    for ax,field,label in zip(axes,['train_std_normalized_distance','mean_cosine'],['Train-scale latent distance','Latent cosine']):
        ax.plot(clock,[values[str(h)]['latent_consistency'][field] for h in hs],'o-')
        ax.set(xlabel='Horizon (s)',ylabel=label,title='Offline Joint encoder consistency'); ax.grid(alpha=.2)
    save(fig,'joint_world_model_latent_consistency.png')
    fig,ax=plt.subplots(figsize=(10,4.6)); variance=report['joint']['latent_variance']
    for split in ('train','test'):
        row=variance[split]; ax.plot(np.arange(64),row['per_dimension_std'],'.-',label=f'{split}, effective rank={row["effective_rank"]:.2f}')
    ax.axhline(1e-6,ls=':',color='k',label='Near-zero diagnostic threshold'); ax.set(xlabel='Latent coordinate index (no semantic interpretation)',ylabel='Raw latent std',yscale='log')
    ax.legend(); ax.grid(alpha=.2); save(fig,'joint_world_model_latent_variance.png')
    selected=report['selected_window']; ep=evaluation['test'][selected['episode_id']]; t=selected['start']; h=selected['horizon']; stats=context['statistics']; ls=context['latent_statistics']
    with torch.inference_mode():
        out=predict_window(evaluation['model'],ep,t,h,stats)
        joint=out['normalized_observations'][0].numpy().astype(np.float64)*np.asarray(stats['obs']['std'])+np.asarray(stats['obs']['mean'])
        explicit=latent_rollout(evaluation['baseline'],context['decoder'],ep['latent'][t:t+1],ep['action'][None,t:t+h],stats,ls)['observations'][0]
        initial=context['decoder'](torch.from_numpy(latent_normalize(ep['latent'][t:t+1],ls))).numpy().astype(np.float64)*np.asarray(stats['obs']['std'])+np.asarray(stats['obs']['mean'])
        explicit=np.vstack([initial,explicit])
        direct=rollout_windows(evaluation['direct'],'history',ep,stats,np.array([t]),h)['predictions'][0]
        direct=np.vstack([ep['obs'][t],direct])
    actual=np.vstack([ep['obs'][t],ep['next_obs'][t:t+h]]); times=np.arange(h+1)/25
    fig,axes=plt.subplots(4,2,figsize=(12,11))
    for i,ax in enumerate(axes.flat):
        if i==7: ax.axis('off'); continue
        for values,label,style in [(actual,'Recorded','k--'),(joint,'Joint latent v1','-'),(explicit,'Frozen explicit','-'),(direct,'Direct K10',':')]:
            ax.plot(times,values[:,i],style,label=label)
        ax.set(xlabel='Time after prior fixed start (s)',ylabel=DIMENSIONS[i]); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8); fig.suptitle('Prior fixed Test window; current decoder reconstruction shown at t=0')
    save(fig,'selected_joint_world_model_rollout.png'); return paths
