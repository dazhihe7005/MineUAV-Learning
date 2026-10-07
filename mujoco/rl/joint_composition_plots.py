"""Nine static scientific audit plots; raw plot arrays never persisted."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render_figures(report,plot,directory):
    directory=Path(directory); paths=[]; hs=report['horizons_steps']; seconds=np.asarray(hs)/25
    def save(name):
        p=directory/(name+'.png'); plt.tight_layout(); plt.savefig(p,dpi=150); plt.close(); paths.append(str(p))
    def line_labels(ax,xlabel,ylabel): ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    auto=report['autonomous']; local=report['local_one_step']['latent']; correction=report['periodic_correction']
    plt.figure(figsize=(8,4)); plt.plot(seconds,[auto[str(h)]['latent']['normalized_rmse'] for h in hs],'o-',label='Autonomous composition')
    plt.axhline(local['normalized_rmse'],ls='--',color='tab:orange',label='Local one-step (all transitions)')
    plt.xlabel('Recorded-action horizon (s)'); plt.ylabel('Train-scale latent RMSE'); plt.grid(alpha=.2); plt.legend(); save('local_vs_autoregressive_latent_error')
    cs=['1','5','10','25','never']; x=np.arange(len(cs))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,field,key,label in [(axes[0],'observation','normalized_observation_rmse','H50 observation RMSE / Train std'),
                              (axes[1],'latent','normalized_rmse','H50 latent RMSE / Train std')]:
        ax.plot(x,[correction[c][field][key] for c in cs],'o-'); ax.set_xticks(x,cs)
        line_labels(ax,'Correction interval (steps), no endpoint reset',label)
    save('periodic_correction_rmse')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,key,label in zip(axes,['standardized_rms','mahalanobis_rms','pca_residual_l2'],['Train-standardized RMS distance','Regularized Mahalanobis RMS','Train PCA residual L2']):
        for kind,style in [('predicted','o-'),('reference','s--')]:
            ax.plot(seconds,[auto[str(h)]['distribution'][kind][key]['mean'] for h in hs],style,label=kind)
        line_labels(ax,'Horizon (s)',label); ax.legend()
    save('latent_off_manifold_vs_horizon')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for field,label in [('latent_l2','Raw latent L2 deviation'),('decoded_normalized_l2','Normalized decoded L2 deviation')]:
        axes[0].plot(seconds,[auto[str(h)]['decoder_sensitivity'][field]['mean'] for h in hs],'o-',label=label)
    axes[0].legend(); line_labels(axes[0],'Horizon (s)','Empirical deviation (different coordinate units)')
    for key,style in [('mean','o-'),('median','s--')]:
        axes[1].plot(seconds,[auto[str(h)]['decoder_sensitivity']['amplification_ratio'][key] for h in hs],style,label=key)
    axes[1].legend(); line_labels(axes[1],'Horizon (s)','Decoded L2 / raw latent L2 (NOT Jacobian norm)'); save('decoder_sensitivity_vs_horizon')
    s=plot['scatter']; corr=auto['50']['local_residual_future_correlations']['latent_l2']
    plt.figure(figsize=(7,4)); plt.scatter(s['local_residual'],s['future_latent_l2'],s=5,alpha=.25)
    plt.xlabel('Initial local transition residual L2'); plt.ylabel('H50 autonomous latent deviation L2')
    plt.title(f'All-window Pearson r={corr:.3f}' if corr is not None else 'Undefined correlation (constant/insufficient data)')
    plt.grid(alpha=.2); save('local_residual_vs_future_error')
    for field,name,xlabel in [('distance_regions','composition_error_vs_distance','True start distance (m)'),
        ('pi_magnitude_regions','composition_error_vs_pi','True start PI magnitude (Train bins)'),
        ('action_magnitude_regions','composition_error_vs_action_magnitude','Start executed4D action norm (Train bins)')]:
        names=list(auto['50'][field]); x=np.arange(len(names)); plt.figure(figsize=(8,4))
        for offset,c,label in [(-.18,'never','Autonomous'),(.18,'1','C1 diagnostic correction')]:
            rows=correction[c][field]
            values=[rows[n]['observation'].get('normalized_observation_rmse',np.nan) for n in names]
            plt.bar(x+offset,values,.36,label=label)
        plt.xticks(x,names); plt.xlabel(xlabel); plt.ylabel('H50 normalized observation RMSE'); plt.legend(); plt.grid(axis='y',alpha=.2); save(name)
    selected=plot['selected']; time=np.arange(len(selected['truth']))/25
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for c in cs:
        axes[0].plot(time,selected['latent_l2'][c],label=f'C={c}')
        axes[1].plot(time,selected['observations'][c][:,3]); axes[2].plot(time,selected['observations'][c][:,4])
    for ax,dim in [(axes[1],3),(axes[2],4)]: ax.plot(time,selected['truth'][:,dim],'k--',label='Recorded observation')
    for ax,label in zip(axes,['Reference latent deviation L2','vx (m/s)','vy (m/s)']): line_labels(ax,'Time after prior fixed start (s)',label)
    axes[0].legend(); axes[1].legend(); fig.suptitle('Same prior selected window; corrections are diagnostic, not deployable')
    save('selected_latent_composition_trajectory')
    return paths
