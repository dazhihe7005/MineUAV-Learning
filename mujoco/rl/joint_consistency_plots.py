"""Twelve static controlled-objective comparison figures; no raw traces saved."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joint_latent_world_model import load_joint,predict_window


def render_figures(r,plot,context,directory):
    directory=Path(directory); paths=[]; hs=r['horizons_steps']; x=np.asarray(hs)/25
    old=r['baseline']; new=r['audit']; datasets=[('Joint v1',old,'o--'),('Consistency v2',new,'s-')]
    def save(name):
        p=directory/(name+'.png'); plt.tight_layout(); plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close(); paths.append(str(p))
    def labels(ax,xlabel,ylabel): ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    rows=r['training']['history']; epochs=[a['epoch'] for a in rows]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for part,name,style in [('train','observation','-'),('validation','observation','--'),('train','weighted_consistency',':'),('train','total','-.')]:
        axes[0].plot(epochs,[a[part][name] for a in rows],style,label=f'{part} {name}')
    axes[0].legend(); labels(axes[0],'Epoch','MSE objective (distinct normalization scales)')
    for k in ('encoder','transition','decoder'):
        axes[1].plot(epochs,[a['objective_gradient_diagnostics'][k]['mean_consistency_to_observation_ratio'] for a in rows],label=k)
    axes[1].axhline(1,color='gray',ls='--'); axes[1].legend(); labels(axes[1],'Epoch','Mean weighted-consistency /\nobservation gradient norm')
    save('consistency_v2_training_curve')
    fig,ax=plt.subplots(figsize=(8,4))
    for name,data,style in datasets: ax.plot(x,[data['autonomous'][str(h)]['observation']['normalized_observation_rmse'] for h in hs],style,label=name)
    for key,name,style in [('frozen_explicit','Frozen Explicit',':'),('direct_k10','Direct K10','-.')]:
        if key in context['baseline']:
            data=context['baseline'][key]['horizons']
            y=[data[str(h)]['observation']['normalized_observation_rmse'] if key=='frozen_explicit' else data[str(h)]['normalized_observation_rmse'] for h in hs]
            ax.plot(x,y,style,label=name)
    ax.legend(); labels(ax,'Recorded-action horizon (s)','Normalized observation endpoint RMSE'); save('joint_v1_vs_consistency_v2_rmse_horizon')
    fig,axes=plt.subplots(1,3,figsize=(12,4)); names=['v1','v2']
    values=[[old['local_one_step']['latent']['normalized_rmse'],new['local_one_step']['latent']['normalized_rmse']],
        [old['local_one_step']['latent']['mean_l2'],new['local_one_step']['latent']['mean_l2']],
        [r['relative_local_residual'][k]['all']['median'] for k in names]]
    for ax,v,label in zip(axes,values,['Local own-Train-scale RMSE','Local raw L2 (different coordinates)','Relative residual median (epsilon-dependent)']):
        ax.bar(names,v); labels(ax,'','' if not label else label)
    save('local_consistency_comparison')
    cs=['1','5','10','25','never']; xx=np.arange(len(cs)); fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,data,style in datasets:
        for ax,field,key in [(axes[0],'observation','normalized_observation_rmse'),(axes[1],'latent','normalized_rmse')]:
            ax.plot(xx,[data['periodic_correction'][c][field][key] for c in cs],style,label=name)
    for ax,label in zip(axes,['H50 observation RMSE','H50 native Train-scale latent RMSE']):
        ax.set_xticks(xx,cs); ax.legend(); labels(ax,'Diagnostic correction interval; no endpoint reset',label)
    save('periodic_correction_v1_vs_v2')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,label in [(axes[0],'mahalanobis_rms','Own-Train regularized Mahalanobis RMS'),(axes[1],'pca_residual_l2','Own-Train PCA residual L2 (coordinate-dependent)')]:
        for name,data,style in datasets:
            ax.plot(x,[data['autonomous'][str(h)]['distribution']['predicted'][key]['mean'] for h in hs],style,label=f'{name} predicted')
            ax.plot(x,[data['autonomous'][str(h)]['distribution']['reference'][key]['mean'] for h in hs],':',label=f'{name} reference')
        ax.legend(fontsize=8); labels(ax,'Horizon (s)',label)
    save('off_manifold_v1_vs_v2')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,data,style in datasets:
        axes[0].plot(x,[data['autonomous'][str(h)]['decoder_sensitivity']['amplification_ratio']['mean'] for h in hs],style,label=name)
        axes[1].plot(x,[data['autonomous'][str(h)]['decoder_sensitivity']['decoded_normalized_l2']['mean'] for h in hs],style,label=name)
    for ax,label in zip(axes,['Empirical ratio (NOT Jacobian;\ncoordinate-dependent)','Mean normalized D(pred)-D(reference) L2']):
        ax.legend(); labels(ax,'Horizon (s)',label)
    save('decoder_sensitivity_v1_vs_v2')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,data,style in datasets:
        for ax,key in [(axes[0],'horizontal_velocity_rmse'),(axes[1],'vertical_velocity_rmse')]:
            ax.plot(x,[data['autonomous'][str(h)]['observation'][key] for h in hs],style,label=name)
    for ax,label in zip(axes,['Horizontal velocity RMSE (m/s)','vz RMSE (m/s)']): ax.legend(); labels(ax,'Horizon (s)',label)
    save('consistency_v2_velocity_rmse')
    for field,filename,xlabel in [('distance_regions','consistency_v2_error_vs_distance','True start distance'),
        ('pi_magnitude_regions','consistency_v2_error_vs_pi','True start PI magnitude (Train bins)'),
        ('action_magnitude_regions','consistency_v2_error_vs_action','Start executed4D action norm (Train bins)')]:
        names=list(new['autonomous']['50'][field]); xx=np.arange(len(names)); fig,ax=plt.subplots(figsize=(8,4))
        for i,(name,data,_) in enumerate(datasets):
            ax.bar(xx+(i-.5)*.36,[data['autonomous']['50'][field][n]['observation'].get('normalized_observation_rmse',np.nan) for n in names],.36,label=name)
        ax.set_xticks(xx,names); ax.legend(); labels(ax,xlabel,'H50 normalized observation RMSE'); save(filename)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); variances=[('Joint v1',context['baseline']['joint']['latent_variance']),('Consistency v2',r['latent_variance'])]
    for name,data in variances:
        for split,style in [('train','-'),('test','--')]: axes[0].plot(np.sort(data[split]['per_dimension_std']),style,label=f'{name} {split}')
    axes[0].legend(fontsize=8); labels(axes[0],'Sorted latent dimensions (no semantic interpretation)','Native per-dimension std')
    xx=np.arange(2)
    for i,(name,data) in enumerate(variances): axes[1].bar(xx+(i-.5)*.36,[data[s]['effective_rank'] for s in ('train','test')],.36,label=name)
    axes[1].set_xticks(xx,['Train','Test']); axes[1].legend(); labels(axes[1],'','Covariance-entropy effective rank'); save('consistency_v2_latent_variance')
    selected=context['baseline']['selected_window']; dataset=context['dataset']
    from run_joint_consistency_v2 import load_split
    ep=load_split(dataset,'test')[selected['episode_id']]; v1,stats,_=load_joint(context['baseline']['model']['path']); v1.requires_grad_(False)
    with torch.inference_mode(): p=predict_window(v1,ep,selected['start'],selected['horizon'],stats)['normalized_observations'][0].numpy()
    mean=np.asarray(stats['obs']['mean']); std=np.asarray(stats['obs']['std']); p=p*std+mean
    s=plot['selected']; time=np.arange(len(s['truth']))/25; fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,dim,label in [(axes[0],3,'vx (m/s)'),(axes[1],4,'vy (m/s)')]:
        ax.plot(time,s['truth'][:,dim],'k--',label='Recorded'); ax.plot(time,p[:,dim],label='Joint v1')
        ax.plot(time,s['observations']['never'][:,dim],label='Consistency v2'); ax.legend(); labels(ax,'Horizon (s)',label)
    for name,pred in [('Joint v1',p),('Consistency v2',s['observations']['never'])]:
        axes[2].plot(time,np.sqrt(np.mean(((pred-s['truth'])/std)**2,axis=1)),label=name)
    axes[2].legend(); labels(axes[2],'Horizon (s)','Selected-window normalized observation RMSE')
    fig.suptitle('Same original fixed Test start; autonomous T-only recorded-action rollout'); save('selected_consistency_v2_rollout')
    return paths
