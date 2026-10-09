"""Eight scientific figures, real observed trajectories only, no invented outcomes."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_bc_plots import save_plot
from uav_bc_yaw_data import load_selected
from uav_bc_yaw_evaluation import CONTROLLERS

COLORS={'scripted':'#198f76','original':'#d64b4b','nominal_expanded':'#e49a20','yaw_augmented':'#377cc0'}

def confidence_error_lengths(values,ci):
    values=np.asarray(values,dtype=float);ci=np.asarray(ci,dtype=float)
    errors=np.vstack([values-ci[:,0],ci[:,1]-values])
    if not np.isfinite(errors).all() or np.any(errors<-1e-12):raise ValueError('invalid confidence interval for reported proportion')
    # Wilson's zero-success lower endpoint can be +3e-18: only rendering
    # roundoff is clipped; the reported interval remains untouched.
    return np.maximum(errors,0.)

def plots(root,manifest,data,training,evaluation):
    root=Path(root);reports=root/'mujoco/reports';metrics=evaluation['metrics'];offline=evaluation['offline_test'];names=[]
    def save(fig,name):
        relative=Path('uav_bc_yaw_coverage_figures')/name
        save_plot(reports/relative,fig);names.append(str(relative))
    def bars(ax,conditions,field,ylabel,interval=False):
        x=np.arange(len(conditions));width=.18
        for i,n in enumerate(CONTROLLERS):
            values=[];ci=[]
            for c in conditions:
                r=metrics[c][n];v=r[field];v=v['mean'] if isinstance(v,dict) else v
                values.append(np.nan if v is None else v)
                if interval:ci.append(r['success_wilson_95_ci'])
            values=np.asarray(values)
            ax.bar(x+(i-1.5)*width,values,width,color=COLORS[n],label=n)
            if interval:
                ax.errorbar(x+(i-1.5)*width,values,yerr=confidence_error_lengths(values,ci),fmt='none',ecolor='black',capsize=2,lw=.8)
        ax.set_xticks(x,conditions,rotation=25,ha='right');ax.set_ylabel(ylabel)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    from uav_bc_data import load_split
    import json
    old=load_split(json.loads((reports/'uav_bc_dataset_manifest_seed0.json').read_text()),'train')
    for name,yaw in [('original_train',old['observations'][:,6]),('nominal_expanded',load_selected(root,'nominal_expanded')['observations'][:,6]),('yaw_augmented',load_selected(root,'yaw_augmented')['observations'][:,6])]:
        axes[0].hist(np.rad2deg(yaw),bins=80,density=True,histtype='step',label=name)
    axes[0].set(xlabel='Observed yaw error (degrees)',ylabel='Selected-data density');axes[0].set_yscale('log');axes[0].legend(fontsize=8)
    x=np.arange(3)
    for i,n in enumerate(('nominal_expanded','yaw_augmented')):
        a=load_selected(root,n);counts=[np.sum(np.isclose(abs(a['initial_yaw_degrees']),v)) for v in (0,10,30)]
        axes[1].bar(x+(i-.5)*.3,counts,.3,label=n,color=COLORS[n])
    axes[1].set_xticks(x,['0°','±10°','±30°']);axes[1].set(xlabel='Actual initial yaw (not B shadow groups)',ylabel='Selected Train samples');axes[1].legend(fontsize=8)
    save(fig,'yaw_coverage_distribution.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4));x=np.arange(3)
    for ax,kind in zip(axes,['raw','clipped']):
        for i,n in enumerate(CONTROLLERS[1:]):
            v=[offline[n]['by_initial_yaw'][str(g)][kind]['rmse'] for g in (0,10,30)]
            ax.bar(x+(i-1)*.25,v,.25,label=n,color=COLORS[n])
        ax.set_xticks(x,['0°','±10°','±30°']);ax.set(ylabel='4D normalized-command RMSE',title=f'{kind} action error on unseen expert trajectories');ax.legend(fontsize=8)
    save(fig,'offline_action_error_vs_yaw.png')
    fig,ax=plt.subplots(figsize=(10,4));bars(ax,['nominal','yaw_small','yaw_large'],'success_rate','Success fraction; n=100/controller',True);ax.legend(fontsize=8);ax.set_ylim(0,1.13)
    save(fig,'success_vs_yaw_perturbation.png')
    fig,ax=plt.subplots(figsize=(10,4));bars(ax,['nominal','yaw_small','yaw_large'],'maximum_speed_m_s','Mean episode peak world speed (m/s)');ax.legend(fontsize=8)
    save(fig,'peak_speed_vs_yaw.png')
    fig,axes=plt.subplots(1,2,figsize=(14,4));conds=['nominal','yaw_small','yaw_large','combined_small','combined_large']
    bars(axes[0],conds,'near_target_actual_speed_m_s','Actual speed at d<0.1m (m/s)');bars(axes[1],conds,'near_target_command_speed_m_s','Command speed at d<0.1m (m/s)');axes[0].legend(fontsize=8)
    save(fig,'near_target_speed_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(14,4));conds=['nominal','position_small','position_large','velocity_small','velocity_large']
    bars(axes[0],conds,'success_rate','Success fraction',True);axes[0].set_ylim(0,1.13);bars(axes[1],conds,'final_distance_m','All-episode mean final distance (m)');axes[0].legend(fontsize=8)
    save(fig,'nominal_regression_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(12,4));conds=['combined_small','combined_large']
    bars(axes[0],conds,'success_rate','Success fraction; n=100/controller',True);axes[0].set_ylim(0,1.13);bars(axes[1],conds,'maximum_speed_m_s','Mean episode peak speed (m/s)');axes[0].legend(fontsize=8)
    save(fig,'combined_perturbation_success.png')
    fig,axes=plt.subplots(1,3,figsize=(16,4));key='yaw_large/final/000'
    for n in CONTROLLERS:
        r=next(r for r in evaluation['rows'] if r['key']==key and r['controller']==n)
        with np.load(r['trace_path'],allow_pickle=False) as a:
            mask=a['physical_state_valid'];o=a['observations'][mask];pos=a['positions'][mask];t=np.flatnonzero(mask)*.04
        label=f"{n}: {r['termination_reason']}";color=COLORS[n]
        axes[0].plot(pos[:,0],pos[:,1],label=label,color=color);axes[1].plot(t,np.linalg.norm(o[:,:3],axis=1),label=n,color=color)
        axes[2].plot(t,np.linalg.norm(o[:,3:6],axis=1),label=n,color=color)
    target=np.asarray(r['target']);axes[0].scatter(*target[:2],marker='x',color='black',s=60);axes[0].set(xlabel='World x (m)',ylabel='World y (m)');axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Time (s)',ylabel='Target distance (m)');axes[2].set(xlabel='Time (s)',ylabel='World speed (m/s)')
    fig.suptitle('Pre-fixed yaw30 final target index0; no result-based representative selection')
    save(fig,'representative_yaw30_trajectories.png')
    return names
