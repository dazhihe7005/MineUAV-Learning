"""Aggregate fixed-cohort figures and honest, preselected episode000 traces."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_bc_safety import atomic_bytes
from uav_bc_external_disturbance import conditions
from uav_bc_external_evaluation import CONTROLLERS
from uav_bc_yaw_plots import confidence_error_lengths

NAMESPACE='uav_bc_external_disturbance_figures'
LABELS={'scripted':'Scripted','original':'Original BC','yaw_augmented':'Yaw-Augmented BC'}
COLORS={'scripted':'#4c78a8','original':'#e45756','yaw_augmented':'#59a14f'}

def plots(root,ev):
    reports=Path(root)/'mujoco/reports';out=reports/NAMESPACE;out.mkdir(parents=True,exist_ok=True)
    names=[];metrics=ev['metrics'];cs=list(conditions());x=np.arange(len(cs))
    def save(fig,name):
        fig.tight_layout();atomic_bytes(out/name,lambda f:fig.savefig(f,format='png',dpi=150));plt.close(fig)
        names.append(NAMESPACE+'/'+name)
    def axes(title,ylabel):
        fig,ax=plt.subplots(figsize=(10,4.5));ax.set_title(title);ax.set_ylabel(ylabel)
        ax.set_xticks(x);ax.set_xticklabels([c.replace('_','\n') for c in cs]);return fig,ax
    fig,ax=axes('Frozen policies: real lateral COM force (100 targets/condition)','Success (%)')
    for j,n in enumerate(CONTROLLERS):
        rates=np.array([metrics[c][n]['success_rate'] for c in cs]);ci=np.array([metrics[c][n]['success_wilson_95_ci'] for c in cs])
        ax.bar(x+(j-1)*.24,rates*100,.24,color=COLORS[n],label=LABELS[n],yerr=confidence_error_lengths(rates,ci)*100,capsize=2)
    ax.set_ylim(0,110);ax.legend();save(fig,'success_vs_external_force.png')

    def trace(condition,n):
        r=next(r for r in ev['rows'] if r['condition']==condition and r['controller']==n and r['key'].endswith('/000'))
        with np.load(r['trace_path'],allow_pickle=False) as f:z={k:f[k].copy() for k in f.files}
        return r,z
    for group,chosen,name in [('Constant',['constant_low','constant_medium','constant_high'],'constant_disturbance_recovery.png'),
                              ('Gust',['gust_medium','gust_high'],'gust_disturbance_recovery.png')]:
        fig,axs=plt.subplots(2,len(chosen),figsize=(5*len(chosen),7),squeeze=False)
        for j,c in enumerate(chosen):
            for n in CONTROLLERS:
                r,z=trace(c,n);v=z['physical_state_valid'];t=z['policy_times'][v];o=z['observations'][v]
                axs[0,j].plot(t,np.linalg.norm(o[:,:3],axis=1),color=COLORS[n],label=LABELS[n]+(' success' if r['success'] else ' failure'))
                axs[1,j].plot(t,np.linalg.norm(o[:,3:6],axis=1),color=COLORS[n])
            axs[0,j].set_title(c+' — fixed target000');axs[0,j].set_ylabel('Target distance (m)');axs[1,j].set_ylabel('Speed (m/s)')
            axs[1,j].set_xlabel('Actual simulation time (s)');axs[0,j].legend(fontsize=8)
            axs[0,j].axhline(.1,color='grey',ls=':');axs[1,j].axhline(.15,color='grey',ls=':')
            if group=='Gust':
                for ax in axs[:,j]:ax.axvspan(2,4,color='grey',alpha=.12)
        fig.suptitle(group+' recovery: unextended original task, traces stop at termination');save(fig,name)

    fig,ax=axes('Peak translational velocity (episode mean of peaks)','Peak speed (m/s)')
    for j,n in enumerate(CONTROLLERS):ax.bar(x+(j-1)*.24,[metrics[c][n]['maximum_speed_m_s']['mean'] for c in cs],.24,label=LABELS[n],color=COLORS[n])
    ax.legend();save(fig,'peak_velocity_comparison.png')

    fig,ax=plt.subplots(figsize=(9,4.5));constant=cs[1:4]
    for j,n in enumerate(CONTROLLERS):
        for i,c in enumerate(constant):
            s=metrics[c][n]['steady_position_error_m'];v=s['statistics']['mean'] if s['statistics'] is not None else np.nan
            ax.bar(i+(j-1)*.24,v,.24,label=LABELS[n] if i==0 else None,color=COLORS[n])
            if np.isfinite(v):ax.annotate(f"n={s['measured_episodes']}",(i+(j-1)*.24,v),ha='center',va='bottom',fontsize=8)
    ax.set_xticks(range(3));ax.set_xticklabels(constant);ax.set_ylabel('Terminal-window mean distance (m)')
    ax.set_title('Finite last1s proxy, duration≥5s only; NOT asymptotic holding');ax.legend();save(fig,'steady_state_position_error.png')

    fig,ax=axes('Paired success difference, same physical states','Difference (percentage points)')
    for j,(p,label) in enumerate([('yaw_augmented_minus_original','Yaw-Augmented − Original'),('yaw_augmented_minus_scripted','Yaw-Augmented − Scripted')]):
        ax.bar(x+(j-.5)*.32,[metrics[c]['paired'][p]['success_delta_percentage_points'] for c in cs],.32,label=label)
    ax.axhline(0,color='black',lw=.6);ax.legend();save(fig,'paired_controller_success.png')

    fig,ax=axes('Command-element saturation (|normalized action|≥.95)','Saturated elements (%)')
    for j,n in enumerate(CONTROLLERS):ax.bar(x+(j-1)*.24,[100*metrics[c][n]['action_saturation_fraction'] for c in cs],.24,label=LABELS[n],color=COLORS[n])
    ax.legend();save(fig,'action_saturation_under_disturbance.png')

    fig,axs=plt.subplots(1,2,figsize=(11,5))
    for ax,c in zip(axs,['constant_high','gust_high']):
        for n in CONTROLLERS:
            r,z=trace(c,n);v=z['physical_state_valid'];p=z['positions'][v]
            ax.plot(p[:,0],p[:,1],color=COLORS[n],label=LABELS[n]+(' success' if r['success'] else ' failure'))
            ax.scatter(p[-1,0],p[-1,1],s=20,color=COLORS[n])
        ax.scatter(r['target'][0],r['target'][1],marker='*',s=150,color='black',label='Target')
        ax.set_title(c+' — target000');ax.set_xlabel('World X (m)');ax.set_ylabel('World Y (m)');ax.axis('equal');ax.legend(fontsize=8)
    fig.suptitle('Fixed preselected real trajectories; no invented successful/failed samples');save(fig,'representative_disturbance_trajectories.png')
    return names
