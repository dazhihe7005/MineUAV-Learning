"""Static figures from completed fixed-target results; no model selection."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render_figures(report,directory):
    directory=Path(directory); files=[]
    models=('zero','scripted','mpc'); tasks=('benchmark','holdout')
    labels={'zero':'Zero command','scripted':'Scripted waypoint','mpc':'Latent shooting MPC'}
    colors=['#999999','#D99720','#247AAA']
    def save(fig,name):
        fig.tight_layout(); fig.savefig(directory/name,dpi=160); plt.close(fig); files.append(str(directory/name))
    def bars(field,name,ylabel):
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for ax,task in zip(axes,tasks):
            values=[report['results'][m][task]['summary'][field] for m in models]
            ax.bar(range(3),[np.nan if v is None else v for v in values],color=colors)
            ax.set_xticks(range(3),[labels[m] for m in models],rotation=15); ax.set_title(task); ax.set_ylabel(ylabel)
            for j,v in enumerate(values):
                if v is not None:
                    text=f'{v:.3g}'
                    if field=='mean_completion_time_s': text+=f'\n(n={report["results"][models[j]][task]["summary"]["successes"]})'
                    ax.text(j,v,text,ha='center',va='bottom',fontsize=9)
                else: ax.text(j,0,'N/A',ha='center',va='bottom',fontsize=8)
            ax.margins(y=.15)
        save(fig,name)
    bars('success_rate','latent_mpc_success_comparison.png','Success fraction (100 targets each)')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axes,tasks):
        ax.boxplot([[r['final_distance_m'] for r in report['results'][m][task]['episodes']] for m in models],tick_labels=[labels[m] for m in models])
        ax.tick_params(axis='x',rotation=15); ax.set_title(task); ax.set_ylabel('Final distance (m), all episodes')
    save(fig,'latent_mpc_final_distance.png')
    bars('mean_completion_time_s','latent_mpc_completion_time.png','Completion time (s)\nSuccessful episodes only')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,task in zip(axes,tasks):
        x=np.arange(3)
        for i,(field,label) in enumerate([('near_target_actual_speed_m_s','Actual'),('near_target_command_speed_m_s','Command')]):
            vals=[report['results'][m][task]['summary'][field] for m in models]
            ax.bar(x+(i-.5)*.32,[np.nan if v is None else v for v in vals],.32,label=label)
        ax.set_xticks(x,[labels[m] for m in models],rotation=15); ax.set_title(task); ax.set_ylabel('Speed (m/s), true distance <0.1 m'); ax.legend()
        for j,m in enumerate(models):
            if report['results'][m][task]['summary']['near_target_steps']==0:
                ax.text(j,0,'N/A: no visits',ha='center',va='bottom',fontsize=8)
    save(fig,'latent_mpc_near_target_speed.png')
    traces=[]
    for task in tasks:
        row=report['results']['mpc'][task]['representative_local_traces'][0]
        with np.load(row['path'],allow_pickle=False) as raw: traces.append({k:raw[k] for k in raw.files})
    fig,axes=plt.subplots(4,2,figsize=(11,9),sharex='col')
    for j,(task,trace) in enumerate(zip(tasks,traces)):
        for dim,label in enumerate(['vx command','vy command','vz command','yaw rate command']):
            axes[dim,j].plot(np.arange(len(trace['actions']))*.04,trace['actions'][:,dim],lw=1)
            axes[dim,j].set_ylabel(f'{label}\n(normalized)'); axes[dim,j].set_ylim(-1.05,1.05)
        axes[0,j].set_title(f'{task} fixed target 0 (not outcome-selected)'); axes[-1,j].set_xlabel('Time (s)')
    save(fig,'latent_mpc_selected_action_trace.png')
    fig,axes=plt.subplots(3,2,figsize=(11,9),sharex='col')
    for j,(task,trace) in enumerate(zip(tasks,traces)):
        check=trace['check_predicted']; truth=trace['check_actual']; x=(trace['check_starts']+10)*.04
        for ax,dim,label in zip(axes[:,j],[0,3,6],['Target error x (m)','vx (m/s)','Yaw error (rad)']):
            if len(x):
                ax.plot(x,truth[:,1,dim],'s-',ms=4,label='Actual'); ax.plot(x,check[:,1,dim],'o-',ms=3,label='Pure latent forecast under realized 10-step actions')
            ax.set_ylabel(label); ax.legend(fontsize=7)
        axes[0,j].set_title(f'{task}, fixed target 0: realized-prefix sanity check'); axes[-1,j].set_xlabel('Forecast endpoint time (s)')
    save(fig,'latent_mpc_predicted_vs_actual.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    train=report['train_action_statistics']
    arrays=[]
    for task in tasks:
        with np.load(report['results']['mpc'][task]['aggregates']['path'],allow_pickle=False) as raw: arrays.append({k:raw[k] for k in raw.files})
    for i,dim in enumerate(['vx','vy','vz','yaw-rate']):
        vals=np.concatenate([a['actions'][:,i] for a in arrays]); axes[0].hist(vals,bins=35,histtype='step',label=dim,density=True)
    axes[0].set_xlabel('Selected normalized action'); axes[0].set_ylabel('Density'); axes[0].legend()
    axes[1].bar(np.arange(4),report['overall_mpc']['action_ood']['outside_train_95_per_dimension_fraction'])
    axes[1].set_xticks(range(4),['vx','vy','vz','yaw']); axes[1].set_ylabel('Fraction outside coordinatewise Train 95%'); axes[1].set_ylim(0,1)
    save(fig,'latent_mpc_action_ood.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for task,array in zip(tasks,arrays):
        ms=array['planning_seconds']*1000
        axes[0].hist(ms,bins=70,histtype='step',label=task,density=True)
        quantiles=np.linspace(0,1,201); axes[1].plot(np.quantile(ms,quantiles),quantiles,label=task)
    for ax in axes: ax.axvline(40,c='red',ls='--',label='40 ms budget'); ax.legend(); ax.set_xlabel('Planning wall time (ms)')
    axes[0].set_ylabel('Density'); axes[1].set_ylabel('Empirical cumulative fraction')
    save(fig,'latent_mpc_planning_time.png')
    return files
