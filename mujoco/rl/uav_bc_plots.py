"""Small aggregate scientific figures; raw episodes remain local-only."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_bc_safety import atomic_bytes

FIGURES=['bc_training_curves.png','bc_action_prediction_scatter.png','bc_vs_scripted_success.png',
    'bc_final_distance_distribution.png','bc_representative_successful_trajectory.png',
    'bc_representative_failed_trajectory.png','bc_action_saturation_statistics.png']

def save_plot(path,figure):
    figure.tight_layout();atomic_bytes(path,lambda f:figure.savefig(f,format='png',dpi=140));plt.close(figure)

def missing_trajectory_notice(success,count):
    outcome='success' if success else 'failure'
    cohort='successful' if success else 'failed'
    return f'No BC {outcome} among the fixed {count} evaluated episodes.\nNo {cohort} trajectory was manufactured.'

def plots(reports,training,evaluation,test,predictions):
    reports=Path(reports);history=training['history']
    fig,ax=plt.subplots(figsize=(7,4))
    for key,label in [('train_loss','Train'),('validation_loss','Validation')]:ax.semilogy([r['epoch'] for r in history],[r[key] for r in history],label=label)
    ax.axvline(training['best_epoch'],color='k',ls=':',label='Val-best epoch');ax.set(xlabel='Epoch',ylabel='Train-action-std normalized MSE');ax.legend()
    save_plot(reports/FIGURES[0],fig)
    fig,axes=plt.subplots(2,2,figsize=(8,6))
    idx=np.linspace(0,len(predictions)-1,min(2000,len(predictions)),dtype=int)
    for i,ax in enumerate(axes.flat):
        x=test['actions'][idx,i];y=predictions[idx,i];lo=min(x.min(),y.min());hi=max(x.max(),y.max())
        ax.scatter(x,y,s=3,alpha=.4);ax.plot([lo,hi],[lo,hi],color='k',ls=':')
        ax.set(title=['vx_cmd','vy_cmd','vz_cmd','yaw_rate_cmd'][i],xlabel='Expert normalized command',ylabel='BC raw command')
    save_plot(reports/FIGURES[1],fig)
    summaries=evaluation['closed_loop'];records=evaluation['records'];conditions=['scripted','bc'];tasks=['benchmark','holdout']
    fig,ax=plt.subplots(figsize=(7,4))
    for j,c in enumerate(conditions):ax.bar(np.arange(2)+j*.32,[summaries[c][t]['success_rate']*100 for t in tasks],width=.32,label=c)
    ax.set(xticks=np.arange(2)+.16,xticklabels=tasks,ylabel='Success (%)',ylim=(0,110));ax.legend();save_plot(reports/FIGURES[2],fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for t,ax in zip(tasks,axes):
        ax.boxplot([[r['final_distance_m'] for r in records[c][t]] for c in conditions],tick_labels=conditions,showmeans=True)
        ax.axhline(.1,ls=':',color='k');ax.set(title=t,ylabel='Final distance (m)')
    save_plot(reports/FIGURES[3],fig)
    all_bc=[r for t in tasks for r in records['bc'][t]]
    for success,name in [(True,FIGURES[4]),(False,FIGURES[5])]:
        row=next((r for r in all_bc if r['success']==success),None)
        if row is None:
            fig,ax=plt.subplots(figsize=(8,4));ax.axis('off');ax.text(.5,.5,missing_trajectory_notice(success,len(all_bc)),ha='center',va='center');save_plot(reports/name,fig);continue
        with np.load(row['path'],allow_pickle=False) as a:
            o=a['observations'];action=a['actions'];time=np.arange(len(o))*.04
            fig,axes=plt.subplots(1,3,figsize=(12,3.7))
            p=np.asarray(row['target'])-o[:,:3];axes[0].plot(p[:,0],p[:,1]);axes[0].scatter(*np.asarray(row['target'])[:2],marker='x',label='Target')
            axes[0].set(xlabel='x (m)',ylabel='y (m)',title=f'{row["task"]} episode {row["target_id"]}');axes[0].legend()
            axes[1].plot(time,np.linalg.norm(o[:,:3],axis=1),label='Distance (m)');axes[1].plot(time,np.linalg.norm(o[:,3:6],axis=1),label='Speed (m/s)');axes[1].legend();axes[1].set(xlabel='Time (s)')
            for i,key in enumerate(['vx','vy','vz','yaw']):axes[2].plot(time[:-1],action[:,i],label=key)
            axes[2].set(xlabel='Time (s)',ylabel='Normalized command');axes[2].legend()
            save_plot(reports/name,fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for j,c in enumerate(conditions):ax.bar(np.arange(2)+j*.32,[summaries[c][t]['action_saturation_fraction']*100 for t in tasks],width=.32,label=c)
    ax.set(xticks=np.arange(2)+.16,xticklabels=tasks,ylabel='Command elements with |a| >= .95 (%)',ylim=(0,1));ax.legend();save_plot(reports/FIGURES[6],fig)
    return FIGURES
