"""Static aggregate figures; no extrapolation past termination or invented examples."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from uav_bc_plots import save_plot

FIGURES=['success_vs_perturbation.png','bc_vs_scripted_final_distance.png','bc_vs_scripted_completion_time.png',
    'velocity_recovery.png','action_saturation.png','paired_success_difference.png','representative_perturbed_trajectories.png']
LABELS=['Nominal','Pos .10','Pos .30','Vel .15','Vel .40','Yaw 10°','Yaw 30°','Combined S','Combined L']
COLORS={'scripted':'#218c74','bc':'#c94c4c'}


def plots(reports,rows,summary,representatives):
    reports=Path(reports);names=list(summary);x=np.arange(len(names));controllers=['scripted','bc']
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for split,ax in zip(['benchmark','holdout'],axes):
        for j,c in enumerate(controllers):
            s=[summary[n][split][c] for n in names];y=np.array([r['success_rate']*100 for r in s]);ci=np.array([r['success_wilson_95_ci'] for r in s])*100
            ax.bar(x+(j-.5)*.36,y,.36,label=c,color=COLORS[c]);ax.errorbar(x+(j-.5)*.36,y,yerr=np.maximum(0,[y-ci[:,0],ci[:,1]-y]),fmt='none',color='k',capsize=2)
        ax.set(title=f'{split}: n=100 / condition / controller',xticks=x,xticklabels=LABELS,ylabel='Success (%)',ylim=(0,110));ax.tick_params(axis='x',rotation=45);ax.legend()
    save_plot(reports/FIGURES[0],fig)
    for key,filename,ylabel in [('final_distance_m',FIGURES[1],'Final distance (m), all episodes'),
        ('completion_time_s',FIGURES[2],'Completion (s), successes only')]:
        fig,ax=plt.subplots(figsize=(12,4.8))
        for j,c in enumerate(controllers):
            data=[[r[key] for r in rows if r['condition']==n and r['controller']==c and r[key] is not None] for n in names]
            values=[v if v else [np.nan] for v in data]
            boxes=ax.boxplot(values,positions=x+(j-.5)*.33,widths=.28,patch_artist=True,showfliers=False,manage_ticks=False)
            for box in boxes['boxes']:box.set_facecolor(COLORS[c]);box.set_alpha(.5)
            ax.plot([],[],color=COLORS[c],label=c)
            if key=='completion_time_s':
                for i,v in enumerate(data):ax.text(i+(j-.5)*.33,.99 if j else .93,f'n={len(v)}',transform=ax.get_xaxis_transform(),ha='center',fontsize=7,color=COLORS[c])
        ax.set(xticks=x,xticklabels=LABELS,ylabel=ylabel);ax.tick_params(axis='x',rotation=35);ax.legend(loc='upper left');save_plot(reports/filename,fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for col,name in enumerate(['velocity_large','yaw_large','combined_large']):
        for c in controllers:
            selected=[r for r in rows if r['condition']==name and r['controller']==c]
            sums=np.zeros((2,76));counts=np.zeros(76)
            for r in selected:
                with np.load(r['trace_path'],allow_pickle=False) as a:
                    o=a['observations'][:76];l=len(o);valid=a['physical_state_valid'][:76]
                    sums[0,:l]+=np.linalg.norm(o[:,3:6],axis=1)*valid;sums[1,:l]+=np.abs(o[:,6])*180/np.pi*valid;counts[:l]+=valid
            for row in range(2):axes[row,col].plot(np.arange(76)*.04,np.divide(sums[row],counts,out=np.full(76,np.nan),where=counts>0),label=f'{c} (remaining n@3s={int(counts[-1])})',color=COLORS[c])
        axes[0,col].set(title=name,ylabel='Mean actual speed (m/s)');axes[1,col].set(xlabel='Time (s)',ylabel='Mean |yaw error| (degrees)');axes[0,col].legend(fontsize=7)
    fig.suptitle('Recovery diagnostics: observed states only; terminated trajectories not carried forward')
    save_plot(reports/FIGURES[3],fig)
    fig,ax=plt.subplots(figsize=(11,4.5))
    for j,c in enumerate(controllers):ax.bar(x+(j-.5)*.36,[100*summary[n]['pooled'][c]['action_saturation_fraction'] for n in names],.36,label=c,color=COLORS[c])
    ax.set(xticks=x,xticklabels=LABELS,ylabel='Command elements with |a| >= .95 (%)');ax.tick_params(axis='x',rotation=35);ax.legend();save_plot(reports/FIGURES[4],fig)
    fig,ax=plt.subplots(figsize=(11,4.5))
    d=np.array([summary[n]['pooled']['paired']['success_delta_percentage_points'] for n in names]);ci=np.array([summary[n]['pooled']['paired']['exploratory_paired_bootstrap_95_ci'] for n in names])*100
    ax.bar(x,d,color='#5666b4');ax.errorbar(x,d,yerr=np.maximum(0,[d-ci[:,0],ci[:,1]-d]),fmt='none',color='k',capsize=3);ax.axhline(0,color='k',lw=.7)
    ax.set(xticks=x,xticklabels=LABELS,ylabel='BC - Scripted success (percentage points)',title='Paired n=200 / condition; descriptive bootstrap 95% CI');ax.tick_params(axis='x',rotation=35);save_plot(reports/FIGURES[5],fig)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for row,(kind,pair) in enumerate(representatives.items()):
        if pair is None:
            for ax in axes[row]:ax.axis('off');ax.text(.5,.5,f'No real {kind} paired trajectory exists.\nNo sample manufactured.',ha='center',va='center')
            continue
        for r in pair:
            with np.load(r['trace_path'],allow_pickle=False) as a:
                o=a['observations'].astype(float);p=a['positions'].copy();valid=a['physical_state_valid'];o[~valid]=np.nan;p[~valid]=np.nan;t=np.arange(len(o))*.04;c=r['controller']
                axes[row,0].plot(p[:,0],p[:,1],label=f'{c}: {r["termination_reason"]}',color=COLORS[c])
                axes[row,1].plot(t,np.linalg.norm(o[:,:3],axis=1),label=c+' distance',color=COLORS[c])
                axes[row,1].plot(t,np.linalg.norm(o[:,3:6],axis=1),ls=':',label=c+' speed',color=COLORS[c])
                axes[row,2].plot(t,o[:,6]*180/np.pi,label=c,color=COLORS[c])
        axes[row,0].scatter(pair[0]['target'][0],pair[0]['target'][1],marker='x',s=60,color='k',label='target');axes[row,0].set(xlabel='world x (m)',ylabel='world y (m)',title=pair[0]['key'])
        axes[row,1].set(xlabel='Time (s)',ylabel='Distance (m) / speed (m/s)');axes[row,2].set(xlabel='Time (s)',ylabel='Yaw error (degrees)')
        for ax in axes[row]:ax.legend(fontsize=7)
    save_plot(reports/FIGURES[6],fig);return FIGURES
