"""Static comparison of complete support-control cohorts; no tuning or filtering."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def render_figures(report,directory):
    directory=Path(directory); paths=[]
    conditions=('unconstrained','train_supported'); splits=('benchmark','holdout')
    labels=('Unconstrained','Train central95% box'); colors=('#B76835','#247AAA')
    def save(fig,name):
        fig.tight_layout(); fig.savefig(directory/name,dpi=160); plt.close(fig); paths.append(str(directory/name))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,splits):
        values=[report['results'][c][split]['summary']['successes'] for c in conditions]
        ax.bar(labels,values,color=colors); ax.set_ylim(0,110); ax.set_title(split); ax.set_ylabel('Successes / 100 (all episodes)')
        for i,v in enumerate(values): ax.text(i,v+2,str(v),ha='center')
    save(fig,'supported_vs_unconstrained_success.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,splits):
        values=[[r['final_distance_m'] for r in report['results'][c][split]['episodes']] for c in conditions]
        ax.boxplot(values,tick_labels=labels); ax.set_ylabel('Final distance (m), all 100 episodes'); ax.set_title(split)
    save(fig,'supported_vs_unconstrained_final_distance.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,splits):
        values=[report['results'][c][split]['action_support']['outside_train_95_any_dimension_fraction']*100 for c in conditions]
        ax.bar(labels,values,color=colors); ax.set_title(split); ax.set_ylabel('Selected actions outside any Train95% axis (%)'); ax.set_ylim(0,105)
        for i,v in enumerate(values): ax.text(i,v+2,f'{v:.2f}%',ha='center')
    save(fig,'selected_action_support_departure.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,splits):
        for j,c in enumerate(conditions):
            d=report['results'][c][split]['action_support']['standardized_norm']
            ax.plot(['Mean','Median','p95','Max'],[d[k] for k in ('mean','median','p95','max')],'o-',c=colors[j],label=labels[j])
        ax.set_title(split); ax.set_ylabel('Standardized selected-action L2 norm'); ax.legend(fontsize=8)
    save(fig,'selected_action_standardized_norm.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,split in zip(axes,splits):
        for j,c in enumerate(conditions):
            value=report['results'][c][split]['prediction_sanity']; common=report['common_window_prediction_check'][split][c]
            if value['windows']:
                ax.plot([1,10],value['normalized_observation_rmse'],'o-',c=colors[j],label=f'{labels[j]}: {value["windows"]} windows')
            if common['windows']:
                ax.plot([1,10],common['normalized_observation_rmse'],'x--',c=colors[j],label=f'Common identities: {common["windows"]}')
        ax.set_xticks([1,10]); ax.set_title(split); ax.set_xlabel('Forecast horizon (actual executed action prefix)')
        ax.set_ylabel('Normalized observation RMSE'); ax.legend(fontsize=7)
    fig.suptitle('Fixed first 3 episodes; stride25, legal10step windows. Ordinary yaw wrap caveat.',fontsize=10)
    save(fig,'supported_vs_unconstrained_prediction_error.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for dim,ax in enumerate(axes):
        for j,c in enumerate(conditions):
            rows=[r for split in splits for r in report['results'][c][split]['episodes']]
            target=np.array([r['target'] for r in rows])[:,dim]-(1 if dim==2 else 0)
            command=np.array([r['first_action'] for r in rows])[:,dim]
            corr=report['overall'][c]['first_decision_target_response']['per_axis'][('x','y','z')[dim]]['first_command_target_correlation']
            ax.scatter(target,command,s=10,alpha=.45,c=colors[j],label=f'{labels[j]} r={corr:.3f}' if corr is not None else f'{labels[j]} r=N/A')
        ax.set_xlabel(f'Initial target error {("x","y","z")[dim]} (m)'); ax.set_ylabel('First normalized command'); ax.set_ylim(-1.05,1.05); ax.legend(fontsize=7)
    fig.suptitle('Same initial state, latent and epsilon; all200 target pairs',fontsize=10)
    save(fig,'first_action_target_response.png')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,splits):
        values=[report['results'][c][split]['summary']['action_saturation_fraction']*100 for c in conditions]
        ax.bar(labels,values,color=colors); ax.set_ylabel('Selected command elements |a| >=0.95 (%)'); ax.set_title(split)
        for i,v in enumerate(values): ax.text(i,v+1,f'{v:.2f}%',ha='center')
        ax.set_ylim(0,max(50,max(values)*1.2))
    save(fig,'command_saturation_comparison.png')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,key,title in zip(axes,('planning_time','total_decision_time'),('Planning','Full decision (Encoder + diagnostics)')):
        for j,c in enumerate(conditions):
            d=report['overall'][c][key]
            ax.plot(['Mean','Median','p95','Max'],[d[k+'_ms'] for k in ('mean','median','p95','max')],'o-',c=colors[j],label=labels[j])
        ax.axhline(40,c='red',ls='--',label='40 ms budget'); ax.set_ylabel('Wall time (ms)'); ax.set_title(title); ax.legend(fontsize=8)
    save(fig,'planning_latency_comparison.png')
    return paths
