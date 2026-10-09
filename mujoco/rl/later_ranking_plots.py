"""Scientific aggregate figures only; no policy or cost changes."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from latent_dynamics_data import file_hash
from later_ranking_core import STAGES
from later_ranking_analysis import CONDITIONS

NAMES=('ranking_spearman_vs_episode_progress','normalized_regret_vs_episode_progress',
    'selected_true_rank_vs_progress','best_tail_ranking_vs_progress','prediction_error_vs_progress',
    'state_ood_vs_progress','state_ood_vs_regret','scripted_percentile_vs_progress',
    'oracle_progress_vs_episode_progress','paired_episode_ranking_degradation')
COLORS=('#1565c0','#ef6c00')

def render(report,directory):
    directory=Path(directory); figures=[]; x=np.arange(4)
    groups={c:report['results'][c]['all']['stages'] for c in CONDITIONS}
    def values(c,path,scale=1):
        out=[]
        for s in STAGES:
            node=groups[c][s]
            for key in path.split('.'): node=node[key]
            v=node['mean']; out.append(np.nan if v is None else scale*v)
        return out
    def finish(fig,name):
        fig.tight_layout(); p=directory/f'{name}.png'; fig.savefig(p,dpi=150); plt.close(fig)
        figures.append(dict(path=str(p),sha256=file_hash(p)))
    def stages(ax,ylabel):
        ax.set_xticks(x,['S0 0%','S1 25%','S2 50%','S3 75%']); ax.set_xlabel('Relative realized episode progress')
        ax.set_ylabel(ylabel); ax.grid(alpha=.25)
    for name,path,label,scale in [
        (NAMES[0],'metrics.spearman','Mean whole-set Spearman rho',1),
        (NAMES[1],'metrics.normalized_regret','Mean range-normalized regret',1),
        (NAMES[2],'metrics.model_selected_true_rank_percentile','Mean selected true rank percentile (0=best)',100)]:
        fig,ax=plt.subplots(figsize=(7,4))
        for c,color in zip(CONDITIONS,COLORS):
            ax.plot(x,values(c,path,scale),'o-',color=color,label=c)
            if name==NAMES[1]: ax.plot(x,values(c,'metrics.random_normalized_regret'),'--',color=color,alpha=.6,label=c+' random expectation')
        stages(ax,label); ax.legend(fontsize=8); finish(fig,name)
    fig,ax=plt.subplots(figsize=(7,4))
    for c,color in zip(CONDITIONS,COLORS):
        ax.plot(x,values(c,'metrics.spearman'),'o-',color=color,label=c+' whole512')
        ax.plot(x,values(c,'best_tail.spearman'),'s--',color=color,label=c+' true best26')
    stages(ax,'Mean Spearman rho (undefined counts in report)'); ax.legend(fontsize=8); finish(fig,NAMES[3])
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for c,ax in zip(CONDITIONS,axs):
        for key,label in [('whole_set_h10_nrmse','Whole-set pooled'),('selected_h10_nrmse','Model selected'),('true_best_h10_nrmse','True-best'),('scripted_h10_nrmse','Scripted')]:
            ax.plot(x,values(c,'prediction.'+key),'o-',label=label)
        stages(ax,'Mean H10 terminal NRMSE'); ax.set_title(c); ax.legend(fontsize=8)
    finish(fig,NAMES[4])
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for ax,key,label in zip(axs,('observation_standardized_rms','mahalanobis_rms','action_standardized_rms'),
            ('Observation Train-standardized RMS','Latent regularized Mahalanobis RMS','Selected action standardized RMS')):
        for c,color in zip(CONDITIONS,COLORS): ax.plot(x,values(c,'ood.'+key),'o-',color=color,label=c)
        stages(ax,label); ax.legend(fontsize=7)
    finish(fig,NAMES[5])
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for c,ax in zip(CONDITIONS,axs):
        for stage,color in zip(STAGES,('blue','green','orange','red')):
            rows=[r for r in report['states'] if r['condition']==c and r['stage']==stage]
            ax.scatter([r['ood']['mahalanobis_rms'] for r in rows],[r['metrics']['normalized_regret'] for r in rows],
                s=12,alpha=.5,color=color,label=stage)
        ax.set_xlabel('Latent Mahalanobis RMS'); ax.set_ylabel('Normalized regret'); ax.set_title(c); ax.legend(fontsize=8); ax.grid(alpha=.25)
    finish(fig,NAMES[6])
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for c,ax in zip(CONDITIONS,axs):
        ax.plot(x,values(c,'metrics.script_true_percentile.cost_percentile',100),'o-',label='Scripted true cost')
        ax.plot(x,values(c,'metrics.script_pred_percentile.cost_percentile',100),'s--',label='Scripted predicted cost')
        stages(ax,'Mean scripted percentile (0=best)'); ax.set_title(c); ax.legend(fontsize=8)
    finish(fig,NAMES[7])
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for c,ax in zip(CONDITIONS,axs):
        ax.plot(x,values(c,'metrics.oracle.distance_reduction'),'o-',label='True-cost best')
        ax.plot(x,values(c,'metrics.model_selected.distance_reduction'),'s-',label='Model selected')
        ax.plot(x,values(c,'scripted.distance_reduction'),'^--',label='Scripted')
        stages(ax,'Mean H10 distance reduction (m)'); ax.axhline(0,color='gray',lw=.8); ax.set_title(c); ax.legend(fontsize=8)
    finish(fig,NAMES[8])
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for c,color in zip(CONDITIONS,COLORS):
        paired=report['results'][c]['all']['paired']['individual']
        dx=[p['spearman_delta'] for p in paired if p['spearman_delta'] is not None]
        dy=[p['regret_delta'] for p in paired]
        axs[0].hist(dx,bins=20,alpha=.45,label=c,color=color); axs[1].hist(dy,bins=20,alpha=.45,label=c,color=color)
    for ax,label in zip(axs,('Within-episode Spearman(S3)-Spearman(S0)','Within-episode regret(S3)-regret(S0)')):
        ax.set_xlabel(label); ax.set_ylabel('Episodes'); ax.axvline(0,color='black',lw=1); ax.legend(fontsize=8)
    finish(fig,NAMES[9]); return figures
