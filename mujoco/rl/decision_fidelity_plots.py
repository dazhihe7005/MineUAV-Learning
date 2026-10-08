"""All-state descriptive figures; fixed-stride candidate display, no selection."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from latent_dynamics_data import file_hash
from run_world_model_decision_fidelity import CONDITIONS

EXAMPLE=('benchmark',0)
FIGURES=('predicted_vs_true_cost_scatter.png','ranking_metrics_unconstrained_vs_supported.png',
    'normalized_regret_distribution.png','model_selected_true_rank_percentile.png','oracle_candidate_task_progress.png',
    'scripted_cost_percentile.png','cost_vs_task_progress.png','prediction_error_vs_ranking_quality.png',
    'true_cost_margin_vs_ranking_error.png','selected_state_candidate_ranking_example.png')
LABELS=('Unconstrained','Train-supported'); COLORS=('#2878b5','#e1812c')


def render_figures(report,directory):
    directory=Path(directory); rows=report['states']; summaries=report['results']; artifacts=[]
    data=[]
    for row in rows:
        item=row['local_arrays']
        if file_hash(item['path'])!=item['sha256']: raise AssertionError('plot arrays changed')
        with np.load(item['path'],allow_pickle=False) as raw: data.append({k:raw[k] for k in raw.files})
    def values(c,key): return np.array([r['conditions'][c][key] for r in rows])
    def save(fig,index):
        fig.tight_layout(); path=directory/FIGURES[index]; fig.savefig(path,dpi=160); plt.close(fig)
        artifacts.append(dict(path=str(path),sha256=file_hash(path)))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for c,label,ax,color in zip(CONDITIONS,LABELS,axes,COLORS):
        true=np.concatenate([d[c+'_true_cost'][::16] for d in data]); pred=np.concatenate([d[c+'_pred_cost'][::16] for d in data])
        ax.scatter(true,pred,s=3,alpha=.25,color=color); lo=min(true.min(),pred.min()); hi=max(true.max(),pred.max())
        ax.plot([lo,hi],[lo,hi],'k--',lw=1); ax.set(title=label,xlabel='MuJoCo true cost',ylabel='Frozen model predicted cost')
    fig.suptitle('All200 states; candidates displayed at fixed stride16 only'); save(fig,0)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); x=np.arange(2)
    for j,key in enumerate(('spearman','kendall')):
        axes[0].bar(x+(j-.5)*.3,[summaries[c]['all'][key]['mean'] for c in CONDITIONS],width=.3,label=key)
    for j,key in enumerate(('top1_exact','pred_best_in_true_top5','true_best_in_pred_top5')):
        axes[1].bar(x+(j-1)*.24,[summaries[c]['all'][key]['fraction'] for c in CONDITIONS],width=.24,label=key.replace('_',' '))
    for ax in axes: ax.set_xticks(x,LABELS); ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    axes[0].set(ylabel='Mean per-state coefficient',ylim=(-.1,1)); axes[1].set(ylabel='State fraction',ylim=(0,1)); save(fig,1)
    fig,ax=plt.subplots(figsize=(8,4)); boxes=[]; labels=[]
    for c,label in zip(CONDITIONS,LABELS):
        boxes.extend([values(c,'normalized_regret'),values(c,'random_normalized_regret')]); labels.extend([label+' model',label+' random expectation'])
    ax.boxplot(boxes,tick_labels=labels,showmeans=True); ax.tick_params(axis='x',labelsize=8); ax.set(ylabel='Regret / true candidate cost range',title='All200 states; random baseline analytically averages512 choices'); save(fig,2)
    fig,ax=plt.subplots(figsize=(8,4))
    for c,label,color in zip(CONDITIONS,LABELS,COLORS): ax.hist(values(c,'model_selected_true_rank_percentile'),bins=np.linspace(0,1,21),alpha=.5,label=label,color=color)
    ax.set(xlabel='Model-selected true rank percentile (0best,1worst)',ylabel='States'); ax.legend(); save(fig,3)
    fig,axes=plt.subplots(1,2,figsize=(12,4)); boxes=[[],[]]; labels=[]
    for c,label in zip(CONDITIONS,LABELS):
        for key in ('oracle','model_selected','scripted'):
            records=[r['scripted'] if key=='scripted' else r['conditions'][c][key] for r in rows]
            boxes[0].append([v['distance_reduction'] for v in records]); boxes[1].append([v['terminal_speed'] for v in records]); labels.append(label+' '+key.replace('_',' '))
    for ax,b,y in zip(axes,boxes,('H10 distance reduction [m]','H10 terminal speed [m/s]')):
        ax.boxplot(b,tick_labels=labels,showmeans=True); ax.tick_params(axis='x',rotation=35,labelsize=7); ax.set_ylabel(y); ax.axhline(0,color='k',lw=.6)
    save(fig,4)
    fig,ax=plt.subplots(figsize=(9,4)); b=[]; labels=[]
    for c,label in zip(CONDITIONS,LABELS):
        for key in ('script_true_percentile','script_pred_percentile'):
            b.append([r['conditions'][c][key]['cost_percentile'] for r in rows]); labels.append(label+(' true' if key=='script_true_percentile' else ' predicted'))
    ax.boxplot(b,tick_labels=labels,showmeans=True); ax.set(ylabel='Scripted cost empirical percentile (0best)',title='Scripted is a reference, never added to512 candidates'); save(fig,5)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c,label,color,ax in zip(CONDITIONS,LABELS,COLORS,axes):
        gain=np.concatenate([r['initial_distance']-d[c+'_distance'][::16] for r,d in zip(rows,data)])
        scaled=np.concatenate([(d[c+'_true_cost'][::16]-d[c+'_true_cost'].min())/(np.ptp(d[c+'_true_cost'])+1e-12) for d in data])
        ax.scatter(gain,scaled,s=3,alpha=.25,color=color); ax.set(title=label,xlabel='H10 distance reduction [m]',ylabel='Within-state true cost range fraction')
    fig.suptitle('Display normalization only; original cost/rank unchanged'); save(fig,6)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for j,(c,label,color) in enumerate(zip(CONDITIONS,LABELS,COLORS)):
        err=[r['conditions'][c]['prediction_error']['whole_set_h10_nrmse'] for r in rows]
        for i,key in enumerate(('spearman','normalized_regret')):
            axes[i,j].scatter(err,values(c,key),s=12,alpha=.5,color=color); axes[i,j].set(title=label,xlabel='Whole-set H10 observation NRMSE',ylabel=key)
    save(fig,7)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for c,label,color in zip(CONDITIONS,LABELS,COLORS):
        for ax,key in zip(axes,('true_best_margin','true_cost_spread')):
            ax.scatter(values(c,key),values(c,'normalized_regret'),s=12,alpha=.5,label=label,color=color); ax.set(xlabel=key.replace('_',' '),ylabel='Normalized regret'); ax.legend(fontsize=8)
    save(fig,8)
    i=next(i for i,r in enumerate(rows) if (r['split'],r['index'])==EXAMPLE); row=rows[i]; d=data[i]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for c,label,ax in zip(CONDITIONS,LABELS,axes):
        true=d[c+'_true_cost']; pred=d[c+'_pred_cost']; m=row['conditions'][c]
        ax.scatter(true,pred,s=8,alpha=.4); ip=m['pred_best_index']; it=m['true_best_index']
        ax.scatter([true[ip]],[pred[ip]],s=80,marker='x',color='r',label='Model selected')
        ax.scatter([true[it]],[pred[it]],s=70,marker='*',color='g',label='True oracle')
        ax.scatter([row['scripted']['cost']],[row['scripted_pred_cost']],s=60,marker='D',color='k',label='Scripted reference')
        ax.set(title=label,xlabel='True cost',ylabel='Predicted cost'); ax.legend(fontsize=8)
    fig.suptitle('Predeclared example: benchmark episode0 (not selected by outcome)'); save(fig,9)
    return artifacts
