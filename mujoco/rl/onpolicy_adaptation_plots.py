"""Eight fixed aggregate figures; no raw candidate traces committed."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from onpolicy_adaptation_analysis import MODELS

FIGURES=('adaptation_training_curves.png','later_state_prediction_comparison.png','later_state_spearman_comparison.png',
    'later_state_regret_comparison.png','best_tail_ranking_comparison.png','selected_action_true_rank_comparison.png',
    'nominal_regression_comparison.png','ranking_vs_state_ood.png')
LABELS={'original':'Frozen original','replay':'Replay control','mpc_state':'MPC-state adapted'}

def plot(report,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    def save(fig,name):fig.tight_layout();fig.savefig(directory/name,dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for source in ('replay','mpc_state'):
        h=report['training'][source]['history'];x=[r['step'] for r in h]
        axes[0].plot(x,[r['train']['observation'] for r in h],label=LABELS[source]);axes[1].plot(x,[r['validation']['observation'] for r in h],label=LABELS[source])
    axes[0].set_title('Train L_obs (last50 update mean)');axes[1].set_title('Own independent validation L_obs')
    for ax in axes:ax.set_xlabel('Optimizer updates');ax.legend();ax.grid(alpha=.2)
    save(fig,FIGURES[0]);stages=list(report['aggregate']);x=np.arange(len(stages))
    specs=[(FIGURES[1],'prediction','Candidate-mean H10 NRMSE'),(FIGURES[2],'spearman','Whole-set Spearman'),
        (FIGURES[3],'normalized_regret','Normalized decision regret'),(FIGURES[4],'best_tail_spearman','True-best26 Spearman'),
        (FIGURES[5],'model_selected_true_rank_percentile','Selected true-rank percentile (lower better)')]
    for name,key,title in specs:
        fig,ax=plt.subplots(figsize=(7,4))
        for i,model in enumerate(MODELS):
            values=[report['aggregate'][s]['models'][model]['prediction']['whole']['10']['mean'] if key=='prediction' else
                report['aggregate'][s]['models'][model][key]['mean'] for s in stages]
            ax.bar(x+(i-1)*.24,values,width=.24,label=LABELS[model])
        ax.set_xticks(x,stages);ax.set_ylabel(title);ax.legend();ax.grid(axis='y',alpha=.2);save(fig,name)
    fig,ax=plt.subplots(figsize=(7,4));horizons=[1,10,25,50]
    for model in MODELS:ax.plot(horizons,[report['nominal'][model][str(h)]['normalized_observation_rmse'] for h in horizons],marker='o',label=LABELS[model])
    ax.set_xlabel('Nominal rollout horizon (steps)');ax.set_ylabel('Normalized observation RMSE');ax.legend();ax.grid(alpha=.2);save(fig,FIGURES[6])
    fig,ax=plt.subplots(figsize=(7,4))
    for model in MODELS:
        rows=[r for r in report['evaluation_rows'] if r['model']==model]
        ax.scatter([r['state_ood']['observation_standardized_rms'] for r in rows],[r['metrics']['normalized_regret'] for r in rows],s=12,alpha=.5,label=LABELS[model])
    ax.set_xlabel('Original Train observation-standardized RMS');ax.set_ylabel('Normalized regret');ax.legend();save(fig,FIGURES[7])
    return [str(directory/n) for n in FIGURES]
