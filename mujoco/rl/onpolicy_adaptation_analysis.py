"""Paired state-cohort aggregation; no model/seed/result selection."""
from decision_fidelity_metrics import summary,correlation

MODELS=('original','replay','mpc_state')
KEYS=('spearman','kendall','normalized_regret','model_selected_true_rank_percentile')

def aggregate(rows):
    result={}
    for stage in ('S0','S2','S3'):
        selected=[r for r in rows if r['stage']==stage]
        if not selected:continue
        groups={m:{r['target_id']:r['metrics'] for r in selected if r['model']==m} for m in MODELS}
        ids=set(groups['original'])
        if any(set(v)!=ids for v in groups.values()) or len(selected)!=len(ids)*3:raise ValueError('unmatched shared state cohort')
        models={}
        for model,g in groups.items():
            values=list(g.values());out={k:summary([r[k] for r in values]) for k in KEYS}
            out.update(state_count=len(values),best_tail_spearman=summary([r['best_tail']['spearman'] for r in values]),
                top1_count=sum(r['top1_exact'] for r in values),pred_best_in_true_top5_count=sum(r['pred_best_in_true_top5'] for r in values),
                true_best_in_pred_top5_count=sum(r['true_best_in_pred_top5'] for r in values),
                prediction={key:{str(h):summary([r['prediction'][key][str(h)]['normalized_observation_rmse'] for r in values]) for h in (1,5,10)} for key in ('whole','selected','true_best')})
            models[model]=out
        paired={k:summary([groups['mpc_state'][i][k]-groups['replay'][i][k] for i in sorted(ids)]) for k in KEYS}
        paired['best_tail_spearman']=summary([groups['mpc_state'][i]['best_tail']['spearman']-groups['replay'][i]['best_tail']['spearman']
            if groups['mpc_state'][i]['best_tail']['spearman'] is not None and groups['replay'][i]['best_tail']['spearman'] is not None else None for i in sorted(ids)])
        paired['h10_prediction']=summary([groups['mpc_state'][i]['prediction']['whole']['10']['normalized_observation_rmse']-
            groups['replay'][i]['prediction']['whole']['10']['normalized_observation_rmse'] for i in sorted(ids)])
        paired['regret_win_count']=sum(groups['mpc_state'][i]['normalized_regret']<groups['replay'][i]['normalized_regret'] for i in ids)
        result[stage]=dict(models=models,paired_mpc_minus_replay=paired)
    return result

def ood_correlations(rows):
    result={}
    for model in MODELS:
        g=[r for r in rows if r['model']==model];x=[r['state_ood']['observation_standardized_rms'] for r in g]
        result[model]={k:{method:correlation(x,[r['metrics'][k] for r in g],method) for method in ('pearson','spearman')} for k in KEYS}
    return result
