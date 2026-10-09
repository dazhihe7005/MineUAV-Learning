"""Per-stage descriptive and strictly within-episode statistics."""
from collections import Counter
import numpy as np
from later_ranking_core import STAGES,stage_indices
from decision_fidelity_metrics import summary,correlation

CONDITIONS=('unconstrained','train_supported')
SPLITS=('benchmark','holdout')

def validate_cohort(records,episodes):
    required={(c,s,i) for c in CONDITIONS for s in SPLITS for i in range(100)}
    keys=[(e['condition'],e['split'],e['index']) for e in episodes]
    if len(keys)!=400 or set(keys)!=required: raise AssertionError('all400 original condition episodes required')
    expected={(e['condition'],e['split'],e['index'],s,i) for e in episodes for s,i in stage_indices(e['episode_steps']).items()}
    actual=[(r['condition'],r['split'],r['index'],r['stage'],r['decision_index']) for r in records]
    if len(actual)!=len(expected) or set(actual)!=expected: raise AssertionError('stage eligibility/deduplication changed')
    return len(expected)

def paired_changes(records,failed_only=False):
    grouped={}
    for r in records:
        if failed_only and r['termination_reason']=='success': continue
        grouped.setdefault((r['split'],r['index']),{})[r['stage']]=r
    pairs=[g for g in grouped.values() if set(g)==set(STAGES)]
    result=dict(episodes=len(pairs),eligible_episode_ids=[dict(split=g['S0']['split'],index=g['S0']['index']) for g in pairs],
        failed_only=failed_only,S3_minus_S0={})
    for key in ('spearman','kendall','normalized_regret','model_selected_true_rank_percentile'):
        deltas=[g['S3']['metrics'][key]-g['S0']['metrics'][key]
            if g['S3']['metrics'][key] is not None and g['S0']['metrics'][key] is not None else None for g in pairs]
        result['S3_minus_S0'][key]=summary(deltas)
    result['individual']=[dict(split=g['S0']['split'],index=g['S0']['index'],
        spearman_delta=g['S3']['metrics']['spearman']-g['S0']['metrics']['spearman']
            if g['S3']['metrics']['spearman'] is not None and g['S0']['metrics']['spearman'] is not None else None,
        regret_delta=g['S3']['metrics']['normalized_regret']-g['S0']['metrics']['normalized_regret']) for g in pairs]
    return result

def numeric_summary(rows):
    if not rows: return {}
    out={}
    for key in rows[0]:
        values=[r[key] for r in rows]
        if isinstance(values[0],dict): out[key]=numeric_summary(values)
        elif all(v is None or isinstance(v,(int,float,bool,np.number)) for v in values): out[key]=summary(values)
    return out

def relationship(rows,x,y):
    pairs=[(x(r),y(r)) for r in rows if x(r) is not None and y(r) is not None]
    a,b=zip(*pairs) if pairs else ([],[])
    return dict(paired_count=len(pairs),**{m:correlation(a,b,m) for m in ('pearson','spearman')})

def summarize_stage(rows):
    out=dict(states=len(rows),candidate_count=512*len(rows),failure_type_composition=dict(Counter(r['termination_reason'] for r in rows)),
        initial_distance=summary([r['initial_distance'] for r in rows]),current_distance=summary([r['distance'] for r in rows]),
        absolute_decision_index=summary([r['decision_index'] for r in rows]),
        realized_progress=summary([r['realized_progress'] for r in rows]),metrics=numeric_summary([r['metrics'] for r in rows]),
        best_tail=numeric_summary([r['best_tail'] for r in rows]),prediction=numeric_summary([r['prediction'] for r in rows]),
        ood=numeric_summary([r['ood'] for r in rows]),scripted=numeric_summary([r['scripted'] for r in rows]))
    for key in ('top1_exact','pred_best_in_true_top5','true_best_in_pred_top5','model_better_than_random'):
        out[key]=dict(count=sum(r['metrics'][key] for r in rows),eligible=len(rows))
    out['correlations']={}
    for xkey in ('observation_standardized_rms','action_standardized_rms','mahalanobis_rms'):
        for ykey in ('spearman','normalized_regret','model_selected_true_rank_percentile'):
            out['correlations'][xkey+'_vs_'+ykey]=relationship(rows,lambda r:r['ood'][xkey],lambda r:r['metrics'][ykey])
        out['correlations'][xkey+'_vs_prediction']=relationship(rows,lambda r:r['ood'][xkey],lambda r:r['prediction']['selected_h10_nrmse'])
    for xkey in ('whole_set_h10_nrmse','selected_h10_nrmse'):
        for ykey in ('spearman','normalized_regret'):
            out['correlations'][xkey+'_vs_'+ykey]=relationship(rows,lambda r:r['prediction'][xkey],lambda r:r['metrics'][ykey])
    for xkey in ('true_best_margin','true_cost_spread'):
        out['correlations'][xkey+'_vs_regret']=relationship(rows,lambda r:r['metrics'][xkey],lambda r:r['metrics']['normalized_regret'])
    out['correlations']['tail_spread_vs_regret']=relationship(rows,lambda r:r['best_tail']['true_cost_spread'],lambda r:r['metrics']['normalized_regret'])
    return out

def aggregate(records):
    result={}
    for c in CONDITIONS:
        result[c]={}
        for split in (*SPLITS,'all'):
            selected=[r for r in records if r['condition']==c and (split=='all' or r['split']==split)]
            result[c][split]=dict(stages={s:summarize_stage([r for r in selected if r['stage']==s]) for s in STAGES},
                paired=paired_changes(selected),failed_only_paired=paired_changes(selected,True),
                pooled_state_relationships=summarize_stage(selected)['correlations'])
    return result
