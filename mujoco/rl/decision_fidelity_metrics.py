"""Per-state ranking/regret; descriptive correlations, no cost redefinition."""
import numpy as np

EPSILON=1e-12


def rankdata(values):
    """One-based average ranks, including exact ties (no arbitrary tie break)."""
    a=np.asarray(values,float); order=np.argsort(a,kind='stable'); ranks=np.empty(len(a),float)
    boundaries=np.r_[0,np.flatnonzero(np.diff(a[order])!=0)+1,len(a)]
    for start,end in zip(boundaries[:-1],boundaries[1:]): ranks[order[start:end]]=(start+1+end)/2
    return ranks


def correlation(x,y,method='spearman'):
    x=np.asarray(x,float); y=np.asarray(y,float)
    if x.shape!=y.shape or not np.isfinite(np.r_[x,y]).all(): return None
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0: return None
    if method=='spearman': x,y=rankdata(x),rankdata(y)
    if method=='kendall':
        i,j=np.triu_indices(len(x),1); sx=np.sign(x[i]-x[j]); sy=np.sign(y[i]-y[j])
        # Exact tau-b: ties in each coordinate enter its denominator separately.
        value=np.sum(sx*sy)/np.sqrt(float(np.count_nonzero(sx))*np.count_nonzero(sy))
    elif method in ('spearman','pearson'):
        dx=x-x.mean(); dy=y-y.mean(); value=np.dot(dx,dy)/np.sqrt(np.dot(dx,dx)*np.dot(dy,dy))
    else: raise ValueError('unknown correlation method')
    return float(value) if np.isfinite(value) else None


def reference_percentile(value,costs):
    c=np.asarray(costs,float)
    return dict(cost_percentile=float((np.sum(c<value)+.5*np.sum(c==value))/len(c)),
        strictly_better_than_fraction=float(np.mean(c>value)),beats_every_candidate=bool(value<c.min()))


def ranking(predicted,true):
    p=np.asarray(predicted,float); t=np.asarray(true,float)
    if p.ndim!=1 or t.shape!=p.shape or len(t)<2 or not np.isfinite(np.r_[p,t]).all():
        raise ValueError('all finite candidate costs required; never drop failed candidates')
    po=np.argsort(p,kind='stable'); to=np.argsort(t,kind='stable'); ip=int(po[0]); it=int(to[0])
    regret=float(t[ip]-t[it]); spread=float(np.ptp(t)); random=float(t.mean()-t[it])
    return dict(pred_best_index=ip,true_best_index=it,spearman=correlation(p,t),kendall=correlation(p,t,'kendall'),
        top1_exact=ip==it,pred_best_in_true_top5=bool(ip in to[:5]),true_best_in_pred_top5=bool(it in po[:5]),
        regret=regret,normalized_regret=regret/(spread+EPSILON),random_expected_regret=random,
        random_normalized_regret=random/(spread+EPSILON),model_better_than_random=bool(regret<random),
        model_selected_true_rank_percentile=float((rankdata(t)[ip]-1)/(len(t)-1)),
        true_best_margin=float(t[to[1]]-t[it]),pred_best_margin=float(p[po[1]]-p[ip]),
        true_cost_spread=spread,normalized_true_best_margin=float((t[to[1]]-t[it])/(spread+EPSILON)),
        calibration_pearson=correlation(p,t,'pearson'),calibration_mae=float(np.mean(np.abs(p-t))),
        calibration_rmse=float(np.sqrt(np.mean((p-t)**2))))


def summary(values):
    valid=[float(v) for v in values if v is not None and np.isfinite(v)]
    if not valid: return dict(count=0,missing=len(values),mean=None,median=None,p75=None,p90=None,min=None,max=None)
    a=np.asarray(valid)
    return dict(count=len(a),missing=len(values)-len(a),mean=float(a.mean()),median=float(np.median(a)),
        p75=float(np.quantile(a,.75)),p90=float(np.quantile(a,.9)),min=float(a.min()),max=float(a.max()),
        sample_std=float(a.std(ddof=1)) if len(a)>1 else None)


def candidate_metrics(pred,true,terminals,pred_terminals,distances,speeds,failures,d0,script,script_pred,stats):
    out=ranking(pred,true); ip=out['pred_best_index']; it=out['true_best_index']
    error=(pred_terminals-terminals)/stats['obs']['std']
    out['prediction_error']=dict(whole_set_h10_nrmse=float(np.sqrt(np.mean(error**2))),
        selected_h10_nrmse=float(np.sqrt(np.mean(error[ip]**2))))
    def utility(i):
        finite=lambda v: float(v) if np.isfinite(v) else None
        return dict(index=int(i),true_cost=float(true[i]),terminal_distance=finite(distances[i]),terminal_speed=finite(speeds[i]),
            distance_reduction=finite(d0-distances[i]),physical_failure=bool(failures[i]),
            distance_improved=bool(d0-distances[i]>1e-6),terminal_speed_below_success_threshold=bool(speeds[i]<.15))
    out.update(oracle=utility(it),model_selected=utility(ip),candidate_failure_fraction=float(np.mean(failures)),
        script_true_percentile=reference_percentile(script['cost'],true),script_pred_percentile=reference_percentile(script_pred,pred),
        cost_task_alignment=dict(distance=correlation(true,distances),distance_reduction=correlation(true,d0-distances),
            speed=correlation(true,speeds)))
    return out
