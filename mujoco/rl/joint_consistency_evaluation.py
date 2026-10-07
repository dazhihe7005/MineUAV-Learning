"""Own-coordinate diagnostics; no consistency training branch at inference."""
import numpy as np
import torch

from joint_latent_world_model import component_hashes,latent_summary,load_joint
from joint_latent_evaluation import encoded_sequences
from joint_composition_core import freeze_joint,fit_manifold,distribution_scores,describe,compose
from joint_composition_evaluation import audit
from run_joint_latent_composition_audit import leakage_checks
from joint_consistency_training import load_consistency


def relative_residual(prediction,source,reference,epsilon=1e-6):
    p,s,r=(np.asarray(a,np.float64) for a in (prediction,source,reference))
    residual=np.linalg.norm(p-r,axis=1); movement=np.linalg.norm(r-s,axis=1)
    ratio=residual/(movement+epsilon); moving=movement>epsilon
    def stats(v):
        row=describe(v)
        if len(v): row.update({name:float(np.quantile(v,q)) for name,q in [('q01',.01),('q25',.25),('q75',.75),('q99',.99)]})
        return row
    return dict(all=stats(ratio),moving_reference_only=stats(ratio[moving]),epsilon=epsilon,
        stationary_reference_count=int((~moving).sum()),reference_movement=describe(movement),residual=describe(residual),
        rule='rawlatent L2 residual/(encoder-reference increment L2+epsilon); not coordinate invariant; all includes stationary denominators, whose ratios can dominate the mean')


@torch.inference_mode()
def local_relative(model,test,stats):
    sequences=encoded_sequences(model,test,stats); source=[]; target=[]; predicted=[]
    for ep,z in zip(test,sequences):
        p=compose(model,torch.from_numpy(z[:-1]),torch.from_numpy(ep['action'][:,None].astype(np.float64)),stats)['latents'][:,-1].numpy()
        source.append(z[:-1]); target.append(z[1:]); predicted.append(p)
    return relative_residual(np.concatenate(predicted),np.concatenate(source),np.concatenate(target))


def evaluate(model_path,context,test):
    model,stats,_=load_consistency(model_path); freeze_joint(model); before=component_hashes(model)
    sequences=encoded_sequences(model,context['splits']['train'],stats)
    train=np.concatenate(sequences)
    manifold=fit_manifold(sequences,[e['action'] for e in context['splits']['train']])
    scores=distribution_scores(train,manifold)
    manifold['train_score_summary']={k:describe(v) for k,v in scores.items()}
    manifold['train_score_quantiles']={k:np.quantile(v,[1/3,2/3]).tolist() for k,v in scores.items()}
    manifold['provenance'].update(dataset_sha256=context['dataset']['source_dataset_sha256']['train'],encoder_parameter_sha256=before['encoder'])
    baseline=context['baseline']; pi=baseline['pi_magnitude_bins']['thresholds_m_s2']; hs=baseline['horizons_steps']
    metrics,plot=audit(model,context['splits']['train'],test,stats,pi,hs,baseline['selected_window'],manifold)
    reference,_stats,_=load_joint(baseline['model']['path']); freeze_joint(reference)
    relative=dict(v1=local_relative(reference,test,stats),v2=local_relative(model,test,stats))
    checks=leakage_checks(dict(source=baseline,model=model,stats=stats,test=test))
    if component_hashes(model)!=before: raise AssertionError('evaluation changed v2 parameters')
    return dict(audit=metrics,manifold=manifold,relative=relative,
        latent_variance=dict(train=latent_summary(train),test=latent_summary(np.concatenate(encoded_sequences(model,test,stats)))),
        verification=dict(checks,no_parameter_mutation=True,consistency_branch_absent_at_inference=True),model=model),plot


def comparison(v1,v2,relative,horizons):
    rows={}
    for h in horizons:
        a=v1['autonomous'][str(h)]['observation']['normalized_observation_rmse']
        b=v2['autonomous'][str(h)]['observation']['normalized_observation_rmse']
        rows[str(h)]=dict(v1=a,v2=b,absolute_delta=b-a,reduction_percent=100*(a-b)/a,
            regime='in_training_horizon' if h<=10 else 'out_of_training_horizon')
    correction={}
    for name,data in [('v1',v1),('v2',v2)]:
        a=data['periodic_correction']['never']['observation']['normalized_observation_rmse']
        b=data['periodic_correction']['1']['observation']['normalized_observation_rmse']
        correction[name]=dict(never=a,C1=b,absolute_gap=a-b,never_to_C1_ratio=a/b,reduction_with_C1_percent=100*(a-b)/a)
    return dict(observation=rows,correction_sensitivity=correction,relative_local_residual=relative,
        latent_comparison_limit='Native scale-aware errors, rawL2 and Decoder ratios use different learned latent geometries; not cross-model physical quantities.')
