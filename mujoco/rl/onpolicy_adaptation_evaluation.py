"""Three frozen models share original-MPC exact states, candidates and truth."""
import numpy as np
import torch
from joint_latent_world_model import encode_episode,predict_latent
from latent_dynamics_multistep import ErrorAccumulator
from later_ranking_core import stage_indices,prediction_trajectory,tail_metrics
from latent_mpc_core import planning_cost
from decision_fidelity_metrics import candidate_metrics,reference_percentile

def final_stages(length):
    return {k:v for k,v in stage_indices(length).items() if k in ('S0','S2','S3')}

@torch.inference_mode()
def predict_from_prefix(model,stats,observations,actions,candidates):
    if len(observations)!=len(actions)+1:raise ValueError('exact current history required')
    previous=np.vstack([np.zeros((1,4),np.float32),actions])
    z=encode_episode(model,observations,previous,stats)[-1:].numpy()
    return prediction_trajectory(model,stats,z,candidates)

def prediction_metrics(pred,truth,std,selected,true_best):
    out={k:{} for k in ('whole','selected','true_best')}
    for h in (1,5,10):
        error=pred[:,h-1]-truth[:,h-1]
        for key,ids in [('whole',np.arange(len(pred))),('selected',[selected]),('true_best',[true_best])]:
            a=ErrorAccumulator(std);a.add(error[ids]);out[key][str(h)]=a.result()
    return out

def evaluate_state(model,stats,raw,state):
    candidates=raw['candidates'];prev=raw['prefix_actions'][-1] if len(raw['prefix_actions']) else np.zeros(4,np.float32)
    prediction=predict_from_prefix(model,stats,raw['prefix_observations'],raw['prefix_actions'],candidates)
    costs,_=planning_cost(prediction[:,-1],candidates,prev)
    scripted=predict_from_prefix(model,stats,raw['prefix_observations'],raw['prefix_actions'],raw['script_actions'][None])
    scost,_=planning_cost(scripted[:,-1],raw['script_actions'][None],prev)
    script=state['script'];metrics=candidate_metrics(costs,raw['true_costs'],raw['truth'][:,-1],prediction[:,-1],
        raw['distances'],raw['speeds'],raw['failures'],state['initial_distance'],script,float(scost[0]),stats)
    metrics['best_tail']=tail_metrics(costs,raw['true_costs'])
    metrics['prediction']=prediction_metrics(prediction,raw['truth'],stats['obs']['std'],metrics['pred_best_index'],metrics['true_best_index'])
    a=ErrorAccumulator(stats['obs']['std']);a.add(scripted[0,-1:]-raw['script_truth'][-1:])
    metrics['scripted_prediction_h10']=a.result()
    metrics['scripted_prediction']={}
    for h in (1,5,10):
        a=ErrorAccumulator(stats['obs']['std']);a.add(scripted[0,h-1:h]-raw['script_truth'][h-1:h])
        metrics['scripted_prediction'][str(h)]=a.result()
    return metrics,costs

@torch.inference_mode()
def nominal_prediction(model,episodes,stats,horizons=(1,10,25,50)):
    model.eval().requires_grad_(False);acc={str(h):ErrorAccumulator(stats['obs']['std']) for h in horizons}
    for ep in episodes:
        z=encode_episode(model,ep['obs'],ep['previous_action'],stats)
        for h in horizons:
            n=max(0,len(ep['obs'])-h+1)
            for begin in range(0,n,512):
                ids=np.arange(begin,min(begin+512,n));actions=ep['action'][ids[:,None]+np.arange(h)]
                out=predict_latent(model,z[ids],torch.as_tensor(actions,dtype=torch.float64),stats)
                prediction=out['normalized_observations'][:,-1].numpy()*stats['obs']['std']+stats['obs']['mean']
                acc[str(h)].add(prediction-ep['next_obs'][ids+h-1])
    return {h:a.result() for h,a in acc.items()}
