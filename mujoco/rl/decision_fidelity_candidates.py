"""Reproduce existing first-decision candidates and every saved decision signal."""
import numpy as np
import torch
from latent_mpc_action_support import ActionSupportMPC
from latent_mpc_support_evaluation import array_hash
from run_latent_random_shooting_mpc import planner_seed
from decision_fidelity_snapshot import capture,fingerprint


@torch.inference_mode()
def reconstruct(ctx,env,split,index):
    seed,target=ctx['targets'][split][index]
    observation,_=env.reset(seed=seed,options={'target_position':target})
    previous=np.zeros(4,np.float32); snapshot=capture(env,previous)
    conditions={}; evidence=dict(candidate_sha256={},raw_epsilon_sha256=[],previous_first_decision_exact=True)
    latent=None
    for condition in ('unconstrained','train_supported'):
        support=None if condition=='unconstrained' else (ctx['support']['lower'],ctx['support']['upper'])
        planner=ActionSupportMPC(ctx['model'],ctx['stats'],env.action_space.low,env.action_space.high,planner_seed(split,index),support)
        z=planner.encode_current(observation,previous); out=planner.plan(z,previous)
        old=ctx['previous_report']['results'][condition][split]['episodes'][index]
        if old['env_seed']!=seed or old['target']!=target or old['target_index']!=index:
            raise AssertionError('original initial target/reset identity changed')
        if array_hash(observation)!=old['first_observation_sha256'] or array_hash(z.numpy())!=old['first_latent_sha256']:
            raise AssertionError('original initial observation/latent changed')
        noise=planner.candidate_audit['epsilon_sha256']
        if noise!=old['first_epsilon_sha256'] or out['index']!=old['first_selected_candidate_index']:
            raise AssertionError('original first candidate noise/selection changed')
        np.testing.assert_array_equal(out['action'],old['first_action'])
        np.testing.assert_array_equal(out['terminal_observation'],old['first_predicted_terminal_observation'])
        for key,oldkey in [('selected_cost','first_selected_cost'),('candidate_cost_median','first_candidate_median_cost'),('candidate_cost_mean','first_candidate_mean_cost')]:
            if out[key]!=old[oldkey]: raise AssertionError(f'original first decision {key} changed')
        evidence['candidate_sha256'][condition]=array_hash(out['candidates'])
        evidence['raw_epsilon_sha256'].append(noise)
        conditions[condition]=out
        if latent is not None: np.testing.assert_array_equal(latent,z.numpy())
        latent=z.numpy().copy()
    if evidence['raw_epsilon_sha256'][0]!=evidence['raw_epsilon_sha256'][1]: raise AssertionError('paired noise changed')
    if fingerprint(capture(env,previous))!=fingerprint(snapshot): raise AssertionError('planning mutated simulator/controller')
    return dict(snapshot=snapshot,z=latent,previous_action=previous,observation=observation,
        conditions=conditions,evidence=evidence,split=split,index=index,env_seed=seed,target=target,
        snapshot_sha256=fingerprint(snapshot))
