"""Replay the ORIGINAL policy, verifying every saved scientific signal.

Only adds read-only snapshot collection to the original planner/evaluation;
there is no modified or newly proposed closed-loop policy.
"""
import copy
import numpy as np
import torch
from decision_fidelity_snapshot import capture,fingerprint
from latent_mpc_support_evaluation import InstrumentedSupportMPC,array_hash
from latent_mpc_evaluation import evaluate_episode
from run_latent_mpc_action_support import load_arrays
from run_latent_random_shooting_mpc import planner_seed
from latent_dynamics_data import file_hash,json_hash
from later_ranking_core import stage_indices

class RecordingPlanner(InstrumentedSupportMPC):
    def __init__(self,env,ctx,condition,stages):
        self.env=env; self.stages=stages; self.decisions={}
        support=None if condition=='unconstrained' else (ctx['support']['lower'],ctx['support']['upper'])
        super().__init__(ctx['model'],ctx['stats'],env.action_space.low,env.action_space.high,0,support)

    @torch.inference_mode()
    def plan(self,z,previous):
        i=self.planning_count; before=copy.deepcopy(self.rng.bit_generator.state)
        result=super().plan(z,previous)
        for stage,index in self.stages.items():
            if i==index:
                snap=capture(self.env,previous)
                self.decisions[stage]=dict(stage=stage,decision_index=i,snapshot=snap,snapshot_sha256=fingerprint(snap),
                    z=z.numpy().copy(),observation=self.env._get_obs().copy(),previous_action=previous.copy(),plan=result,
                    candidate_sha256=array_hash(result['candidates']),epsilon_sha256=self.epsilon_hashes[-1],
                    planner_rng_before_sha256=json_hash(before),planner_rng_after_sha256=json_hash(self.rng.bit_generator.state),
                    pi_magnitude=float(np.linalg.norm(self.env.velocity_controller.integral_acceleration_world)))
        return result

def archived_episode(ctx,condition,split,index):
    result=ctx['previous_report']['results'][condition][split]
    cache=ctx.setdefault('_original_arrays',{})
    key=(condition,split)
    if key not in cache: cache[key]=load_arrays(result)
    arrays=cache[key]; lo,hi=arrays['offsets'][index:index+2]
    return result,result['episodes'][index],{k:arrays[k][lo:hi] for k in ('actions','costs','epsilon_hashes')}

def replay_episode(ctx,env,condition,split,index):
    result,old,archive=archived_episode(ctx,condition,split,index)
    stages=stage_indices(old['episode_steps']); planner=RecordingPlanner(env,ctx,condition,stages)
    seed,target=ctx['targets'][split][index]
    if old['env_seed']!=seed or old['target']!=target or old['target_index']!=index:
        raise AssertionError('target/reset identity changed')
    row,trace=evaluate_episode(env,'mpc',seed,target,planner,planner_seed(split,index))
    for name in ('actions','costs'): np.testing.assert_array_equal(trace[name],archive[name])
    np.testing.assert_array_equal(planner.epsilon_hashes,archive['epsilon_hashes'])
    if json_hash(planner.epsilon_hashes)!=old['epsilon_stream_sha256']: raise AssertionError('original noise stream changed')
    # Wall-clock timing is intentionally not a reproducible scientific signal.
    for key,value in row.items():
        if key not in ('planning_time','encoding_time','total_decision_time') and value!=old[key]:
            raise AssertionError(f'original episode aggregate changed: {key}')
    if array_hash(trace['observations'][0])!=old['first_observation_sha256'] or planner.first!= {k:old[k] for k in planner.first}:
        raise AssertionError('first decision signal changed')
    representative=False
    for artifact in result['representative_local_traces']:
        if artifact['episode_index']!=index: continue
        if file_hash(artifact['path'])!=artifact['sha256']: raise AssertionError('original representative trace changed')
        with np.load(artifact['path'],allow_pickle=False) as raw:
            for key in ('observations','previous_actions','latents','selected_indices','candidate_terminals','actions','costs'):
                np.testing.assert_array_equal(trace[key],raw[key])
        representative=True
    for d in planner.decisions.values():
        i=d['decision_index']; np.testing.assert_array_equal(d['plan']['action'],archive['actions'][i])
        if d['plan']['index']!=int(trace['selected_indices'][i]): raise AssertionError('selected candidate mismatch')
        np.testing.assert_array_equal(d['plan']['candidates'][d['plan']['index'],0],trace['actions'][i])
    return dict(decisions=planner.decisions,episode=row,
        verification=dict(all_actions_costs_noise_exact=True,representative_observation_latent_trace_exact=representative,
            action_sha256=array_hash(trace['actions']),cost_sha256=json_hash(trace['costs'].tolist()),
            epsilon_stream_sha256=json_hash(planner.epsilon_hashes),original_steps=old['episode_steps']))
