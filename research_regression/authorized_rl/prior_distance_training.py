"""R10 narrow initialization/reuse/diagnostic hooks. No independent PPO algorithm."""
import copy
import math
from research_ops.storage import read_json,sha256
from .runtime import ROOT
from .prior_distance_profile import PRIOR_POLICY_SHA,R3_PRIOR_SHA


def initialize_prior_distance(model,scheduler,manager,*,identity,seed):
    from .std_training import initialize_std_training
    initialization,event,probe=initialize_std_training(model,scheduler,manager,identity=identity,seed=seed,variant='PRIOR')
    if initialization['assigned_policy_sha256']!=PRIOR_POLICY_SHA:raise ValueError('R10 prior initialization differs')
    initialization['execution_profile']='PRIOR_DISTANCE_R10_V1'
    return initialization,event,probe


def reuse_prior_original_step0(policy_sha256):
    path=ROOT/'.research_local/authorized_rl_v1/runs/ARL_R3_PRIOR_STD03_SEED0/summary.json'
    if policy_sha256!=PRIOR_POLICY_SHA or sha256(path)!=R3_PRIOR_SHA:
        raise ValueError('R10 original step0 requires frozen identical PRIOR policy, never RANDOM')
    summary=read_json(path);event=summary['events'][0]
    if event['actual_step']!=0 or event['weight_sha256']!=policy_sha256:
        raise ValueError('R3 PRIOR initial evidence absent')
    result=copy.deepcopy(event['validation'])
    result.update(reused_historical_evidence=True,newly_executed_episodes=0,source_summary_sha256=R3_PRIOR_SHA,
        claim='historical R3 PRIOR original-target step0, NOT a new R10 episode')
    return result


def annotate_prior_validation(result):
    """Correct generic evaluator metadata without mutating historical records."""
    actual=copy.deepcopy(result);step=actual['training_timesteps_before_after']
    if type(step) is not int or step<0:raise ValueError('invalid prior evaluation timestep')
    if step==0:
        if actual['policy_sha256_before_after']!=PRIOR_POLICY_SHA:raise ValueError('initial prior label requires exact full policy')
        actual['model_status']='ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED'
    actual['initialization']='ANALYTIC_STATE_MEAN_TANH_V1'
    return actual


class GoalStopDiagnostics:
    """Observed current samples only; never modifies environment/task outcome."""
    def __init__(self):self.near=0;self.fast=0;self.joint=0;self.contiguous=0;self.maximum=0

    def observe(self,distance,speed):
        if not math.isfinite(distance) or not math.isfinite(speed) or distance<0 or speed<0:
            raise ValueError('nonfinite/negative goal diagnostic sample')
        near=distance<.1;joint=near and speed<.15
        self.near+=int(near);self.fast+=int(near and not joint);self.joint+=int(joint)
        self.contiguous=self.contiguous+1 if joint else 0;self.maximum=max(self.maximum,self.contiguous)

    def record(self):
        return {'near_goal_samples':self.near,'near_goal_fast_samples':self.fast,'joint_samples':self.joint,
                'max_consecutive_joint_samples':self.maximum,'last_consecutive_joint_samples':self.contiguous}
