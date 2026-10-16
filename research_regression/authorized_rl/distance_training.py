"""Narrow hooks for unchanged PPO training and distance-only Validation."""
import copy
from research_ops.storage import read_json,sha256,digest
from .runtime import ROOT
from .distance_profile import CONTROL_SHA,INITIAL_POLICY_SHA
from .short_goal import short_manifest
from lidar_integration.ppo_formal_validation import LidarPPOValidationEvaluator


def initialize_distance_training(model,scheduler,manager,*,identity,seed):
    from .std_training import initialize_std_training
    initialization,event,probe=initialize_std_training(model,scheduler,manager,
        identity=identity,seed=seed,variant='RANDOM')
    if initialization['assigned_policy_sha256']!=INITIAL_POLICY_SHA:
        raise ValueError('distance initialization differs from frozen historical control')
    initialization['execution_profile']='DISTANCE_CURRICULUM_R6_V1'
    return initialization,event,probe


class ShortValidationEvaluator(LidarPPOValidationEvaluator):
    def __init__(self,prereg,upstream_root,*,failure_recorder=None):
        super().__init__(prereg,upstream_root,failure_recorder=failure_recorder)
        self.manifest_sha256=digest(short_manifest(prereg))
        self.rows=tuple(copy.deepcopy(short_manifest(prereg)['validation']))

    def evaluate(self,model,*,max_policy_steps=375):
        result=super().evaluate(model,max_policy_steps=max_policy_steps)
        result.update(task_distribution='DISTANCE_ONLY_0_20M_VALIDATION',source='SIM_GROUND_TRUTH_STATE',
            short_manifest_sha256=self.manifest_sha256)
        return result


def reuse_original_step0(policy_sha256):
    path=ROOT/'.research_local/authorized_rl_v1/runs/ARL_R3_RANDOM_STD03_SEED0/summary.json'
    if policy_sha256!=INITIAL_POLICY_SHA or sha256(path)!=CONTROL_SHA:
        raise ValueError('original step0 reuse requires identical frozen policy/control')
    summary=read_json(path);event=summary['events'][0]
    if event['actual_step']!=0 or event['weight_sha256']!=policy_sha256:
        raise ValueError('missing original frozen step0 evidence')
    result=copy.deepcopy(event['validation'])
    result.update(reused_historical_evidence=True,newly_executed_episodes=0,
        source_summary_sha256=CONTROL_SHA,claim='historical original-target evaluation, NOT a new R6 episode')
    return result
