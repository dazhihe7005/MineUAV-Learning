"""Fixed PPO worker, reachable only with a live reservation and supervisor."""
from __future__ import annotations
import argparse
import ctypes
import os
import signal
import time
from pathlib import Path
from research_ops.storage import OpsError,read_json,atomic_json,sha256,digest,lock_held
from .budget import BudgetStore
from .runtime import ROOT,HOME,run_directory,validate_launch


def launch_context(identity):
    output=run_directory(HOME,identity)
    permit=read_json(output/'permit.json');validate_launch(permit,identity)
    # Linux parent-death signal: no orphaned learning if Codex/supervisor dies.
    if ctypes.CDLL(None,use_errno=True).prctl(1,signal.SIGTERM)!=0:raise OpsError('parent death protection unavailable')
    validate_launch(permit,identity)
    from research_ops_v2.manager import OperationsV2
    from research_ops.identity import source_identity
    manager=OperationsV2(ROOT)
    if not lock_held(manager.lock_path):raise OpsError('supervisor has no shared OPS lock')
    registration=manager._load()[1]['registration']
    protocol=read_json(output/'protocol.json')
    if (permit['protocol_sha256']!=digest(protocol) or protocol['registration_sha256']!=digest(registration)
        or source_identity(ROOT,registration['source_paths'])!=registration['source_identity']):
        raise OpsError('worker frozen identity mismatch')
    entry=next((v for v in BudgetStore(HOME).status()['attempts'] if v['experiment_id']==identity),None)
    if not entry or entry['status']!='RESERVED_OR_INTERRUPTED' or entry['protocol']!=protocol:
        raise OpsError('worker budget reservation missing')
    return output,permit,protocol,registration


def runtime_sources(prereg):
    from lidar_integration.ppo_training_orchestration import _runtime_paths
    paths=set(_runtime_paths())|{ROOT/'mujoco'/p for p in prereg['preflight_source_sha256']}
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}


def save_checkpoint_probe(model,scheduler,manager,*,run_id,seed):
    """Exclude only the passive train override; preserve actual optimizer state."""
    import copy
    had_override='train' in vars(model)
    override=vars(model).get('train')
    if had_override:del model.train
    try:record=manager.save(model,scheduler,run_id=run_id,seed=seed)
    finally:
        if had_override:model.train=override
    obs={k:v.copy() for k,v in model._last_obs.items()}
    action=model.predict(obs,deterministic=True)[0].copy()
    return record,(obs,action,copy.deepcopy(model.policy.optimizer.state_dict()))


def run(identity):
    output,permit,p,registration=launch_context(identity)
    from .lifetime import start_guard,stop_guard
    watcher=start_guard(permit['parent_identity'],time.monotonic()+p['wall_seconds'],ROOT)
    try:return train_worker(output,permit,p,registration,identity)
    finally:stop_guard(watcher)


def train_worker(output,permit,p,registration,identity):
    import json
    import numpy as np
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from lidar_integration.ppo_preflight import make_ppo
    from lidar_integration.ppo_formal_seed0_health import FormalHealthMetricsEnv,_append_jsonl
    from lidar_integration.ppo_training_orchestration import _optimizer_step_count,_finite_model_and_gradients
    from lidar_integration.ppo_training_runner import _weights_sha,_historical_model_hashes,_stats
    from lidar_integration.ppo_formal_checkpoint import CheckpointManager
    from lidar_integration.ppo_formal_validation import LidarPPOValidationEvaluator
    from lidar_integration.failure_evidence import FailureEvidenceRecorder
    from research_ops.identity import source_identity
    from .curriculum import make_scheduler
    torch.set_num_threads(1)
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    before_history=_historical_model_hashes()
    if before_history!=prereg['historical_checkpoint_sha256']:raise OpsError('historical checkpoint mismatch')
    frozen=runtime_sources(prereg);started=time.monotonic()
    upstream=ROOT/'.research_local/dependencies/MuJoCo-LiDAR'
    store=BudgetStore(HOME)

    def guard():
        validate_launch(permit,identity)
        if store.status()['stop_requested']:raise OpsError('stop requested')
        if time.monotonic()-started>=p['wall_seconds']:raise OpsError('worker absolute wall deadline')

    if p['schema'] in (8,10):
        from .repaired_distance_training import make_repaired_scheduler
        scheduler=make_repaired_scheduler(prereg,upstream,seed=p['seed'])
    elif p['schema']==6:
        from .short_goal_env import make_short_scheduler
        scheduler=make_short_scheduler(prereg,upstream,seed=p['seed'])
    else:scheduler=make_scheduler(prereg,upstream,p['arm'],p['seed'])
    failures=FailureEvidenceRecorder(output/'failures',run_id=identity,attempt_id=identity,origin='AUTHORIZED_RL_RUNTIME')

    class Measured(FormalHealthMetricsEnv):
        def reset(self,**kwargs):
            result=super().reset(**kwargs)
            self.minimum_goal_error=float(np.linalg.norm(result[0]['state'][:3]))
            self.reward_components={};self.boundary_count=0;self.near_goal_steps=0
            if p['schema']==10:
                from .prior_distance_training import GoalStopDiagnostics
                self.goal_stop_diagnostics=GoalStopDiagnostics()
            return result

        def step(self,action):
            guard()
            if self.policy_steps>=p['timesteps']:raise OpsError('environment timestep budget exhausted')
            obs,r,t,tr,info=super().step(action)
            self.minimum_goal_error=min(self.minimum_goal_error,float(info['distance_m']))
            for key,value in info['reward_breakdown'].items():
                self.reward_components[key]=self.reward_components.get(key,0.)+float(value)
            self.boundary_count+=int(np.any(np.abs(action)>=.999999))
            self.near_goal_steps+=int(info['distance_m']<.1)
            if p['schema']==10:self.goal_stop_diagnostics.observe(float(info['distance_m']),float(info['speed_m_s']))
            if t or tr:
                _append_jsonl(output/'episode_details.jsonl',{
                    **self.episode_diagnostics[-1],'minimum_goal_error_m':self.minimum_goal_error,
                    'near_goal_steps':self.near_goal_steps,'reward_components':self.reward_components,
                    'collision_contact_pairs':info.get('contact_pairs',[]),
                    'action_boundary_fraction':self.boundary_count/self._episode_steps,
                    **({'distance_stage':info['distance_stage'],'base_scenario_identity_sha256':info['base_scenario_identity_sha256'],
                        'task_target_xyz':info['task_target_xyz']} if p['schema'] in (6,8,10) else {}),
                    **({'goal_stop_diagnostics':self.goal_stop_diagnostics.record()} if p['schema']==10 else {})})
            if len(self.episode_diagnostics)<4 and self.policy_steps%5==0:
                plant=scheduler.active.unwrapped
                _append_jsonl(output/'bounded_traces.jsonl',{'step':self.policy_steps,'sim_time':float(plant.data.time),
                    'scenario_identity_sha256':scheduler.current_identity,'position':plant.data.qpos[:3].tolist(),
                    'velocity':plant.data.qvel[:3].tolist(),'action':np.asarray(action).tolist(),
                    'goal_error_m':float(info['distance_m']),'lidar_age_s':info['lidar_age_s']})
            return obs,r,t,tr,info

    measured=Measured(scheduler,failure_recorder=failures,diagnostics_dir=output)
    manager=CheckpointManager(output,prereg,execution_protocol_sha256=digest(p))
    model=None
    try:
        model=make_ppo(measured,n_steps=128,seed=p['seed'])
        if model.num_timesteps or model.policy.optimizer.state:raise OpsError('not a new random initialization')
        initialization={'kind':'FRESH_RANDOM'}
        events=[];checkpoint_probes={}
        if p['schema']==10:
            from .prior_distance_training import initialize_prior_distance
            initialization,event,probe=initialize_prior_distance(model,scheduler,manager,identity=identity,seed=p['seed'])
            events.append(event);checkpoint_probes[0]=probe
        elif p['schema']==8:
            from .repaired_distance_training import initialize_repaired_training
            initialization,event,probe=initialize_repaired_training(model,scheduler,manager,identity=identity,seed=p['seed'])
            events.append(event);checkpoint_probes[0]=probe
        elif p['schema']==6:
            from .distance_training import initialize_distance_training
            initialization,event,probe=initialize_distance_training(model,scheduler,manager,identity=identity,seed=p['seed'])
            events.append(event);checkpoint_probes[0]=probe
        elif p['schema']==3:
            from .std_training import initialize_std_training
            initialization,event,probe=initialize_std_training(model,scheduler,manager,
                identity=identity,seed=p['seed'],variant=p['variant'])
            events.append(event);checkpoint_probes[0]=probe
        elif p['schema']==2:
            from .prior_training import initialize_training_prior
            initialization,event,probe=initialize_training_prior(model,scheduler,manager,identity=identity,seed=p['seed'])
            events.append(event);checkpoint_probes[0]=probe
        atomic_json(output/'initialization.json',initialization,exclusive=True)
        initial={'actor':_weights_sha(model.policy.action_net),'critic':_weights_sha(model.policy.value_net),
                 'policy':_weights_sha(model.policy)}
        atomic_json(output/'initial_weights.json',initial,exclusive=True)
        updates=[];original_train=model.train

        def train():
            guard()
            if source_identity(ROOT,registration['source_paths'])!=registration['source_identity']:
                raise OpsError('registered code changed before optimizer')
            for rel,value in frozen.items():
                if sha256(ROOT/rel)!=value:raise OpsError('historical source changed before optimizer')
            begun=time.monotonic();original_train()
            if not _finite_model_and_gradients(model):raise OpsError('nonfinite model or gradients')
            values={k:float(v) for k,v in model.logger.name_to_value.items()
                    if k.startswith('train/') and np.isscalar(v)}
            if not all(np.isfinite(v) for v in values.values()):raise OpsError('nonfinite PPO diagnostics')
            row={'actual_step':int(model.num_timesteps),'ppo_n_updates':int(model._n_updates),
                'optimizer_steps':_optimizer_step_count(model),'update_wall_ms':(time.monotonic()-begun)*1000,
                'raw_action_clip_fraction':float(np.mean(np.abs(model.rollout_buffer.actions)>1)),**values}
            _append_jsonl(output/'ppo_updates.jsonl',row);updates.append(row['update_wall_ms'])
            if source_identity(ROOT,registration['source_paths'])!=registration['source_identity']:
                raise OpsError('registered code changed during training')
            for rel,value in frozen.items():
                if sha256(ROOT/rel)!=value:raise OpsError('frozen historical runtime changed')
        model.train=train

        class Guard(BaseCallback):
            def _on_step(self):
                guard()
                if self.num_timesteps>p['timesteps']:raise OpsError('PPO sampling exceeded authorized steps')
                return True

        from lidar_integration.ppo_formal_validation import _optimizer_states_equal
        for target in p['validation_steps']:
            if target==0 or target>p['timesteps']:continue
            model.learn(target-int(model.num_timesteps),reset_num_timesteps=False,callback=Guard())
            if model.num_timesteps!=target or model._n_updates!=2*(target//128):raise OpsError('rollout/update boundary inconsistent')
            record,probe=save_checkpoint_probe(model,scheduler,manager,run_id=f'{identity}_STEP{target}',seed=p['seed'])
            checkpoint_probes[target]=probe
            events.append({'actual_step':target,'checkpoint':record,'weight_sha256':_weights_sha(model.policy),
                           'scheduler':scheduler.scheduler_state()})
        if model.num_timesteps!=p['timesteps']:raise OpsError('registered final milestone did not execute exact budget')
        if measured.policy_steps!=model.num_timesteps or scheduler.completed_policy_steps!=model.num_timesteps:
            raise OpsError('environment/scheduler/model step counts disagree')
        if len(updates)!=p['timesteps']//128 or sum(measured.terminal_counts.values())!=len(measured.episode_diagnostics):
            raise OpsError('rollout/episode accounting mismatch')
        if len((output/'train_episodes.jsonl').read_text().splitlines())!=len(measured.episode_diagnostics):
            raise OpsError('episode raw record count differs')
        if _optimizer_step_count(model)!=4*(p['timesteps']//128):raise OpsError('optimizer accounting differs')
        final={'actor':_weights_sha(model.policy.action_net),'critic':_weights_sha(model.policy.value_net)}
        if final['actor']==initial['actor'] or final['critic']==initial['critic']:raise OpsError('policy/value parameters did not update')
        scheduler_state=scheduler.scheduler_state();measured.close()
        # Evaluate frozen checkpoints only after releasing Train's MuJoCo instance.
        # Validation does not influence this run's training RNG or model selection.
        for event in events:
            guard()
            restored=manager.load_verified(None,run_id=event['checkpoint']['run_id'])
            if _weights_sha(restored.policy)!=event['weight_sha256']:raise OpsError('checkpoint reloaded with different weights')
            obs_probe,action_probe,optimizer_probe=checkpoint_probes[event['actual_step']]
            if (restored.num_timesteps!=event['actual_step']
                or not _optimizer_states_equal(optimizer_probe,restored.policy.optimizer.state_dict())
                or not np.array_equal(action_probe,restored.predict(obs_probe,deterministic=True)[0])):
                raise OpsError('checkpoint optimizer/prediction identity mismatch')
            event['optimizer_and_prediction_reload_verified']=True
            if p['schema'] in (6,8,10):
                from .distance_training import ShortValidationEvaluator,reuse_original_step0
                if p['schema']==10 and event['actual_step']==0:
                    from .prior_distance_training import reuse_prior_original_step0
                    event['validation']=reuse_prior_original_step0(event['weight_sha256'])
                else:event['validation']=(reuse_original_step0(event['weight_sha256']) if event['actual_step']==0
                        else LidarPPOValidationEvaluator(prereg,upstream,failure_recorder=failures).evaluate(restored))
                event['short_validation']=ShortValidationEvaluator(prereg,upstream,failure_recorder=failures).evaluate(restored)
                if p['schema']==10:
                    from .prior_distance_training import annotate_prior_validation
                    event['short_validation']=annotate_prior_validation(event['short_validation'])
                if p['schema']==8 and event['actual_step']==0:
                    event['validation']['claim']='historical original-target evaluation, NOT a new R8 episode'
            else:event['validation']=LidarPPOValidationEvaluator(prereg,upstream,failure_recorder=failures).evaluate(restored)
            if p['schema'] in (2,3):
                event['validation']['initialization']=initialization['kind']
                if event['actual_step']==0:
                    event['validation']['model_status']=('ANALYTIC_HAND_WRITTEN_PRIOR_NOT_LEARNED'
                        if initialization['kind']=='ANALYTIC_STATE_MEAN_TANH_V1' else 'UNTRAINED_RANDOM_MEAN')
            event['validation_execution']='DEFERRED_UNTIL_TRAIN_ENV_CLOSED; frozen milestone weights'
            if p['schema'] in (6,8,10) and event['actual_step']==0:
                event['validation_execution']='ORIGINAL_STEP0_REUSED_FROM_R3; SHORT_VALIDATION_DEFERRED_UNTIL_TRAIN_CLOSED'
            atomic_json(output/f"validation_{event['actual_step']}.json",event,exclusive=True)
            del restored
        if _historical_model_hashes()!=before_history:raise OpsError('historical checkpoint changed')
        summary={'experiment_id':identity,'protocol_sha256':digest(p),'actual_timesteps':int(model.num_timesteps),
            'optimizer_step_count':_optimizer_step_count(model),'rollout_count':model.num_timesteps//128,
            'ppo_n_updates':int(model._n_updates),'finite_model_and_gradients':_finite_model_and_gradients(model),
            'training_physics_ticks':measured.physics_ticks,'terminal_counts':measured.terminal_counts,
            'episode_count':len(measured.episode_returns),'mean_return':float(np.mean(measured.episode_returns)),
            'initial_weights':initial,'initialization':initialization,'final_weights':final,'scheduler_state':scheduler_state,
            'events':events,'test_episode_count':0,'runtime_sources':frozen,
            'environment_step_ms':_stats(measured.step_ms),'scan_ms':_stats(measured.scan_ms),
            'encoding_ms':_stats(measured.encode_ms),'ppo_update_ms':_stats(updates),
            'rss_peak_mib':measured.rss_peak_mib,'min_available_mib':measured.min_available_mib,
            'reset_rss_mib':measured.reset_rss_mib,'elapsed_wall_seconds':time.monotonic()-started,
            'resume':'RESTART_REQUIRED','model_selection':'FINAL_CHECKPOINT'}
        if p['schema'] in (6,8,10):
            actual_validation=sum(len(e['short_validation']['episodes'])+
                (0 if e['actual_step']==0 else len(e['validation']['episodes'])) for e in events)
            if actual_validation!=30:raise OpsError('distance Validation evidence coverage incomplete')
            summary.update(short_manifest_sha256=p['short_manifest_sha256'],
                actual_new_validation_episodes=actual_validation,reused_original_step0_episodes=6,
                checkpoint_l6_manifest_scope='unchanged base geometry/start/yaw metadata; distance task manifest bound by execution protocol and scheduler')
        atomic_json(output/'summary.json',summary,exclusive=True)
        atomic_json(output/'complete.json',{'summary_sha256':sha256(output/'summary.json'),
                    'actual_timesteps':summary['actual_timesteps']},exclusive=True)
    finally:
        measured.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--experiment-id',required=True)
    args=parser.parse_args();run(args.experiment_id)


if __name__=='__main__':main()
