"""AUDIT-01 read-only aggregation. No simulator, policy, tensor or optimizer import."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import statistics
import time

from research_ops.identity import file_record,safe_path
from research_ops.storage import (FileLock,OpsError,atomic_json,digest,read_json,sha256,now)

CLASSES=('success','collision','time_limit','physical_failure','sensor_failure')
UNKNOWN='NOT OBSERVABLE'
REPORTS='mujoco/reports/'
L9=REPORTS+'mujoco_lidar_ppo_formal_seed0_parts/run_01'
L9S=REPORTS+'mujoco_lidar_ppo_repaired_integration_parts/run_01'
L8=REPORTS+'mujoco_lidar_ppo_orchestration_parts/run_01'


def finite(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('finite number required')
    return value


def read_jsonl(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    def reject(value):raise ValueError('nonfinite JSON constant')
    rows=[]
    with Path(path).open() as stream:
        for line in stream:
            if len(line)>65536:raise ValueError('oversized evidence row')
            row=json.loads(line,object_pairs_hook=pairs,parse_constant=reject,parse_float=lambda x:finite(float(x)))
            if not isinstance(row,dict):raise ValueError('evidence row must be object')
            rows.append(row)
            if len(rows)>10000:raise ValueError('bounded audit row limit')
    return rows


def verify_seal(directory,expected):
    if not expected:raise ValueError('missing raw evidence seal')
    for name,wanted in expected.items():
        if Path(name).name!=name or len(wanted)!=64:raise ValueError('invalid seal path/hash')
        if sha256(safe_path(directory,name))!=wanted:raise ValueError('sealed evidence changed: '+name)


def stats(values):
    values=[finite(v) for v in values]
    if not values:return {'count':0,'mean':None,'min':None,'median':None,'max':None}
    return {'count':len(values),'mean':statistics.mean(values),'min':min(values),
            'median':statistics.median(values),'max':max(values)}


def group(rows):
    return {'episodes':len(rows),'counts':{k:sum(r['outcome']==k for r in rows) for k in CLASSES},
            **{k:stats(r[k] for r in rows) for k in ('episode_return','episode_policy_steps',
              'final_target_distance_m','mean_action_norm','trajectory_length_m','net_displacement_m')},
            'motionless_episodes':sum(r['motionless_episode'] for r in rows)}


def summarize_episodes(rows,train):
    previous=0
    for r in rows:
        identity=r['scenario_identity_sha256']
        if identity not in train or train[identity]['split']!='train':raise ValueError('non-Train episode identity')
        if r['outcome'] not in CLASSES or r.get('terminal_class',r['outcome'])!=r['outcome']:
            raise ValueError('unknown or inconsistent terminal class')
        n=r['episode_policy_steps'];end=r['policy_end_step']
        if type(n) is not int or type(end) is not int or not 0<n<=375 or end<=previous or end<n:
            raise ValueError('invalid completed episode step sequence')
        previous=end
        for key in ('episode_return','final_target_distance_m','mean_action_norm','trajectory_length_m',
                    'net_displacement_m','sim_end_time_s'):finite(r[key])
        if (not 0<=r['mean_action_norm']<=2.000001 or r['final_target_distance_m']<0
                or r['trajectory_length_m']+1e-8<r['net_displacement_m']
                or not 0<=r['net_displacement_m'] or type(r['motionless_episode']) is not bool):
            raise ValueError('invalid motion diagnostics')
        if r['outcome']=='time_limit' and (n!=375 or abs(r['sim_end_time_s']-15)>1e-6):
            raise ValueError('timeout not at frozen 15s/375step limit')
    layouts=sorted({v['layout'] for v in train.values()})
    out=group(rows)
    out.update(by_layout={l:group([r for r in rows if train[r['scenario_identity_sha256']]['layout']==l]) for l in layouts},
        by_identity={k:group([r for r in rows if r['scenario_identity_sha256']==k]) for k in train},
        final_distance_threshold_counts={f'<{d:g}m':sum(r['final_target_distance_m']<d for r in rows) for d in (.1,.25,.5,1.)},
        closest_approach=UNKNOWN,collision_contact_location=UNKNOWN,
        episode_weighting='completed episodes, not independent training seeds')
    out['step_blocks']=[]
    for low,high in ((0,30000),(30000,60000),(60000,90000),(90000,120000)):
        q=[r for r in rows if low<r['policy_end_step']<=high]
        out['step_blocks'].append({'lower_exclusive':low,'upper_inclusive':high,**group(q),
            'by_layout':{l:group([r for r in q if train[r['scenario_identity_sha256']]['layout']==l]) for l in layouts}})
    return out


def reward_bounds(row,scene):
    n=row['episode_policy_steps'];m=row['mean_action_norm']
    initial=math.dist(scene['start_xyz'],scene['target_xyz'])
    # Jensen: N*mean(norm)^2 <= sum(norm^2) <= 2*N*mean(norm), norm<=2.
    bounds=[-.005*2*n*m,-.005*n*m*m]
    cost=-.01*n-(2. if row['outcome']=='time_limit' else 0.)
    result={'action_penalty_bounds':bounds,'time_and_timeout_sum':cost,
            'initial_distance_m':initial,'endpoint_distance_reduction_m':initial-row['final_target_distance_m'],
            'progress_sum':UNKNOWN,'brake_sum':UNKNOWN,'source':'MATHEMATICAL_BOUNDS_FROM_EPISODE_AGGREGATES'}
    if row['outcome']=='time_limit':
        progress=10*(initial-row['final_target_distance_m'])
        result['progress_sum']=progress
        result['brake_penalty_bounds']=[row['episode_return']-progress-cost-bounds[1],
                                        row['episode_return']-progress-cost-bounds[0]]
    # Failure-transition progress is explicitly suppressed in source. Endpoint
    # displacement cannot reconstruct its omitted progress or braking history.
    return result


def summarize_ppo_snapshot(metrics,step):
    if not metrics:raise ValueError('no recorded PPO diagnostic')
    for value in metrics.values():finite(value)
    return {'actual_step':step,'snapshot_count':1,'metrics':metrics,'temporal_trend':UNKNOWN,
            'raw_actor_action_series':UNKNOWN,'raw_advantages_or_value_targets':UNKNOWN,
            'entropy_note':'negative entropy_loss is unsquashed Gaussian entropy, not clipped action diversity'}


def summarize_causality(rows,train):
    ages=[];previous_step=-1;steps=0;resets=0;ticks=0
    for row in rows:
        if row['scenario_identity_sha256'] not in train:raise ValueError('non-Train causal row')
        sim=finite(row['simulation_time']);sample=finite(row['lidar_sample_time']);age=finite(row['lidar_age_s'])
        step=row['policy_step'];physics=row['physics_steps']
        if (sample>sim+1e-8 or age< -1e-8 or age>.15+1e-8 or abs(sim-sample-age)>1e-8
                or row['lidar_status']!='FRESH' or step<previous_step
                or type(physics) is not int or not 0<=physics<=20):raise ValueError('invalid causal observation')
        previous_step=step;ages.append(age);ticks+=physics
        if physics:steps+=1
        else:resets+=1
    return {'rows':len(rows),'step_rows':steps,'zero_tick_rows':resets,'physics_ticks':ticks,
            'max_age_s':max(ages) if ages else None,'mean_age_s':statistics.mean(ages) if ages else None,
            'future_samples':0,'raw_pose_or_action':UNKNOWN}


def collect(root):
    """Source hashes are checked by OPS; all historical raw seals checked here."""
    tracked={}
    def load(relative):
        path=safe_path(root,relative);tracked[relative]=file_record(path);return read_json(path)
    prereg=load(REPORTS+'mujoco_lidar_ppo_training_prereg_seed0.json')
    manifest=prereg['scenario_manifest']
    if digest(manifest)!=prereg['scenario_manifest_sha256']:raise ValueError('scenario manifest identity changed')
    train={r['identity_sha256']:r for r in manifest['train']}
    if len(train)!=12 or len(manifest['validation'])!=6:raise ValueError('wrong frozen split counts')
    l9=load(REPORTS+'mujoco_lidar_ppo_formal_seed0.json')
    if l9['overall_result']!='FAIL' or l9['formal_training_complete']:raise ValueError('original L9 not preserved as FAIL')
    verify_seal(root/L9,l9['raw_evidence_sha256'])
    seal=load(L9S+'/pilot.complete.json');verify_seal(root/L9S,seal['evidence_sha256'])
    l8_report=load(REPORTS+'mujoco_lidar_ppo_orchestration.json')
    l8_seal=load(L8+'/pilot.complete.json')
    verify_seal(root/L8,{'pilot_summary.json':l8_seal['summary_sha256'],
        'L8_SEED0_STEP0008192_event.json':l8_report['raw_evidence']['milestone_event_sha256'],
        'L8_SEED0_STEP0008192.zip':l8_report['raw_evidence']['checkpoint_sha256']})
    for folder,hashes in ((L9,l9['raw_evidence_sha256']),(L9S,seal['evidence_sha256'])):
        for name in hashes:tracked[folder+'/'+name]=file_record(root/folder/name)
    l8=load(L8+'/pilot_summary.json');l9s=load(L9S+'/pilot_summary.json')
    repair=load(REPORTS+'mujoco_lidar_ppo_termination_repair.json')
    repair_identity=repair['frozen_repair_identity']
    if (digest(repair_identity['protocol'])!=repair_identity['protocol_sha256']
            or repair_identity['protocol']['formal_retrain_authorized'] is not False):raise ValueError('L9R denial or identity changed')
    for name,expected in repair_identity['protocol']['repair_source_sha256'].items():
        relative='mujoco/lidar_integration/'+name
        if sha256(root/relative)!=expected:raise ValueError('repaired runtime source changed')
        tracked[relative]=file_record(root/relative)
    if (sha256(root/'mujoco/models/mine_uav_dynamics_v2.xml')!=prereg['historical_model_sha256']
            or sha256(root/'mujoco/lidar_integration/navigation_env.py')!=prereg['reward_implementation_sha256']
            or sha256(root/'mujoco/lidar_integration/ppo_preflight.py')!=prereg['network_implementation_sha256']):
        raise ValueError('model/reward/CNN identity changed')
    for name,expected in prereg['historical_checkpoint_sha256'].items():
        relative='mujoco/rl/models/'+name
        if sha256(root/relative)!=expected:raise ValueError('historical checkpoint changed')
        tracked[relative]=file_record(root/relative)
    result={'L8':{'steps':l8['training_actual_timesteps'],'counts':l8['terminal_counts'],
        'mean_return':l8['mean_completed_episode_return'],'episode_detail':UNKNOWN},'L9R_authorization':False}
    raw_rows={}
    for tag,folder in (('L9',L9),('L9S',L9S)):
        rows=read_jsonl(root/folder/'train_episodes.jsonl');raw_rows[tag]=rows
        if sum(r['episode_policy_steps'] for r in rows)!=rows[-1]['policy_end_step']:
            raise ValueError('completed episode accounting gap')
        result[tag]=summarize_episodes(rows,train)
        rb=[reward_bounds(r,train[r['scenario_identity_sha256']]) for r in rows]
        timeout=[r for r in rb if 'brake_penalty_bounds' in r]
        result[tag]['reward_reconstruction']={'per_step_components':UNKNOWN,
          'endpoint_distance_reduction_m':stats(r['endpoint_distance_reduction_m'] for r in rb),
          'timeout_progress_sum':stats(r['progress_sum'] for r in timeout),
          'timeout_action_sum_lower_bound':stats(r['action_penalty_bounds'][0] for r in timeout),
          'timeout_action_sum_upper_bound':stats(r['action_penalty_bounds'][1] for r in timeout),
          'timeout_brake_sum_lower_bound':stats(r['brake_penalty_bounds'][0] for r in timeout),
          'timeout_brake_sum_upper_bound':stats(r['brake_penalty_bounds'][1] for r in timeout)}
    if result['L9']['counts']!={'success':0,'collision':656,'time_limit':50,'physical_failure':0,'sensor_failure':0}:
        raise ValueError('L9 differs from original persisted counts')
    if result['L9S']['counts']!=l9s['accounting']['terminal_counts']:raise ValueError('L9S count mismatch')
    result['L9']['last_resource_step']=read_jsonl(root/L9/'resource_samples.jsonl')[-1]['policy_step']
    result['L9']['last_completed_episode_step']=raw_rows['L9'][-1]['policy_end_step']
    result['L9']['exact_failure_step']='NOT RECORDED'
    result['L9']['failing_transition_physical_cause']='NOT RECORDED'
    result['L9S']['causality']=summarize_causality(read_jsonl(root/L9S/'causal_observations.jsonl'),train)
    if result['L9S']['causality']['step_rows']!=4096:raise ValueError('incomplete L9S causal evidence')
    event=load(L9+'/L8_SEED0_STEP0100096_event.json')
    health=load(L9+'/L9_HEALTH_STEP0100096.json')
    validations={'L8':l8['events'][0]['validation'],'L9':event['validation'],'L9S':l9s['validation']}
    result['validation']={}
    expected={r['identity_sha256'] for r in manifest['validation']}
    for tag,validation in validations.items():
        episodes=validation['episodes']
        if len(episodes)!=6 or {e['scenario_identity_sha256'] for e in episodes}!=expected:raise ValueError('invalid Validation split')
        if validation.get('optimizer_unchanged') is not True:raise ValueError('unverified isolated Validation')
        result['validation'][tag]={'episodes':episodes,'overall':validation['overall'],
            'positive_x_boundary_contact_episodes':sum(any('nav_wall_x_pos' in pair for pair in e['collision_contact_pairs']) for e in episodes),
            'obstacle_contact_episodes':sum(any(any('nav_obstacle_' in v for v in pair) for pair in e['collision_contact_pairs']) for e in episodes)}
    result['ppo_snapshots']={'L9':summarize_ppo_snapshot(health['ppo_diagnostics'],100096),
                             'L9S':summarize_ppo_snapshot(l9s['health']['ppo_diagnostics'],4096)}
    result['snapshot_comparison']='different fresh seed0 attempts, NOT a within-run time series or independent seeds'
    result['source_formula']={'reward_version':'NAV_V2_COLLISION_PREFLIGHT_V1','gamma':.99,'gae_lambda':.95,
      'rollout_seconds':128/25,'gamma_375':.99**375,'gamma_lambda_trace_seconds':1/(1-.99*.95)/25,
      'stationary_far_goal_timeout_return':-.01*375-2,
      'stationary_note':'MATHEMATICAL_COUNTERFACTUAL: fixed pose, zero velocity/action, no contact; not a new rollout'}
    l5=load(REPORTS+'mujoco_lidar_navigation_env.json')
    result['historical_L5_scripted_reference']=[{k:e[k] for k in ('layout','scene_seed','steps','success','collision','return','final_distance_m','reward_component_sums')} for e in l5['episodes']]
    result['L5_comparability']='different small preflight scene manifest and starts/Yaw; not matched expert proof for L9'
    result['task_distribution']={'train_layout_counts':{l:sum(s['layout']==l for s in train.values()) for l in sorted({s['layout'] for s in train.values()})},
      'initial_distance_m':stats(math.dist(s['start_xyz'],s['target_xyz']) for s in train.values()),
      'all_train_targets_z_m':sorted({s['target_xyz'][2] for s in train.values()}),
      'scenario_manifest_sha256':prereg['scenario_manifest_sha256']}
    result['source_evidence']={k:tracked[k] for k in sorted(tracked)}
    result['interpretation_limits']={'train_contact_location':UNKNOWN,'positions_or_actions_per_step':UNKNOWN,
      'reached_obstacle_side':UNKNOWN,'ever_entered_goal_neighborhood':UNKNOWN,
      'rollout_advantages_and_returns':UNKNOWN,'reward_components_per_training_step':UNKNOWN,
      'source_data_test_episode_count':0,'audit_new_test_episodes':0,'independent_training_seeds':1}
    return result,raw_rows,tracked


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots',action='store_true');args=parser.parse_args(argv)
    from research_ops_v2.manager import OperationsV2
    root=Path(__file__).resolve().parents[1];manager=OperationsV2(root)
    out=safe_path(root,str(args.output))
    if not out.is_relative_to(root/'.research_local/audit_01'):raise OpsError('unique local AUDIT-01 output required')
    with FileLock(manager.lock_path):
        manager.verify();_,protocol,state,_=manager._load()
        if state['phase_id']!='AUDIT-01' or state['training_budget']!=0:raise OpsError('AUDIT-01 registration with zero budget required')
        out.mkdir(parents=True,exist_ok=False)
        atomic_json(out/'audit.attempt.json',{'phase_id':'AUDIT-01','protocol_sha256':state['protocol_sha256'],
                    'registration_sha256':digest(protocol['registration']),'optimizer_updates':0,'time':now()},exclusive=True)
        started=time.monotonic()
        try:
            result,rows,identities=collect(root)
            if args.plots:
                from .navigation_diagnosis_plots import plot_all
                result['figures']=plot_all(root,result,rows)
            if any(file_record(root/p)!=record for p,record in identities.items()):raise ValueError('historical evidence changed during audit')
            manager.verify()
            result['audit_resource']={'wall_s':time.monotonic()-started,
                'peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,'optimizer_updates':0}
            atomic_json(out/'analysis.json',result,exclusive=True)
            atomic_json(out/'audit.complete.json',{'analysis_sha256':sha256(out/'analysis.json'),
                'historical_files_unchanged':True,'optimizer_updates':0},exclusive=True)
            print(json.dumps({'status':'PASS','output':str(out),'L9_counts':result['L9']['counts']},ensure_ascii=False))
        except BaseException as error:
            atomic_json(out/'audit.failure.json',{'type':type(error).__name__,'message':str(error)[:1000]},exclusive=True)
            raise


if __name__=='__main__':main()
