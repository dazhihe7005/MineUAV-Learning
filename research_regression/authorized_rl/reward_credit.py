"""Fixed R19 read-only reward arithmetic; no simulator or ML imports."""
import math
from pathlib import Path
from research_ops.storage import OpsError, read_json, sha256, digest
from .runtime import ROOT
from .feedback_followup import eligible_budget, verify_released_raw

PARTS=('progress','action','brake','success','collision','timeout_or_time_cost','physical_failure','sensor_failure')
CLASSES=('success','collision','time_limit','physical_failure','sensor_failure')
R18_REPORT='f071966d76110fde45dae74cbc1ba8e421e258bc802222fe3055b588ce39e294'


def configuration():
    return {'experiment_id':'AUTORL_R19_REWARD_CREDIT_ONLY','kind':'READ_ONLY_RECORDED_REWARD_ARITHMETIC',
        'gammas':[.99,.995,.999,1.],'actual_training_gamma':.99,'actual_gae_lambda':.95,'policy_hz':25,
        'training_timesteps':0,'optimizer_updates':0,'physics_steps':0,'new_episodes':0,'test_episodes':0,
        'model_inference_calls':0,'wall_seconds':120,'workers':1,'device':'cpu',
        'claim_limit':'three recorded development traces and two completed Train histories; no new policy performance or saved advantages'}


def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('nonfinite or nonnumeric reward')
    return float(value)


def components(parts,reward):
    if not isinstance(parts,dict) or set(parts)!=set(PARTS)|{'total'}:raise ValueError('reward fields differ')
    values={k:finite(parts[k]) for k in PARTS}
    if not math.isclose(math.fsum(values.values()),finite(reward),rel_tol=1e-10,abs_tol=1e-8):
        raise ValueError('reward/component sum mismatch')
    if not math.isclose(finite(parts['total']),finite(reward),rel_tol=1e-10,abs_tol=1e-8):
        raise ValueError('recorded total differs')
    return values


def discounted_components(rows,gamma):
    if isinstance(gamma,bool) or gamma not in configuration()['gammas']:raise ValueError('unregistered gamma')
    if not rows:raise ValueError('no recorded rewards')
    values=[components(row['reward_breakdown'],row['reward']) for row in rows]
    result={k:math.fsum((gamma**i)*p[k] for i,p in enumerate(values)) for k in PARTS}
    return {'gamma':gamma,'steps':len(rows),'return':math.fsum(result.values()),'components':result,
        'terminal_discount_factor':gamma**(len(rows)-1),
        'interpretation':'ACTUAL_TRAINING_GAMMA' if gamma==.99 else 'OFFLINE_COUNTERFACTUAL_ARITHMETIC_NOT_TRAINING'}


def episode_exposure(rows,base_layout_map,*,tail_steps,tail_base,expected_policy_steps):
    if (isinstance(tail_steps,bool) or not isinstance(tail_steps,int) or tail_steps<0
        or isinstance(expected_policy_steps,bool) or not isinstance(expected_policy_steps,int) or expected_policy_steps<=0
        or tail_base not in base_layout_map):raise ValueError('invalid unfinished tail identity/steps')
    by={name:{'ended_episodes':0,'closed_policy_steps':0,'policy_steps_with_tail':0,
        'terminal_counts':dict.fromkeys(CLASSES,0),'reward_components':dict.fromkeys(PARTS,0.)}
        for name in sorted(set(base_layout_map.values()))}
    for row in rows:
        base=row['base_scenario_identity_sha256'];steps=row['episode_policy_steps'];outcome=row['terminal_class']
        if (row['distance_stage']!='ORIGINAL' or base not in base_layout_map or outcome not in CLASSES
            or isinstance(steps,bool) or not isinstance(steps,int) or steps<=0):raise ValueError('invalid ended original Train episode')
        p=components(row['reward_components'],row['episode_return']);v=by[base_layout_map[base]]
        v['ended_episodes']+=1;v['closed_policy_steps']+=steps;v['policy_steps_with_tail']+=steps;v['terminal_counts'][outcome]+=1
        for k in PARTS:v['reward_components'][k]+=p[k]
    by[base_layout_map[tail_base]]['policy_steps_with_tail']+=tail_steps
    if sum(v['policy_steps_with_tail'] for v in by.values())!=expected_policy_steps:raise ValueError('ended plus tail steps mismatch')
    for v in by.values():
        v['ended_episode_fraction']=v['ended_episodes']/len(rows) if rows else 0.
        v['policy_step_fraction']=v['policy_steps_with_tail']/expected_policy_steps
    return {'by_layout':by,'ended_episodes':len(rows),'closed_policy_steps':sum(v['closed_policy_steps'] for v in by.values()),
        'unfinished_tail_steps':tail_steps,'unfinished_tail_base':tail_base,'policy_steps_with_tail':expected_policy_steps,
        'tail_reward_and_terminal':'NOT_RECORDED_NOT_IMPUTED','interpretation':'EXPOSURE_NOT_GRADIENT_OR_ADVANTAGE'}


def reset_exposure(selected_tasks,base_layout_map,*,closed_episodes,tail_base):
    if any(not isinstance(v,str) for v in selected_tasks):raise ValueError('scheduler records must be task identity strings')
    original=[v for v in selected_tasks if v in base_layout_map]
    if len(original)!=closed_episodes+1 or not original or original[-1]!=tail_base:
        raise ValueError('reset count/tail differs from completed episodes')
    counts={name:0 for name in sorted(set(base_layout_map.values()))}
    for identity in original:counts[base_layout_map[identity]]+=1
    return {'original_reset_counts':counts,'original_resets':len(original),
        'reset_fractions':{k:v/len(original) for k,v in counts.items()}}


def references(root=ROOT):
    from .stopping_dynamics import references as old_references, SUMMARIES
    root=Path(root)
    try:
        models=old_references(root);path=root/'mujoco/reports/autonomous_rl_stopping_dynamics_r18.json'
        if sha256(path)!=R18_REPORT:raise OpsError('R19 fixed released R18 report changed')
        report=read_json(path)
        for v in (report['review'],report['result_review'],report['publication_addendum']['independent_disposition']):
            if v['status']!='GO' or v['critical'] or v['important'] or sha256(root/v['evidence']['path'])!=v['evidence']['sha256']:
                raise OpsError('R19 R18 actual independent review differs')
        verify_released_raw(report,root)
        directory=root/'.research_local/autorl_r18_stopping_dynamics/run_01'
        if sha256(directory/'result.json')!='bd56b076e5f2241ce94a6843c6760709eaa70dc81ba64ee74f8edc4b8703384b':
            raise OpsError('R19 R18 direct result identity differs')
        r18=read_json(directory/'result.json')
        if read_json(directory/'complete.json')['result_sha256']!=sha256(directory/'result.json'):raise OpsError('R18 incomplete')
        if [v['reference'] for v in r18['models']]!=models:raise OpsError('R19 trace/checkpoint pairing differs')
        manifest=root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json'
        if digest(read_json(manifest)['scenario_manifest'])!='f6ee34a153e6cb4d95fa43e7214d6d67c9b534835362be275363e14ec87b75d5':
            raise OpsError('original scenario manifest changed')
        files=[path,directory/'result.json',directory/'complete.json',manifest]
        files.extend(root/v['trace_file']['path'] for v in r18['models'])
        for identity in SUMMARIES:
            d=root/'.research_local/authorized_rl_v1/runs'/identity
            files.extend(d/n for n in ('summary.json','episode_details.jsonl','complete.json'))
        return {'models':models,'files':{str(p.relative_to(root)):sha256(p) for p in files},
            'publication_warning':'R18 nonfunctional whitespace/process Minor preserved; independent disposition GO'}
    except (OSError,KeyError,TypeError,ValueError) as e:raise OpsError('missing/invalid R19 fixed history') from e


def make_protocol(registration,root=ROOT):
    from .budget import BudgetStore
    root=Path(root);budget=BudgetStore(root/'.research_local/authorized_rl_v1').status();eligible_budget(budget)
    return {**configuration(),'registration_sha256':digest(registration),'source_identity_sha256':digest(registration['source_identity']),
        'references':references(root),'budget_snapshot_sha256':digest(budget)}


def verify_checks(registration,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r19_checks_01.json';a=read_json(path);source=digest(registration['source_identity'])
        if a['registration_sha256']!=digest(registration) or a['source_identity_sha256']!=source:raise OpsError('stale R19 admission')
        vv={}
        for k in ('operations','tensor','historical','runtime_binding'):
            rel=f'.research_local/autorl_r19_{k}_final_01.json';ref=a['evidence'][k]
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('R19 evidence path/hash differs')
            vv[k]=read_json(root/rel)
        op,te,hi,bi=(vv[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=187
            or len(op['results'])!=187 or any(v['outcome']!='PASS' for v in op['results'])
            or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run']!=63 or te['optimizer_updates'] or 'Ran 63 tests' not in te['stderr']
            or te['source_identity_sha256']!=source or te['registration_sha256']!=digest(registration)
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
            or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
            or bi['registration_sha256']!=digest(registration) or bi['final']!=a['evidence']['historical']):raise OpsError('R19 complete tests missing')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as e:raise OpsError('missing/invalid R19 current checks') from e
