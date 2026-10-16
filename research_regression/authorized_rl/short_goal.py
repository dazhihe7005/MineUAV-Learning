"""Distance-only tasks derived from frozen L6 metadata, no simulation imports."""
import math
import copy
from pathlib import Path
from research_ops.storage import OpsError,digest,read_json,sha256
from .runtime import ROOT

BASE_MANIFEST_SHA='f6ee34a153e6cb4d95fa43e7214d6d67c9b534835362be275363e14ec87b75d5'


def short_target(start,target):
    if (not isinstance(start,(list,tuple)) or not isinstance(target,(list,tuple))
        or len(start)!=3 or len(target)!=3
        or any(type(v) not in (int,float) or not math.isfinite(v) for v in [*start,*target])
        or start[2]!=target[2] or math.dist(start,target)<=.20):raise OpsError('invalid original distance task')
    distance=math.dist(start,target)
    return [float(a+.20*(b-a)/distance) for a,b in zip(start,target)]


def short_manifest(prereg):
    original=prereg['scenario_manifest']
    if digest(original)!=BASE_MANIFEST_SHA:raise OpsError('L6 base manifest changed')
    result={}
    for split in ('train','validation'):
        result[split]=[]
        for base in original[split]:
            row=copy.deepcopy(base);row.pop('identity_sha256')
            row.update(base_identity_sha256=base['identity_sha256'],task_variant='DISTANCE_ONLY_0_20M_SIM_TASK',
                target_xyz=short_target(base['start_xyz'],base['target_xyz']))
            row['identity_sha256']=digest(row);result[split].append(row)
    return result


def verify_short_manifest(manifest,prereg):
    if manifest!=short_manifest(prereg):raise OpsError('frozen short task manifest differs')


def stage_for_steps(steps):
    if type(steps) is not int or steps<0:raise OpsError('invalid completed policy count')
    return 'SHORT' if steps<32768 else 'ORIGINAL'


def validate_feasibility(result,manifest):
    rows=result.get('rows',[])
    if (result.get('status')!='PASS' or result.get('experiment_id')!='AUTORL_R5_SHORT_FEASIBILITY_ONLY'
        or len(rows)!=12 or [r.get('task_identity_sha256') for r in rows]!=[r['identity_sha256'] for r in manifest['train']]
        or any(type(result.get(k)) is not int or result[k]!=0 for k in ('optimizer_updates','training_timesteps','test_episodes','validation_episodes'))):
        raise OpsError('missing/wrong physical feasibility coverage')
    total_ticks=0
    for r,task in zip(rows,manifest['train']):
        trace=r.get('trace',[])
        if (r.get('terminal_class')!='success' or r.get('raw_termination_reason')!='success'
            or r.get('base_identity_sha256')!=task['base_identity_sha256']
            or r.get('start_xyz')!=task['start_xyz'] or r.get('target_xyz')!=task['target_xyz']
            or type(r.get('steps')) is not int
            or not 5<=r['steps']<=375 or len(trace)!=r['steps']
            or any(not math.isfinite(v[k]) for v in trace for k in ('distance_m','speed_m_s','sim_time'))
            or any(not 0<=v['distance_m']<.1 or not 0<=v['speed_m_s']<.15 for v in trace[-5:])
            or any(abs(v['sim_time']-(i+1)*.04)>1e-8 for i,v in enumerate(trace))):
            raise OpsError('short task did not satisfy original successful stop')
        for v in trace:
            if (type(v.get('physics_ticks')) is not int or v['physics_ticks']!=20
                or any(not isinstance(v.get(k),list) or len(v[k])!=3
                       or any(type(x) not in (int,float) or not math.isfinite(x) for x in v[k]) for k in ('position','velocity'))
                or abs(math.dist(v['position'],task['target_xyz'])-v['distance_m'])>1e-7
                or abs(math.sqrt(sum(x*x for x in v['velocity']))-v['speed_m_s'])>1e-7):
                raise OpsError('short task physical trace contradicts success summary')
            total_ticks+=v['physics_ticks']
    if type(result.get('physics_ticks')) is not int or result['physics_ticks']!=total_ticks:
        raise OpsError('short task total physical ticks differ')


def verify_feasibility(root=ROOT):
    root=Path(root)
    try:
        run=root/'.research_local/autorl_r5_short_goal/run_01';result=read_json(run/'result.json')
        if read_json(run/'complete.json')['result_sha256']!=sha256(run/'result.json'):raise OpsError('short task proof SHA differs')
        prereg=read_json(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        registration=read_json(root/'mujoco/reports/autonomous_rl_short_goal_r5_registration.json')
        protocol=read_json(root/'mujoco/reports/autonomous_rl_short_goal_r5_prereg.json')
        if (result.get('registration_sha256')!=digest(registration)
            or result.get('source_identity_sha256')!=digest(registration['source_identity'])
            or result.get('protocol_sha256')!=digest(protocol)
            or protocol['registration_sha256']!=digest(registration)
            or protocol['source_identity_sha256']!=result['source_identity_sha256']):
            raise OpsError('short task proof runtime/protocol identity differs')
        manifest=short_manifest(prereg)
        if result['short_manifest_sha256']!=digest(manifest):raise OpsError('short task proof manifest differs')
        validate_feasibility(result,manifest)
        return {'status':'PASS','result_sha256':sha256(run/'result.json')}
    except (OSError,KeyError,TypeError,ValueError) as e:raise OpsError('missing/invalid short task feasibility') from e


def verify_short_checks(registration,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r5_checks.json')
        if (a['registration_sha256']!=digest(registration)
            or a['source_identity_sha256']!=digest(registration['source_identity'])):raise OpsError('stale R5 admission')
        values={}
        for kind in ('operations','tensor','historical'):
            relative=f'.research_local/autorl_r5_{kind}_final_02.json';ref=a['evidence'][kind]
            if ref['path']!=relative or sha256(root/relative)!=ref['sha256']:raise OpsError('R5 test evidence differs')
            values[kind]=read_json(root/relative)
        op,te,hi=(values[k] for k in ('operations','tensor','historical'))
        if (op['complete'] is not True or op['failures'] or op['errors'] or op['optimizer_updates']
            or op['source_identity_sha256']!=a['source_identity_sha256'] or op['tests_run']<99
            or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or len(op['results'])!=op['tests_run'] or any(r['outcome']!='PASS' for r in op['results'])
            or te['returncode'] or te['optimizer_updates'] or te['tests_run']!=19 or 'Ran 19 tests' not in te['stderr']
            or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=a['registration_sha256']
            or hi['complete'] is not True or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(r['outcome']!='passed' for r in hi['results'])):raise OpsError('incomplete/current/unsafe R5 checks')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r5_checks.json')}
    except (OSError,KeyError,TypeError) as e:raise OpsError('missing/invalid R5 admission') from e
