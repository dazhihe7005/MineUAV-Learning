"""Fixed R9 non-learning contract. No simulation imports at module load."""
import math
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .short_goal import short_manifest,verify_feasibility

IDENTITY='AUTORL_R9_SHORT_PRIOR_FEASIBILITY_ONLY'
PRIOR_SHA='6312e99c87fd5a4dca066bd1ef64acd17d5ee44b563c152c59c133ee659d7f74'
R8_REPORT_SHA='9771e19e56568a186c0f150729795411a5df756cf4cdbb3399cc5b221968e844'


def require_safe_coverage_terminal(terminal_class):
    if terminal_class not in ('success','collision','time_limit'):
        raise OpsError('R9 unsafe or unknown terminal; coverage stopped: '+str(terminal_class))


def short_prior_spec(root=ROOT):
    root=Path(root)
    prereg=read_json(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    return {'experiment_id':IDENTITY,'version':'SHORT_ANALYTIC_PRIOR_R9_V1',
        'initialization':'FRESH_SEED0_ANALYTIC_MEAN_STD03_NO_OPTIMIZER','policy_sha256':PRIOR_SHA,
        'short_manifest':short_manifest(prereg),'train_episodes':12,'validation_episodes':0,'test_episodes':0,
        'optimizer_updates':0,'training_timesteps':0,'max_policy_steps_per_episode':375,
        'wall_seconds':180,'success':{'distance_m_strictly_below':.1,'speed_m_s_strictly_below':.15,'consecutive_policy_steps':5},
        'r8_report_sha256':R8_REPORT_SHA,'claim':'NON_LEARNING_FEASIBILITY; HAND_WRITTEN_STATE_ONLY_NOT_LEARNED_AVOIDANCE'}


def initialize_short_prior(model):
    import torch
    from lidar_integration.ppo_training_runner import _weights_sha
    from .actor_prior import initialize_actor_prior
    from .std_profile import BASE_POLICY_SHA
    if (model._last_obs is not None or model.num_timesteps or model.policy.optimizer.state
        or model._n_updates or _weights_sha(model.policy)!=BASE_POLICY_SHA):
        raise ValueError('R9 requires untouched fresh seed0 weights; no checkpoint initialization')
    configuration=initialize_actor_prior(model)
    with torch.no_grad():model.policy.log_std.fill_(math.log(.3))
    model.policy.set_training_mode(False)
    actual=_weights_sha(model.policy)
    if actual!=PRIOR_SHA:raise ValueError('R9 analytic initialization identity mismatch')
    return {'policy_sha256':actual,'configuration':configuration,'optimizer_updates':0}


def validate_short_prior_result(result,manifest):
    rows=result.get('rows',[])
    if (result.get('status')!='PASS' or result.get('experiment_id')!=IDENTITY
        or len(rows)!=12 or len(manifest['train'])!=12
        or any(type(result.get(k)) is not int or result[k]!=0 for k in ('optimizer_updates','training_timesteps','validation_episodes','test_episodes'))):
        raise OpsError('R9 complete non-learning Train coverage missing')
    ticks=0
    for row,task in zip(rows,manifest['train']):
        trace=row.get('trace',[])
        if (row.get('task_identity_sha256')!=task['identity_sha256'] or row.get('base_identity_sha256')!=task['base_identity_sha256']
            or row.get('start_xyz')!=task['start_xyz'] or row.get('target_xyz')!=task['target_xyz']
            or row.get('terminal_class')!='success' or row.get('raw_termination_reason')!='success'
            or type(row.get('steps')) is not int or not 5<=row['steps']<=375 or len(trace)!=row['steps']):
            raise OpsError('R9 task identity or success trace incomplete')
        for i,v in enumerate(trace):
            if (any(type(v.get(k)) not in (int,float) or not math.isfinite(v[k]) for k in ('sim_time','distance_m','speed_m_s','lidar_age_s'))
                or abs(v['sim_time']-(i+1)*.04)>1e-8 or type(v.get('physics_ticks')) is not int or v['physics_ticks']!=20
                or v['distance_m']<0 or v['speed_m_s']<0 or not 0<=v['lidar_age_s']<=.15
                or any(not isinstance(v.get(k),list) or len(v[k])!=n or any(type(x) not in (int,float) or not math.isfinite(x) for x in v[k]) for k,n in [('position',3),('velocity',3),('action',4)])
                or any(abs(x)>1 for x in v['action'])
                or abs(math.dist(v['position'],task['target_xyz'])-v['distance_m'])>1e-7
                or abs(math.sqrt(sum(x*x for x in v['velocity']))-v['speed_m_s'])>1e-7):
                raise OpsError('R9 impossible/invalid observed physical trace')
            ticks+=v['physics_ticks']
        if any(v['distance_m']>=.1 or v['speed_m_s']>=.15 for v in trace[-5:]):
            raise OpsError('R9 original joint stop was not observed')
    if type(result.get('physics_ticks')) is not int or result['physics_ticks']!=ticks:
        raise OpsError('R9 total physics count differs')


def verify_short_prior_admission(reg,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root);a=read_json(root/'.research_local/autorl_r9_checks.json')
    if (a['registration_sha256']!=digest(reg) or a['source_identity_sha256']!=digest(reg['source_identity'])):
        raise OpsError('R9 stale admission')
    records={}
    for kind in ('operations','tensor','historical','runtime_binding'):
        ref=a['evidence'][kind];expected=f'.research_local/autorl_r9_{kind}_final_03.json'
        if ref['path']!=expected or sha256(root/expected)!=ref['sha256']:raise OpsError('R9 admission evidence changed')
        records[kind]=read_json(root/expected)
    op,te,hi,bi=(records[k] for k in ('operations','tensor','historical','runtime_binding'))
    if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=118
        or op['source_identity_sha256']!=a['source_identity_sha256'] or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
        or len(op['results'])!=118 or any(v['outcome']!='PASS' for v in op['results'])
        or te['returncode'] or te['optimizer_updates'] or te['tests_run']!=34 or 'Ran 34 tests' not in te['stderr']
        or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=digest(reg)
        or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
        or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results'])
        or bi['source_before_sha256']!=a['source_identity_sha256'] or bi['source_after_sha256']!=a['source_identity_sha256']
        or bi['registration_sha256']!=digest(reg) or bi['returncode'] or bi['final']['sha256']!=a['evidence']['historical']['sha256']):
        raise OpsError('R9 incomplete/current/unsafe no-update admission')
    if sha256(root/'mujoco/reports/autonomous_rl_repaired_distance_r8.json')!=R8_REPORT_SHA:
        raise OpsError('R8 released result differs')
    verify_feasibility(root)
    return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r9_checks.json')}
