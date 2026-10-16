"""R3 fixed initial-std profiles, never a new grant or arbitrary experiment API."""
from pathlib import Path
import re
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .prior_profile import CONTROL_SUMMARY_SHA,PRIOR_POLICY_SHA,prior_experiment,verify_prior_proof

R2_SUMMARY_SHA='03fdb73a394880432eabbe19b7e28ce0325186e8d96d9346aae1dd319ec3d928'
BASE_POLICY_SHA='bc1adbd197b52b4bfd0bb759a749c16e351092960557f2d341ba3321876b39ba'


def std_experiment(variant):
    if variant not in ('RANDOM','PRIOR'):raise OpsError('unsupported R3 initialization variant')
    p=prior_experiment()
    p.update(schema=3,experiment_id=f'ARL_R3_{variant}_STD03_SEED0',variant=variant,
        initial_std=.3,std_remains_trainable=True,execution_profile='INITIAL_STD_R3_V1',
        initialization='FRESH_RANDOM' if variant=='RANDOM' else 'ANALYTIC_STATE_MEAN_TANH_V1',
        control_summary_sha256=CONTROL_SUMMARY_SHA if variant=='RANDOM' else R2_SUMMARY_SHA)
    p.pop('expected_prior_policy_sha256')
    p['evaluation']={'selection':'FINAL_CHECKPOINT',
        'initial_std_effect':'final successes > same-mean historical std1 final successes and collisions <= control',
        'learning_over_step0':'final successes > own step0 successes and collisions <= own step0 collisions',
        'candidate_success_min':4,'candidate_collision_max':1,'candidate_physical_sensor_failures_max':0,
        'claim_limit':'single seed historical control; no learned LiDAR dependence or generalization proof'}
    return p


def validate_std_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(std_experiment(p.get('variant'))):raise OpsError('unsupported fixed initial-std profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):
            raise OpsError('invalid std source/registration identity')


def verify_std_controls(p,root=ROOT):
    validate_std_profile(p);root=Path(root)
    # Reuse proof of unchanged physics/network and old Scripted feasibility.
    verify_prior_proof(prior_experiment(),root)
    base=root/'.research_local/authorized_rl_v1/runs'
    control=read_json(base/'ARL_R1_UNIFORM_SEED0/summary.json')
    path=base/'ARL_R2_PRIOR_SEED0/summary.json';prior=read_json(path)
    if (sha256(path)!=R2_SUMMARY_SHA or read_json(path.parent/'complete.json')['summary_sha256']!=R2_SUMMARY_SHA
        or prior['actual_timesteps']!=65536 or prior['test_episode_count']!=0
        or prior['runtime_sources']!=control['runtime_sources']
        or prior['initial_weights']['policy']!=PRIOR_POLICY_SHA
        or control['initial_weights']['policy']!=BASE_POLICY_SHA):
        raise OpsError('R3 frozen controls missing or changed')
    return {'status':'PASS','R1_summary_sha256':CONTROL_SUMMARY_SHA,'R2_summary_sha256':R2_SUMMARY_SHA}


def verify_std_checks(p,root=ROOT):
    """Current actual no-update admission; not itself a budget grant."""
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        admission=read_json(root/'.research_local/autorl_r3_checks.json')
        if (admission['source_identity_sha256']!=p['source_identity_sha256']
            or admission['registration_sha256']!=p['registration_sha256']):raise OpsError('stale R3 admission')
        values={}
        for name in ('operations','tensor','historical'):
            relative=f'.research_local/autorl_r3_{name}_final.json';record=admission['evidence'][name]
            if record['path']!=relative or sha256(root/relative)!=record['sha256']:
                raise OpsError('R3 pre-run evidence hash/path differs')
            values[name]=read_json(root/relative)
        ops,tensor,history=(values[k] for k in ('operations','tensor','historical'))
        if (ops['complete'] is not True or ops['failures'] or ops['errors'] or ops['optimizer_updates']
            or ops['source_identity_sha256']!=p['source_identity_sha256'] or ops['tests_run']<84
            or len(ops['results'])!=ops['tests_run'] or any(r['outcome']!='PASS' for r in ops['results'])
            or ops['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or tensor['returncode'] or tensor['optimizer_updates'] or tensor['tests_run']!=13
            or tensor['source_identity_sha256']!=p['source_identity_sha256']
            or tensor['registration_sha256']!=p['registration_sha256'] or 'Ran 13 tests' not in tensor['stderr']
            or history['complete'] is not True or history['exit_code'] or history['actual_optimizer_updates']
            or history['forbidden_calls'] or len(history['results'])!=885
            or any(r['outcome']!='passed' for r in history['results'])):
            raise OpsError('R3 tests incomplete/failed/noncurrent')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r3_checks.json')}
    except (OSError,KeyError,TypeError) as error:raise OpsError('missing or invalid R3 admission') from error
