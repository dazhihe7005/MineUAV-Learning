"""One distance-only curriculum profile using the existing user grant."""
from pathlib import Path
import re
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT
from .short_goal import verify_feasibility,short_manifest

CONTROL_SHA='7651327c99f27940ce047d4b03b57b40f4ab59aff37632757ae70350445d36d1'
FEASIBILITY_SHA='395fe3891efc2b7917d087a939922d945bd9d3c1aace4e632efa95b366fd0a94'
FEASIBILITY_REPORT_SHA='f97b5266ecec75e9e8067ad9611026a3fd7d8e0db1bcb6743062f81919809ee7'
SHORT_MANIFEST_SHA='f93b6dce91abeadbde8ebaf619a3c5839cbb9312cca2402f2868472345d5faf3'
INITIAL_POLICY_SHA='db33ba4df5f7aaf76f98b7915675cbfc80dbe52273045ed86811b5cc074dfa06'


def distance_experiment():
    from .budget import experiment
    p=experiment('UNIFORM',65536,3600,revision='R6')
    p.update(schema=6,experiment_id='ARL_R6_DISTANCE_SEED0',execution_profile='DISTANCE_CURRICULUM_R6_V1',
        initialization='FRESH_RANDOM',initial_std=.3,std_remains_trainable=True,
        curriculum_rule='twelve short0.20m tasks until32768; original targets only on next natural reset',
        validation_steps=[0,32768,65536],short_manifest_sha256=SHORT_MANIFEST_SHA,
        feasibility_result_sha256=FEASIBILITY_SHA,feasibility_report_sha256=FEASIBILITY_REPORT_SHA,
        control_summary_sha256=CONTROL_SHA,expected_initial_policy_sha256=INITIAL_POLICY_SHA,
        evaluation={'selection':'FINAL_CHECKPOINT','short_final_success_min':4,'short_final_collision_max':1,
            'short_final_physical_sensor_failures_max':0,'original_final_success_min':1,'original_final_collision_max':6,
            'learning':'short final successes > own step0 short successes',
            'original_step0':'reuse fixed R3 RANDOM step0 identical policy evidence, not a new episode',
            'actual_new_validation_episodes':30,'claim_limit':'single seed small Validation; no generalized obstacle avoidance claim'})
    return p


def validate_distance_profile(p):
    clean={k:v for k,v in p.items() if k not in ('source_identity_sha256','registration_sha256')}
    if digest(clean)!=digest(distance_experiment()):raise OpsError('unsupported distance curriculum profile')
    for k in ('source_identity_sha256','registration_sha256'):
        if k in p and (not isinstance(p[k],str) or not re.fullmatch('[0-9a-f]{64}',p[k])):raise OpsError('invalid distance source identity')


def verify_distance_proof(p,root=ROOT):
    validate_distance_profile(p);root=Path(root)
    try:
        report_path=root/'mujoco/reports/autonomous_rl_short_goal_r5.json'
        if sha256(report_path)!=FEASIBILITY_REPORT_SHA:raise OpsError('released R5 report differs')
        report=read_json(report_path)
        if report['review']['status']!='GO' or report['result_review']['status']!='GO':raise OpsError('R5 review not complete')
        if verify_feasibility(root)['result_sha256']!=FEASIBILITY_SHA:raise OpsError('released R5 proof differs')
        reg=read_json(root/'mujoco/reports/autonomous_rl_short_goal_r5_registration.json')
        for rel in ('research_regression/authorized_rl/short_goal.py','research_regression/authorized_rl/short_goal_env.py'):
            if sha256(root/rel)!=reg['source_identity'][rel]['sha256']:raise OpsError('tested short task source changed')
        for ref in report['raw_files']:
            if sha256(root/ref['path'])!=ref['sha256'] or (root/ref['path']).stat().st_size!=ref['bytes']:raise OpsError('R5 raw proof changed')
        control_path=root/'.research_local/authorized_rl_v1/runs/ARL_R3_RANDOM_STD03_SEED0/summary.json'
        if sha256(control_path)!=CONTROL_SHA or read_json(control_path.parent/'complete.json')['summary_sha256']!=CONTROL_SHA:
            raise OpsError('R3 historical control changed')
        control=read_json(control_path)
        if (control['initial_weights']['policy']!=INITIAL_POLICY_SHA or control['actual_timesteps']!=65536
            or control['test_episode_count']!=0):raise OpsError('R3 initial policy or budget differs')
        for rel,expected in control['runtime_sources'].items():
            if sha256(root/rel)!=expected:raise OpsError('frozen historical physics/network/Reward changed')
        prereg=read_json(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
        if digest(short_manifest(prereg))!=SHORT_MANIFEST_SHA:raise OpsError('short task manifest changed')
        return {'status':'PASS','R5_result_sha256':FEASIBILITY_SHA,'control_sha256':CONTROL_SHA}
    except (OSError,KeyError,TypeError,ValueError) as e:raise OpsError('missing/invalid released distance proof/control') from e


def verify_distance_checks(p,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r6_checks.json')
        if a['registration_sha256']!=p['registration_sha256'] or a['source_identity_sha256']!=p['source_identity_sha256']:
            raise OpsError('stale distance admission')
        values={}
        for kind in ('operations','tensor','historical'):
            rel=f'.research_local/autorl_r6_{kind}_final.json';ref=a['evidence'][kind]
            if ref['path']!=rel or sha256(root/rel)!=ref['sha256']:raise OpsError('distance test evidence differs')
            values[kind]=read_json(root/rel)
        op,te,hi=(values[k] for k in ('operations','tensor','historical'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']<105
            or op['source_identity_sha256']!=a['source_identity_sha256'] or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or len(op['results'])!=op['tests_run'] or any(r['outcome']!='PASS' for r in op['results'])
            or te['returncode'] or te['optimizer_updates'] or te['tests_run']!=21 or 'Ran 21 tests' not in te['stderr']
            or te['source_identity_sha256']!=a['source_identity_sha256'] or te['registration_sha256']!=a['registration_sha256']
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(r['outcome']!='passed' for r in hi['results'])):raise OpsError('distance checks incomplete/unsafe')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r6_checks.json')}
    except (OSError,KeyError,TypeError) as e:raise OpsError('missing/invalid distance checks') from e
