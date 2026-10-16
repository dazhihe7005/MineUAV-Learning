"""Fixed read-only input panel and SHA-bound completed model references."""
import math
from pathlib import Path
from research_ops.storage import OpsError,read_json,sha256,digest
from .runtime import ROOT

SUMMARIES={
    'ARL_R1_UNIFORM_SEED0':'f12df386cf9a0cadd8ac979dbbb8799db9e0af4b2a0127af1c31d7d14b61c927',
    'ARL_R2_PRIOR_SEED0':'03fdb73a394880432eabbe19b7e28ce0325186e8d96d9346aae1dd319ec3d928',
    'ARL_R3_RANDOM_STD03_SEED0':'7651327c99f27940ce047d4b03b57b40f4ab59aff37632757ae70350445d36d1',
    'ARL_R3_PRIOR_STD03_SEED0':'afbeac0d092aa4f38c318f4b75909006db65f2f8eaa4435783ae6b7a887ada4e',
}


def probe_cases():
    cases=[]
    for axis,name in enumerate(('x','y','z')):
        errors=(-2.,-.5,-.1,0.,.1,.5,2.) if axis<2 else (-.5,-.1,0.,.1,.5)
        velocities=(-1.,0.,1.) if axis<2 else (-.5,0.,.5)
        for error in errors:
            for velocity in velocities:
                state=[0.]*7;state[axis]=error;state[axis+3]=velocity
                cases.append({'case_id':f'{name}_error{error}_velocity{velocity}',
                    'axis':name,'state':state,'provenance':'OFFLINE_COUNTERFACTUAL_INPUT'})
    for error in (-.5,-.1,0.,.1,.5):
        cases.append({'case_id':f'yaw_error{error}','axis':'yaw','state':[0.]*6+[error],
            'provenance':'OFFLINE_COUNTERFACTUAL_INPUT'})
    return cases


def validate_case(c):
    if (not isinstance(c,dict) or set(c)!={'case_id','axis','state','provenance'}
        or c['axis'] not in ('x','y','z','yaw') or not isinstance(c['case_id'],str)
        or c['provenance']!='OFFLINE_COUNTERFACTUAL_INPUT' or not isinstance(c['state'],list)
        or len(c['state'])!=7 or any(type(v) not in (int,float) or not math.isfinite(v) for v in c['state'])
        or any(abs(v)>100 for v in c['state'][:6]) or abs(c['state'][6])>math.pi):
        raise OpsError('invalid/privileged counterfactual input')


def verify_file(path,expected):
    try:
        if sha256(path)!=expected:raise OpsError('fixed feedback evidence SHA mismatch')
    except OSError as e:raise OpsError('fixed feedback evidence missing') from e


def checkpoint_references(root=ROOT):
    root=Path(root);refs=[]
    try:
        for identity,expected in SUMMARIES.items():
            run=root/'.research_local/authorized_rl_v1/runs'/identity;path=run/'summary.json'
            verify_file(path,expected);s=read_json(path)
            if (read_json(run/'complete.json')['summary_sha256']!=expected or s['experiment_id']!=identity
                or s['actual_timesteps']!=65536 or s['test_episode_count']!=0):raise OpsError('incomplete fixed feedback experiment')
            wanted=[32768,65536] if identity=='ARL_R1_UNIFORM_SEED0' else [0,32768,65536]
            if [e['actual_step'] for e in s['events']]!=wanted:raise OpsError('feedback checkpoint coverage differs')
            for event in s['events']:
                step=event['actual_step'];record=event['checkpoint'];name=f'{identity}_STEP{step}'
                sidecar=run/(name+'.json');archive=run/(name+'.zip')
                actual=read_json(sidecar)
                if (actual!=record or record['run_id']!=name or record['training_timesteps']!=step
                    or record['execution_protocol_sha256']!=s['protocol_sha256']
                    or not event['optimizer_and_prediction_reload_verified']):raise OpsError('fixed feedback sidecar identity differs')
                payload={k:v for k,v in actual.items() if k!='sidecar_payload_sha256'}
                if digest(payload)!=actual['sidecar_payload_sha256']:raise OpsError('feedback sidecar integrity differs')
                verify_file(archive,record['checkpoint_sha256'])
                refs.append({'experiment_id':identity,'actual_step':step,'run_id':name,
                    'protocol_sha256':s['protocol_sha256'],'summary_sha256':expected,
                    'checkpoint_path':str(archive.relative_to(root)),'checkpoint_sha256':record['checkpoint_sha256'],
                    'sidecar_path':str(sidecar.relative_to(root)),'sidecar_sha256':sha256(sidecar),
                    'policy_sha256':event['weight_sha256'],'usage':'READ_ONLY_INFERENCE_NOT_RESUME'})
    except (OSError,KeyError,TypeError) as e:raise OpsError('missing/invalid frozen feedback reference') from e
    return refs


def verify_probe_checks(registration,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        a=read_json(root/'.research_local/autorl_r4_checks.json')
        if (a['registration_sha256']!=digest(registration)
            or a['source_identity_sha256']!=digest(registration['source_identity'])):
            raise OpsError('stale feedback admission')
        evidence={}
        for kind in ('operations','tensor','historical'):
            relative=f'.research_local/autorl_r4_{kind}_final_02.json';ref=a['evidence'][kind]
            if ref['path']!=relative:raise OpsError('unexpected feedback evidence path')
            verify_file(root/relative,ref['sha256']);evidence[kind]=read_json(root/relative)
        op,te,hi=(evidence[k] for k in ('operations','tensor','historical'))
        if (op['complete'] is not True or op['failures'] or op['errors'] or op['optimizer_updates']
            or op['source_identity_sha256']!=a['source_identity_sha256'] or op['tests_run']<91
            or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or len(op['results'])!=op['tests_run'] or any(r['outcome']!='PASS' for r in op['results'])
            or te['returncode'] or te['optimizer_updates'] or te['tests_run']!=16
            or 'Ran 16 tests' not in te['stderr'] or te['source_identity_sha256']!=a['source_identity_sha256']
            or te['registration_sha256']!=a['registration_sha256']
            or hi['complete'] is not True or hi['exit_code'] or hi['actual_optimizer_updates']
            or hi['forbidden_calls'] or len(hi['results'])!=885 or any(r['outcome']!='passed' for r in hi['results'])):
            raise OpsError('feedback tests not complete/current/no-update')
        return {'status':'PASS','admission_sha256':sha256(root/'.research_local/autorl_r4_checks.json')}
    except (OSError,KeyError,TypeError) as e:raise OpsError('missing/invalid no-update feedback admission') from e
