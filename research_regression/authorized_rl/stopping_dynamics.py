"""R18 fixed three-policy one-scene diagnostic, no training authority extension."""
from pathlib import Path
from research_ops.storage import read_json,digest,sha256,OpsError
from .runtime import ROOT
from .feedback_followup import eligible_budget,verify_released_raw

REPORTS={'autonomous_rl_centered_noise_r16.json':'60c231a2d42afed7ba16b890f7be6774bc62f13bfb7501f5bd9f219893f3f174',
         'autonomous_rl_empty_rehearsal_r17.json':'d87993588f64db612daab4ee3091f64437a090eeb2b82532906f770a740a12f6'}
SUMMARIES={'ARL_R16_CENTERED_STD01_SEED0':'bf3814e86f0402ed02748483453a8279079aabca3e4df0c93264a82fcc71668c',
           'ARL_R17_EMPTY_REHEARSAL_SEED0':'950a588ea593f5eb0ece276748624e590efb231472a5c5aacb67098946da2f72'}


def configuration():
    return {'experiment_id':'AUTORL_R18_STOPPING_DYNAMICS_ONLY','kind':'FROZEN_POLICY_NEW_DIAGNOSTIC_EPISODES_NOT_TRAINING',
        'model_count':3,'episodes':3,'max_policy_steps_per_model':375,'training_timesteps':0,'optimizer_updates':0,'test_episodes':0,
        'validation_identity':'fe85ac443a0dc1bd2c676b6455f28ef0a4d15ea046287b692cc7eb3f6eefe06f',
        'wall_seconds':180,'workers':1,'device':'cpu','success_contract':'distance<.10m speed<.15m/s joint5policy samples',
        'claim_limit':'three frozen policies one development Validation scene; new diagnostic traces, not historical raw or generalization',
        'historical_summary_reproduction_tolerance':1e-8}


def references(root=ROOT):
    root=Path(root)
    try:
        for name,value in REPORTS.items():
            path=root/'mujoco/reports'/name
            if sha256(path)!=value:raise OpsError('R18 released source report changed')
            report=read_json(path)
            if report['implementation_status']!='PASS' or report['test_episodes']!=0:raise OpsError('R18 historical stage incomplete')
            for k in ('review','result_review'):
                v=report[k]
                if v['status']!='GO' or sha256(root/v['evidence']['path'])!=v['evidence']['sha256']:raise OpsError('R18 independent history review differs')
            verify_released_raw(report,root)
        refs=[]
        for identity,step in (('ARL_R17_EMPTY_REHEARSAL_SEED0',32768),('ARL_R16_CENTERED_STD01_SEED0',65536),('ARL_R17_EMPTY_REHEARSAL_SEED0',65536)):
            directory=root/'.research_local/authorized_rl_v1/runs'/identity;s=read_json(directory/'summary.json')
            if sha256(directory/'summary.json')!=SUMMARIES[identity] or read_json(directory/'complete.json')['summary_sha256']!=SUMMARIES[identity]:
                raise OpsError('R18 fixed completed summary differs')
            e=next(v for v in s['events'] if v['actual_step']==step);record=e['checkpoint'];run_id=record['run_id']
            side=directory/(run_id+'.json');archive=directory/(run_id+'.zip')
            if (read_json(side)!=record or sha256(archive)!=record['checkpoint_sha256'] or not e['optimizer_and_prediction_reload_verified']
                or digest({k:v for k,v in record.items() if k!='sidecar_payload_sha256'})!=record['sidecar_payload_sha256']):
                raise OpsError('R18 checkpoint sidecar/archive invalid')
            expected=next(v for v in e['validation']['episodes'] if v['scenario_identity_sha256']==configuration()['validation_identity'])
            refs.append({'experiment_id':identity,'actual_step':step,'run_id':run_id,'summary_sha256':SUMMARIES[identity],
                'protocol_sha256':s['protocol_sha256'],'checkpoint_path':str(archive.relative_to(root)),
                'checkpoint_sha256':sha256(archive),'sidecar_path':str(side.relative_to(root)),'sidecar_sha256':sha256(side),
                'policy_sha256':e['weight_sha256'],'historical_same_scene_summary':expected,'usage':'READ_ONLY_NEW_DIAGNOSTIC_NOT_RESUME'})
        return refs
    except (OSError,KeyError,TypeError,ValueError,StopIteration) as e:raise OpsError('missing/corrupt R18 fixed history') from e


def make_protocol(registration,root=ROOT):
    from .budget import BudgetStore
    root=Path(root);b=BudgetStore(root/'.research_local/authorized_rl_v1').status();eligible_budget(b)
    return {**configuration(),'registration_sha256':digest(registration),'source_identity_sha256':digest(registration['source_identity']),
        'models':references(root),'budget_snapshot_sha256':digest(b)}


def verify_checks(registration,root=ROOT):
    from research_ops.identity import source_identity
    root=Path(root)
    try:
        path=root/'.research_local/autorl_r18_checks_01.json';a=read_json(path);source=digest(registration['source_identity'])
        if a['registration_sha256']!=digest(registration) or a['source_identity_sha256']!=source:raise OpsError('stale R18 admission')
        vv={}
        for k in ('operations','tensor','historical','runtime_binding'):
            rel=f'.research_local/autorl_r18_{k}_final_01.json';ref=a['evidence'][k]
            if ref['path']!=rel:raise OpsError('unexpected R18 evidence path')
            if sha256(root/rel)!=ref['sha256']:raise OpsError('R18 admission hash differs')
            vv[k]=read_json(root/rel)
        op,te,hi,bi=(vv[k] for k in ('operations','tensor','historical','runtime_binding'))
        if (not op['complete'] or op['failures'] or op['errors'] or op['optimizer_updates'] or op['tests_run']!=178
            or len(op['results'])!=178 or any(v['outcome']!='PASS' for v in op['results'])
            or op['source_identity_sha256']!=source or op['test_source_identity']!=source_identity(root,['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run']!=63 or te['optimizer_updates'] or 'Ran 63 tests' not in te['stderr']
            or te['source_identity_sha256']!=source or te['registration_sha256']!=digest(registration)
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results'])!=885 or any(v['outcome']!='passed' for v in hi['results']) or bi['returncode']
            or bi['source_before_sha256']!=source or bi['source_after_sha256']!=source
            or bi['registration_sha256']!=digest(registration) or bi['final']!=a['evidence']['historical']):
            raise OpsError('R18 current complete no-update checks missing')
        return {'status':'PASS','admission_sha256':sha256(path)}
    except (OSError,KeyError,TypeError,ValueError) as error:raise OpsError('missing/invalid R18 current tests') from error



