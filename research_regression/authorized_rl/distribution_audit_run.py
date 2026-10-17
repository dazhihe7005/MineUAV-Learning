"""R23 unique frozen-data audit; no model, optimizer, simulator or training entry."""
import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity
from research_ops.environment import enforce_resources
from research_ops.identity import source_identity
from research_regression.navigation_diagnosis import read_jsonl
from .runtime import ROOT,HOME,verify_current
from .budget import BudgetStore
from .closed_failure_proof import eligible_after_closed_r20
from .feedback_spec import verify_file
from .feedback_followup import verify_released_raw
from .distribution_audit import heading_coverage,original_train_groups,validation_groups,counts

OUT=ROOT/'.research_local/autorl_r23_distribution_audit/run_01'
FROZEN={
    'ARL_R17_EMPTY_REHEARSAL_SEED0':('mujoco/reports/autonomous_rl_empty_rehearsal_r17.json',
        'd87993588f64db612daab4ee3091f64437a090eeb2b82532906f770a740a12f6',
        '950a588ea593f5eb0ece276748624e590efb231472a5c5aacb67098946da2f72'),
    'ARL_R22_GAE_CREDIT_SEED0':('mujoco/reports/autonomous_rl_gae_credit_r22.json',
        'ea1f49e4df91af36b158f318022137e8fa76e0d39ec14ba10b76a219509ba8b5',
        'b0c45293b8713e20a7861fa835b6063d00146572e93860ca9f05637e8a8f8c14')}


def make_protocol(reg,root=ROOT):
    root=Path(root);budget=BudgetStore(root/'.research_local/authorized_rl_v1').status()
    eligible_after_closed_r20(budget,root);refs=[]
    for identity,(report_name,report_sha,summary_sha) in FROZEN.items():
        report=root/report_name;verify_file(report,report_sha);released=read_json(report)
        if released['implementation_status']!='PASS' or released['test_episodes']!=0:
            raise OpsError('R23 source report not completed/isolation incorrect')
        for key in ('review','result_review'):
            v=released[key]
            if v['status']!='GO':raise OpsError('R23 source actual review not GO')
            verify_file(root/v['evidence']['path'],v['evidence']['sha256'])
        verify_released_raw(released,root)
        run=root/'.research_local/authorized_rl_v1/runs'/identity;s=read_json(run/'summary.json')
        verify_file(run/'summary.json',summary_sha)
        if (s['experiment_id']!=identity or s['actual_timesteps']!=65536 or s['test_episode_count']
                or read_json(run/'complete.json')['summary_sha256']!=summary_sha):raise OpsError('R23 incomplete source run')
        refs.append({'experiment_id':identity,'report_path':report_name,'report_sha256':report_sha,
            'summary_sha256':summary_sha,'run_path':str(run.relative_to(root)),
            'episode_details_sha256':sha256(run/'episode_details.jsonl'),'train_episodes_sha256':sha256(run/'train_episodes.jsonl')})
    return {'experiment_id':'AUTORL_R23_SPLIT_DISTRIBUTION_ONLY','kind':'READ_ONLY_ARITHMETIC_NOT_TRAINING',
        'registration_sha256':digest(reg),'source_identity_sha256':digest(reg['source_identity']),
        'models_loaded':0,'training_timesteps':0,'optimizer_updates':0,'physics_steps':0,'new_scans':0,
        'episodes':0,'validation_episodes':0,'test_episodes':0,'wall_seconds':120,'workers':1,'device':'cpu',
        'references':refs,'budget_snapshot_sha256':digest(budget),
        'manifest_file_sha256':sha256(root/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json'),
        'hypothesis':'initial heading/task distribution mismatch is a candidate,not a proven cause',
        'claim_limit':'nonstationary stochastic Train aggregates are not final deterministic Validation generalization rates'}


def verify_checks(reg):
    evidence=read_json(ROOT/'.research_local/autorl_r23_checks_02.json');source=digest(reg['source_identity'])
    if evidence['registration_sha256']!=digest(reg) or evidence['source_identity_sha256']!=source:
        raise OpsError('stale R23 no-update checks')
    for kind,ref in evidence['evidence'].items():
        verify_file(ROOT/ref['path'],ref['sha256']);v=read_json(ROOT/ref['path'])
        if kind=='operations':
            valid=(v['complete'] and v['tests_run']==217 and not v['failures'] and not v['errors'] and not v['optimizer_updates']
                and v['source_identity_sha256']==source and v['test_source_identity']==source_identity(ROOT,['tests/research_ops_v2'])
                and len(v['results'])==217 and all(a['outcome']=='PASS' for a in v['results']))
        elif kind=='historical':valid=(v['complete'] and v['exit_code']==0 and not v['actual_optimizer_updates'] and not v['forbidden_calls']
                and len(v['results'])==885 and all(a['outcome']=='passed' for a in v['results']))
        elif kind=='tensor':valid=(v['returncode']==0 and v['tests_run']==70 and not v['optimizer_updates']
                and v['source_identity_sha256']==source and v['registration_sha256']==digest(reg) and 'Ran 70 tests' in v['stderr'])
        elif kind=='runtime_binding':valid=(v['returncode']==0 and v['source_before_sha256']==source==v['source_after_sha256']
                and v['registration_sha256']==digest(reg) and v['final']==evidence['evidence']['historical'])
        else:raise OpsError('unexpected R23 evidence kind')
        if not valid:raise OpsError('incomplete R23 no-update tests')
    if set(evidence['evidence'])!={'operations','historical','tensor','runtime_binding'}:raise OpsError('missing R23 test category')


def calculate(spec,guard):
    manifest=read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')['scenario_manifest']
    train,validation=manifest['train'],manifest['validation']
    if len(train)!=12 or len(validation)!=6:raise OpsError('fixed split sizes differ')
    coverage=heading_coverage(train,validation);runs=[]
    for ref in spec['references']:
        guard();run=ROOT/ref['run_path'];s=read_json(run/'summary.json')
        rows=read_jsonl(run/'episode_details.jsonl');raw=read_jsonl(run/'train_episodes.jsonl')
        if len(rows)!=len(raw) or len(rows)!=s['episode_count'] or any(any(v[k]!=value for k,value in a.items()) for a,v in zip(raw,rows)):
            raise OpsError('raw/detail completed episode records differ')
        if counts(rows)!={k:s['terminal_counts'].get(k,0) for k in ('success','collision','time_limit','physical_failure','sensor_failure')}:
            raise OpsError('source total terminal counts differ')
        original=[v for v in rows if v['distance_stage']=='ORIGINAL'];groups=original_train_groups(original,train)
        if [e['actual_step'] for e in s['events']]!=[0,32768,65536]:raise OpsError('source checkpoints incomplete')
        vals=[]
        for event in s['events']:
            ev=event['validation'];overall=ev['overall']
            if overall['total']!=6 or overall['probe_cutoff']!=0 or counts(ev['episodes'])!={k:overall[k] for k in counts(ev['episodes'])}:
                raise OpsError('incomplete source actual Validation')
            vals.append({'actual_step':event['actual_step'],**validation_groups(ev['episodes'],validation)})
        switch=s['scheduler_state']['phase_switch_step'];tail=65536-switch-groups['overall']['policy_steps']
        if type(tail) is not int or tail<0 or tail!=65536-rows[-1]['policy_end_step']:
            raise OpsError('source incomplete tail/accounting differs')
        runs.append({'experiment_id':ref['experiment_id'],'original_completed_train':groups,'validation':vals,
            'actual_switch_step':switch,'unfinished_tail_policy_steps':tail,
            'tail_terminal_and_return':'NOT_IMPUTED','counterfactual_success_on_other_heading':'NOT_TESTED',
            'causal_attribution':'NOT_TESTED_NO_NEW_TRAJECTORIES_OR_POLICY_INTERVENTION'})
    return {'heading_coverage':coverage,'train_tasks':train,'validation_tasks':validation,'runs':runs,
        'unobserved':['per-step advantages','collision approach trajectory','policy actions conditioned on heading'],
        'causal_conclusion':'SUSPECTED distribution effect,not established by retrospective correlation'}


def run():
    if os.getpgrp()!=os.getpid():raise OpsError('R23 requires owned isolated process group')
    from research_ops_v2.manager import OperationsV2
    from .lifetime import start_guard,stop_guard
    manager=OperationsV2(ROOT);parent=process_identity(os.getppid())
    with FileLock(manager.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R23-SPLIT-DISTRIBUTION':raise OpsError('wrong R23 registered phase')
        verify_checks(reg);enforce_resources(ROOT);spec=read_json(ROOT/'mujoco/reports/autonomous_rl_distribution_r23_prereg.json')
        if spec!=make_protocol(reg):raise OpsError('R23 frozen identity differs')
        watcher=start_guard(parent,time.monotonic()+120,ROOT)
        try:
            OUT.mkdir(parents=True,exist_ok=False)
            atomic_json(OUT/'attempt.json',{'experiment_id':spec['experiment_id'],'protocol_sha256':digest(spec),
                'source_identity_sha256':digest(reg['source_identity']),'owner':process_identity(),'parent':parent},exclusive=True)
            try:
                started=time.monotonic();cpu=time.process_time();minimum=float('inf')
                budget=BudgetStore(HOME).status();before={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
                def guard():
                    nonlocal minimum
                    minimum=min(minimum,enforce_resources(ROOT)['mem_available_mib'])
                    if time.monotonic()-started>115 or BudgetStore(HOME).status()['stop_requested']:raise OpsError('R23 time/stop protection')
                guard();data=calculate(spec,guard);guard()
                after={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
                if (before!=after or BudgetStore(HOME).status()!=budget or make_protocol(reg)!=spec
                        or source_identity(ROOT,reg['source_paths'])!=reg['source_identity']):raise OpsError('R23 immutable source/history/budget changed')
                if any(n in sys.modules for n in ('torch','stable_baselines3','mujoco','mujoco_lidar')):
                    raise OpsError('R23 arithmetic unexpectedly loaded ML/simulator module')
                result={'status':'PASS',**spec,**data,'protocol_sha256':digest(spec),'all_authorization_files_unchanged':True,
                    'authorization_file_count':len(before),'authorization_files_sha256':before,'budget_after':budget,
                    'wall_elapsed_seconds':time.monotonic()-started,'cpu_elapsed_seconds':time.process_time()-cpu,
                    'worker_hwm_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
                    'sampled_min_mem_available_mib':minimum,'history':manager.verify()['history']}
                atomic_json(OUT/'result.json',result,exclusive=True)
                atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
                return {'status':'PASS','result_sha256':sha256(OUT/'result.json'),'optimizer_updates':0}
            except BaseException as error:
                import traceback
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(error).__name__,'message':str(error)[:2000],
                    'traceback':traceback.format_exc()[-12000:],'context':'arithmetic only,no completion or causal effect assumed'},exclusive=True)
                raise
        finally:stop_guard(watcher)


if __name__=='__main__':
    argparse.ArgumentParser(description=__doc__).parse_args();print(json.dumps(run(),allow_nan=False))
