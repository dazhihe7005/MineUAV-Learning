"""Unique R19 fixed-data arithmetic audit, no model/simulator execution."""
import argparse, json, math, os, resource, time, zipfile
from research_ops.storage import FileLock, OpsError, atomic_json, read_json, digest, sha256, process_identity
from research_ops.environment import enforce_resources
from research_ops.identity import source_identity
from research_regression.navigation_diagnosis import read_jsonl
from .runtime import ROOT, HOME, verify_current
from .budget import BudgetStore
from .reward_credit import configuration, make_protocol, verify_checks, discounted_components, episode_exposure, reset_exposure

OUT=ROOT/'.research_local/autorl_r19_reward_credit/run_01'


def calculate(spec,guard):
    r18=read_json(ROOT/'.research_local/autorl_r18_stopping_dynamics/run_01/result.json');traces=[]
    for model in r18['models']:
        guard();ref=model['reference']
        # SB3 data is JSON; never deserialize its embedded pickle strings or load .pth.
        with zipfile.ZipFile(ROOT/ref['checkpoint_path']) as z:
            if z.getinfo('data').file_size>4*1024*1024:raise OpsError('oversized checkpoint JSON')
            metadata=json.loads(z.read('data'))
        if metadata['gamma']!=.99 or metadata['gae_lambda']!=.95:raise OpsError('actual saved temporal parameters differ')
        rows=read_jsonl(ROOT/model['trace_file']['path'])
        if (len(rows)!=model['policy_steps'] or any(v['step']!=i+1 for i,v in enumerate(rows))
            or rows[-1]['terminal_class']!=model['outcome']
            or any(v['terminal_class'] is not None for v in rows[:-1])
            or not math.isclose(math.fsum(v['reward'] for v in rows),model['episode_return'],abs_tol=1e-8)):
            raise OpsError('recorded complete trace sequence/return differs')
        traces.append({'run_id':ref['run_id'],'outcome':model['outcome'],'policy_steps':len(rows),
            'actual_gamma':metadata['gamma'],'actual_gae_lambda':metadata['gae_lambda'],
            'discounted':[discounted_components(rows,g) for g in spec['gammas']],
            'trace_sha256':model['trace_file']['sha256'],'final_target_error_m':model['final_target_error_m']})
    manifest=read_json(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')['scenario_manifest']
    mapping={v['identity_sha256']:v['layout'] for v in manifest['train']};training=[]
    if len(mapping)!=12 or any(v['split']!='train' for v in manifest['train']):raise OpsError('only frozen Train allowed')
    for identity in ('ARL_R16_CENTERED_STD01_SEED0','ARL_R17_EMPTY_REHEARSAL_SEED0'):
        guard();d=HOME/'runs'/identity;s=read_json(d/'summary.json');rows=read_jsonl(d/'episode_details.jsonl')
        if (len(rows)!=s['episode_count'] or s['actual_timesteps']!=65536 or s['test_episode_count']
            or s['scheduler_state']['phase_switch_step']!=32850):raise OpsError('Train history totals/switch differ')
        ended=[v for v in rows if v['distance_stage']=='ORIGINAL'];closed=sum(v['episode_policy_steps'] for v in ended)
        expected=65536-s['scheduler_state']['phase_switch_step'];tail=expected-closed
        exposure=episode_exposure(ended,mapping,tail_steps=tail,
            tail_base=s['scheduler_state']['active_task']['identity_sha256'],expected_policy_steps=expected)
        resets=reset_exposure(s['scheduler_state']['selected_tasks'],mapping,closed_episodes=len(ended),
            tail_base=s['scheduler_state']['active_task']['identity_sha256'])
        training.append({'experiment_id':identity,**exposure,**resets,'summary_sha256':sha256(d/'summary.json'),
            'episode_details_sha256':sha256(d/'episode_details.jsonl')})
    gamma=.99;kernel=gamma*.95
    decay={name:{'base':base,'half_life_policy_steps':math.log(.5)/math.log(base),
        'half_life_seconds':math.log(.5)/math.log(base)/25,
        'geometric_sum_steps':1/(1-base),'geometric_sum_seconds':1/(1-base)/25}
        for name,base in (('gamma',gamma),('gamma_lambda',kernel))}
    success=traces[0]['discounted'][0]['return'];failed=[v['discounted'][0]['return'] for v in traces[1:]]
    return {'recorded_traces':traces,'original_train_exposure':training,'decay_arithmetic':decay,
        'observed_reward_prefers_two_failures_to_success_hypothesis':
            'REFUTED_ON_THREE_TRACES_ONLY' if all(success>v for v in failed) else 'NOT_REFUTED_ON_THREE_TRACES',
        'actual_saved_ppo_advantages':'NOT_OBSERVABLE_NOT_PRESERVED',
        'causal_gradient_attribution':'NOT_TESTED_REWARD_MASS_AND_DECAY_ARE_NOT_GRADIENTS'}


def run():
    if os.getpgrp()!=os.getpid():raise OpsError('R19 requires owned isolated process group')
    from research_ops_v2.manager import OperationsV2
    from .lifetime import start_guard, stop_guard
    m=OperationsV2(ROOT);parent=process_identity(os.getppid())
    with FileLock(m.lock_path):
        _,reg=verify_current()
        if reg['phase_id']!='AUTORL-R19-REWARD-CREDIT':raise OpsError('wrong R19 phase')
        verify_checks(reg);enforce_resources(ROOT)
        spec=read_json(ROOT/'mujoco/reports/autonomous_rl_reward_credit_r19_prereg.json')
        if spec!=make_protocol(reg):raise OpsError('R19 frozen identity differs')
        watcher=start_guard(parent,time.monotonic()+spec['wall_seconds'],ROOT)
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
                    if time.monotonic()-started>spec['wall_seconds']-5 or BudgetStore(HOME).status()['stop_requested']:
                        raise OpsError('R19 time/stop protection')
                guard();data=calculate(spec,guard);guard()
                after={str(p.relative_to(ROOT)):sha256(p) for p in HOME.rglob('*') if p.is_file()}
                if (after!=before or BudgetStore(HOME).status()!=budget or make_protocol(reg)!=spec
                    or source_identity(ROOT,reg['source_paths'])!=reg['source_identity']):raise OpsError('R19 source/history/budget changed')
                result={'status':'PASS',**configuration(),**data,'protocol_sha256':digest(spec),
                    'source_identity_sha256':digest(reg['source_identity']),'registration_sha256':digest(reg),
                    'authorization_files_sha256':before,'all_authorization_files_unchanged':True,
                    'budget_snapshot_sha256':digest(budget),'wall_seconds':time.monotonic()-started,
                    'cpu_seconds':time.process_time()-cpu,'worker_hwm_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
                    'sampled_min_mem_available_mib':minimum,'history':m.verify()['history']}
                atomic_json(OUT/'result.json',result,exclusive=True)
                atomic_json(OUT/'complete.json',{'result_sha256':sha256(OUT/'result.json')},exclusive=True)
                return {'status':'PASS','result_sha256':sha256(OUT/'result.json')}
            except BaseException as e:
                import traceback
                atomic_json(OUT/'failure.json',{'status':'FAIL','exception_type':type(e).__name__,'message':str(e)[:2000],
                    'traceback':traceback.format_exc()[-12000:],'context':'arithmetic audit only; no complete result assumed'},exclusive=True)
                raise
        finally:stop_guard(watcher)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args();print(json.dumps(run(),allow_nan=False))


if __name__=='__main__':main()
