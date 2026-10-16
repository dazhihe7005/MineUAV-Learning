"""Single fixed-module supervised execution. No shell/script injection entry."""
from __future__ import annotations
import os
import re
import signal
import subprocess
import time
from pathlib import Path
from research_ops.environment import runtime_environment,resource_status
from research_ops.identity import git,source_identity
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,sha256,process_identity,lock_held
from research_ops_v2.manager import OperationsV2
from research_ops_v2.execution import group_alive,stop_group
from .budget import BudgetStore,validate_experiment

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'.research_local/authorized_rl_v1'


def run_directory(home,identity):
    if not isinstance(identity,str) or not re.fullmatch(r'ARL_[A-Z0-9_]{3,90}',identity):raise OpsError('invalid run identity')
    return Path(home)/'runs'/identity


def check_resources(mem_mib,disk_bytes):
    if mem_mib<1536 or disk_bytes<2*1024**3:raise OpsError('memory/disk guard reached')


def monitor_process(child,wall_seconds,probe):
    """Monotonic absolute deadline; failures terminate this owned process group."""
    started=time.monotonic()
    try:
        while child.poll() is None:
            if time.monotonic()-started>=wall_seconds:raise OpsError('absolute authorized wall deadline')
            probe()
            time.sleep(.05)
        if time.monotonic()-started>wall_seconds:raise OpsError('worker finished after absolute wall deadline')
        if group_alive(child.pid):raise OpsError('worker leader exited with living descendants')
    except BaseException:
        if child.poll() is None or group_alive(child.pid):stop_group(child)
        raise


def validate_launch(permit,identity):
    if (permit.get('experiment_id')!=identity or permit.get('parent_pid')!=os.getppid()
        or permit.get('parent_identity')!=process_identity(os.getppid())
        or not isinstance(permit.get('protocol_sha256'),str)):
        raise OpsError('worker has no live supervising permit')


def verify_current(root=ROOT):
    m=OperationsV2(root);v=m.verify();_,version,state,_=m._load()
    if state['stage_status'] in ('RUNNING','STOP_REQUESTED','SYNC_PENDING','FAIL','BLOCKED','INTERRUPTED'):
        raise OpsError('operations stage not eligible for authorized research')
    return m,version['registration']


def supervise(protocol,*,root=ROOT):
    validate_experiment(protocol)
    home=root/'.research_local/authorized_rl_v1';store=BudgetStore(home)
    manager=OperationsV2(root)
    with FileLock(manager.lock_path):
        _,registration=verify_current(root)
        if (protocol.get('registration_sha256')!=digest(registration)
            or protocol.get('source_identity_sha256')!=digest(registration['source_identity'])):
            raise OpsError('experiment source/registration not the independently reviewed version')
        if protocol['schema']==20:
            if registration['phase_id']!='AUTORL-R20-TEMPORAL-CREDIT':raise OpsError('temporal credit requires reviewed R20 phase')
            from .temporal_credit_profile import verify_temporal_proof,verify_temporal_checks
            verify_temporal_proof(protocol,root);verify_temporal_checks(protocol,root)
        elif protocol['schema']==17:
            if registration['phase_id']!='AUTORL-R17-EMPTY-REHEARSAL':raise OpsError('rehearsal requires reviewed R17 phase')
            from .empty_rehearsal_profile import verify_rehearsal_proof,verify_rehearsal_checks
            verify_rehearsal_proof(protocol,root);verify_rehearsal_checks(protocol,root)
        elif protocol['schema']==16:
            if registration['phase_id']!='AUTORL-R16-CENTERED-STD01':raise OpsError('centered noise requires reviewed R16 phase')
            from .centered_noise_profile import verify_noise_proof,verify_noise_checks
            verify_noise_proof(protocol,root);verify_noise_checks(protocol,root)
        elif protocol['schema']==14:
            if registration['phase_id']!='AUTORL-R14-CENTERED-ACTOR':raise OpsError('centered Actor requires reviewed R14 phase')
            from .centered_training_profile import verify_centered_proof,verify_centered_checks
            verify_centered_proof(protocol,root);verify_centered_checks(protocol,root)
        elif protocol['schema']==11:
            if registration['phase_id']!='AUTORL-R11-LOWER-LR':raise OpsError('lower-LR requires reviewed R11 phase')
            from .lower_lr_profile import verify_lower_lr_proof,verify_lower_lr_checks
            verify_lower_lr_proof(protocol,root);verify_lower_lr_checks(protocol,root)
        elif protocol['schema']==10:
            if registration['phase_id']!='AUTORL-R10-PRIOR-DISTANCE':raise OpsError('prior-distance requires its reviewed R10 phase')
            from .prior_distance_profile import verify_prior_distance_proof,verify_prior_distance_checks
            verify_prior_distance_proof(protocol,root);verify_prior_distance_checks(protocol,root)
        elif protocol['schema']==8:
            if registration['phase_id']!='AUTORL-R8-REPAIRED-DISTANCE':raise OpsError('repaired run requires its reviewed R8 phase')
            from .repaired_distance_profile import verify_repaired_proof,verify_repaired_checks
            verify_repaired_proof(protocol,root);verify_repaired_checks(protocol,root)
        elif protocol['schema']==6:
            if registration['phase_id']!='AUTORL-R6-DISTANCE':raise OpsError('distance run needs its reviewed training phase')
            from .distance_profile import verify_distance_proof,verify_distance_checks
            verify_distance_proof(protocol,root);verify_distance_checks(protocol,root)
        elif protocol['schema']==3:
            if registration['phase_id']!='AUTORL-R3-STD':raise OpsError('std run needs its own reviewed training registration')
            from .std_profile import verify_std_controls,verify_std_checks
            verify_std_controls(protocol,root);verify_std_checks(protocol,root)
        elif protocol['schema']==2:
            if registration['phase_id']!='AUTORL-R2-PRIOR':raise OpsError('prior run needs its own reviewed training registration')
            from .prior_profile import verify_prior_proof,verify_training_checks
            verify_prior_proof(protocol,root)
            verify_training_checks(protocol,root)
        else:
            from .baseline import verify_baseline
            verify_baseline(registration)
            baseline=read_json(home/'baseline.json')
            for rel,value in baseline['runtime_sources'].items():
                if sha256(root/rel)!=value:raise OpsError('baseline runtime identity differs')
        output=run_directory(home,protocol['experiment_id'])
        if output.exists():raise OpsError('runtime directory already used; never retry')
        available=resource_status(root);check_resources(available['mem_available_mib'],available['disk_free_bytes'])
        store.reserve(protocol)
        output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'protocol.json',protocol,exclusive=True)
        permit={'experiment_id':protocol['experiment_id'],'parent_pid':os.getpid(),
            'parent_identity':process_identity(os.getpid()),'protocol_sha256':digest(protocol)}
        atomic_json(output/'permit.json',permit,exclusive=True)
        lock=read_json(root/'research_ops/environment_lock.json')
        started=time.monotonic();child=None
        try:
            with (output/'console.txt').open('xb') as console:
                child=subprocess.Popen([lock['interpreter'],'-m','research_regression.authorized_rl.worker',
                    '--experiment-id',protocol['experiment_id']],cwd=root,env=runtime_environment(root,lock),
                    stdout=console,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_json(output/'child.json',{'pid':child.pid,'identity':process_identity(child.pid)},exclusive=True)
                def probe():
                    resources=resource_status(root)
                    check_resources(resources['mem_available_mib'],resources['disk_free_bytes'])
                    if store.status()['stop_requested']:raise OpsError('safe stop requested')
                monitor_process(child,protocol['wall_seconds']-(time.monotonic()-started),probe)
            elapsed=time.monotonic()-started
            if child.returncode!=0:raise OpsError(f'fixed training worker failed: {child.returncode}')
            summary=read_json(output/'summary.json');complete=read_json(output/'complete.json')
            if complete.get('summary_sha256')!=sha256(output/'summary.json') or summary['protocol_sha256']!=digest(protocol):
                raise OpsError('worker summary/complete identity mismatch')
            verify_current(root)
            for rel,expected in summary['runtime_sources'].items():
                if sha256(root/rel)!=expected:raise OpsError('runtime source changed')
            store.complete(protocol['experiment_id'],{'protocol_sha256':digest(protocol),
                'actual_timesteps':summary['actual_timesteps'],'optimizer_step_count':summary['optimizer_step_count'],
                'finite_model_and_gradients':summary['finite_model_and_gradients'],
                'summary_sha256':sha256(output/'summary.json'),'observed_wall_seconds':elapsed})
            return {'status':'PASS','experiment_id':protocol['experiment_id'],'actual_timesteps':summary['actual_timesteps'],
                    'summary_sha256':sha256(output/'summary.json'),'observed_wall_seconds':elapsed}
        except BaseException as error:
            if child is not None and (child.poll() is None or group_alive(child.pid)):stop_group(child)
            atomic_json(output/'supervisor_failure.json',{'error_type':type(error).__name__,
                'message':str(error)[:2000],'elapsed_wall_seconds':time.monotonic()-started,
                'resume':'RESTART_REQUIRED; no automatic retry'},exclusive=True)
            store.fail(protocol['experiment_id'],str(error))
            raise
