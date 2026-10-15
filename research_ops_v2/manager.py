"""Independent version store. The V1 state/protocol/ledger are read-only inputs."""
from __future__ import annotations

import copy
from pathlib import Path
import uuid

from research_ops.environment import enforce_resources,inspect_environment
from research_ops.identity import (file_record,git,safe_path,source_identity,verify_baseline,
                                  verify_checkpoint,verify_original_l9)
from research_ops.lifecycle import Operations
from research_ops.storage import (AUTHORIZATION,ZERO_BUDGET,FileLock,OpsError,atomic_json,
    atomic_bytes,digest,lock_held,now,process_identity,read_json,sha256,STATUSES)
from .registration import validate_registration,verify_review


class OperationsV2:
    def __init__(self,root):
        self.root=Path(root).resolve();self.legacy=Operations(self.root)
        self.home=safe_path(self.root,'.research_local/ops_v2')
        self.lock_path=self.legacy.lock_path

    def _v1_paths(self):
        names=['research_state.json','protocol.json','budget_ledger.json','history_baseline.json',
               'initialization.attempt.json','initialization.complete.json']
        paths=[self.legacy.local/n for n in names]
        runs=safe_path(self.root,'.research_local/runs')
        if runs.exists():paths.extend(p for p in runs.rglob('*') if p.is_file())
        return paths

    def _verify_history(self,registration,baseline):
        effective=copy.deepcopy(baseline)
        for relative,change in registration['approved_changes'].items():
            if baseline['protected_files'].get(relative)!=change['before']:
                raise OpsError('approved change does not match immutable V1 history baseline')
            effective['protected_files'][relative]=change['after']
        # Explicit versioned overlay is limited to reviewed test source changes.
        # Every raw/model/report/other historical identity remains checked.
        return verify_baseline(self.root,effective)

    def _verify_v1(self,registration):
        state=self.legacy._read()
        if state['stage_status'] in ('RUNNING','STOP_REQUESTED','SYNC_PENDING'):
            raise OpsError('V1 must be stopped and synchronized before migration')
        protocol=read_json(self.legacy.local/'protocol.json')
        if digest(protocol)!=state['protocol_sha256']:
            raise OpsError('V1 protocol identity mismatch')
        if source_identity(self.root,self.legacy.config['runtime_paths'])!=protocol['source_identity']:
            raise OpsError('V1 runtime must remain intact')
        if sha256(self.legacy.local/'history_baseline.json')!=protocol['baseline_sha256']:
            raise OpsError('V1 historical baseline changed')
        if read_json(self.legacy.local/'budget_ledger.json')!={'scope':'OPS_ONLY_NO_TRAINING','authorization':ZERO_BUDGET,'training_experiments':[]}:
            raise OpsError('V1 zero budget ledger differs')
        self._verify_history(registration,read_json(self.legacy.local/'history_baseline.json'))
        verify_original_l9(self.root,self.legacy.config);verify_checkpoint(self.root,self.legacy.config)
        return state

    def register(self,registration,review,*,migrating=False):
        validate_registration(self.root,registration);verify_review(registration,review)
        with FileLock(self.lock_path):
            enforce_resources(self.root)
            exists=(self.home/'active.json').exists()
            if migrating==exists:raise OpsError('use migrate once; later phases use explicit register')
            parent=None
            previous_files={}
            anchor=self.legacy._read().get('latest_successful_push')
            if exists:
                # A new reviewed runtime may replace the working copy, never the
                # prior immutable source archive/protocol or original V1 code.
                self.verify(check_environment=False,candidate=registration)
                current=self._load()
                if current[2]['stage_status'] in ('RUNNING','STOP_REQUESTED','SYNC_PENDING'):
                    raise OpsError('current V2 phase must be stopped/synchronized')
                parent=current[0]['protocol_sha256']
                for previous in (self.home/'versions').iterdir():
                    for path in previous.rglob('*'):
                        if path.is_file():previous_files[str(path.relative_to(self.home))]=file_record(path)
                anchor=current[2].get('latest_successful_push')
                if not isinstance(anchor,dict):anchor=current[1]['publication_anchor']
            v1=self._verify_v1(registration)
            if not isinstance(anchor,dict):
                if self.legacy.config.get('expected_remote_url'):raise OpsError('no verified publication anchor')
                anchor={'commit':read_json(self.legacy.local/'initialization.complete.json')['base_commit'],
                        'scope':'LOCAL_GIT_TEST_FIXTURE_INITIAL_BASE'}
            if git(self.root,'merge-base','--is-ancestor',anchor['commit'],'HEAD',check=False).returncode:
                raise OpsError('publication anchor is not in current ancestry')
            if registration['branch']!=self.legacy.branch() or registration['base_commit']!=self.legacy.head():
                raise OpsError('registration is not for this exact branch/commit')
            version_id='v2_'+digest(registration)
            directory=self.home/'versions'/version_id
            directory.mkdir(parents=True,exist_ok=False)
            atomic_json(directory/'migration.attempt.json',{'version_id':version_id,'time':now(),
                        'parent_protocol_sha256':parent or v1['protocol_sha256']},exclusive=True)
            backup={}
            for path in self._v1_paths():
                relative=str(path.relative_to(self.legacy.local));dest=directory/'v1_backup'/relative
                atomic_bytes(dest,path.read_bytes(),exclusive=True)
                backup[relative]=file_record(dest)
            protocol={'schema_version':2,'version_id':version_id,'registration':registration,
                      'review':review,'parent_protocol_sha256':parent or v1['protocol_sha256'],
                      'publication_anchor':anchor,
                      'previous_version_files':previous_files,
                      'v1_original_files':backup,'v1_protocol_sha256':v1['protocol_sha256']}
            for relative in registration['source_identity']:
                atomic_bytes(directory/'source_archive'/relative,safe_path(self.root,relative).read_bytes(),exclusive=True)
            atomic_json(directory/'protocol.json',protocol,exclusive=True)
            atomic_bytes(directory/'budget_ledger.json',(self.legacy.local/'budget_ledger.json').read_bytes(),exclusive=True)
            state={'schema_version':2,'phase_id':registration['phase_id'],'stage_status':'PENDING',
                   'protocol_sha256':digest(protocol),'training_budget':0,'authorization_scope':AUTHORIZATION,
                   'checkpoint_resume_class':'RESTART_REQUIRED','run_id':'NOT_STARTED','task_id':'NOT_STARTED',
                   'owner':'NOT_VERIFIED','latest_commit':self.legacy.head(),'latest_successful_push':'NOT_VERIFIED',
                   'last_update_time':now(),'last_shutdown_status':'NOT_VERIFIED','phase_complete':False}
            atomic_json(directory/'state.json',{'state':state,'sha256':digest(state)},exclusive=True)
            atomic_json(directory/'migration.complete.json',{'protocol_sha256':digest(protocol),
                        'state_initial_sha256':sha256(directory/'state.json')},exclusive=True)
            # Activation is the final operation. Failed candidates remain forensic evidence.
            atomic_json(self.home/'active.json',{'version_id':version_id,'protocol_sha256':digest(protocol)})
            return {'status':'PASS',**self.status()}

    def _load(self):
        pointer=read_json(self.home/'active.json')
        version=pointer.get('version_id','')
        import re
        if not re.fullmatch('v2_[0-9a-f]{64}',version):raise OpsError('invalid active version')
        directory=safe_path(self.root,str((self.home/'versions'/version).relative_to(self.root)))
        protocol=read_json(directory/'protocol.json')
        complete=read_json(directory/'migration.complete.json')
        if protocol.get('version_id')!=version or digest(protocol)!=pointer['protocol_sha256'] or complete.get('protocol_sha256')!=pointer['protocol_sha256']:
            raise OpsError('version identity or migration completion differs')
        envelope=read_json(directory/'state.json');state=envelope['state']
        if digest(state)!=envelope.get('sha256') or state.get('protocol_sha256')!=pointer['protocol_sha256']:
            raise OpsError('V2 state corrupt or refers to another version')
        if (type(state.get('training_budget')) is not int or state['training_budget']!=0
                or digest(state.get('authorization_scope'))!=digest(AUTHORIZATION)
                or state.get('checkpoint_resume_class')!='RESTART_REQUIRED'
                or state.get('phase_id')!=protocol['registration']['phase_id']):
            raise OpsError('V2 state cannot grant training authority')
        if state.get('stage_status') not in STATUSES or type(state.get('phase_complete')) is not bool:
            raise OpsError('invalid V2 state status')
        run=state.get('run_id','')
        if run!='NOT_STARTED' and not re.fullmatch('OPS2_[0-9a-f]{32}',run):raise OpsError('invalid run identity')
        if run!='NOT_STARTED' and state.get('task_id') not in {t['task_id'] for t in protocol['registration']['tasks']}:
            raise OpsError('current task not registered')
        return pointer,protocol,state,directory

    def _save(self,state):
        pointer,protocol,_,directory=self._load()
        state=copy.deepcopy(state);state['last_update_time']=now()
        atomic_json(directory/'state.json',{'state':state,'sha256':digest(state)})

    def status(self):
        pointer,protocol,state,directory=self._load()
        return {'version_id':pointer['version_id'],'operations_version':'RESEARCH_OPS_V2',
                'state':state,'actual_branch':self.legacy.branch(),'actual_head':self.legacy.head(),
                'lock_held':lock_held(self.lock_path),'AUTO_TRAINING_AUTHORIZED':'NO'}

    def verify(self,*,check_environment=True,candidate=None):
        pointer,protocol,state,directory=self._load();registration=protocol['registration']
        for relative,record in protocol['previous_version_files'].items():
            path=safe_path(self.root,str((self.home/relative).relative_to(self.root)))
            if file_record(path)!=record:raise OpsError('previous operations version history changed')
        validate_registration(self.root,registration,check_source=candidate is None);verify_review(registration,protocol['review'])
        for relative,record in registration['source_identity'].items():
            if file_record(directory/'source_archive'/relative)!=record:raise OpsError('archived reviewed source changed')
        if candidate is None and registration['branch']!=self.legacy.branch():raise OpsError('current branch differs from registered phase')
        if git(self.root,'merge-base','--is-ancestor',registration['base_commit'],'HEAD',check=False).returncode:
            raise OpsError('current commit is not descended from registration')
        for relative,record in protocol['v1_original_files'].items():
            old=safe_path(self.root,'.research_local/'+relative)
            backup=directory/'v1_backup'/relative
            if file_record(old)!=record or file_record(backup)!=record:
                raise OpsError('original V1 state/protocol/attempt or its backup changed')
        if (directory/'budget_ledger.json').read_bytes()!=(self.legacy.local/'budget_ledger.json').read_bytes():
            raise OpsError('versioned zero-budget ledger changed')
        baseline=read_json(directory/'v1_backup/history_baseline.json')
        history=self._verify_history(candidate or registration,baseline)
        checkpoint=verify_checkpoint(self.root,self.legacy.config)
        original=verify_original_l9(self.root,self.legacy.config)
        from .execution import verify_runs
        verify_runs(self,state,registration)
        environment=inspect_environment(self.root,self.legacy.environment_lock) if check_environment else 'NOT_TESTED'
        return {'status':'PASS','version_id':pointer['version_id'],'protocol_sha256':pointer['protocol_sha256'],
                'history':history,'original_l9':original,'checkpoint':checkpoint,
                'resources':enforce_resources(self.root),'environment':environment,'training_budget':0}

    def doctor(self,*,restore_dependencies=False):
        with FileLock(self.lock_path):
            if restore_dependencies:
                self.verify(check_environment=False)
                from research_ops.environment import restore_upstream
                restore_upstream(self.root,self.legacy.environment_lock)
            return self.verify()

    def start(self,task_id):
        from .execution import start
        with FileLock(self.lock_path):return start(self,task_id)

    def request_stop(self,*,sync=True):
        from .execution import request_stop
        result=request_stop(self)
        if sync and result.get('status')=='SAFE_STOPPED':return self.sync()
        return result

    def recover(self):
        if lock_held(self.lock_path):return {'status':'BLOCKED','reason':'writer holds flock; no takeover'}
        with FileLock(self.lock_path):
            self._recover_unlaunched_attempt()
            self.verify();pointer,protocol,state,directory=self._load()
            if state['stage_status'] in ('RUNNING','STOP_REQUESTED'):
                owner=state['owner']
                if isinstance(owner,dict) and process_identity(owner['pid'])==owner:
                    raise OpsError('owner still alive without flock')
                child=directory/'runs'/state['run_id']/'child.json'
                if (directory/'runs'/state['run_id']/'launch-intent.json').exists() and not child.exists():
                    raise OpsError('launch identity missing: cannot prove task is absent; preserve and audit')
                if child.exists():
                    child_record=read_json(child);identity=child_record['identity']
                    if isinstance(identity,dict) and process_identity(identity['pid'])==identity:
                        raise OpsError('task child still alive; audit without automatic takeover')
                    from .execution import group_alive
                    if group_alive(child_record['pgid']):raise OpsError('task descendants still alive; no takeover')
                state.update(stage_status='INTERRUPTED',last_shutdown_status='UNCLEAN_INTERRUPTION',phase_complete=False)
                self._save(state)
            return self.status()

    def _recover_unlaunched_attempt(self):
        _,protocol,state,directory=self._load();runs=directory/'runs'
        if not runs.exists():return
        orphans=[]
        for run in runs.iterdir():
            if run.name==state['run_id'] or any((run/n).exists() for n in ('complete.json','stopped.json','failure.json')):continue
            attempt=read_json(run/'attempt.json')
            tasks={t['task_id']:t for t in protocol['registration']['tasks']}
            if (attempt.get('run_id')!=run.name or attempt.get('protocol_sha256')!=state['protocol_sha256']
                    or attempt.get('task')!=tasks.get(attempt.get('task_id'))):raise OpsError('unknown orphan attempt')
            if (run/'launch-intent.json').exists():raise OpsError('orphan launch may exist; no automatic takeover')
            orphans.append((run,attempt))
        if len(orphans)>1:raise OpsError('multiple orphan attempts require manual audit')
        if orphans:
            run,attempt=orphans[0]
            state.update(run_id=run.name,task_id=attempt['task_id'],stage_status='INTERRUPTED',
                         owner='NOT_VERIFIED',last_shutdown_status='INTERRUPTED_BEFORE_LAUNCH',phase_complete=False)
            self._save(state)
            atomic_json(run/'failure.json',{'run_id':run.name,'error_type':'INTERRUPTED_BEFORE_LAUNCH',
                        'message':'attempt existed; no launch intent was published','observed_at':now()},exclusive=True)

    def sync(self):
        from .publication import synchronize
        with FileLock(self.lock_path):return synchronize(self)
