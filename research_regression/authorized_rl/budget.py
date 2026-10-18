"""Append-only, conservative reservations. Standard library only, no training import."""
from __future__ import annotations
import math
import re
import time
from pathlib import Path
from research_ops.storage import FileLock,OpsError,atomic_json,read_json,digest,now


class BudgetLock:
    """Short bounded transactions, unlike the exclusive whole-experiment lock."""
    def __init__(self,path):self.path=path;self.inner=None
    def __enter__(self):
        deadline=time.monotonic()+1.
        while True:
            candidate=FileLock(self.path)
            try:
                candidate.__enter__();self.inner=candidate;return self
            except OpsError:
                if time.monotonic()>=deadline:raise
                time.sleep(.002)
    def __exit__(self,*args):return self.inner.__exit__(*args)


def grant():
    return {'authorization_id':'AUTONOMOUS-RL-RESEARCH-V1','version':1,
        'basis':'Explicit user instruction following AUDIT-01 commit e77479eaccf08cacfd9844b5ea4f0df98e4ad0d6',
        'max_training_timesteps':2000000,'max_training_wall_seconds':86400,
        'max_single_run_timesteps':1000064,'workers':1,'device':'cpu',
        'allowed_executor':'PPO_ONLY_V1','PX4':False,'hardware':False,'test_episodes':False,
        'interruption':'RESTART_REQUIRED; no retry of consumed experiment identity',
        'wall_accounting':'Full wall reservation permanently charged; actual elapsed also recorded, no automatic refund',
        'step_accounting':'All optimizer-collecting environment steps charged; no free optimizer tests or supervised updates'}


def experiment(arm,steps,wall,*,seed=0,revision='R1'):
    return {'schema':1,'experiment_id':f'ARL_{revision}_{arm}_SEED{seed}',
        'arm':arm,'seed':seed,'algorithm':'PPO','timesteps':steps,'wall_seconds':wall,
        'workers':1,'device':'cpu','test_episode_count':0,'initialization':'FRESH_RANDOM',
        'n_steps':128,'reward_version':'NAV_V2_COLLISION_PREFLIGHT_V1',
        'manifest_sha256':'f6ee34a153e6cb4d95fa43e7214d6d67c9b534835362be275363e14ec87b75d5',
        'curriculum_rule':'empty until 32768; switch only on next natural episode reset',
        'validation_steps':[32768,65536],'resume':'RESTART_REQUIRED'}


def validate_experiment(p):
    if p.get('schema')==33:
        from .early_distance_profile import validate_early_mix_profile
        return validate_early_mix_profile(p)
    if p.get('schema')==28:
        from .held_goal_profile import validate_held_goal_profile
        return validate_held_goal_profile(p)
    if p.get('schema')==24:
        from .yaw_coverage_profile import validate_yaw_profile
        return validate_yaw_profile(p)
    if p.get('schema')==22:
        from .gae_credit_profile import validate_gae_profile
        return validate_gae_profile(p)
    if p.get('schema')==20:
        from .temporal_credit_profile import validate_temporal_profile
        return validate_temporal_profile(p)
    if p.get('schema')==17:
        from .empty_rehearsal_profile import validate_rehearsal_profile
        return validate_rehearsal_profile(p)
    if p.get('schema')==16:
        from .centered_noise_profile import validate_noise_profile
        return validate_noise_profile(p)
    if p.get('schema')==14:
        from .centered_training_profile import validate_centered_profile
        return validate_centered_profile(p)
    if p.get('schema')==11:
        from .lower_lr_profile import validate_lower_lr_profile
        return validate_lower_lr_profile(p)
    if p.get('schema')==10:
        from .prior_distance_profile import validate_prior_distance_profile
        return validate_prior_distance_profile(p)
    if p.get('schema')==8:
        from .repaired_distance_profile import validate_repaired_profile
        return validate_repaired_profile(p)
    if p.get('schema')==6:
        from .distance_profile import validate_distance_profile
        return validate_distance_profile(p)
    if p.get('schema')==3:
        from .std_profile import validate_std_profile
        return validate_std_profile(p)
    if p.get('schema')==2:
        from .prior_profile import validate_prior_profile
        return validate_prior_profile(p)
    keys=set(experiment('UNIFORM',65536,3600))|{'source_identity_sha256','registration_sha256'}
    if set(p)-keys or not set(experiment('UNIFORM',65536,3600))<=set(p):raise OpsError('experiment schema invalid')
    if (not re.fullmatch(r'ARL_[A-Z0-9_]{3,90}',p['experiment_id']) or p['arm'] not in ('UNIFORM','CURRICULUM')
        or type(p['seed']) is not int or not 0<=p['seed']<1000
        or p['algorithm']!='PPO' or type(p['workers']) is not int or p['workers']!=1
        or p['device']!='cpu' or type(p['test_episode_count']) is not int or p['test_episode_count']!=0
        or p['initialization']!='FRESH_RANDOM' or p['resume']!='RESTART_REQUIRED'):
        raise OpsError('unsupported experiment authority')
    if (type(p['timesteps']) is not int or not 0<p['timesteps']<=1000064 or p['timesteps']%128
        or type(p['wall_seconds']) not in (int,float) or not math.isfinite(p['wall_seconds'])
        or not 0<p['wall_seconds']<=86400):raise OpsError('invalid rollout or wall budget')
    baseline=experiment(p['arm'],p['timesteps'],p['wall_seconds'],seed=p['seed'])
    for k in ('schema','n_steps','reward_version','manifest_sha256','curriculum_rule','validation_steps'):
        if p[k]!=baseline[k]:raise OpsError('frozen algorithm/protocol field changed')


class BudgetStore:
    def __init__(self,home):
        self.home=Path(home)
        self.lock=self.home.parent/(self.home.name+'.lock')

    def initialize(self,*,expected_grant_sha256):
        if expected_grant_sha256!=digest(grant()):raise OpsError('user grant SHA mismatch')
        with BudgetLock(self.lock):
            self.home.mkdir(exist_ok=False,parents=True)
            atomic_json(self.home/'grant.json',grant(),exclusive=True)
            (self.home/'events').mkdir()
            atomic_json(self.home/'head.json',{'count':0,'sha256':digest(grant())},exclusive=True)

    def _read(self):
        if read_json(self.home/'grant.json')!=grant():raise OpsError('immutable user grant changed')
        head=read_json(self.home/'head.json');files=sorted((self.home/'events').iterdir())
        if type(head.get('count')) is not int or len(files)!=head['count']:raise OpsError('budget event deletion/orphan; audit required')
        previous=digest(grant());attempts={};stopped=False
        for i,file in enumerate(files,1):
            event=read_json(file)
            if file.name!=f'{i:06d}.json' or event.get('sequence')!=i or event.get('previous_sha256')!=previous:
                raise OpsError('budget journal chain invalid')
            previous=digest(event);kind=event['kind'];data=event['data']
            if kind=='RESERVE':
                validate_experiment(data)
                key=data['experiment_id']
                if key in attempts or stopped:raise OpsError('duplicate/stopped reservation')
                attempts[key]={'protocol':data,'status':'RESERVED_OR_INTERRUPTED'}
            elif kind in ('FAIL','COMPLETE'):
                key=data['experiment_id']
                if key not in attempts or attempts[key]['status']!='RESERVED_OR_INTERRUPTED':raise OpsError('invalid attempt finalization')
                attempts[key]['status']=kind;attempts[key]['result']=data
            elif kind=='STOP':stopped=True
            else:raise OpsError('unknown budget event')
        if previous!=head.get('sha256'):raise OpsError('budget head checksum invalid')
        steps=sum(v['protocol']['timesteps'] for v in attempts.values())
        wall=sum(v['protocol']['wall_seconds'] for v in attempts.values())
        if steps>2000000 or wall>86400:raise OpsError('cumulative authorization exceeded')
        return head,attempts,stopped,steps,wall

    def _append(self,kind,data,head):
        event={'sequence':head['count']+1,'previous_sha256':head['sha256'],'kind':kind,'data':data,'time':now()}
        atomic_json(self.home/'events'/f"{event['sequence']:06d}.json",event,exclusive=True)
        atomic_json(self.home/'head.json',{'count':event['sequence'],'sha256':digest(event)})

    def status(self):
        with BudgetLock(self.lock):
            _,attempts,stopped,steps,wall=self._read()
            return {'authorization_id':grant()['authorization_id'],'grant_sha256':digest(grant()),
                'reserved_timesteps':steps,'remaining_timesteps':2000000-steps,
                'reserved_wall_seconds':wall,'remaining_wall_seconds':86400-wall,
                'attempts':[{'experiment_id':k,**v} for k,v in attempts.items()],
                'stop_requested':stopped,'training_authorized':not stopped}

    def reserve(self,p):
        validate_experiment(p)
        with BudgetLock(self.lock):
            head,attempts,stopped,steps,wall=self._read()
            if stopped or p['experiment_id'] in attempts:raise OpsError('stopped or consumed experiment; no retry')
            if any(v['status']=='RESERVED_OR_INTERRUPTED' for v in attempts.values()):
                raise OpsError('unresolved attempt: recover audit before new scheduling')
            if steps+p['timesteps']>2000000 or wall+p['wall_seconds']>86400:raise OpsError('insufficient cumulative budget')
            self._append('RESERVE',p,head)
        return digest(p)

    def request_stop(self):
        with BudgetLock(self.lock):
            head,_,stopped,_,_=self._read()
            if not stopped:self._append('STOP',{'reason':'USER_OR_OPERATOR_STOP; new scheduling disabled'},head)

    def fail(self,experiment_id,reason):
        with BudgetLock(self.lock):
            head,attempts,_,_,_=self._read()
            if experiment_id not in attempts or attempts[experiment_id]['status']!='RESERVED_OR_INTERRUPTED':raise OpsError('attempt not pending')
            self._append('FAIL',{'experiment_id':experiment_id,'reason':str(reason)[:2000]},head)

    def complete(self,experiment_id,evidence):
        with BudgetLock(self.lock):
            head,attempts,_,_,_=self._read();p=attempts[experiment_id]['protocol']
            if attempts[experiment_id]['status']!='RESERVED_OR_INTERRUPTED':raise OpsError('attempt not pending')
            if (evidence.get('protocol_sha256')!=digest(p) or evidence.get('actual_timesteps')!=p['timesteps']
                or evidence.get('optimizer_step_count')!=p['timesteps']//128*4
                or evidence.get('finite_model_and_gradients') is not True
                or not isinstance(evidence.get('summary_sha256'),str) or len(evidence['summary_sha256'])!=64
                or type(evidence.get('observed_wall_seconds')) not in (float,int)
                or not 0<evidence['observed_wall_seconds']<=p['wall_seconds']):raise OpsError('completion evidence invalid')
            self._append('COMPLETE',{'experiment_id':experiment_id,**evidence},head)
