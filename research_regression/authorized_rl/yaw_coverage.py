"""R24 independent initial-yaw tasks; old L6 and distance manifests untouched."""
import copy
import math
from research_ops.storage import digest
from .short_goal import short_manifest,BASE_MANIFEST_SHA

VERSION='TRAIN_INITIAL_YAW_FIVE_OPTIONS_R24_V1'


def yaw_manifest(prereg):
    if digest(prereg['scenario_manifest'])!=BASE_MANIFEST_SHA:
        raise ValueError('R24 requires immutable original L6 manifest')
    short=short_manifest(prereg);bases=prereg['scenario_manifest']['train']
    by_short={v['base_identity_sha256']:v for v in short['train']}
    result={'version':VERSION,'base_manifest_sha256':BASE_MANIFEST_SHA,'short_manifest_sha256':digest(short)}
    for stage in ('SHORT','ORIGINAL'):
        rows=[]
        for base in bases:
            parent=by_short[base['identity_sha256']] if stage=='SHORT' else base
            for option,yaw in enumerate((base['start_yaw_rad'],0.,math.pi/2,math.pi,-math.pi/2)):
                row=copy.deepcopy(parent);row.pop('identity_sha256')
                row.update(base_identity_sha256=base['identity_sha256'],distance_task_identity_sha256=parent['identity_sha256'],
                    task_variant='DISTANCE_AND_INITIAL_YAW_SIM_TASK_R24_V1',yaw_profile=VERSION,distance_stage=stage,
                    yaw_option=option,base_start_yaw_rad=base['start_yaw_rad'],start_yaw_rad=yaw)
                row['identity_sha256']=digest(row);rows.append(row)
        result[stage]=rows
    if len({v['identity_sha256'] for stage in ('SHORT','ORIGINAL') for v in result[stage]})!=120:
        raise ValueError('R24 task identities not unique')
    return result


def verify_yaw_manifest(manifest,prereg):
    if manifest!=yaw_manifest(prereg):raise ValueError('R24 frozen yaw manifest differs')


def resolve_yaw_task(manifest,stage,base_identity,option):
    if stage not in ('SHORT','ORIGINAL') or type(option) is not int or not 0<=option<5:
        raise ValueError('only explicit Train stage and five yaw options allowed')
    rows=[v for v in manifest[stage] if v['base_identity_sha256']==base_identity and v['yaw_option']==option]
    if len(rows)!=1:raise ValueError('foreign/duplicate Train yaw identity')
    row=rows[0]
    if (row['split']!='train' or row['distance_stage']!=stage or row['yaw_profile']!=VERSION
            or digest({k:v for k,v in row.items() if k!='identity_sha256'})!=row['identity_sha256']):
        raise ValueError('corrupt Train yaw task payload')
    return copy.deepcopy(row)


def verify_yaw_temporal_configuration(model):
    actual={'gamma':model.gamma,'gae_lambda':model.gae_lambda,
        'rollout_buffer_gamma':model.rollout_buffer.gamma,'rollout_buffer_gae_lambda':model.rollout_buffer.gae_lambda}
    expected={'gamma':.99,'gae_lambda':.95,'rollout_buffer_gamma':.99,'rollout_buffer_gae_lambda':.95}
    if actual!=expected:raise ValueError('R24 must retain complete R17 temporal configuration')
    return actual
