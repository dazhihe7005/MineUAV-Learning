"""Frozen fixed-protocol identity and passive continuous-training target audit."""
import json
import platform
from pathlib import Path
import numpy as np
import torch
import mujoco
import stable_baselines3 as sb3
from ppo_pi_env import MineUAVPIEnv
from latent_dynamics_data import file_hash,json_hash
from uav_bc_safety import atomic_json
from uav_bc_yaw_data import assert_isolated
from uav_ppo_bc_nominal import SOURCE_NAMES,MODELS,initial_snapshot

BASE='77ce81a99cc6306167a91a9fd4f804417433f655'
BRANCH='feat/uav-ppo-multiseed-replication'
PARTS='uav_ppo_multiseed_replication_parts'
MANIFEST='uav_ppo_multiseed_preregistration.json'
REPORT='uav_ppo_multiseed_replication.json'
ROOT=Path(__file__).resolve().parents[2]

def seal(value):return dict(value,sha256=json_hash(value))

def verify_seal(value):
    if value.get('sha256')!=json_hash({k:v for k,v in value.items() if k!='sha256'}):
        raise ValueError('immutable manifest/record hash mismatch')

def config():
    return dict(seeds=[0,1,2,3,4],initialization='fresh random, never load a historical policy',
        policy='MlpPolicy',net_arch=dict(pi=[64,64],vf=[64,64]),activation='Tanh',log_std_init=-2.,
        optimizer='Adam',adam_eps=1e-5,learning_rate=.0003,n_steps=256,n_envs=8,
        vec_env='DummyVecEnv synchronous in one process',workers=1,batch_size=256,n_epochs=10,
        gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
        normalize_advantage=True,clip_range_vf=None,target_kl=None,use_sde=False,
        requested_timesteps=100000,actual_timesteps=100352,rollout_size=2048,rollouts=49,
        optimization_epochs=490,optimizer_steps=3920,checkpoint_selection='fixed final budget only',
        normalization='raw7D observations; no VecNormalize; environment clipped normalized4D actions',
        reward_version='v2',target_distribution='full',physics_hz=500,control_hz=100,policy_hz=25,
        timeout_seconds=15.,success='distance<0.10 AND speed<0.15 for5consecutive policy steps',
        validation_milestones=[20480,51200,100352],validation_targets=40,final_targets=200,
        device='cpu',torch_threads=1,deterministic_evaluation=True,
        training_target_rng='Gymnasium seed=training_seed+rank, continued across resets; actual visits logged',
        resume='skip verified completed seeds; interrupted seed restarts from scratch, retained attempts')

class AuditedPIEnv(MineUAVPIEnv):
    """Only observe resets/returns; unchanged simulator, sampler and reward."""
    def __init__(self,rank,forbidden):
        super().__init__(render_mode=None,reward_version='v2',target_distribution='full')
        self.rank=rank;self.forbidden=np.asarray(forbidden,float).reshape(-1,3)
        self.visits=[];self.completed=[];self.components={}
    def reset(self,**kwargs):
        result=super().reset(**kwargs);t=self.target_position.copy()
        if len(self.forbidden) and np.any(np.linalg.norm(self.forbidden-t,axis=1)<1e-8):
            raise ValueError('Train target collides with validation/final: refuse, no resampling')
        self.visits.append(dict(rank=self.rank,episode_index=len(self.visits),target=t.tolist(),
            target_id='train-coordinate-'+json_hash(t.tolist()),reset_seed=kwargs.get('seed')))
        self.components={};return result
    def step(self,action):
        result=super().step(action)
        for k,v in result[4]['reward_breakdown'].items():self.components[k]=self.components.get(k,0.)+float(v)
        if result[2] or result[3]:
            self.completed.append(dict(self.visits[-1],steps=self.episode_steps,
                termination_reason=result[4]['termination_reason'],reward_components=dict(self.components)))
        return result

def assert_model_protocol(model):
    c=config()
    for k in ['learning_rate','n_steps','n_epochs','batch_size','gamma','gae_lambda','ent_coef','vf_coef',
              'max_grad_norm','normalize_advantage','clip_range_vf','target_kl','use_sde']:
        if getattr(model,k)!=c[k]:raise ValueError('PPO factory drift: '+k)
    if model.n_envs!=8 or model.clip_range(1.)!=.2 or model.policy.net_arch!=c['net_arch']:
        raise ValueError('PPO rollout/network drift')
    if model.policy.optimizer.defaults['eps']!=1e-5:raise ValueError('Adam epsilon drift')

def verify_completion(record,manifest,seed):
    verify_seal(record)
    if record['manifest_sha256']!=manifest['sha256'] or record['seed']!=seed or record['status']!='complete':
        raise ValueError('wrong experiment/seed or partial training')
    if (record['actual_timesteps'],record['optimization_epochs'],len(record['history']))!=(100352,490,49):
        raise ValueError('incomplete training budget')
    if file_hash(record['checkpoint_path'])!=record['checkpoint_sha256']:raise ValueError('checkpoint corrupt')
    for prefix in ['actual_train_targets','training_episodes']:
        if file_hash(record[prefix+'_path'])!=record[prefix+'_sha256']:raise ValueError('training provenance corrupt')

def read_manifest(root=ROOT):
    m=json.loads((Path(root)/'mujoco/reports'/MANIFEST).read_text());verify_seal(m)
    if m['config']!=config():raise ValueError('locked protocol changed')
    for path,sha in m['source_sha256'].items():
        if file_hash(Path(root)/path)!=sha:raise ValueError('frozen source changed: '+path)
    for name,meta in m['frozen_models'].items():
        if file_hash(Path(root)/meta['path'])!=meta['sha256']:raise ValueError('historical checkpoint changed: '+name)
    sbroot=Path(sb3.__file__).parent
    for path,sha in m['sb3_source_sha256'].items():
        if file_hash(sbroot/path)!=sha:raise ValueError('SB3 implementation changed')
    return m

def build_manifest(root=ROOT):
    root=Path(root);r=root/'mujoco/reports';path=r/MANIFEST
    if path.exists():return read_manifest(root)
    from uav_bc_external_disturbance import excluded_targets
    old=excluded_targets(root)+json.loads((r/'uav_bc_external_disturbance_manifest_seed0.json').read_text())['targets']
    old+=json.loads((r/'uav_ppo_bc_nominal_manifest_seed0.json').read_text())['targets']
    rng=np.random.default_rng(2026101601);splits={}
    for split,count,base in [('validation',40,950160000),('final',200,950170000)]:
        rows=[dict(target_id=f'ppo-multiseed-{split}-2026101601-{i:03d}',env_seed=base+i,
            target=np.r_[rng.uniform(-2,2,2),rng.uniform(.7,1.5)].tolist()) for i in range(count)]
        assert_isolated(rows,old);old+=rows;splits[split]=rows
    env=MineUAVPIEnv(reward_version='v2');states=[]
    try:
        for i,t in enumerate(splits['final']):
            _,meta=initial_snapshot(env,t)
            states.append(dict(t,key=f'final/{i:03d}',condition='nominal',split='final',index=i,initial_state=meta))
    finally:env.close()
    frozen={n:dict(path='mujoco/rl/models/'+p,sha256=sha) for n,(p,sha) in MODELS.items()}
    frozen['world_model']=dict(path='mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt',
        sha256='42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9')
    sources=['mujoco/'+n for n in SOURCE_NAMES]+['mujoco/rl/'+n for n in
        ['train_ppo_waypoint.py','train_ppo_pi_lowstd.py','train_ppo_lowstd_multiseed.py',
         'uav_ppo_multiseed.py','uav_ppo_multiseed_train.py','uav_ppo_multiseed_eval.py']]
    history=json.loads((r/'ppo_waypoint_pi_lowstd_seed0.json').read_text())
    sbroot=Path(sb3.__file__).parent
    value=dict(base_commit=BASE,branch=BRANCH,config=config(),config_sha256=json_hash(config()),
        preregistered_before_optimizer_step=True,splits=splits,states=states,
        train_specification=dict(distribution='unchanged independent uniforms x,y[-2,2],z[.7,1.5]',
            audit='every actual target reset logged; exact coordinate exclusion from Val/Final; no resampling',
            cross_seed_target_stream_overlap='seed+rank rule retained; neighboring seeds share7 of8 underlying target RNG seeds; independent network/action/minibatch seeds, not disjoint environment streams'),
        excluded_known_targets=len(old)-240,excluded_targets_sha256=json_hash(old[:-240]),
        historical=dict(report_sha256=file_hash(r/'ppo_waypoint_pi_lowstd_seed0.json'),
            actual_timesteps=history['actual_timesteps'],n_steps=history['n_steps'],n_envs=history['n_envs'],
            provenance_limits='historical report lacks original checkpoint SHA and full Train stream; source/current ZIP metadata verified, exact historical stochastic training replay not claimed'),
        frozen_models=frozen,source_sha256={p:file_hash(root/p) for p in sources},
        sb3_source_sha256={n:file_hash(sbroot/n) for n in ['ppo/ppo.py','common/on_policy_algorithm.py','common/policies.py','common/buffers.py']},
        runtime=dict(python=platform.python_version(),numpy=np.__version__,torch=str(torch.__version__),
            mujoco=mujoco.__version__,sb3=sb3.__version__),
        statistics=dict(seed_unit='five independently initialized/trained policies',sd='sample ddof1',
            seed_ci='exploratory t95 interval with df4; same fixed200targets, not200seeds',
            target_ci='Wilson95 per policy, descriptive target-cohort uncertainty',
            paired='same physical snapshot; time/length common-success targets only'))
    m=seal(value);atomic_json(path,m);return read_manifest(root)
