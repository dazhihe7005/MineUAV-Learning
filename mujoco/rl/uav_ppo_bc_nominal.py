"""Immutable identities and matched nominal inputs; no training entry point."""
import json
import platform
import subprocess
from pathlib import Path
import mujoco
import numpy as np
import torch
import stable_baselines3 as sb3
from latent_dynamics_data import file_hash, json_hash
from uav_bc_safety import atomic_json
from uav_bc_policy import load, parameter_hash
from uav_bc_robustness import prepare_snapshot
from uav_bc_yaw_data import assert_isolated

BASE='7e93edda8e1625862dc6dafb4baae58dd62ca3f5'
BRANCH='feat/uav-ppo-bc-nominal-comparison'
PARTS='uav_ppo_bc_nominal_comparison_parts'
MANIFEST='uav_ppo_bc_nominal_manifest_seed0.json'
REPORT='uav_ppo_bc_nominal_comparison.json'
SEED=2026101501
CONTROLLERS=('scripted','original_bc','yaw_bc','ppo7d')
MODELS={
 'original_bc':('uav_bc_mlp_seed0.pt','7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c'),
 'yaw_bc':('uav_bc_yaw_augmented_seed0.pt','d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce'),
 'ppo7d':('ppo_waypoint_pi_lowstd_100k.zip','a5bc3c2ed9032dfa8b11e41f1ea9e0d94eef1a2287a5b2611ef829cf265e6953')}
SOURCE_NAMES=['rl/mine_uav_env.py','rl/ppo_pi_env.py','rl/audit_velocity_pi.py',
 'rl/uav_bc_policy.py','rl/uav_bc_robustness.py','rl/uav_bc_evaluation.py','rl/latent_mpc_evaluation.py',
 'rl/decision_fidelity_snapshot.py','rl/test_env_scripted_policy.py','rl/uav_bc_safety.py',
 'control/velocity_command_controller.py','control/hover_controller.py','control/position_controller.py',
 'control/control_allocator.py','models/mine_uav_dynamics_v2.xml','models/rotor_actuators.xml','models/rotor_config.json']
TARGET_SOURCES=['uav_bc_target_splits_seed0.json','onpolicy_adaptation_target_splits_seed0.json',
 'uav_bc_yaw_coverage_manifest_seed0.json','uav_bc_external_disturbance_manifest_seed0.json']

def fresh_targets(old):
    rng=np.random.default_rng(SEED)
    rows=[dict(target_id=f'ppo-bc-nominal-{SEED}-{i:03d}',env_seed=950150000+i,
        target=np.r_[rng.uniform(-2,2,2),rng.uniform(.7,1.5)].tolist()) for i in range(200)]
    assert_isolated(rows,old)
    return rows

def initial_snapshot(env,target):
    return prepare_snapshot(env,target,dict(position_m=[0.,0.,0.],velocity_m_s=[0.,0.,0.],yaw_rad=0.))

def frozen_controller(root,name):
    if name=='scripted':
        from test_env_scripted_policy import scripted_action
        return scripted_action,None,dict(inputs='7D observation only',parameters=0)
    filename,sha=MODELS[name];path=Path(root)/'mujoco/rl/models'/filename
    if file_hash(path)!=sha:raise ValueError('checkpoint identity changed')
    if name=='ppo7d':
        model=sb3.PPO.load(path,device='cpu')
        model.policy.set_training_mode(False);module=model.policy.requires_grad_(False)
        if model.seed!=0 or model.num_timesteps!=100352:raise ValueError('wrong PPO training provenance')
        def predict(obs):
            x=np.asarray(obs,np.float32)
            if x.shape!=(7,) or not np.isfinite(x).all():raise ValueError('finite 7D input only')
            return model.predict(x,deterministic=True)[0].astype(np.float32)
        meta=dict(training_seed=int(model.seed),actual_training_timesteps=model.num_timesteps,
            requested_timesteps=100000,architecture='separate actor/critic 7-64Tanh-64Tanh, means4/value1, global log_std4',
            observation_normalization='none (no VecNormalize)',action_distribution='unsquashed diagonal Gaussian; deterministic clipped mean',
            learning_rate=model.learning_rate,n_steps=model.n_steps,batch_size=model.batch_size,n_epochs=model.n_epochs,
            gamma=model.gamma,gae_lambda=model.gae_lambda,clip_range=model.clip_range(1.),ent_coef=model.ent_coef,
            vf_coef=model.vf_coef,max_grad_norm=model.max_grad_norm,clip_range_vf=None,target_kl=model.target_kl,
            normalize_advantage=model.normalize_advantage,use_sde=model.use_sde,log_std=module.log_std.detach().tolist(),
            training_source='mujoco/rl/train_ppo_pi_lowstd.py::train',factory='train_ppo_waypoint.py::make_ppo')
    else:
        policy,training_meta=load(path);module=policy.actor;predict=policy.predict
        meta=dict(architecture='7-128ReLU-128ReLU-4',normalization='checkpoint Train-only observation/action statistics',
            normalization_sha256=json_hash(policy.stats),training_metadata=training_meta)
    return predict,module,dict(meta,path=str(path.relative_to(root)),checkpoint_sha256=sha,
        parameter_sha256=parameter_hash(module),parameters=sum(p.numel() for p in module.parameters()),
        eval_mode=not module.training,requires_grad_false=all(not p.requires_grad for p in module.parameters()))

def compatibility(env,policy):
    if env.__class__.__name__!='MineUAVPIEnv' or env.reward_version!='v2':raise ValueError('unchanged nominal PI RewardV2 required')
    if policy.observation_space!=env.observation_space or policy.action_space!=env.action_space:
        raise ValueError('policy/environment observation or action spaces incompatible')
    if (env.physics_dt,env.control_dt,env.policy_dt,env.max_episode_steps)!=(.002,.01,.04,375):
        raise ValueError('timing incompatible')
    return dict(compatible=True,observation_shape=[7],action_shape=[4],ki_xy=env.velocity_controller.ki_xy,
        ki_z=env.velocity_controller.ki_z,semantics='world target-error xyz m; world velocity xyz m/s; wrapped target-minus-yaw rad',
        action='normalized world velocity xyz *[1.5,1.5,1]m/s and yaw_rate*1rad/s, clip[-1,1]',
        low_level='same configure_pi_env; same attitude/allocator/physics/task, nominal zero external force',
        success='distance<0.10m AND speed<0.15m/s for5 policy steps',timeout_s=15,
        failures='nonfinite_state,z_below_zero,outside_flight_area(radius6m/z4m),excessive_tilt(60deg)',
        privileged_inputs=False,partial_observability='7D omits PI integral and attitude/angular state for ALL main policies')

def inventory(root):
    names=subprocess.check_output(['rg','--files','--no-ignore','mujoco/rl/models','-g','*.zip'],cwd=root,text=True).splitlines()
    groups={}
    for n in names:
        parts=Path(n).parts;g='top_level' if len(parts)==4 else parts[3]
        groups[g]=groups.get(g,0)+1
    return dict(zip_count=len(names),groups=groups,paths_sha256=json_hash(sorted(names)),
        note='ZIP count is NOT independent seed count; nested diagnostic rollouts/replays/intermediate models excluded')

def provenance(root):
    r=Path(root)/'mujoco/reports'
    p=json.loads((r/'ppo_waypoint_pi_lowstd_seed0.json').read_text())
    old=json.loads((r/'ppo_waypoint_v2_lowstd_multiseed.json').read_text())
    extra=json.loads((r/'ppo_waypoint_pi_observable_lowstd_seed0.json').read_text())
    return dict(main_eligible_seeds=[0],main_checkpoint_predeclared='final100k; no choice using current evaluation',
        historical_pi7d=dict(report_sha256=file_hash(r/'ppo_waypoint_pi_lowstd_seed0.json'),
            final_evaluation=p['milestones']['100k']['evaluation'],actual_timesteps=p['actual_timesteps'],
            training_time_checkpoint_sha256=p['milestones']['100k'].get('checkpoint_sha256'),
            identity_limitation='historical report lacks checkpoint SHA; current hash/ZIP seed,budget,network and replay establish current identity, not a missing historical hash'),
        old_p_loop=dict(training_seeds=old['training_seed_order'],aggregate=old['aggregate'],
            report_sha256=file_hash(r/'ppo_waypoint_v2_lowstd_multiseed.json'),
            evaluation='100 benchmark reset seeds20261001..20261100,100 holdout20271001..20271100, deterministic final100k per seed',
            main_excluded_reason='trained/evaluated with P velocity loop, not current PI;95.6% not same protocol'),
        pi10d_extra_information=dict(observation='7D plus3 Ki*integral-error world acceleration m/s²',
            extra_information=True,final_evaluation=extra['milestones']['100k']['evaluation'],
            main_excluded_reason='not same observation as BC; historical reference only'))

def build_manifest(root):
    from ppo_pi_env import MineUAVPIEnv
    from uav_bc_external_disturbance import excluded_targets
    root=Path(root);r=root/'mujoco/reports'
    old=excluded_targets(root)+json.loads((r/TARGET_SOURCES[-1]).read_text())['targets']
    bc=json.loads((r/TARGET_SOURCES[0]).read_text());new=fresh_targets(old)
    states=[];models={};env=MineUAVPIEnv(reward_version='v2')
    try:
        for name in CONTROLLERS:
            _,module,meta=frozen_controller(root,name);models[name]=meta
            if name=='ppo7d':compatible=compatibility(env,module)
            del module
        for cohort,entries in [('fresh',new)]+[(s,v['targets']) for s,v in bc['provenance']['evaluation'].items()]:
            for i,t in enumerate(entries):
                target=dict(t,target_id=t.get('target_id',f'historical-{cohort}-{i:03d}'))
                _,meta=initial_snapshot(env,target)
                states.append(dict(target,key=f'{cohort}/{i:03d}',condition='nominal',split=cohort,index=i,initial_state=meta))
    finally:env.close()
    sbroot=Path(sb3.__file__).parent
    identity=dict(models=models,source_sha256={n:file_hash(root/'mujoco'/n) for n in SOURCE_NAMES},
        runner_sha256={n:file_hash(root/'mujoco/rl'/n) for n in ['uav_ppo_bc_nominal.py','uav_ppo_bc_nominal_run.py']},
        sb3_source_sha256={n:file_hash(sbroot/n) for n in ['ppo/ppo.py','common/policies.py','common/buffers.py',
            'common/on_policy_algorithm.py','common/distributions.py','common/torch_layers.py']},
        runtime=dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,torch=str(torch.__version__),sb3=sb3.__version__))
    value=dict(base_commit=BASE,branch=BRANCH,generation_seed=SEED,workers=1,compatibility=compatible,identity=identity,
        targets=new,states=states,exclusion_count=len(old),exclusion_sha256=json_hash(old),
        target_sources={n:file_hash(r/n) for n in TARGET_SOURCES},
        isolation='strict target/ID/reset-seed exclusion from available BC/adaptation/yaw/external manifests; full PPO training target stream not persisted, global PPO training nonoverlap unprovable',
        protocol=dict(main_episodes=800,historical_pi7d_episodes=200,technical_repeat_episodes=4,
            deterministic=True,training_steps_this_stage=0,selection='final-budget PI7D checkpoint predeclared; no evaluation selection',
            target_distribution='full task: independent uniform x,y[-2,2]m,z[.7,1.5]m',
            historical_replay='separate known benchmark/holdout, NOT new-test performance; no old P or10D flights'),
        inventory=inventory(root),provenance=provenance(root))
    manifest=dict(value,sha256=json_hash(value));path=r/MANIFEST
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('frozen manifest identity changed')
    if not path.exists():atomic_json(path,manifest)
    return manifest
