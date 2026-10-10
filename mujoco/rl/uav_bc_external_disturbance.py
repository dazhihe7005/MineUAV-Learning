"""Frozen-controller evaluation plant: world COM force before each physics tick."""
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from ppo_pi_env import MineUAVPIEnv
from latent_dynamics_data import file_hash,json_hash
from uav_bc_safety import atomic_json
from uav_bc_robustness import prepare_snapshot,manifest_identity

BASE='e30984ad879c2263081c0f9244b3ad1dc1bea8d0'
SEED=2026101201
PARTS='uav_bc_external_disturbance_seed0_parts'
MANIFEST='uav_bc_external_disturbance_manifest_seed0.json'
REPORT='uav_bc_external_disturbance_seed0.json'
MODELS={
 'original':('uav_bc_mlp_seed0.pt','7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c'),
 'yaw_augmented':('uav_bc_yaw_augmented_seed0.pt','d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce')}

def conditions():
    return {'nominal':dict(fraction_mg=0.,start_tick=0,end_tick=None),
        **{f'constant_{n}':dict(fraction_mg=f,start_tick=0,end_tick=None)
            for n,f in [('low',.05),('medium',.10),('high',.20)]},
        **{f'gust_{n}':dict(fraction_mg=f,start_tick=1000,end_tick=2000)
            for n,f in [('medium',.10),('high',.20)]}}

class DisturbanceEnv(MineUAVPIEnv):
    """No new control logic. xfrc_applied is a WORLD wrench at body COM."""
    def __init__(self,**kwargs):
        super().__init__(render_mode=None,reward_version='v2',**kwargs)
        self.uav_body_id=self.model.body('mine_uav').id
        if self.model.nbody!=2 or self.model.body_mass[self.uav_body_id]!=mujoco.mj_getTotalmass(self.model):
            raise ValueError('single rigid vehicle body required for COM resultant')
        self.mass_kg=float(mujoco.mj_getTotalmass(self.model))
        self.gravity=float(abs(self.model.opt.gravity[2]))
        self.configure('nominal',[1.,0.,0.]);self.force_times=[];self.force_vectors=[]
        self.policy_times=[0.];self.allocator_saturations=[];self.control_updates=[]

    def configure(self,condition,direction):
        v=np.asarray(direction,float)
        if condition not in conditions() or v.shape!=(3,) or not np.isfinite(v).all() or abs(v[2])>1e-12 or not np.isclose(np.linalg.norm(v),1.,atol=1e-12,rtol=0):
            raise ValueError('fixed condition and unit world XY force direction required')
        self.force_condition=condition;self.force_direction=v.copy()
        self.data.xfrc_applied[:]=0.

    def reset(self,**kwargs):
        result=super().reset(**kwargs)
        self.force_times=[];self.force_vectors=[];self.data.xfrc_applied[:]=0.
        self.policy_times=[float(self.data.time)];self.allocator_saturations=[];self.control_updates=[]
        return result

    def step(self,action):
        result=super().step(action)
        self.policy_times.append(float(self.data.time))
        self.allocator_saturations.append(result[4]['allocator_saturation_count'])
        self.control_updates.append(result[4]['controller_updates'])
        return result

    def force_at_tick(self,tick):
        c=conditions()[self.force_condition]
        active=tick>=c['start_tick'] and (c['end_tick'] is None or tick<c['end_tick'])
        return self.force_direction*(self.mass_kg*self.gravity*c['fraction_mg'] if active else 0.)

    def _before_physics_step(self):
        # Integer ticks avoid floating-clock 1.99999999997 delaying the gust.
        tick=int(round(float(self.data.time)/self.physics_dt));force=self.force_at_tick(tick)
        self.data.xfrc_applied[:]=0. # assignment, not addition: never double apply.
        self.data.xfrc_applied[self.uav_body_id,:3]=force
        self.force_times.append(float(self.data.time));self.force_vectors.append(force.copy())

def preflight(env):
    c=env.velocity_controller;mg=env.mass_kg*env.gravity;force=.2*mg
    thrust=math.hypot(mg,force);tilt=math.atan(.2)
    max_thrust=c.attitude_controller.max_total_thrust_n
    feasible=thrust<max_thrust and .2*env.gravity<c.horizontal_accel_limit and tilt<math.radians(60)
    if not feasible:raise ValueError('High exceeds static physical/controller feasibility; stop before evaluation')
    return dict(mass_kg=env.mass_kg,mass_provenance='compiled model:7kg engineering estimate, not measured vehicle mass',
        gravity_m_s2=env.gravity,body_name='mine_uav',body_id=env.uav_body_id,
        com_body_m=env.model.body_ipos[env.uav_body_id].tolist(),application_point='body COM, follows world xipos',
        interface='xfrc_applied[body_id,:3] world force; [3:] zero direct torque; assigned before every mj_step',
        forces_n={n:f*mg for n,f in [('low',.05),('medium',.1),('high',.2)]},
        high_equilibrium_thrust_n=thrust,high_equilibrium_tilt_degrees=math.degrees(tilt),
        max_total_thrust_n=max_thrust,horizontal_accel_limit_m_s2=c.horizontal_accel_limit,
        termination_tilt_limit_degrees=60.,pi_integral_accel_limit_m_s2=c.integral_accel_limit_xy,
        high_static_thrust_tilt_feasible=feasible,high_exceeds_pi_integral_acceleration_limit=.2*env.gravity>c.integral_accel_limit_xy,
        feasibility_note='Static feasibility is not guaranteed closed-loop recovery; High needs1.962m/s², exceeding1.5m/s² PI integral compensation. No parameter tuning.',
        physics_hz=500,control_hz=100,policy_hz=25,pressure_test_not_wind_speed=True)

def directions():
    rng=np.random.default_rng(SEED+1)
    angles=(np.arange(100)+rng.uniform(0,1,100))*2*np.pi/100
    angles=angles[rng.permutation(100)]
    return np.c_[np.cos(angles),np.sin(angles),np.zeros(100)]

def excluded_targets(root):
    from uav_bc_yaw_data import exclusion_targets
    r=Path(root)/'mujoco/reports'
    old=exclusion_targets(json.loads((r/'uav_bc_target_splits_seed0.json').read_text()),
        json.loads((r/'onpolicy_adaptation_target_splits_seed0.json').read_text()))
    return old+json.loads((r/'uav_bc_yaw_coverage_manifest_seed0.json').read_text())['final_targets']

def targets(old):
    from uav_bc_yaw_data import assert_isolated
    rng=np.random.default_rng(SEED)
    new=[dict(target_id=f'external-final-{SEED}-{i:03d}',env_seed=950140000+i,
        target=np.r_[rng.uniform(-2,2,2),rng.uniform(.7,1.5)].tolist()) for i in range(100)]
    assert_isolated(new,old);return new

def initial_snapshot(env,target):
    return prepare_snapshot(env,target,dict(position_m=[0.,0.,0.],velocity_m_s=[0.,0.,0.],yaw_rad=0.))

def build_manifest(root):
    root=Path(root);reports=root/'mujoco/reports'
    identity=manifest_identity(root)
    identity['external_source_sha256']=file_hash(Path(__file__))
    identity['models']={n:dict(path='mujoco/rl/models/'+name,sha256=sha) for n,(name,sha) in MODELS.items()}
    for n,(name,sha) in MODELS.items():
        if file_hash(root/'mujoco/rl/models'/name)!=sha:raise ValueError('frozen checkpoint changed')
    old=excluded_targets(root);new=targets(old);vectors=directions();env=DisturbanceEnv();states=[]
    try:
        physical=preflight(env)
        for name in conditions():
            for i,t in enumerate(new):
                env.configure(name,vectors[i]);_,meta=initial_snapshot(env,t)
                states.append(dict(t,key=f'{name}/final/{i:03d}',condition=name,split='final',index=i,
                    direction_world=vectors[i].tolist(),initial_state=meta))
    finally:env.close()
    config=dict(base_commit=BASE,branch='feat/uav-bc-external-disturbance',seed=SEED,identity=identity,preflight=physical,
        conditions=conditions(),targets=new,states=states,total_controller_episodes=1800,
        directions='100 permuted angular bins with fixed-seed uniform within-bin jitter; shared across conditions/controllers',
        excluded_target_count=len(old),excluded_target_sha256=json_hash(old),
        target_sources={n:file_hash(reports/n) for n in ['uav_bc_target_splits_seed0.json','onpolicy_adaptation_target_splits_seed0.json','uav_bc_yaw_coverage_manifest_seed0.json']},
        recovery_definition='At t>=4s, first5consecutive valid25Hz samples with distance<.10m AND speed<.15m/s; onset/confirmation lag from4s. Early termination censored, never unexposed recovery.',
        steady_definition='Constant-only terminal1s sample mean/std/slope/hold fraction for duration>=5s. Finite terminal-window proxy, NOT demonstrated asymptotic holding. No extension after task success.',
        workers=1)
    value=dict(config,sha256=json_hash(config));path=reports/MANIFEST
    if path.exists() and json.loads(path.read_text())!=value:raise ValueError('frozen manifest/source changed, refuse stale resume')
    if not path.exists():atomic_json(path,value)
    return value
