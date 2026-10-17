"""Heading-only observation wrapper and bounded existing-physics instrumentation."""
import copy,math,time
import gymnasium as gym
import numpy as np
import torch
from lidar_integration.navigation_env import LidarNavigationEnv
from lidar_integration.ppo_preflight import HeadingObservation
from lidar_integration.ppo_scenarios import build_ppo_scene
from lidar_integration.terminal_accounting import classify_terminal
from lidar_integration.ppo_training_runner import _weights_sha
from lidar_integration.ppo_formal_validation import _optimizer_states_equal
from research_ops.environment import enforce_resources
from .runtime import ROOT
from .stopping_trace import stop_metrics

class HoldInitialHeading(gym.Wrapper):
    """Change declared task heading only, never qpos/qvel/control/clock/LiDAR."""
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs);plant=self.unwrapped
        w,x,y,z=plant.data.qpos[3:7]
        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        plant.target_yaw=yaw
        obs=dict(obs);obs['state']=plant._get_obs()
        return obs,{**info,'target_yaw_rad':yaw,'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW'}
    def step(self,action):
        o,r,t,tr,info=self.env.step(action)
        return o,r,t,tr,{**info,'target_yaw_rad':self.unwrapped.target_yaw,'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW'}

def collect(model,task,upstream,*,guard,on_record):
    row=task['scenario']
    if row['layout']!='empty' or row['split']!='validation' or task['goal_yaw_mode']!='HOLD_INITIAL_ENU_YAW':raise ValueError('R27 only fixed altered empty Validation')
    scene=build_ppo_scene(row)
    if scene.geometry_hash!=row['obstacle_configuration_hash']:raise ValueError('R27 geometry differs')
    env=HoldInitialHeading(HeadingObservation(LidarNavigationEnv('empty',row['scene_seed'],upstream,scene=scene)))
    fingerprint=_weights_sha(model.policy);optimizer=copy.deepcopy(model.policy.optimizer.state_dict());steps=model.num_timesteps
    rows=[];ticks=0;total=0.;started=time.monotonic()
    try:
        obs,info=env.reset(seed=row['scene_seed'],options={'start_seed':row['start_seed'],'target_seed':row['target_seed'],
            'start_position':row['start_xyz'],'target_position':row['target_xyz'],'start_yaw_rad':row['start_yaw_rad']})
        plant=env.unwrapped
        if (info['obstacle_configuration_hash']!=row['obstacle_configuration_hash'] or abs(info['target_yaw_rad']-task['goal_yaw_rad'])>1e-9
                or abs(float(obs['state'][6]))>1e-7):raise ValueError('R27 altered heading reset differs')
        outcome=None;raw=None
        for step in range(1,376):
            guard();enforce_resources(ROOT);pre=float(plant.data.time);pre_qpos=plant.data.qpos.copy();pre_qvel=plant.data.qvel.copy()
            state=obs['state'].copy();heading=obs['heading'].copy();age=float(obs['lidar_age'][0])
            with torch.inference_mode():
                tensor,_=model.policy.obs_to_tensor(obs);mean=model.policy.get_distribution(tensor).distribution.mean[0].cpu().numpy().copy()
                action,_=model.predict(obs,deterministic=True)
            if np.asarray(action).shape!=(4,) or mean.shape!=(4,) or not np.isfinite(action).all() or not np.isfinite(mean).all() or np.any(np.abs(action)>1):raise ValueError('R27 action invalid')
            obs,reward,t,tr,info=env.step(action);terminal=classify_terminal(info,t,tr)
            record={'step':step,'pre_time_s':pre,'post_time_s':float(plant.data.time),'pre_qpos':pre_qpos.tolist(),'pre_qvel':pre_qvel.tolist(),
                'state_pre':state.tolist(),'heading_pre':heading.tolist(),'lidar_age_pre_s':age,'raw_action_mean':mean.tolist(),
                'action_clipped':np.asarray(action).tolist(),'velocity_command_enu_m_s':np.asarray(info['velocity_command_m_s']).tolist(),
                'yaw_rate_command_enu_rad_s':float(info['yaw_rate_command_rad_s']),'post_qpos':plant.data.qpos.tolist(),'post_qvel':plant.data.qvel.tolist(),
                'distance_m':float(info['distance_m']),'speed_m_s':float(info['speed_m_s']),'physics_steps':int(info['physics_steps']),
                'lidar_source_time_s':float(info['lidar_source_time_s']),'lidar_age_post_s':float(info['lidar_age_s']),
                'reward':float(reward),'reward_breakdown':info['reward_breakdown'],'terminal_class':terminal.terminal_class,
                'raw_termination_reason':terminal.raw_termination_reason,'goal_yaw_rad':float(plant.target_yaw),'goal_yaw_mode':'HOLD_INITIAL_ENU_YAW',
                'pi_integral_error_post':plant.velocity_controller.integral_error.tolist(),
                'desired_acceleration_post_enu_m_s2':plant.velocity_controller.last_desired_acceleration_world.tolist(),'contact_pairs':info.get('contact_pairs',[])}
            if (record['post_time_s']<=pre or record['lidar_source_time_s']>record['post_time_s']+1e-9 or age<0
                    or abs(record['goal_yaw_rad']-task['goal_yaw_rad'])>1e-9
                    or any(not np.isfinite(np.asarray(record[k],dtype=float)).all() for k in
                    ('pre_qpos','pre_qvel','post_qpos','post_qvel','state_pre','heading_pre','pi_integral_error_post','desired_acceleration_post_enu_m_s2'))
                    or not math.isfinite(float(reward))):raise ValueError('R27 invalid/noncausal state')
            rows.append(record);ticks+=record['physics_steps'];total+=float(reward);on_record(record)
            if t or tr:outcome=terminal.terminal_class;raw=terminal.raw_termination_reason;break
        if outcome is None:raise ValueError('R27 missing natural terminal at375')
        if _weights_sha(model.policy)!=fingerprint or model.num_timesteps!=steps or not _optimizer_states_equal(optimizer,model.policy.optimizer.state_dict()):raise ValueError('R27 model/optimizer changed')
        return {'task_identity_sha256':task['identity_sha256'],'scenario_identity_sha256':row['identity_sha256'],
            'goal_yaw_rad':task['goal_yaw_rad'],'goal_yaw_mode':task['goal_yaw_mode'],'outcome':outcome,'raw_termination_reason':raw,
            'policy_steps':len(rows),'physics_ticks':ticks,'episode_return':total,'final_target_error_m':rows[-1]['distance_m'],
            'policy_sha256_before_after':fingerprint,'optimizer_unchanged':True,'training_timesteps_before_after':steps,
            'stop_metrics':stop_metrics(rows),'wall_seconds':time.monotonic()-started,'state_source':'SIM_GROUND_TRUTH_STATE',
            'coordinate_contract':'existing freejoint worldENU qpos/qvel,wxyz;not COM/SLAM/PX4 calibration'}
    finally:env.close()
