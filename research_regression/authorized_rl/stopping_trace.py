"""Instrument existing plant without altering control, rewards or observations."""
import copy,math,time
import numpy as np
import torch
from lidar_integration.navigation_env import LidarNavigationEnv
from lidar_integration.ppo_preflight import HeadingObservation
from lidar_integration.ppo_scenarios import build_ppo_scene
from lidar_integration.ppo_training_runner import _weights_sha
from lidar_integration.ppo_formal_validation import _optimizer_states_equal
from lidar_integration.terminal_accounting import classify_terminal
from research_ops.environment import enforce_resources
from .runtime import ROOT


def stop_metrics(rows):
    near=fast=joint=consecutive=maximum=0;last=-1.;first_near=first_five=None
    for row in rows:
        d,v,t=(row[k] for k in ('distance_m','speed_m_s','post_time_s'))
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (d,v,t)) or d<0 or v<0 or t<=last:
            raise ValueError('nonfinite/invalid/noncausal stopping samples')
        last=t;n=d<.1;j=n and v<.15
        near+=int(n);fast+=int(n and not j);joint+=int(j)
        consecutive=consecutive+1 if j else 0;maximum=max(maximum,consecutive)
        if n and first_near is None:first_near=t
        if consecutive>=5 and first_five is None:first_five=t
    return {'near_goal_samples':near,'near_goal_fast_samples':fast,'joint_samples':joint,
        'max_consecutive_joint_samples':maximum,'first_near_time_s':first_near,'first_five_joint_time_s':first_five}


def collect_trace(model,row,upstream,*,max_steps=375,guard=lambda:None,on_record=lambda record:None):
    if type(max_steps) is not int or not 1<=max_steps<=375:raise ValueError('invalid bounded diagnostic steps')
    if row['split']!='validation' or row['layout']!='empty':raise ValueError('only frozen empty Validation diagnostic')
    scene=build_ppo_scene(row)
    if scene.geometry_hash!=row['obstacle_configuration_hash']:raise ValueError('diagnostic geometry differs')
    env=HeadingObservation(LidarNavigationEnv(row['layout'],row['scene_seed'],upstream,scene=scene))
    before=_weights_sha(model.policy);optimizer=copy.deepcopy(model.policy.optimizer.state_dict());steps=model.num_timesteps
    traces=[];reward_sum=0.;ticks=0;started=time.monotonic()
    try:
        obs,info=env.reset(seed=row['scene_seed'],options={'start_seed':row['start_seed'],'target_seed':row['target_seed'],
            'start_position':row['start_xyz'],'target_position':row['target_xyz'],'start_yaw_rad':row['start_yaw_rad']})
        plant=env.unwrapped;outcome='probe_cutoff';raw_reason=None
        if info['obstacle_configuration_hash']!=row['obstacle_configuration_hash']:raise ValueError('reset geometry differs')
        for step in range(1,max_steps+1):
            guard();enforce_resources(ROOT)
            pre=float(plant.data.time);pre_qpos=plant.data.qpos.copy();pre_qvel=plant.data.qvel.copy()
            pre_state=np.asarray(obs['state']).copy();pre_heading=np.asarray(obs['heading']).copy();age=float(obs['lidar_age'][0])
            with torch.inference_mode():
                tensor,_=model.policy.obs_to_tensor(obs)
                raw_mean=model.policy.get_distribution(tensor).distribution.mean[0].cpu().numpy().copy()
                action,_=model.predict(obs,deterministic=True)
            if (np.asarray(action).shape!=(4,) or raw_mean.shape!=(4,) or not np.isfinite(raw_mean).all()
                or not np.isfinite(action).all() or np.any(np.abs(action)>1)):
                raise ValueError('invalid deterministic diagnostic action')
            obs,reward,terminated,truncated,info=env.step(action)
            c=classify_terminal(info,terminated,truncated)
            record={'step':step,'pre_time_s':pre,'post_time_s':float(plant.data.time),
                'pre_qpos':pre_qpos.tolist(),'pre_qvel':pre_qvel.tolist(),'state_pre':pre_state.tolist(),'heading_pre':pre_heading.tolist(),
                'lidar_age_pre_s':age,'raw_action_mean':raw_mean.tolist(),'action_clipped':np.asarray(action).tolist(),
                'velocity_command_enu_m_s':np.asarray(info['velocity_command_m_s']).tolist(),'yaw_rate_command_enu_rad_s':float(info['yaw_rate_command_rad_s']),
                'post_qpos':plant.data.qpos.tolist(),'post_qvel':plant.data.qvel.tolist(),
                'distance_m':float(info['distance_m']),'speed_m_s':float(info['speed_m_s']),
                'physics_steps':int(info['physics_steps']),'lidar_source_time_s':float(info['lidar_source_time_s']),
                'lidar_age_post_s':float(info['lidar_age_s']),'reward':float(reward),'reward_breakdown':info['reward_breakdown'],
                'terminal_class':c.terminal_class,'raw_termination_reason':c.raw_termination_reason,
                'pi_integral_error_post':plant.velocity_controller.integral_error.tolist(),
                'desired_acceleration_post_enu_m_s2':plant.velocity_controller.last_desired_acceleration_world.tolist(),
                'contact_pairs':info.get('contact_pairs',[])}
            if (record['post_time_s']<=pre or record['lidar_source_time_s']>record['post_time_s']+1e-9
                or age<0 or any(not np.isfinite(np.asarray(record[k],dtype=float)).all() for k in
                ('pre_qpos','pre_qvel','state_pre','heading_pre','post_qpos','post_qvel','pi_integral_error_post',
                 'desired_acceleration_post_enu_m_s2','velocity_command_enu_m_s'))
                or not math.isfinite(float(reward))):raise ValueError('invalid/noncausal physical trace')
            traces.append(record);on_record(record);ticks+=record['physics_steps'];reward_sum+=float(reward)
            if terminated or truncated:
                outcome=c.terminal_class;raw_reason=c.raw_termination_reason;break
        if (_weights_sha(model.policy)!=before or model.num_timesteps!=steps
            or not _optimizer_states_equal(optimizer,model.policy.optimizer.state_dict())):
            raise ValueError('read-only diagnostic changed policy/optimizer/training steps')
        return {'scenario_identity_sha256':row['identity_sha256'],'outcome':outcome,'raw_termination_reason':raw_reason,
            'policy_steps':len(traces),'physics_ticks':ticks,'episode_return':reward_sum,'final_target_error_m':traces[-1]['distance_m'],
            'policy_sha256_before_after':before,'optimizer_unchanged':True,'training_timesteps_before_after':steps,
            'stop_metrics':stop_metrics(traces),'trace':traces,'wall_seconds':time.monotonic()-started,
            'coordinate_contract':'existing environment freejoint world ENU position/linear velocity, qpos quaternion wxyz; not SLAM/PX4/COM calibration'}
    finally:env.close()
