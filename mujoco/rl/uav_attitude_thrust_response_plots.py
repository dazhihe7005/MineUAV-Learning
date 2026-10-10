"""Five static figures; recorded signals and model-derived force clearly named."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from uav_gust_recovery_audit import sha
from uav_pi_telemetry_analysis import attitude_error_degrees


def plots(reports,arrays,direction):
    reports=Path(reports);folder=reports/'uav_attitude_thrust_response_figures';folder.mkdir(exist_ok=True)
    axis=np.asarray(direction);results={}
    def save(fig,name):
        fig.tight_layout();path=folder/(name+'.png');fig.savefig(path,dpi=140);plt.close(fig)
        results[str(path.relative_to(reports))]=sha(path)
    def panels():
        fig,axs=plt.subplots(2,2,figsize=(12,7),sharex=True)
        for row,(case,_) in enumerate(arrays.items()):
            for ax in axs[row]:
                ax.set_title(case.replace('_',' ').title());ax.set_xlim(3,8)
                ax.axvline(4,c='black',ls=':',label='Gust off' if case.startswith('gust') else 't4 reference; force persists')
                ax.grid(alpha=.25);ax.set_xlabel('simulation time, s')
        return fig,axs
    fig,axs=panels()
    for row,(case,(z,d)) in enumerate(arrays.items()):
        t=z['control_time_s'];actual=d['actual_body_z_world'];desired=z['control_desired_rotation_body_to_world'][:,:,2]
        for signal,label in ((desired,'desired'),(actual,'actual recorded')):
            axs[row,0].plot(t,np.degrees(np.arctan2(signal@axis,signal[:,2])),label=label)
        axs[row,0].set_ylabel('force-axis body-Z tilt, deg');axs[row,0].legend(fontsize=8)
        axs[row,1].plot(t,attitude_error_degrees(z['control_actual_quaternion_wxyz'],z['control_desired_quaternion_wxyz']),label='quaternion tracking error')
        axs[row,1].set_ylabel('attitude error, deg');axs[row,1].legend(fontsize=8)
    save(fig,'desired_vs_actual_attitude')
    fig,axs=panels()
    for row,(case,(z,d)) in enumerate(arrays.items()):
        t=z['control_time_s'];thrust=np.linalg.norm(d['rotor_force_world_n'],axis=1)/7
        for key,label in [('control_p_contribution_m_s2_derived','P (derived)'),('control_i_contribution_m_s2','I (recorded)'),('control_output_after_limiting_m_s2','PI command')]:
            axs[row,0].plot(t,z[key]@axis,label=label)
        axs[row,0].set_ylabel('force-axis acceleration demand, m/s²');axs[row,0].legend(fontsize=8)
        axs[row,1].plot(t,thrust*(z['control_desired_rotation_body_to_world'][:,:,2]@axis),label='desired direction × allocated T/m')
        axs[row,1].plot(t,d['rotor_force_world_n']@axis/7,label='model-derived actual direction × T/m')
        axs[row,1].set_ylabel('force-axis rotor contribution, m/s²');axs[row,1].legend(fontsize=8)
    save(fig,'pi_output_vs_attitude_response')
    fig,axs=panels()
    for row,(case,(z,d)) in enumerate(arrays.items()):
        t=z['control_time_s']
        axs[row,0].plot(t,z['control_desired_thrust_n'],label='desired scalar T')
        axs[row,0].plot(t,np.linalg.norm(d['rotor_force_world_n'],axis=1),ls='--',label='model-derived rotor resultant')
        axs[row,0].set_ylabel('scalar thrust magnitude, N');axs[row,0].legend(fontsize=8)
        for signal,label,style in [(z['control_output_after_limiting_m_s2'],'PI requested','-'),(d['com_acceleration_world_m_s2'],'model-derived COM','-'),
                                   (d['origin_acceleration_world_m_s2'],'model-derived origin','--'),(z['control_external_force_world_n']/7,'external force/m',':')]:
            axs[row,1].plot(t,signal@axis,ls=style,label=label)
        axs[row,1].set_ylabel('force-axis acceleration, m/s²');axs[row,1].legend(fontsize=8)
    save(fig,'thrust_and_acceleration_response')
    fig,axs=panels()
    for row,(case,(z,d)) in enumerate(arrays.items()):
        t=z['control_time_s'];v=z['control_measured_velocity_m_s']
        axs[row,0].plot(t,v@axis,label='actual force-axis velocity');axs[row,0].plot(t,z['control_commanded_velocity_m_s']@axis,label='desired force-axis velocity')
        axs[row,0].plot(t,np.linalg.norm(v,axis=1),label='actual 3D speed');axs[row,0].set_ylabel('velocity/speed, m/s');axs[row,0].legend(fontsize=8)
        axs[row,1].plot(t,np.einsum('ij,ij->i',v,z['control_output_after_limiting_m_s2']),label='v · PI command')
        axs[row,1].plot(t,np.einsum('ij,ij->i',v,d['origin_acceleration_world_m_s2']),label='v · current model acceleration')
        axs[row,1].axhline(0,c='gray',lw=.8);axs[row,1].set_ylabel('speed-energy derivative proxy, m²/s³');axs[row,1].legend(fontsize=8)
    save(fig,'velocity_braking_timeline')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,(case,(z,d)) in zip(axs,arrays.items()):
        t=z['control_time_s'];error=(z['control_target_m']-z['control_position_m'])@axis;v=z['control_measured_velocity_m_s']@axis
        ax.plot(error,v,lw=.7,alpha=.6);sc=ax.scatter(error,v,c=t,vmin=0,vmax=15,s=5,cmap='viridis')
        ax.axvspan(-.1,.1,color='green',alpha=.08);ax.axhspan(-.15,.15,color='green',alpha=.08)
        ax.set_title(case.replace('_',' ').title());ax.set_xlabel('force-axis position error, m');ax.set_ylabel('force-axis velocity, m/s');ax.grid(alpha=.2)
    fig.colorbar(sc,ax=axs,label='simulation time, s')
    fig.suptitle('Projected phase response: shaded bands are necessary, NOT sufficient, for 3D success',fontsize=10)
    path=folder/'position_velocity_phase.png';fig.savefig(path,dpi=140);plt.close(fig);results[str(path.relative_to(reports))]=sha(path)
    return results
