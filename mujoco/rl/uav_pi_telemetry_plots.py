"""Static scientific plots from verified recorded signals only."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from uav_bc_safety import atomic_bytes
from uav_gust_recovery_audit import sha
from uav_pi_telemetry_analysis import body_z_world

COLORS={'constant_medium':'#2878b5','gust_medium':'#d95f02'}
LABELS={'constant_medium':'Constant-Medium (success)','gust_medium':'Gust-Medium (timeout)'}


def decoration(axes):
    for ax in np.asarray(axes,dtype=object).ravel():
        ax.axvline(2,color='gray',ls=':',lw=.8);ax.axvline(4,color='gray',ls='--',lw=.8)
        ax.grid(alpha=.22);ax.set_xlim(0,15)


def plots(root,arrays,direction):
    root=Path(root);out=root/'mujoco/reports/uav_pi_internal_telemetry_figures';out.mkdir(parents=True,exist_ok=True)
    axis=np.asarray(direction,float);figures={}
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'figure.dpi':130})
    def save(fig,name):
        fig.tight_layout();path=out/name
        atomic_bytes(path,lambda f:fig.savefig(f,format='png',dpi=150,metadata={'Software':'MineUAV passive telemetry'}))
        plt.close(fig);figures[str(path.relative_to(root/'mujoco/reports'))]=sha(path)
    fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    for c,z in arrays.items():
        ft=z['physics_force_times'];f=z['physics_forces']
        ax[0].plot(ft,np.linalg.norm(f,axis=1),label=LABELS[c],color=COLORS[c])
        ax[1].plot(ft,f@axis,label=LABELS[c],color=COLORS[c])
    ax[0].set(ylabel='Force norm (N)',title='Actual applied force at every 500 Hz physics tick')
    ax[1].set(ylabel='Force-axis projection (N)',xlabel='Simulation time (s)');ax[0].legend();decoration(ax)
    save(fig,'external_force_timeline.png')

    fig,ax=plt.subplots(2,2,figsize=(11,7),sharex=True)
    for c,z in arrays.items():
        t=z['control_time_s'];e=z['control_target_m']-z['control_position_m'];v=z['control_measured_velocity_m_s']
        for a,val in zip(ax.ravel(),[np.linalg.norm(e,axis=1),np.linalg.norm(v,axis=1),e@axis,v@axis]):
            a.plot(t,val,color=COLORS[c],label=LABELS[c])
    ax[0,0].axhline(.1,color='black',ls=':',label='0.10 m threshold');ax[0,1].axhline(.15,color='black',ls=':',label='0.15 m/s threshold')
    ax[0,0].set(ylabel='Distance (m)',title='Position error magnitude');ax[0,1].set(ylabel='Speed (m/s)',title='Translational speed')
    ax[1,0].set(ylabel='Error projection (m)',xlabel='Time (s)',title='Signed error along force axis')
    ax[1,1].set(ylabel='Velocity projection (m/s)',xlabel='Time (s)',title='Signed velocity along force axis')
    ax[0,0].legend(fontsize=8);decoration(ax);save(fig,'position_velocity_response.png')

    fig,ax=plt.subplots(2,3,figsize=(12,6),sharex=True)
    for row,(c,z) in enumerate(arrays.items()):
        t=z['control_time_s']
        for k in range(3):
            ax[row,k].plot(t,z['control_commanded_velocity_m_s'][:,k],label='Command (25 Hz held)',ls='--',color='#555555')
            ax[row,k].plot(t,z['control_measured_velocity_m_s'][:,k],label='Actual (100 Hz)',color=COLORS[c])
            ax[row,k].set(title=LABELS[c]+f' / {"XYZ"[k]}',ylabel='Velocity (m/s)',xlabel='Time (s)')
    ax[0,0].legend();decoration(ax);save(fig,'desired_vs_actual_velocity.png')

    fig,ax=plt.subplots(3,1,figsize=(10,7),sharex=True)
    for c,z in arrays.items():
        t=z['control_time_s'];i=z['control_integral_after_m']
        for k,val in enumerate([np.linalg.norm(i[:,:2],axis=1),i[:,2],z['control_i_contribution_m_s2']@axis]):
            ax[k].plot(t,val,color=COLORS[c],label=LABELS[c])
    ax[0].axhline(3,color='black',ls=':',label='XY state norm cap: 3 m')
    ax[1].axhline(1.875,color='black',ls=':');ax[1].axhline(-1.875,color='black',ls=':')
    ax[0].set(ylabel='Integral XY norm (m)',title='Actual post-update PI state; neither clamp was triggered')
    ax[1].set(ylabel='Integral Z (m)');ax[2].set(ylabel='I along force axis (m/s²)',xlabel='Time (s)')
    ax[0].legend();decoration(ax);save(fig,'pi_integral_state.png')

    fig,ax=plt.subplots(3,2,figsize=(12,8),sharex=True)
    for col,(c,z) in enumerate(arrays.items()):
        t=z['control_time_s']
        for k in range(3):
            for f,label,color in [('p_contribution_m_s2_derived','P (derived)','#2878b5'),
                                  ('i_contribution_m_s2','I (recorded)','#d95f02'),
                                  ('output_after_limiting_m_s2','Committed output','#333333')]:
                ax[k,col].plot(t,z['control_'+f][:,k],label=label,color=color,lw=1.1)
            ax[k,col].set(title=LABELS[c]+f' / {"XYZ"[k]}',ylabel='Acceleration (m/s²)',xlabel='Time (s)')
    ax[0,0].legend(fontsize=8);decoration(ax);save(fig,'p_i_contributions.png')

    fig,ax=plt.subplots(3,2,figsize=(12,8),sharex=True)
    for col,(c,z) in enumerate(arrays.items()):
        t=z['control_time_s']
        for f,label,color in [('output_before_limiting_m_s2','Pre-limit','#2878b5'),('output_after_limiting_m_s2','Post-limit','#d95f02'),
                                  ('cached_qacc_world_m_s2','Cached MjData qacc (prior tick)','#444444')]:
            ax[0,col].plot(t,z['control_'+f]@axis,label=label,color=color)
        ax[1,col].plot(t,body_z_world(z['control_desired_quaternion_wxyz'])@axis,label='Desired body Z',color='#2878b5')
        ax[1,col].plot(t,body_z_world(z['control_actual_quaternion_wxyz'])@axis,label='Actual body Z',color='#d95f02')
        ax[2,col].plot(t,z['control_desired_thrust_n'],label='Desired thrust magnitude',color='#2878b5')
        ax[2,col].plot(t,z['control_commanded_thrust_n'],label='Allocator commanded thrust',ls='--',color='#d95f02')
        ax[0,col].set(title=LABELS[c],ylabel='Force-axis accel. (m/s²)');ax[1,col].set(ylabel='Body Z · force direction')
        ax[2,col].set(ylabel='Thrust command (N)',xlabel='Time (s)')
        for k in range(3):ax[k,col].legend(fontsize=7)
    decoration(ax);save(fig,'controller_output_timeline.png')

    fig,ax=plt.subplots(1,2,figsize=(11,5))
    for c,z in arrays.items():
        e=z['control_target_m']-z['control_position_m'];v=z['control_measured_velocity_m_s'];t=z['control_time_s']
        ax[0].plot(np.linalg.norm(e,axis=1),np.linalg.norm(v,axis=1),label=LABELS[c],color=COLORS[c],alpha=.8)
        ax[1].plot(e@axis,v@axis,label=LABELS[c],color=COLORS[c],alpha=.8)
        k=np.argmin(abs(t-4));ax[1].scatter(e[k]@axis,v[k]@axis,color=COLORS[c],marker='x',s=55)
    from matplotlib.patches import Rectangle
    ax[0].add_patch(Rectangle((0,0),.1,.15,color='green',alpha=.25,label='Joint threshold region'))
    ax[0].set(title='Position/speed gates are out of phase',xlabel='Distance (m)',ylabel='Speed (m/s)');ax[0].legend(fontsize=8)
    ax[1].axhline(0,color='gray',lw=.7);ax[1].axvline(0,color='gray',lw=.7)
    ax[1].set(title='Signed force-axis phase trajectory (x: t=4 s)',xlabel='Target error along force axis (m)',ylabel='Velocity along force axis (m/s)')
    for a in ax:a.grid(alpha=.2)
    save(fig,'position_velocity_phase_trajectory.png')

    fig,ax=plt.subplots(3,1,figsize=(11,7),sharex=True)
    for c,z in arrays.items():
        t=z['policy_times'];d=np.linalg.norm(z['observations'][:,:3],axis=1);v=np.linalg.norm(z['observations'][:,3:6],axis=1)
        ax[0].plot(t,d,color=COLORS[c],label=LABELS[c]);ax[1].plot(t,v,color=COLORS[c],label=LABELS[c])
        ax[2].step(t,z['success_streak'],where='post',color=COLORS[c],label=LABELS[c])
    ax[0].axhline(.1,color='black',ls=':');ax[1].axhline(.15,color='black',ls=':');ax[2].axhline(5,color='black',ls=':')
    ax[0].set(title='Unchanged 25 Hz success/termination rule',ylabel='Distance (m)');ax[1].set(ylabel='Speed (m/s)')
    ax[2].set(ylabel='Consecutive joint-gate samples',xlabel='Time (s)');ax[0].legend();decoration(ax)
    save(fig,'success_threshold_timeline.png')
    return figures
