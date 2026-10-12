"""Static, offline plots of the one saved Phase2F PX4/MuJoCo flight."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .analyze_yaw import simulation_time
from .diagnostics import sha256


def _save(fig, directory, name):
    path = directory / name
    fig.savefig(path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    return sha256(path)


def _mark_step(axes, step):
    if step is None:
        return
    for ax in axes:
        ax.axvline(step, color='0.45', linestyle=':', linewidth=1,
                   label='commanded yaw step' if ax is axes[0] else None)


def write_figures(directory, trace, raw, aux, report):
    """Plot saved signals; the supplied PX4 yaw estimate is already resampled."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    t = np.asarray(trace['t'], dtype=float)
    step = (raw.get('phase2') or {}).get('actual_step_time_s')
    made = {}

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.step(t, np.degrees(trace['desired']), where='post', label='PX4 desired NED yaw')
    ax.plot(t, np.degrees(trace['estimate']), '.', ms=2, label='PX4 estimated NED yaw')
    ax.plot(t, np.degrees(trace['true']), label='MuJoCo true NED yaw', linewidth=1.3)
    _mark_step([ax], step)
    ax.set(xlabel='Simulation time (s)', ylabel='NED yaw (deg)', title='Commanded, estimated, and physical yaw')
    ax.grid(alpha=.25); ax.legend(loc='best')
    made['target_estimated_true_yaw.png'] = _save(fig, directory, 'target_estimated_true_yaw.png')

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(t, trace['rate'], label='MuJoCo body FRD-z yaw rate', linewidth=1.3)
    rate = aux.get('vehicle_rates_setpoint', {})
    if 'timestamp' in rate and 'yaw' in rate:
        ax.step(simulation_time(rate['timestamp']), rate['yaw'], where='post',
                label='PX4 yaw-rate setpoint', linewidth=1.1)
    _mark_step([ax], step)
    ax.set(xlabel='Simulation time (s)', ylabel='Yaw rate (rad/s)', title='Yaw-rate response')
    ax.grid(alpha=.25); ax.legend(loc='best')
    made['yaw_rate_response.png'] = _save(fig, directory, 'yaw_rate_response.png')

    torque = aux.get('vehicle_torque_setpoint', {})
    alloc = aux.get('control_allocator_status', {})
    fig, ax = plt.subplots(figsize=(9, 4))
    if 'timestamp' in torque and 'xyz[2]' in torque:
        ax.step(simulation_time(torque['timestamp']), torque['xyz[2]'], where='post',
                label='PX4 requested yaw torque', linewidth=1.2)
    if 'timestamp' in alloc and 'unallocated_torque[2]' in alloc:
        ax.step(simulation_time(alloc['timestamp']), alloc['unallocated_torque[2]'], where='post',
                label='PX4 unallocated yaw torque', linewidth=1.2)
    _mark_step([ax], step)
    ax.set(xlabel='Simulation time (s)', ylabel='PX4 normalized torque units',
           title='Requested and unallocated yaw torque (not N·m)')
    ax.grid(alpha=.25); ax.legend(loc='best')
    made['requested_unallocated_yaw_torque.png'] = _save(fig, directory, 'requested_unallocated_yaw_torque.png')

    fig, ax = plt.subplots(figsize=(9, 4))
    for i in range(4):
        ax.step(t, np.asarray(trace['motors'])[:, i], where='post', linewidth=1,
                label=f'Motor {i+1}')
    _mark_step([ax], step)
    ax.set(xlabel='Simulation time (s)', ylabel='Normalized motor output',
           title='Executed four-motor commands')
    ax.set_ylim(-.03, 1.03); ax.grid(alpha=.25); ax.legend(loc='best')
    made['four_motor_outputs.png'] = _save(fig, directory, 'four_motor_outputs.png')

    position = np.asarray(trace['position'], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, position[:, 2], label='MuJoCo body-origin ENU height')
    axes[0].axhline(2.051, color='0.5', linestyle='--', label='Post-ascend body-origin target')
    axes[0].set_ylabel('ENU height (m)'); axes[0].legend(loc='best')
    axes[1].plot(t, np.linalg.norm(position[:, :2], axis=1), label='Horizontal offset from origin')
    axes[1].set(xlabel='Simulation time (s)', ylabel='XY offset (m)')
    _mark_step(axes, step)
    for ax in axes: ax.grid(alpha=.25)
    made['height_xy_position.png'] = _save(fig, directory, 'height_xy_position.png')

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=False,
                             constrained_layout=True)
    for key, label, ax in (('local', 'heading_good_for_control', axes[0]),
                            ('flags', 'cs_mag_aligned_in_flight', axes[1])):
        topic = trace.get(key, {})
        if 'timestamp' in topic and label in topic:
            ax.step(simulation_time(topic['timestamp']), topic[label], where='post',
                    label=label)
        ax.set(ylim=(-.1, 1.1), ylabel='uORB flag', xlabel='Simulation time (s)')
        ax.legend(loc='best')
    queries = (raw.get('runtime_measurements') or {}).get('queries') or []
    if queries:
        start = np.asarray([q['start_wall_s'] for q in queries], dtype=float)
        duration_ms = np.asarray([(q['end_wall_s']-q['start_wall_s'])*1000 for q in queries])
        axes[2].plot(start, duration_ms, '.', ms=3, label='Safety-query response')
        axes[2].axhline(150, linestyle='--', color='tab:red', label='150 ms deadline')
    axes[2].set(xlabel='Bridge wall-clock time (s)', ylabel='Query response (ms)')
    axes[2].legend(loc='best')
    _mark_step(axes[:2], step)
    for ax in axes: ax.grid(alpha=.25)
    fig.suptitle('EKF status publications and safety-query timing (distinct clocks)')
    made['ekf_health_safety_query_timing.png'] = _save(fig, directory,
                                                         'ekf_health_safety_query_timing.png')
    return made
