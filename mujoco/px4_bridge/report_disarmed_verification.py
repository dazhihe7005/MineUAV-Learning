"""Offline-only Phase2E report/figures from the sole saved raw SITL run."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from .analyze_disarmed_verification import assess_run,_stats
from .diagnostics import atomic_json,sha256
from .simulation_runner import streaming_runtime_identity
from .streaming_diagnostics import StreamingDiagnostics
from .allocation_runner import allocation_status

ROOT=Path(__file__).resolve().parents[2]
RAW=ROOT/'mujoco/reports/px4_disarmed_verification_parts/disarmed_01/smoke_summary.json'
OUT=ROOT/'mujoco/reports/px4_mujoco_disarmed_verification.json'
FIG=ROOT/'mujoco/reports/figures/px4_disarmed_verification'
PROTECTED_MODELS={
    'mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt':'42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9',
    'mujoco/rl/models/uav_bc_mlp_seed0.pt':'7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c',
    'mujoco/rl/models/uav_bc_yaw_augmented_seed0.pt':'d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce'}
ORIGINAL_PX4_BINARY=Path('/home/keyousuke/PX4-Autopilot/build/px4_sitl_default/bin/px4')
ORIGINAL_PX4_SHA256='4f491ce85d6c79fbd5713894112c8911903fcefd1bf45b7da3579cd1812beac4'


def console_evidence(raw):
    text=raw['px4_console'];parser=StreamingDiagnostics();topics={};rows=[]
    for i in range(0,len(text),1024):
        for event in parser.feed(text[i:i+1024]):
            if event['kind']=='topic':topics[event['topic']]=topics.get(event['topic'],0)+1
            elif event['kind']=='yaw_row':rows.append(event['value'])
    expected=raw['phase2']['allocation_fix']['expected_yaw_row']
    return {'console_chars':len(text),'topic_record_counts':topics,
        'allocator_row_records':len(rows),
        'allocator_row_all_match':bool(rows) and all(allocation_status('MineUAV yaw axis: '+row,expected) for row in rows),
        'sensor_combined_listener_seen':bool(re.search(r'(?m)^TOPIC: sensor_combined\b',text)),
        'vehicle_magnetometer_listener_seen':bool(re.search(r'(?m)^TOPIC: vehicle_magnetometer\b',text)),
        'px4_console_warnings':{'SIM_GZ_EN_parameter_not_found':'Parameter SIM_GZ_EN not found' in text,
            'logger_subscription_capacity':'Too many subscriptions, failed to add' in text},
        'parser_replay_peak_chars':parser.peak_buffer_chars}


def write_figures(raw,acceptance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True,exist_ok=True)
    q=raw['runtime_measurements']['queries'];chunks=raw['runtime_measurements']['chunks']
    starts=np.array([v['start_wall_s'] for v in q]);dur=np.array([v['end_wall_s']-v['start_wall_s'] for v in q])
    paths=[]
    def done(name):
        plt.tight_layout();dest=FIG/name;plt.savefig(dest,dpi=160);plt.close();paths.append(str(dest.relative_to(ROOT)))
    plt.figure(figsize=(8,3.5));plt.plot(starts,dur*1000,'o-',label='Live query response')
    plt.ylim(0,max(8,float(dur.max()*1200)))
    plt.text(.98,.94,'150 ms safety deadline (off scale)',ha='right',va='top',
        transform=plt.gca().transAxes,color='red')
    plt.xlabel('Bridge host wall time (s)');plt.ylabel('Response (ms)');plt.legend(loc='upper left');plt.grid(alpha=.2)
    done('live_query_response.png')
    plt.figure(figsize=(8,3.5));plt.plot(starts[1:],np.diff(starts)*1000,'o-',label='Live query interval')
    plt.axhline(500,color='green',linestyle='--',label='500 ms target')
    plt.axhline(650,color='red',linestyle='--',label='650 ms service bound')
    plt.xlabel('Bridge host wall time (s)');plt.ylabel('Interval (ms)');plt.legend();plt.grid(alpha=.2)
    done('live_query_interval.png')
    plt.figure(figsize=(8,3.5));plt.scatter([v['wall_s'] for v in chunks],
        [v['parse_validate_s']*1000 for v in chunks],s=12,alpha=.65)
    plt.xlabel('Bridge host wall time (s)');plt.ylabel('Parse + safety validation (ms)');plt.grid(alpha=.2)
    done('live_incremental_parse_cost.png')
    plt.figure(figsize=(8,3.5));plt.plot(starts,[v['start_clock_usec']/1e6-1 for v in q],'o-',label='MuJoCo simulated')
    plt.plot([0,raw['wall_seconds']],[0,raw['wall_seconds']],linestyle='--',label='1:1 wall time')
    plt.xlabel('Bridge host wall time (s)');plt.ylabel('Simulation time (s)');plt.legend();plt.grid(alpha=.2)
    done('live_sim_vs_wall.png')
    return paths


def build(rawpath=RAW,external_peak_rss_kib=None):
    raw=json.loads(rawpath.read_text());assessment=assess_run(raw)
    if assessment['overall']!='PASS':raise ValueError('saved SITL evidence failed: '+','.join(assessment['first_failed_checks']))
    old=json.loads((ROOT/'mujoco/reports/px4_mujoco_query_scheduling.json').read_text())
    ulogs=sorted(rawpath.parent.glob('log/**/*.ulg'))
    if len(ulogs)!=1:raise ValueError('exactly one owned PX4 ULog required')
    console=console_evidence(raw)
    if console['allocator_row_records']<len(raw['runtime_measurements']['queries']) or not console['allocator_row_all_match']:
        raise ValueError('saved allocation rows do not all match physical model')
    metrics=raw['runtime_measurements'];queries=metrics['queries'];fresh=raw['phase2']['telemetry_fix']
    receipt=[e for e in fresh['events'] if e['kind']=='receipt']
    publications={}
    for name in ('vehicle_local_position','estimator_status_flags','vehicle_status'):
        values=[e['publication_usec'] for e in receipt if e['topic']==name]
        publications[name]={'receipts':len(values),'distinct_publications':len(set(values)),
            'first_publication_usec':min(values),'last_publication_usec':max(values),
            'strictly_nondecreasing':all(a<=b for a,b in zip(values,values[1:]))}
    if any(p['receipts']!=len(queries) or not p['strictly_nondecreasing'] for p in publications.values()):
        raise ValueError('source receipt identity/cadence mismatch')
    model_hashes={name:sha256(ROOT/name) for name in PROTECTED_MODELS}
    original_binary_hash=sha256(ORIGINAL_PX4_BINARY)
    if model_hashes!=PROTECTED_MODELS or original_binary_hash!=ORIGINAL_PX4_SHA256:
        raise ValueError('protected canonical models or original PX4 binary changed')
    current_hashes=streaming_runtime_identity()
    if current_hashes!=raw['streaming_runtime_hashes']:
        raise ValueError('running Bridge source no longer matches sole live SITL run')
    report={
        'phase':'PX4-MuJoCo Phase2E Disarmed Runtime Integration Verification',
        'date':'2026-10-12','branch':'feat/px4-mujoco-disarmed-verification',
        'base_commit':'c9174292e3e75cfab43706bc63b5b53b689a136c',
        'experiment':{'authorized_sitl_runs':1,'armed_runs':0,'actual_sitl_runs':1,
            'mode':'allocation_runner --mode disarmed','simulation_duration_s':raw['simulation_seconds'],
            'bridge_wall_duration_s':raw['wall_seconds'],'physics_hz':raw['physics_hz'],
            'sensor_hz':raw['sensor_hz'],'workers':raw['workers'],
            'fresh_runtime':True,'no_flight_command':raw['flight_command_sent'] is False},
        'acceptance':assessment,
        'status':{'bridge_disarmed_integration':assessment['overall'],
            'live_queries':'PASS' if assessment['checks']['query_deadlines']=='PASS' else 'FAIL',
            'allocator_yaw_axis':'PASS' if assessment['checks']['allocator_yaw_row']=='PASS' else 'FAIL',
            'ekf_final_inflight_alignment':'NOT TESTED','yaw_flight_control':'NOT TESTED',
            'next_restricted_yaw_test_communication_prerequisites':'PASS' if assessment['overall']=='PASS' else 'BLOCKED',
            'phase3_entry':'NOT TESTED'},
        'px4':raw['px4'],'mujoco_version':raw['mujoco_version'],
        'raw_evidence':{'summary_path':str(rawpath.relative_to(ROOT)),'summary_sha256':sha256(rawpath),
            'console_path':str((rawpath.parent/'px4_console.bin').relative_to(ROOT)),
            'console_sha256':sha256(rawpath.parent/'px4_console.bin'),
            'ulog_path':str(ulogs[0].relative_to(ROOT)),'ulog_sha256':sha256(ulogs[0]),
            'runtime_source_sha256':current_hashes,'raw_local_only':True},
        'communications':{'sent':raw['messages_sent'],'received':raw['messages_received'],
            'stale_bootstrap_messages_discarded':raw['stale_bootstrap_messages_discarded'],
            'first_actuator_at_sim_s':raw['first_actuator_at_s'],
            'actuator_interval_sim_usec':raw['actuator_interval_usec'],
            'px4_source_publications':publications,'console':console,
            'incoming_px4_command_long_note':'One COMMAND_LONG was received from PX4 by the Bridge; Bridge sent only HIL_SENSOR and HIL_GPS. Incoming message payload was not logged.'},
        'live_timing':{'query_response_wall_s':metrics['samples']['query_response_seconds'],
            'query_start_interval_wall_s':assessment['query_interval_wall_s'],
            'actuator_receipt_interval_wall_s':metrics['samples']['actuator_interval_wall_seconds'],
            'incremental_parse_and_validation_s':metrics['samples']['parse_validate_seconds'],
            'console_service_s':metrics['samples']['console_service_seconds'],
            'scheduler_max_lateness_s':metrics['scheduler']['max_lateness_s'],
            'scheduled_query_starts':metrics['scheduler']['starts'],
            'completed_queries':len(queries),'row_valid_at_completions':metrics['row_valid_at_completions'],
            'max_pending_queries':metrics['pending_query_max'],
            'monitor_fault':fresh['fault'],'recording_dropped':metrics['dropped'],
            'hard_deadlines_s':{'query_response':.15,'actuator_watchdog':.25,'query_service':.65},
            'parse_semantics':metrics['measurement_semantics'],
            'query_latency_first_half_mean_s':float(np.mean([v['end_wall_s']-v['start_wall_s'] for v in queries[:10]])),
            'query_latency_second_half_mean_s':float(np.mean([v['end_wall_s']-v['start_wall_s'] for v in queries[10:]]))},
        'bounded_storage':{'parser_processed_chars':metrics['parser_processed_chars'],
            'parser_peak_buffer_chars':metrics['parser_peak_buffer_chars'],
            'recent_console_peak_chars':metrics['recent_peak_chars'],
            'console_archive_bytes':raw['console_archive']['total_bytes'],
            'console_archive_limit_bytes':raw['console_archive']['lifetime_limit_bytes'],
            'max_queued_console_write_bytes':metrics['write_peak_bytes'],
            'measurement_sample_limit':metrics['sample_limit'],'measurement_query_limit':metrics['query_limit']},
        'safety':{'flight_command_sent':raw['flight_command_sent'],
            'four_motor_ctrl_zero_on_exit':raw['zero_control_on_stop'],
            'all_recorded_motor_ctrl_zero':all(r['ctrl_omega_squared']==[0,0,0,0] for r in raw['records']),
            'px4_exit_code':raw['px4_exit_code'],'exception':raw['exception'],
            'last_telemetry_decision':fresh['last_decision'],
            'final_physical_state':raw['final_physical_state'],
            'no_watchdog_or_safety_stop':fresh['fault'] is None},
        'resources':{'bridge_peak_rss_bytes':raw['bridge_peak_rss_bytes'],
            'px4_sampled_peak_rss_bytes':raw['px4_sampled_peak_rss_bytes'],
            'system_min_available_bytes':raw['system_min_available_bytes'],
            'bridge_process_cpu_seconds':raw['bridge_cpu_seconds'],
            'px4_sampled_cpu_seconds':metrics['px4_sampled_cpu_seconds'],
            'px4_cpu_sample_count':metrics['px4_cpu_sample_count'],
            'external_time_peak_rss_bytes':None if external_peak_rss_kib is None else external_peak_rss_kib*1024,
            'oom_observed_in_owned_run':False,'swap_changed':False},
        'historical_comparison':{'phase2c_live_failed_query_service':old['phase2c_stopping_evidence'],
            'phase2d_offline_incremental_265kb_us':next(x['incremental_us'] for x in old['parsing_benchmark'] if x['history_chars']==265803),
            'phase2d_offline_legacy_265kb_us':next(x['old_us'] for x in old['parsing_benchmark'] if x['history_chars']==265803),
            'comparison_boundary':'Phase2D microbenchmark used synthetic 94-character deltas; Phase2E live parse+validation includes different chunk sizes and scheduler load. Their latency numbers are not like-for-like.'},
        'preservation':{'history_hashes_before':raw['history_hashes_before'],
            'history_hashes_after':raw['history_hashes_after'],'unchanged':raw['history_unchanged'],
            'isolated_binary_hash_after':sha256(Path(raw['px4']['binary'])),
            'isolated_binary_unchanged_after':sha256(Path(raw['px4']['binary']))==raw['px4']['binary_sha256'],
            'original_px4_binary_sha256':original_binary_hash,
            'original_px4_binary_unchanged':original_binary_hash==ORIGINAL_PX4_SHA256,
            'protected_models_sha256':model_hashes,
            'protected_models_unchanged':model_hashes==PROTECTED_MODELS},
        'limitations':['Single16s unarmed session; no takeoff, yaw step, or final inflight magnetic alignment.',
            'QUERY and parse timings are sampled inside the Python Bridge, not PX4 internal profiler measurements.',
            'PX4 CPU and RSS are sampled at50Hz; between-sample peaks are not observable.',
            'Incoming PX4 COMMAND_LONG payload was not archived; direction is known from wire counters.',
            'Console shows SIM_GZ_EN parameter-not-found and logger subscription-capacity warnings; no safety stop occurred.',
            'Nominal local SITL only; no hard real-time or hardware claim.']}
    report['figures']=write_figures(raw,assessment)
    atomic_json(OUT,report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,default=RAW)
    p.add_argument('--external-peak-rss-kib',type=int)
    args=p.parse_args();out=build(args.raw,args.external_peak_rss_kib)
    print(json.dumps({'overall':out['acceptance']['overall'],'raw_sha256':out['raw_evidence']['summary_sha256'],
        'report':str(OUT),'figures':out['figures']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
