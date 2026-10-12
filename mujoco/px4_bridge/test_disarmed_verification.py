"""Offline tests only; never launch PX4 or a simulation episode."""
from pathlib import Path
import pytest
from .test_interfaces import module
from .test_yaw_validation import safety_console


def measurements():
    q=module('query_scheduler')
    assert hasattr(q,'RuntimeMeasurements'), 'bounded passive measurements missing'
    return q.RuntimeMeasurements()


def test_measurements_keep_storage_bounded_and_report_dropped_samples():
    m=measurements()
    for i in range(m.LIMIT+3):m.sample('parse_validate_seconds',.001)
    assert len(m.samples['parse_validate_seconds'])==m.LIMIT
    assert m.dropped['parse_validate_seconds']==3
    assert m.summary()['dropped']['parse_validate_seconds']==3


def test_measurements_do_not_turn_missing_timings_into_zero():
    m=measurements()
    assert m.summary()['samples']=={}


def test_console_service_records_real_start_to_completion_not_topic_receipt(tmp_path):
    q=module('query_scheduler')
    p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    chunks=[];clock=[0.]
    archive=q.ConsoleArchive(tmp_path/'console.bin')
    s=q.ConsoleService(p,archive,lambda:chunks.pop(0) if chunks else b'',lambda b:len(b))
    s.wall_now=lambda:clock[0]
    s.service(7000000,0.)
    chunks.append(safety_console(7020000).encode());clock[0]=.02
    s.service(7020000,.02)
    assert p.monitor.fault is None and p.monitor.query is None
    assert hasattr(s,'measurements'),'live query completion record missing'
    assert s.measurements.queries==[{'start_clock_usec':7000000,'start_wall_s':0.,
        'end_clock_usec':7020000,'end_wall_s':.02}]
    assert s.measurements.summary()['samples']['query_response_seconds']['max']==pytest.approx(.02)
    archive.close()


def test_console_write_peak_counts_a_fully_flushed_batch(tmp_path):
    q=module('query_scheduler')
    p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    archive=q.ConsoleArchive(tmp_path/'console.bin')
    s=q.ConsoleService(p,archive,lambda:b'',lambda b:len(b))
    s.service(7000000,0.)
    assert s.measurements.write_peak_bytes==len(p.queries().encode())
    archive.close()


def test_console_write_peak_also_counts_diagnostic_batches(tmp_path):
    q=module('query_scheduler')
    p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    archive=q.ConsoleArchive(tmp_path/'console.bin')
    s=q.ConsoleService(p,archive,lambda:b'',lambda b:len(b))
    assert s.diagnostic(b'ver all\n',0.)
    assert s.measurements.write_peak_bytes==len(b'ver all\n')
    archive.close()


def test_missing_px4_cpu_proc_samples_are_not_reported_as_zero():
    m=measurements()
    assert m.summary()['px4_sampled_cpu_seconds'] is None
    assert m.summary()['px4_cpu_sample_count']==0
    m.process_cpu(None);m.process_cpu(1.25)
    assert m.summary()['px4_sampled_cpu_seconds']==1.25
    assert m.summary()['px4_cpu_sample_count']==1


def test_passive_recording_cannot_clear_a_late_query_fault(tmp_path):
    q=module('query_scheduler')
    p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    chunks=[];archive=q.ConsoleArchive(tmp_path/'console.bin')
    s=q.ConsoleService(p,archive,lambda:chunks.pop(0) if chunks else b'',lambda b:len(b))
    s.service(7000000,0.)
    chunks.append(safety_console(7152000).encode())
    s.service(7152000,.152)
    assert p.monitor.fault=='query response timeout/truncated topic'
    assert hasattr(s,'measurements'),'bounded measurement missing'
    assert s.measurements.queries==[]
    archive.close()


def assessor():
    assert (Path(__file__).parent/'analyze_disarmed_verification.py').exists(),'offline evidence assessor missing'
    return module('analyze_disarmed_verification').assess_run


def fixture():
    hashes=module('simulation_runner').streaming_runtime_identity()
    return {'complete':True,'kind':'disarmed_communication_smoke','simulation_seconds':16.,
        'final_physical_state':{'time_s':16.},'flight_command_sent':False,'exception':None,
        'px4_exit_code':0,'zero_control_on_stop':[0.,0.,0.,0.],'history_unchanged':True,
        'history_hashes_before':{'model':'abc'},'history_hashes_after':{'model':'abc'},
        'messages_sent':{'HIL_SENSOR':4000,'HIL_GPS':160},
        'messages_received':{'HIL_ACTUATOR_CONTROLS':4000},'stale_bootstrap_messages_discarded':0,
        'streaming_runtime_hashes':hashes,
        'phase2':{'active':False,'actual_step_time_s':None,
            'allocation_fix':{'expected_yaw_row':[-.017207708,.017207708,-.017207708,.017207708]},
            'telemetry_fix':{'fault':None,'last_decision':{'safe_stop':False,'clock_usec':17000000,'wall_s':16.},'events':[
                {'kind':'query','wall_s':6.+i*.5,'clock_usec':7000000+i*500000} for i in range(20)]}},
        'runtime_measurements':{'samples':{'query_response_seconds':{'n':20,'max':.02},
            'actuator_interval_wall_seconds':{'n':3999,'max':.008},
            'parse_validate_seconds':{'n':20,'max':.001}},'dropped':{},
            'queries':[{'start_clock_usec':7000000+i*500000,'start_wall_s':6.+i*.5,
                'end_clock_usec':7020000+i*500000,'end_wall_s':6.02+i*.5} for i in range(20)],
            'scheduler':{'starts':20},'row_valid_at_completions':20,'actuator_receipts':4000,
            'parser_processed_chars':1000,'parser_peak_buffer_chars':1024,'recent_peak_chars':2048},
        'system_min_available_bytes':8*1024**3,
        'wall_seconds':16.5,
        'records':[{'time_s':16.,'ctrl_omega_squared':[0.,0.,0.,0.]}]}


def test_assessor_all_evidence_required_and_no_arm_path():
    result=assessor()(fixture())
    assert result['overall']=='PASS'
    assert result['checks']['no_arm']=='PASS'


def test_runtime_identity_covers_passive_measurement_source():
    from .simulation_runner import streaming_runtime_identity
    from .diagnostics import sha256
    path=Path(__file__).parent/'runtime_measurements.py'
    assert streaming_runtime_identity()['runtime_measurements.py']==sha256(path)


def test_stale_bootstrap_is_counted_as_received_but_not_accepted():
    raw=fixture();raw['messages_received']['HIL_ACTUATOR_CONTROLS']=4005
    raw['stale_bootstrap_messages_discarded']=5
    assert assessor()(raw)['overall']=='PASS'


@pytest.mark.parametrize('mutation', ['incomplete','arm','cleanup','fault','missing_fault','missing_dropped',
    'missing_metrics','expired_row','missing_query','late_query','overlap','commands','history',
    'sparse_queries','wrong_runtime_hash','slow_sim_queries','missing_last_query',
    'shifted_event_clock','negative_query_wall','coordinated_early_stop_evidence',
    'stale_final_safety_decision'])
def test_assessor_fail_closed_for_missing_or_unsafe_live_evidence(mutation):
    r=fixture()
    if mutation=='incomplete':r['complete']=False;r['simulation_seconds']=15.484
    if mutation=='arm':r['flight_command_sent']=True
    if mutation=='cleanup':r['zero_control_on_stop'][0]=1.
    if mutation=='fault':r['phase2']['telemetry_fix']['fault']='safety query service stopped'
    if mutation=='missing_fault':r['phase2']['telemetry_fix'].pop('fault')
    if mutation=='missing_dropped':r['runtime_measurements'].pop('dropped')
    if mutation=='missing_metrics':r.pop('runtime_measurements')
    if mutation=='expired_row':r['runtime_measurements']['row_valid_at_completions']=0
    if mutation=='missing_query':r['runtime_measurements']['queries'].pop()
    if mutation=='late_query':r['runtime_measurements']['samples']['query_response_seconds']['max']=.151
    if mutation=='overlap':r['runtime_measurements']['queries'][0]['end_wall_s']=7.
    if mutation=='commands':r['messages_sent']['COMMAND_LONG']=1
    if mutation=='history':r['history_hashes_after']['model']='changed'
    if mutation=='wrong_runtime_hash':r['streaming_runtime_hashes']={'anything':'invalid'}
    if mutation=='slow_sim_queries':
        for q in r['runtime_measurements']['queries'][1:]:
            q['start_clock_usec']+=1000000;q['end_clock_usec']+=1000000
    if mutation=='missing_last_query':
        r['runtime_measurements']['queries'].pop()
        r['runtime_measurements']['scheduler']['starts']-=1
        r['runtime_measurements']['row_valid_at_completions']-=1
        r['runtime_measurements']['samples']['query_response_seconds']['n']-=1
        r['phase2']['telemetry_fix']['events'].pop()
    if mutation=='shifted_event_clock':r['phase2']['telemetry_fix']['events'][0]['clock_usec']=1
    if mutation=='negative_query_wall':
        for q in r['runtime_measurements']['queries']:
            q['start_wall_s']-=20;q['end_wall_s']-=20
        for e in r['phase2']['telemetry_fix']['events']:e['wall_s']-=20
    if mutation=='coordinated_early_stop_evidence':
        for _ in range(3):
            r['runtime_measurements']['queries'].pop()
            r['phase2']['telemetry_fix']['events'].pop()
        r['runtime_measurements']['scheduler']['starts']-=3
        r['runtime_measurements']['row_valid_at_completions']-=3
        r['runtime_measurements']['samples']['query_response_seconds']['n']-=3
        r['phase2']['telemetry_fix']['last_decision']['clock_usec']=15500000
        r['phase2']['telemetry_fix']['last_decision']['wall_s']=14.5
    if mutation=='stale_final_safety_decision':
        r['phase2']['telemetry_fix']['last_decision']['clock_usec']=16800000
        r['phase2']['telemetry_fix']['last_decision']['wall_s']=15.8
    if mutation=='sparse_queries':
        r['runtime_measurements']['queries']=r['runtime_measurements']['queries'][:1]
        r['runtime_measurements']['scheduler']['starts']=1
        r['runtime_measurements']['row_valid_at_completions']=1
        r['runtime_measurements']['samples']['query_response_seconds']['n']=1
        r['phase2']['telemetry_fix']['events']=r['phase2']['telemetry_fix']['events'][:1]
    assert assessor()(r)['overall']=='FAIL'


def test_saved_single_run_report_recomputes_from_immutable_raw_when_available():
    import json
    from .diagnostics import sha256
    root=Path(__file__).resolve().parents[2]
    rawpath=root/'mujoco/reports/px4_disarmed_verification_parts/disarmed_01/smoke_summary.json'
    if not rawpath.exists():pytest.skip('one-run raw evidence is local-only')
    reportpath=root/'mujoco/reports/px4_mujoco_disarmed_verification.json'
    assert reportpath.exists(),'live result report not generated'
    raw=json.loads(rawpath.read_text());report=json.loads(reportpath.read_text())
    assert report['raw_evidence']['summary_sha256']==sha256(rawpath)
    assert report['acceptance']==assessor()(raw)
    assert report['experiment']['authorized_sitl_runs']==1
    assert report['experiment']['armed_runs']==0
    assert report['preservation']['protected_models_unchanged'] is True
    assert report['preservation']['original_px4_binary_unchanged'] is True
