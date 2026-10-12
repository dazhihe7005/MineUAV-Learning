"""Adversarial streaming monitor regressions with actual safety semantics."""
import pytest
from .test_interfaces import module
from .test_yaw_validation import safety_console

def protocol():return module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})

def deliver(p,clock=33000000,wall=0.,text=None):
    p.query_requested(clock,wall,'')
    p.on_console_chunk(text or safety_console(clock),clock,wall)
    p.on_actuator(clock,wall)

@pytest.mark.parametrize('replacement',[
    ('cs_mag_fault: False','cs_mag_fault: True'),
    ('cs_gnss_fault: False','cs_gnss_fault: True'),
    ('cs_gnss_yaw_fault: False','cs_gnss_yaw_fault: True'),
    ('cs_gnss_hgt_fault: False','cs_gnss_hgt_fault: True'),
    ('failsafe: False','failsafe: True'),
    ('heading_good_for_control: True','heading_good_for_control: False'),
    ('cs_mag_aligned_in_flight: True','cs_mag_aligned_in_flight: False'),
    ('xy_valid: True','xy_valid: False')])
def test_streamed_real_fault_latches_and_later_health_does_not_resurrect(replacement):
    p=protocol();deliver(p)
    deliver(p,33004000,.004,safety_console(33004000).replace(*replacement))
    assert p.monitor.decision(33004000,.004)['safe_stop']
    deliver(p,33008000,.008)
    assert p.monitor.decision(33008000,.008)['safe_stop']

@pytest.mark.parametrize('stamp',['nan','inf','-1','34000000','32999999'])
def test_bad_stream_publication_is_not_receipt_freshness(stamp):
    p=protocol();deliver(p)
    deliver(p,33004000,.004,safety_console(33004000).replace('timestamp: 33004000','timestamp: '+stamp))
    assert p.monitor.decision(33004000,.004)['safe_stop']

def test_duplicate_source_does_not_renew_lease_despite_live_queries():
    p=protocol();deliver(p)
    for i in range(1,164):
        clock=33000000+i*4000;wall=i*.004
        if i==125:deliver(p,clock,wall,safety_console(33000000))
        else:p.on_actuator(clock,wall)
        decision=p.monitor.decision(clock,wall)
    assert decision['safe_stop'] and 'publication expired' in decision['reason']

def test_partial_new_query_never_reuses_previous_healthy_snapshot():
    p=protocol();deliver(p);p.query_requested(33004000,.004,'')
    p.on_console_chunk('TOPIC: vehicle_status\n    timestamp: 33004000\n',33004000,.004)
    assert not p.monitor.decision(33004000,.004)['allow_new_maneuver']
    p.on_console_chunk(safety_console(33160000),33160000,.160)
    assert p.monitor.fault and not p.monitor.decision(33160000,.160)['allow_new_maneuver']

def test_duplicate_fields_cannot_hide_true_failure():
    p=protocol();deliver(p,text=safety_console(33000000).replace('    failsafe: False','    failsafe: True\n    failsafe: False'))
    assert p.monitor.fault is not None

def test_repeated_instance_headers_stay_bounded():
    p=module('streaming_diagnostics').StreamingDiagnostics()
    with pytest.raises(ValueError,match='bound'):
        p.feed('TOPIC: vehicle_status\n'+'Instance 0:\n'*4000)

def test_monitor_event_memory_has_a_failure_bound():
    m=module('telemetry_freshness').TelemetryFreshness()
    for _ in range(9000):m.begin_query(33000000,0.);m.query=None
    assert len(m.events)<=8192 and m.fault is not None

@pytest.mark.parametrize('replacement',[
    ('failsafe: False','failsafe: True'),
    ('cs_gnss_fault: False','cs_gnss_fault: True'),
    ('cs_mag_aligned_in_flight: True','cs_mag_aligned_in_flight: False'),
    ('heading_good_for_control: True','heading_good_for_control: False')])
def test_complete_fault_not_overwritten_by_later_good_record_in_same_read(replacement):
    p=protocol();deliver(p)
    p.query_requested(33004000,.004,'')
    bad=safety_console(33004000).replace(*replacement)
    p.on_console_chunk(bad+safety_console(33008000),33008000,.008)
    p.on_actuator(33008000,.008)
    assert p.monitor.decision(33008000,.008)['safe_stop']

def test_tab_indented_duplicate_cannot_hide_failsafe():
    p=protocol();deliver(p,text=safety_console(33000000).replace('    failsafe: False','    failsafe: True\n\t   failsafe: False'))
    assert p.monitor.fault

@pytest.mark.parametrize('prefix',[' ','\t'])
def test_noncanonical_instance_one_cannot_supply_primary_health(prefix):
    p=protocol();text=safety_console(33000000).replace('\n vehicle_local_position\n','\n vehicle_local_position\n'+prefix+'Instance 1:\n')
    deliver(p,text=text)
    assert not p.monitor.decision(33000000,0.)['allow_new_maneuver']

def test_terminal_cursor_command_inside_field_cannot_fabricate_health():
    p=protocol();text=safety_console(33000000).replace('failsafe: False','failsafe: Fa\x1b[2Dlse')
    deliver(p,text=text)
    assert p.monitor.fault

@pytest.mark.parametrize('replacement',[
    ('heading_good_for_control: True','heading_good_for_control: False'),
    ('cs_mag_aligned_in_flight: True','cs_mag_aligned_in_flight: False')])
def test_first_flight_batch_fault_cannot_be_hidden_without_prior_health(replacement):
    p=protocol();p.active=True;p.query_requested(33000000,0.,'')
    p.on_console_chunk(safety_console(33000000).replace(*replacement)+safety_console(33004000),33004000,.004)
    p.on_actuator(33004000,.004)
    assert p.monitor.decision(33004000,.004)['safe_stop']

def test_corrupt_stream_before_first_query_stops_runner_gate():
    from types import SimpleNamespace
    p=protocol();p.on_console_chunk('x'*8193,1000000,0.)
    with pytest.raises(ValueError):p.before_step(SimpleNamespace(time=0.),1000000,'',wall=0.)

@pytest.mark.parametrize('replacement',[
    ('xy_valid: True','xy_valid: 1'),
    ('failsafe: False','failsafe: 0'),
    ('heading: 1.57079632679','heading: nan'),
    ('heading: 1.57079632679','heading: inf')])
def test_invalid_complete_health_payload_cannot_be_hidden_in_first_batch(replacement):
    p=protocol();p.active=True;p.query_requested(33000000,0.,'')
    p.on_console_chunk(safety_console(33000000).replace(*replacement)+safety_console(33004000),33004000,.004)
    assert p.monitor.fault

def test_unrequested_fault_cannot_be_ignored_or_renew_positive_authority():
    p=protocol();deliver(p)
    p.on_console_chunk(safety_console(33004000).replace('failsafe: False','failsafe: True'),33004000,.004)
    assert p.monitor.fault
    assert p.monitor.topics['vehicle_status']['timestamp']==33000000

@pytest.mark.parametrize('whitespace',['\u2003','\u00a0','\f','\v'])
def test_all_legacy_field_whitespace_cannot_hide_duplicate_fault(whitespace):
    p=protocol();p.active=True
    text=safety_console(33000000).replace('    failsafe: False','    failsafe: True\n'+whitespace+'   failsafe: False')
    deliver(p,text=text)
    assert p.monitor.fault and not p.monitor.decision(33000000,0.)['allow_new_maneuver']

def test_clear_line_control_inside_field_key_cannot_create_health():
    p=protocol();p.active=True
    deliver(p,text=safety_console(33000000).replace('    failsafe: False','    fai\x1b[2Klsafe: False'))
    assert p.monitor.fault

def test_clear_line_control_before_field_cannot_fabricate_canonical_indent():
    p=protocol();p.active=True
    deliver(p,text=safety_console(33000000).replace('    failsafe: False','\x1b[2K    failsafe: False'))
    assert p.monitor.fault

@pytest.mark.parametrize('token',['True','False','32999999.5'])
def test_sample_timestamp_is_actual_integer_time_not_boolean_or_fraction(token):
    p=protocol()
    deliver(p,text=safety_console(33000000).replace('    timestamp: 33000000','    timestamp: 33000000\n    timestamp_sample: '+token))
    assert p.monitor.fault

def test_prompt_prefixed_secondary_instance_cannot_supply_primary_health():
    p=protocol();p.active=True
    text=safety_console(33000000).replace('\n vehicle_local_position\n','\n vehicle_local_position\npxh> Instance 1:\n')
    deliver(p,text=text)
    assert not p.monitor.decision(33000000,0.)['allow_new_maneuver']

def test_partial_record_after_full_batch_retains_original_query_deadline():
    p=protocol();p.query_requested(33000000,0.,'')
    partial='TOPIC: vehicle_status\n    timestamp: 33004000\n    failsafe: True'
    p.on_console_chunk(safety_console(33000000)+partial,33004000,.004)
    assert p.parser.pending and p.monitor.query is not None
    for i in range(2,40):
        clock=33000000+i*4000;wall=i*.004
        p.on_actuator(clock,wall);p.monitor.decision(clock,wall)
    assert p.monitor.fault and not p.monitor.decision(clock,wall)['allow_new_maneuver']

@pytest.mark.parametrize('prefix',['I','INFO  [control_allocator] ', 'INFO  [control_allocator] MineUAV yaw axis: 0 0 0 0'])
def test_prefixed_allocator_fragment_cannot_admit_maneuver(prefix):
    from .test_interfaces import module
    row=[-.017207708,.017207708,-.017207708,.017207708]
    p=module('allocation_runner').AllocationYawProtocol({'field_ned_gauss':[.2,.01,.4]},row,active=True)
    good='MineUAV yaw axis: -.017207708 .017207708 -.017207708 .017207708\n'
    deliver(p,text=safety_console(33000000)+good+prefix)
    assert p.parser.pending and p.monitor.query is not None
    assert not p.monitor.decision(33000000,0.)['allow_new_maneuver']

@pytest.mark.parametrize('header',['pxh>\u2003Instance 1:', 'pxh>\vInstance 1:',
    'Instance\t1:', 'Instance\u20031:', 'pxh>\fInstance 1:', 'Instance1:'])
def test_noncanonical_instance_like_header_cannot_merge_secondary_health(header):
    p=protocol();p.active=True
    text=safety_console(33000000).replace('\n vehicle_local_position\n','\n vehicle_local_position\n'+header+'\n')
    deliver(p,text=text)
    assert not p.monitor.decision(33000000,0.)['allow_new_maneuver']

@pytest.mark.parametrize('name',['vehicle_statu','estimator_status_flag','vehicle_local_positio'])
def test_truncated_safety_topic_name_cannot_hide_new_fault_after_health(name):
    p=protocol();p.active=True
    damaged='TOPIC: '+name+'\n    timestamp: 33000000\n    failsafe: True\n\n'
    deliver(p,text=safety_console(33000000)+damaged)
    assert p.monitor.fault and not p.monitor.decision(33000000,0.)['allow_new_maneuver']

@pytest.mark.parametrize('token',['True','nan','34000000','32999999'])
def test_unrequested_invalid_source_timestamp_is_negative_evidence(token):
    p=protocol();deliver(p)
    p.on_console_chunk(safety_console(33004000).replace('timestamp: 33004000','timestamp: '+token),33004000,.004)
    assert p.monitor.fault
    assert p.monitor.topics['vehicle_status']['timestamp']==33000000

@pytest.mark.parametrize('token',['True','nan','34000000','32999999'])
def test_unrequested_invalid_sample_time_is_negative_evidence(token):
    p=protocol()
    initial=safety_console(33000000).replace('    timestamp: 33000000','    timestamp: 33000000\n    timestamp_sample: 33000000')
    deliver(p,text=initial)
    later=safety_console(33004000).replace('    timestamp: 33004000','    timestamp: 33004000\n    timestamp_sample: '+token)
    p.on_console_chunk(later,33004000,.004)
    assert p.monitor.fault

def test_unrequested_publication_order_is_preserved_without_lease_renewal():
    p=protocol();deliver(p)
    p.on_console_chunk(safety_console(33008000),33008000,.008)
    assert p.monitor.fault is None and p.monitor.topics['vehicle_status']['timestamp']==33000000
    deliver(p,33012000,.012,text=safety_console(33004000))
    assert p.monitor.fault

def test_unrequested_duplicate_changed_payload_cannot_pass_timestamp_validation():
    p=protocol();deliver(p)
    p.on_console_chunk(safety_console(33000000).replace('nav_state: 14','nav_state: 3'),33004000,.004)
    assert p.monitor.fault
