"""Failures guarded: history replay, fragment authorization, unsafe overwrite."""
import json
from pathlib import Path
import numpy as np
import pytest
from .test_interfaces import module
from .test_yaw_validation import safety_console

ROW=[-.017207708,.017207708,-.017207708,.017207708]

def parser():
    m=module('streaming_diagnostics')
    assert hasattr(m,'StreamingDiagnostics'), 'incremental bounded parser required'
    return m.StreamingDiagnostics()

def test_new_bytes_only_and_fragment_completion():
    p=parser();events=[];text=safety_console(33000000)
    for char in text:events.extend(p.feed(char))
    assert [e['topic'] for e in events]==['vehicle_local_position','estimator_status_flags','vehicle_status']
    assert not p.feed('')
    assert p.processed_chars==len(text)
    assert p.peak_buffer_chars<=32768

@pytest.mark.parametrize('cut',[1,5,11,35,129,501])
def test_ansi_cross_read_boundaries(cut):
    p=parser();text=safety_console(33000000).replace('timestamp:', '\x1b[0mtimestamp:\x1b[0m')
    out=[]
    for i in range(0,len(text),cut):out+=p.feed(text[i:i+cut])
    assert len(out)==3 and out[0]['fields']['timestamp']==33000000

def test_incomplete_healthy_record_cannot_emit():
    p=parser();assert not p.feed('TOPIC: vehicle_status\n    timestamp: 33000000\n    failsafe: False')
    assert p.pending

def test_instance_one_two_cannot_be_instance_zero_health():
    p=parser();text=safety_console(33000000).split('TOPIC: estimator_status_flags')[0]
    text=text.replace('\n vehicle_local_position\n','\n vehicle_local_position\nInstance 1:\n')
    assert not p.feed(text)

def test_complete_fault_retained_before_partial_next_title():
    p=parser();text=safety_console(33000000,heading=False).split('TOPIC: estimator_status_flags')[0]
    events=p.feed(text+'TOPI')
    assert events[0]['fields']['heading_good_for_control'] is False and p.pending

@pytest.mark.parametrize('text',['x'*8193,'TOPIC: vehicle_status\n'+'    key: 1\n'*4000])
def test_unbounded_line_or_record_fails_closed(text):
    p=parser()
    with pytest.raises(ValueError,match='bound'):p.feed(text)

def test_multiple_instances_never_merge_faults_or_timestamps():
    p=parser();text=safety_console(33000000).split('TOPIC: estimator_status_flags')[0]
    text=text.replace('\n vehicle_local_position\n','\n vehicle_local_position\nInstance 0:\n')
    events=p.feed(text+'Instance 1:\n    timestamp: 34000000\n    heading_good_for_control: True\n\n')
    assert len(events)==1 and events[0]['fields']['timestamp']==33000000

@pytest.mark.parametrize('line',['0 0 0 0','nan 0 0 0','1 2 3','bad'])
def test_bad_row_not_hidden_by_previous_good_row(line):
    p=module('allocation_runner').AllocationYawProtocol({'field_ned_gauss':[.2,.01,.4]},ROW)
    assert hasattr(p,'on_console_chunk'), 'runtime must consume deltas'
    p.query_requested(33000000,0.,'')
    p.on_console_chunk(safety_console(33000000)+'MineUAV yaw axis: '+' '.join(map(str,ROW))+'\n',33000000,.01)
    p.on_console_chunk('MineUAV yaw axis: '+line+'\n',33000000,.02)
    assert p.monitor.fault is not None

def test_allocator_row_is_query_snapshot_with_bounded_lease():
    p=module('allocation_runner').AllocationYawProtocol({'field_ned_gauss':[.2,.01,.4]},ROW)
    assert hasattr(p,'on_console_chunk')
    p.query_requested(33000000,0.,'')
    p.on_console_chunk(safety_console(33000000),33000000,.01)
    assert p.monitor.query is not None
    p.on_console_chunk('MineUAV yaw axis: '+' '.join(map(str,ROW))+'\n',33000000,.02)
    assert p.monitor.query is None and p.row_received_wall==.02
    assert p.row_ready(33000000,.02)
    assert not p.row_ready(33000000,.671)
    assert p.monitor.fault is not None

def test_cumulative_console_growth_no_reparse():
    p=parser();text='ordinary log line\n'*100000
    for i in range(0,len(text),4096):p.feed(text[i:i+4096])
    assert p.processed_chars==len(text) and p.peak_buffer_chars<=8192

@pytest.mark.parametrize('part',['px4_yaw_telemetry_fix_parts/yaw_01','px4_yaw_allocation_parts/yaw_01'])
def test_saved_console_primary_records_equivalent(part):
    path=Path(__file__).resolve().parents[1]/'reports'/part/'smoke_summary.json'
    if not path.exists():pytest.skip('immutable local raw fixture')
    raw=json.loads(path.read_text());p=parser();events=[];text=raw['px4_console']
    for i in range(0,len(text),1024):events+=p.feed(text[i:i+1024])
    latest={e['topic']:e['fields'] for e in events if e['kind']=='topic'}
    legacy=module('yaw_validation').latest_topics(text)
    for name in module('telemetry_freshness').TelemetryFreshness.SOURCE_LIMITS:
        assert latest[name]==legacy[name]
