"""Absolute timing/queue bounds; no PX4 process or simulated aircraft."""
from types import SimpleNamespace
import pytest
from .test_interfaces import module
from .test_yaw_validation import safety_console

def scheduler():
    mod=module('query_scheduler')
    assert hasattr(mod,'BoundedQueryScheduler'),'wall-clock service required'
    protocol=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    return protocol,mod.BoundedQueryScheduler(protocol)

def answer(p,clock,wall):
    p.on_console_chunk(safety_console(clock),clock,wall)
    if p.monitor.last_actuator is None or clock>p.monitor.last_actuator[0]:p.on_actuator(clock,wall)

def test_wall_queries_while_simulation_progress_slows():
    p,s=scheduler();starts=[]
    for i in range(376):
        wall=i*.004;clock=7000000+int(wall*.8*1e6)
        p.on_actuator(clock,wall)
        s.service(clock,wall,lambda b:starts.append(wall))
        if p.monitor.query:answer(p,clock,wall)
    assert starts==pytest.approx([0.,.5,1.,1.5])
    assert p.monitor.fault is None

def test_simulation_due_can_query_before_wall_period():
    p,s=scheduler();starts=[]
    s.service(7000000,0.,lambda b:starts.append(0.));answer(p,7000000,0.)
    for clock in range(7004000,7500000,4000):
        p.on_actuator(clock,(clock-7000000)/1e6*.8)
        s.service(clock,(clock-7000000)/1e6*.8,lambda b:starts.append(-1))
    p.on_actuator(7500000,.4)
    s.service(7500000,.4,lambda b:starts.append(.4))
    assert starts==[0.,.4]

def test_pending_response_never_queues_catchup_queries():
    p,s=scheduler();sent=[]
    s.service(7000000,0.,sent.append)
    for i in range(1,38):s.service(7000000+i*4000,i*.004,sent.append)
    assert len(sent)==1 and p.monitor.fault is None
    s.service(7152000,.152,sent.append)
    assert p.monitor.fault and len(sent)==1

def test_late_healthy_cannot_clear_response_timeout():
    p,s=scheduler();s.service(7000000,0.,lambda b:None)
    s.service(7152000,.152,lambda b:None);fault=p.monitor.fault
    answer(p,7152000,.152)
    assert p.monitor.fault==fault

def test_wall_stall_detected_before_any_new_query_can_renew():
    p,s=scheduler();sent=[]
    s.service(7000000,0.,sent.append);answer(p,7000000,0.)
    s.service(7000000,.651,sent.append)
    assert p.monitor.fault and len(sent)==1

def test_query_service_during_actuator_wait_uses_same_watchdog(monkeypatch):
    p,s=scheduler();r=module('simulation_runner');now=[0.];starts=[]
    monkeypatch.setattr(r.time,'monotonic',lambda:now[0])
    class Wire:
        def receive_until(self,kind,timeout):now[0]+=timeout;raise TimeoutError('unavailable')
    def service():s.service(7000000,now[0],lambda b:starts.append(now[0]))
    with pytest.raises(TimeoutError):
        r.receive_current_actuator(Wire(),7000000,True,.25,service,lambda:p.monitor.next_wall_deadline())
    assert starts==[0.] and now[0]<=.15+1e-12

def test_nonblocking_writer_single_batch_and_partial_progress():
    w=module('query_scheduler').BoundedConsoleWriter()
    w.enqueue(b'abcd',0.,.15)
    def blocked(data):raise BlockingIOError()
    assert not w.flush(blocked,.01)
    assert w.pending_bytes==4
    with pytest.raises(ValueError):w.enqueue(b'other',.01,.15)
    sent=[]
    def partial(data):sent.append(data[:2]);return 2
    assert not w.flush(partial,.02)
    assert w.flush(partial,.03)
    assert b''.join(sent)==b'abcd' and w.pending_bytes==0

def test_blocked_writer_cannot_flush_after_deadline():
    w=module('query_scheduler').BoundedConsoleWriter();w.enqueue(b'query',0.,.15)
    with pytest.raises(TimeoutError):w.flush(lambda data:len(data),.151)

def test_archive_preserves_full_text_without_growing_hot_path(tmp_path):
    archive=module('query_scheduler').ConsoleArchive(tmp_path/'console.txt')
    text='ordinary line\n'*100000
    for i in range(0,len(text),4096):archive.append(text[i:i+4096])
    assert len(archive.recent)<=65536
    assert archive.total_chars==len(text) and archive.read_final()==text
    archive.close()

def test_console_service_reads_bounded_chunks_and_dispatches_query(tmp_path):
    m=module('query_scheduler')
    assert hasattr(m,'ConsoleService'), 'runner must share bounded I/O service during receive'
    p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[safety_console(7000000)];writes=[]
    archive=m.ConsoleArchive(tmp_path/'console.txt')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else '',lambda b:writes.append(b) or len(b))
    service.service(7000000,0.)
    # Prior unsolicited topic blocks cannot satisfy the newly issued query.
    assert p.monitor.query is not None and len(writes)==1
    incoming.append(safety_console(7004000));service.service(7004000,.004)
    p.on_actuator(7004000,.004)
    assert p.monitor.decision(7004000,.004)['allow_new_maneuver']
    archive.close()

def test_console_service_bounded_read_prevents_log_flood_starvation(tmp_path):
    m=module('query_scheduler');assert hasattr(m,'ConsoleService')
    p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    archive=m.ConsoleArchive(tmp_path/'console.txt');reads=[]
    def read():reads.append(1);return 'ordinary log\n'*300
    service=m.ConsoleService(p,archive,read,lambda b:len(b))
    service.service(7000000,0.)
    assert len(reads)<=4 and p.monitor.last_query is None
    archive.close()

def test_actuator_send_honors_absolute_monitor_deadline(monkeypatch):
    r=module('simulation_runner');assert hasattr(r,'send_sensor_bounded')
    now=[10.];monkeypatch.setattr(r.time,'monotonic',lambda:now[0]);sent=[]
    class Wire:
        def send(self,msg,timeout=None):sent.append(timeout);now[0]+=.004
    r.send_sensor_bounded(Wire(),object(),10.006)
    assert sent==pytest.approx([.006])
    now[0]=10.007
    with pytest.raises(TimeoutError):r.send_sensor_bounded(Wire(),object(),10.006)
    assert len(sent)==1

def test_old_disarmed_proof_cannot_certify_changed_runtime():
    r=module('simulation_runner');assert hasattr(r,'validate_streaming_proof')
    with pytest.raises(ValueError):r.validate_streaming_proof({}, {'runner':'new'})
    with pytest.raises(ValueError):r.validate_streaming_proof({'streaming_runtime_hashes':{'runner':'old'}},{'runner':'new'})
    r.validate_streaming_proof({'streaming_runtime_hashes':{'runner':'new'}},{'runner':'new'})

def test_unread_pre_query_backlog_cannot_renew_current_snapshot(tmp_path):
    m=module('query_scheduler');p=module('allocation_runner').AllocationYawProtocol({'field_ned_gauss':[.2,.01,.4]},[-.017207708,.017207708,-.017207708,.017207708])
    incoming=['ordinary log\n']*4+[safety_console(7000000),'MineUAV yaw axis: -.017207708 .017207708 -.017207708 .017207708\n']
    archive=m.ConsoleArchive(tmp_path/'console.txt');writes=[]
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else '',lambda b:writes.append(b) or len(b))
    service.service(7000000,0.)
    assert not writes and p.monitor.query is None
    service.service(7004000,.004)
    assert len(writes)==1 and p.monitor.query is not None and p.row_received_wall is None
    archive.close()

def test_archive_delay_is_in_response_budget_not_aged_as_old_receipt(tmp_path):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[];now=[0.];archive=m.ConsoleArchive(tmp_path/'console.txt')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else '',lambda b:len(b))
    service.wall_now=lambda:now[0]
    service.service(7000000,0.);p.on_actuator(7000000,0.)
    original=archive.append
    def slow(chunk):original(chunk);now[0]+=.020
    archive.append=slow
    incoming.append(safety_console(7000000));now[0]=.149
    service.service(7000000,.149)
    assert p.monitor.fault and not p.monitor.decision(7000000,.169)['allow_new_maneuver']
    archive.close()

def test_invalid_utf8_raw_evidence_retained_before_fail_closed_decode(tmp_path):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    bad=b'previous useful console\n\xffcorrupt evidence\n';incoming=[bad];path=tmp_path/'console.bin'
    archive=m.ConsoleArchive(path);service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:len(b))
    service.service(7000000,0.);archive.close()
    assert path.read_bytes()==bad and p.monitor.fault and p.monitor.query is None

def test_pending_utf8_fragment_cannot_complete_health_or_dispatch_query(tmp_path):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[];archive=m.ConsoleArchive(tmp_path/'console.bin');writes=[]
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:writes.append(b) or len(b))
    service.service(33000000,0.)
    incoming.append(safety_console(33000000).encode()+b'\xe2')
    service.service(33004000,.004);p.on_actuator(33004000,.004)
    assert p.monitor.query is not None
    assert not p.monitor.decision(33004000,.004)['allow_new_maneuver']
    assert p.monitor.fault is None
    incoming.append(b'\x82\xac ordinary log\n')
    service.service(33008000,.008);p.on_actuator(33008000,.008)
    assert p.monitor.query is None and p.monitor.fault is None
    archive.close()

def test_pending_utf8_startup_fragment_has_absolute_bounded_timeout(tmp_path):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[b'\xe2'];archive=m.ConsoleArchive(tmp_path/'console.bin');writes=[]
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:writes.append(b) or len(b))
    service.service(7000000,0.)
    assert not writes
    service.service(7000000,.151)
    assert p.monitor.fault and not writes
    archive.close()

@pytest.mark.parametrize('clock,wall',[(6999999,.004),(7004000,-.004),(9000000,.004)])
def test_scheduler_clock_invalidity_latches(clock,wall):
    p,s=scheduler();s.service(7000000,0.,lambda b:None);answer(p,7000000,0.)
    s.service(clock,wall,lambda b:None)
    assert p.monitor.fault

def test_query_cannot_finish_before_empty_transport_or_skip_unread_failure(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]},active=True)
    incoming=[];archive=m.ConsoleArchive(tmp_path/'console.bin')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:len(b))
    service.service(33000000,0.)
    incoming.extend([safety_console(33000000).encode()]+[b'ordinary log\n']*3+[safety_console(33004000).replace('failsafe: False','failsafe: True').encode()])
    service.service(33004000,.004);p.on_actuator(33004000,.004)
    state=SimpleNamespace(time=33.,qpos=np.array([0,0,2.051,1,0,0,0]),qvel=np.zeros(6))
    p.before_step(state,33004000,'',wall=.004)
    assert p.step_time is None and p.monitor.query is not None
    service.service(33008000,.008)
    assert p.monitor.fault
    archive.close()

def test_partial_second_read_keeps_original_response_deadline(tmp_path):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[];archive=m.ConsoleArchive(tmp_path/'console.bin')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:len(b))
    service.service(33000000,0.)
    incoming.extend([safety_console(33000000).encode(),b'TOPIC: vehicle_status\n    timestamp: 33004000\n    failsafe: True'])
    service.service(33004000,.004);p.on_actuator(33004000,.004)
    assert p.monitor.query is not None
    for i in range(2,40):
        service.service(33000000+i*4000,i*.004);p.on_actuator(33000000+i*4000,i*.004)
    assert p.monitor.fault
    archive.close()

def test_continuous_pre_query_backlog_cannot_postpone_first_service_forever():
    p,s=scheduler();sent=[]
    s.service(7000000,0.,lambda b:sent.append(b),dispatch_ready=False)
    assert p.monitor.fault is None and not sent
    s.service(7000000,.651,lambda b:sent.append(b),dispatch_ready=False)
    assert p.monitor.fault and not sent

@pytest.mark.parametrize('established',[False,True])
def test_utf8_completion_after_slow_archive_cannot_erase_deadline(tmp_path,established):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[];now=[0.];writes=[];archive=m.ConsoleArchive(tmp_path/'console.bin')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:writes.append(b) or len(b))
    service.wall_now=lambda:now[0]
    if established:
        service.service(33000000,0.);incoming.append(safety_console(33000000).encode())
        service.service(33000000,0.);p.on_actuator(33000000,0.)
    incoming.append(b'\xe2');now[0]=.004
    service.service(33004000,.004);count=len(writes)
    original=archive.append
    def slow(chunk):original(chunk);now[0]+=.020
    archive.append=slow;incoming.append(b'\x82\xac ordinary log\n');now[0]=.153
    service.service(33008000,.153)
    assert now[0]==pytest.approx(.173)
    assert p.monitor.fault and not p.monitor.decision(33008000,.173)['allow_new_maneuver']
    assert len(writes)==count
    archive.close()

def test_first_utf8_fragment_deadline_begins_at_receipt_not_archive_completion(tmp_path):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[b'\xe2'];now=[0.];archive=m.ConsoleArchive(tmp_path/'console.bin')
    original=archive.append
    def slow(chunk):original(chunk);now[0]+=.160
    archive.append=slow
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:len(b))
    service.wall_now=lambda:now[0];service.service(7000000,0.)
    assert p.monitor.fault and p.monitor.query is None
    archive.close()

def test_archival_only_cleanup_does_not_create_new_fragment_fault(tmp_path):
    m=module('query_scheduler');p=module('yaw_telemetry_runner').TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
    incoming=[b'\xe2'];archive=m.ConsoleArchive(tmp_path/'console.bin')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:len(b))
    service.service(7000000,0.);service.monitoring=False
    service.drain(7000000,10.)
    assert p.monitor.fault is None
    archive.close()

@pytest.mark.parametrize('header',[' TOPIC: vehicle_status', '\u2003TOPIC: vehicle_status',
    'T','TO','TOP','TOPI','TOPIC','pxh> TOP','pxh>\u2003TOPIC'])
def test_damaged_topic_boundary_cannot_hide_failure_and_admit_allocation_yaw(tmp_path,header):
    from types import SimpleNamespace
    import numpy as np
    m=module('query_scheduler');row=[-.017207708,.017207708,-.017207708,.017207708]
    p=module('allocation_runner').AllocationYawProtocol({'field_ned_gauss':[.2,.01,.4]},row,active=True)
    incoming=[];archive=m.ConsoleArchive(tmp_path/'console.bin')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:len(b))
    service.service(33000000,0.)
    good='MineUAV yaw axis: -.017207708 .017207708 -.017207708 .017207708\n'
    damaged=header+'\n    timestamp: 33000000\n    failsafe: True\n\n'
    incoming.append((safety_console(33000000)+good+damaged).encode())
    service.service(33004000,.004);p.on_actuator(33004000,.004)
    state=SimpleNamespace(time=32.004,qpos=np.array([0,0,2.051,1,0,0,0]),qvel=np.zeros(6))
    try:p.before_step(state,33004000,'',wall=.004)
    except ValueError:pass # the production safe stop is acceptable; admission is not
    assert p.monitor.fault and p.step_time is None
    archive.close()

@pytest.mark.parametrize('variant',[0,1,3,5,8,'header_only','missing_colon','extra_padded_fault','extra_missing_colon'])
def test_started_primary_record_cannot_drop_malformed_body_before_yaw(tmp_path,variant):
    from types import SimpleNamespace
    import numpy as np
    m=module('query_scheduler');row=[-.017207708,.017207708,-.017207708,.017207708]
    p=module('allocation_runner').AllocationYawProtocol({'field_ned_gauss':[.2,.01,.4]},row,active=True)
    incoming=[];archive=m.ConsoleArchive(tmp_path/'console.bin')
    service=m.ConsoleService(p,archive,lambda:incoming.pop(0) if incoming else b'',lambda b:len(b))
    service.service(33000000,0.)
    if isinstance(variant,int):
        indent=' '*variant
        bad='TOPIC: vehicle_status\n vehicle_status\n'+indent+'timestamp: 33000000\n'+indent+'failsafe: True\n\n'
    elif variant=='header_only':bad='TOPIC: vehicle_status\n\n'
    elif variant=='missing_colon':bad='TOPIC: vehicle_status\n vehicle_status\n    failsafe True\n\n'
    else:
        extra='     failsafe: True' if variant=='extra_padded_fault' else '    failsafe True'
        bad=safety_console(33000000).replace('    failsafe: False','    failsafe: False\n'+extra)
    good='MineUAV yaw axis: -.017207708 .017207708 -.017207708 .017207708\n'
    incoming.append((bad+safety_console(33000000)+good).encode())
    service.service(33004000,.004);p.on_actuator(33004000,.004)
    state=SimpleNamespace(time=32.004,qpos=np.array([0,0,2.051,1,0,0,0]),qvel=np.zeros(6))
    try:p.before_step(state,33004000,'',wall=.004)
    except ValueError:pass
    assert p.monitor.fault and p.step_time is None
    archive.close()
