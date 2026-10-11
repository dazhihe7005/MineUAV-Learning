import socket
import sys
import threading
import json
from pathlib import Path
import numpy as np
import pytest
from .test_interfaces import module,fixture

SEARCH=Path.home()/".local/lib/python3.10/site-packages"

def test_sensor_wire_fields_and_crc():
    t=module("transport");mv=t.load_mavlink(str(SEARCH))
    m,d=fixture();s=module("sensor_adapter").SensorAdapter(m).sample(d,123456)
    d.xfrc_applied[1,2]=68.67
    s=module("sensor_adapter").SensorAdapter(m).sample(d,123456)
    encoder=mv.MAVLink(None,srcSystem=42,srcComponent=51)
    parser=mv.MAVLink(None)
    msgs=t.sensor_messages(s,mv)
    received=[parser.parse_char(msg.pack(encoder)) for msg in msgs]
    assert [msg.get_type() for msg in received]==["HIL_SENSOR","HIL_GPS"]
    assert received[0].time_usec==123456
    assert received[0].fields_updated==7167 # all IMU/mag/baro fields except diff pressure
    assert received[0].zacc==pytest.approx(-9.81,abs=1e-5)
    assert received[1].alt==round(s.altitude_m*1000)
    assert received[1].fix_type==3

def test_real_tcp_fragmented_actuator_and_eof():
    t=module("transport");mv=t.load_mavlink(str(SEARCH))
    with t.LoopbackTransport(port=0,timeout=.5,mavlink=mv) as server:
        client=socket.create_connection(("127.0.0.1",server.port))
        server.accept()
        encoder=mv.MAVLink(None,srcSystem=1,srcComponent=1)
        payload=mv.MAVLink_hil_actuator_controls_message(9000,[.2]*16,128,1).pack(encoder)
        client.sendall(payload[:5]);client.sendall(payload[5:])
        msg=server.receive_until("HIL_ACTUATOR_CONTROLS")
        assert msg.controls[:4]==pytest.approx([.2]*4)
        assert msg.time_usec==9000
        client.close()
        with pytest.raises(ConnectionError): server.receive_until("HIL_ACTUATOR_CONTROLS")

def test_tcp_timeout_and_remote_bind_refused():
    t=module("transport");mv=t.load_mavlink(str(SEARCH))
    with pytest.raises(ValueError): t.LoopbackTransport(host="0.0.0.0",mavlink=mv)
    with t.LoopbackTransport(port=0,timeout=.02,mavlink=mv) as server:
        with pytest.raises(TimeoutError):server.accept()

def test_clock_strict_integer_and_bad_stale_commands():
    d=module("diagnostics")
    c=d.SimClock()
    assert c.advance(2)==1004000
    assert c.advance(2)==1008000
    with pytest.raises(ValueError):c.advance(0)
    gate=d.CommandGate()
    gate.accept(1000)
    with pytest.raises(ValueError):gate.accept(1000)
    with pytest.raises(ValueError):gate.accept(999)

def test_memory_reserve_and_atomic_report(tmp_path):
    d=module("diagnostics")
    with pytest.raises(MemoryError):d.require_memory(available_bytes=1024)
    assert d.require_memory(available_bytes=2*1024**3)==2*1024**3
    p=tmp_path/"result.json";d.atomic_json(p,{"complete":True,"n":1})
    assert json.loads(p.read_text())=={"complete":True,"n":1}
    assert list(tmp_path.iterdir())==[p]

def test_bad_crc_never_becomes_actuator_command():
    t=module("transport");mv=t.load_mavlink(str(SEARCH))
    with t.LoopbackTransport(port=0,timeout=.1,mavlink=mv) as server:
        client=socket.create_connection(("127.0.0.1",server.port));server.accept()
        encoder=mv.MAVLink(None,srcSystem=1,srcComponent=1)
        data=bytearray(mv.MAVLink_hil_actuator_controls_message(1000,[.1]*16,128,1).pack(encoder))
        data[-1]^=1;client.sendall(data)
        with pytest.raises(ValueError):server.receive_until("HIL_ACTUATOR_CONTROLS")
        client.close()

def test_model_loading_budget_reserves_os_memory():
    d=module("diagnostics")
    with pytest.raises(MemoryError):d.require_startup_memory(available_bytes=2*1024**3)
    assert d.require_startup_memory(available_bytes=3*1024**3)==3*1024**3
