"""MAVLink2 loopback TCP server; PX4 SimulatorMavlink is the TCP client."""
from collections import deque
import importlib
import math
from pathlib import Path
import socket
import sys
import time

def load_mavlink(search_path=None):
    if search_path:
        p=Path(search_path).expanduser().resolve()
        if not (p/"pymavlink").is_dir(): raise ValueError("pymavlink search path unavailable")
        # Append, never override benchmark venv NumPy or execute .pth files.
        if str(p) not in sys.path:sys.path.append(str(p))
    return importlib.import_module("pymavlink.dialects.v20.common")

def sensor_messages(sample,mv,include_gps=True):
    a=sample.accel_frd;w=sample.gyro_frd;b=sample.mag_frd_gauss
    sensor=mv.MAVLink_hil_sensor_message(sample.time_usec,*a,*w,*b,
             sample.pressure_hpa,0.,sample.altitude_m,15.,7167,id=0)
    messages=[sensor]
    if include_gps:
        n,e,_=sample.position_ned_m;vn,ve,vd=sample.velocity_ned_m_s
        lat=47.397742+n/6378137.*180/math.pi
        lon=8.545594+e/(6378137.*math.cos(math.radians(47.397742)))*180/math.pi
        speed=math.sqrt(vn*vn+ve*ve)
        course=round(math.degrees(math.atan2(ve,vn))%360*100) if speed>.01 else 0
        messages.append(mv.MAVLink_hil_gps_message(sample.time_usec,3,
            round(lat*1e7),round(lon*1e7),round(sample.altitude_m*1000),
            30,50,round(speed*100),round(vn*100),round(ve*100),round(vd*100),
            course,12,id=0))
    return messages

class LoopbackTransport:
    def __init__(self,host="127.0.0.1",port=4560,timeout=3.,mavlink=None):
        if host!="127.0.0.1":raise ValueError("SITL transport is loopback only, no serial/hardware")
        if timeout<=0 or not math.isfinite(timeout):raise ValueError("positive finite timeout required")
        self.mv=mavlink or load_mavlink()
        self.encoder=self.mv.MAVLink(None,srcSystem=42,srcComponent=51)
        self.parser=self.mv.MAVLink(None);self.parser.robust_parsing=True
        self.pending=deque();self.timeout=float(timeout);self.connection=None
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        try:
            self.socket.bind((host,port));self.socket.listen(1);self.socket.settimeout(timeout)
        except BaseException:
            self.socket.close();raise
        self.port=self.socket.getsockname()[1]
        self.counts={};self.sent={}

    def __enter__(self):return self
    def __exit__(self,*args):self.close()

    def accept(self):
        c,addr=self.socket.accept()
        if addr[0]!="127.0.0.1":c.close();raise ConnectionError("non-loopback peer")
        c.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1);c.settimeout(self.timeout)
        self.connection=c

    def send(self,msg):
        if self.connection is None:raise ConnectionError("no PX4 connection")
        self.connection.sendall(msg.pack(self.encoder));self.encoder.seq=(self.encoder.seq+1)%256
        name=msg.get_type();self.sent[name]=self.sent.get(name,0)+1

    def receive_until(self,kind,timeout=None):
        deadline=time.monotonic()+(self.timeout if timeout is None else timeout)
        while True:
            while self.pending:
                msg=self.pending.popleft()
                name=msg.get_type();self.counts[name]=self.counts.get(name,0)+1
                if name=="BAD_DATA":raise ValueError("invalid MAVLink data/CRC")
                if msg.get_srcSystem()!=1 or msg.get_srcComponent()!=1:
                    raise ValueError("unexpected PX4 source identity (instance0 only)")
                if name==kind:return msg
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError(f"PX4 {kind} watchdog expired")
            if self.connection is None:raise ConnectionError("no PX4 connection")
            self.connection.settimeout(remaining)
            data=self.connection.recv(4096)
            if not data:raise ConnectionError("PX4 disconnected")
            self.pending.extend(self.parser.parse_buffer(data) or [])

    def close(self):
        if self.connection is not None:self.connection.close();self.connection=None
        self.socket.close()
