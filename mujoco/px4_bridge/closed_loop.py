"""Explicit opt-in, bounded SITL-only position hold; no bypass/force arming."""
import re
import socket
import numpy as np
from .frame_conversions import rotation

def clean_console(text):
    text=re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",text)
    return "\n".join(line for line in text.splitlines() if not line.startswith("pxh>"))

def estimator_ready(console):
    text=clean_console(console)
    keys={"xy_valid":True,"z_valid":True,"v_xy_valid":True,"v_z_valid":True,
          "pre_flight_checks_pass":True,"cs_tilt_align":True,"cs_yaw_align":True,
          "cs_gnss_pos":True,"cs_mag_fault":False,"cs_baro_fault":False}
    for key,value in keys.items():
        matches=re.findall(r"^\s*"+key+r": (True|False)\s*$",text,re.M)
        if not matches or (matches[-1]=="True")!=value:return False
    return True

def check_envelope(data,maximum_height=1.2):
    r=rotation(data.qpos[3:])
    tilt=np.arccos(np.clip(r[2,2],-1,1))
    if (not np.isfinite(data.qvel).all() or np.linalg.norm(data.qvel[:3])>1.5
        or np.linalg.norm(data.qvel[3:])>2.5 or not -.01<=data.qpos[2]<=maximum_height
        or np.linalg.norm(data.qpos[:2])>.5 or tilt>np.deg2rad(20)):
        raise ValueError("SITL flight safety envelope exceeded; stop, do not tune gains")

def assess_hover(result):
    """A sampled2s hold proof, not a general flight-certification criterion."""
    end=result.get("simulation_seconds",0)
    rows=[r for r in result.get("records",[]) if r["time_s"]>=end-2.-1e-6]
    ack={(x["command"],x["result"]) for x in result.get("command_acks",[])}
    target_z=result.get("initial_qpos",[0,0,0])[2]+.6
    ok=(result.get("complete",False) and result.get("kind")=="bounded_offboard_hover"
        and result.get("flight_command_sent",False) and {(176,0),(400,0)}<=ack
        and len(rows)>=20 and rows[-1]["time_s"]-rows[0]["time_s"]>=1.85)
    for i,r in enumerate(rows):
        tilt=np.arccos(np.clip(rotation(r["quaternion_enu_flu"])[2,2],-1,1))
        ok=ok and (abs(r["position_enu_m"][2]-target_z)<.15
                   and np.linalg.norm(r["position_enu_m"][:2])<.2
                   and np.linalg.norm(r["velocity_origin_enu_m_s"])<.2
                   and tilt<np.deg2rad(10) and min(r["ctrl_omega_squared"])>0)
        if i:ok=ok and r["time_s"]-rows[i-1]["time_s"]<=.12
    return {"pass":bool(ok),"duration_s":2,"sampling_hz":10,"samples":len(rows),
            "target_body_origin_z_m":target_z,"height_tolerance_m":.15,
            "horizontal_tolerance_m":.2,"speed_threshold_m_s":.2,"tilt_threshold_deg":10}

class OffboardLink:
    def __init__(self,mavlink,remote_port=14580,local_port=14540):
        self.mv=mavlink;self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1",local_port));self.socket.setblocking(False)
        self.remote=("127.0.0.1",remote_port)
        self.encoder=mavlink.MAVLink(None,srcSystem=42,srcComponent=191)
        self.parser=mavlink.MAVLink(None);self.acks=[]

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.socket.close()
    def send(self,msg):
        self.socket.sendto(msg.pack(self.encoder),self.remote);self.encoder.seq=(self.encoder.seq+1)%256
    def heartbeat(self):
        self.send(self.mv.MAVLink_heartbeat_message(self.mv.MAV_TYPE_GCS,self.mv.MAV_AUTOPILOT_INVALID,0,0,4,3))
    def setpoint(self,time_usec,height=.6,yaw=np.pi/2):
        if not np.isfinite([height,yaw]).all() or not 0<height<=2.6:
            raise ValueError("finite bounded SITL position/yaw setpoint required")
        self.send(self.mv.MAVLink_set_position_target_local_ned_message(
            time_usec//1000,1,1,self.mv.MAV_FRAME_LOCAL_NED,2552,
            0.,0.,-height,0.,0.,0.,0.,0.,0.,yaw,0.))
    def command(self,command,*params):
        p=list(params)+[0.]*(7-len(params))
        self.send(self.mv.MAVLink_command_long_message(1,1,command,0,*p))
    def poll(self):
        while True:
            try:data,peer=self.socket.recvfrom(65536)
            except BlockingIOError:break
            if peer!=self.remote:raise ValueError("unexpected offboard UDP peer")
            for msg in self.parser.parse_buffer(data) or []:
                if msg.get_type()=="COMMAND_ACK":
                    self.acks.append({"command":msg.command,"result":msg.result})
                    if msg.result not in (self.mv.MAV_RESULT_ACCEPTED,self.mv.MAV_RESULT_IN_PROGRESS):
                        raise ValueError(f"PX4 rejected SITL command {msg.command}: {msg.result}")
