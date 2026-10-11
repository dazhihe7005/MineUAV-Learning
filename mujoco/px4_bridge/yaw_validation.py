"""Pure yaw math and live uORB safety gates; never writes PX4 health flags."""
import re
import numpy as np
from .frame_conversions import attitude_to_px4,rotation
from .closed_loop import clean_console

def angle_error(a,b):return (np.asarray(a)-np.asarray(b)+np.pi)%(2*np.pi)-np.pi

def yaw_ned(quaternion_enu_flu):
    r=rotation(attitude_to_px4(quaternion_enu_flu))
    return float(np.arctan2(r[1,0],r[0,0]))

def latest_topics(console):
    # PX4 listener titles can include "3 instances" despite printing only
    # Instance0. Retain those boundaries; never merge flags into a pose topic.
    text=re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",console)
    text="".join(line[line.index("TOPIC:"):] if "TOPIC:" in line and "pxh>" in line else line for line in text.splitlines(keepends=True))
    text=text.replace("\r\n","\n").replace("\r","\n")
    last=text.splitlines()[-1].strip() if text.splitlines() else ""
    if "pxh>" in last:last=last.split("pxh>")[-1].strip()
    if last in ("T","TO","TOP","TOPI","TOPIC"):return {}
    # Every title is a boundary, including a PTY-fragmented one. Never
    # authorize from old records while a new safety-topic title is pending.
    starts=list(re.finditer(r"(?m)^TOPIC:[^\n]*(?:\n|$)",text));titles=[];result={}
    safety_names=("vehicle_local_position","estimator_status_flags","vehicle_status")
    for start in starts:
        title=re.fullmatch(r"TOPIC:\s+([a-z0-9_]+)(?: +[0-9]+ +instances)?[ \t]*",start[0].rstrip("\n"))
        if title is None:return {}
        name=title[1]
        if any(s.startswith(name) and s!=name for s in safety_names):return {}
        titles.append(name)
    for i,start in enumerate(starts):
        block=text[start.end():starts[i+1].start() if i+1<len(starts) else len(text)]
        fields={}
        instances=list(re.finditer(r"(?m)^Instance ([0-9]+):",block))
        if instances:
            primary=next((j for j,m in enumerate(instances) if m[1]=="0"),None)
            if primary is None:result[titles[i]]=fields;continue
            # Other registered EKF slots may say "never published". Do not
            # reject healthy0 merely because they exist, or merge their data.
            block=block[instances[primary].end():instances[primary+1].start() if primary+1<len(instances) else len(block)]
        # PX4 completes listeners with a blank line or returns the shell
        # prompt directly (LocalPosition uses the latter in captured logs).
        # Do not trust a body that may still be mid-print; never cache health.
        if not re.search(r"(?m)^ {4}[a-zA-Z0-9_]+: [^\n]+\n(?:[ \t]*\n|pxh>)",block):
            result[titles[i]]=fields;continue
        for key,value in re.findall(r"(?m)^\s{4}([a-zA-Z0-9_]+):\s+([^\n]+)$",block):
            token=value.split()[0]
            if token in ("True","False"):fields[key]=token=="True"
            else:
                try:fields[key]=float(token)
                except ValueError:continue
        # Deliberately replace old complete records with partial new records.
        result[titles[i]]=fields
    return result

def live_heading_ready(console,time_usec):
    return heading_topics_ready(latest_topics(console),time_usec)

def heading_topics_ready(topics,time_usec,age_limits=None):
    requirements={"vehicle_local_position":{"xy_valid":True,"z_valid":True,
          "v_xy_valid":True,"v_z_valid":True,"heading_good_for_control":True},
       "estimator_status_flags":{"cs_tilt_align":True,"cs_yaw_align":True,
          "cs_mag_aligned_in_flight":True,"cs_in_air":True,"cs_gnss_pos":True,
          "cs_mag_fault":False,"cs_baro_fault":False,"cs_gnss_fault":False,
          "cs_gnss_yaw_fault":False,"cs_gnss_hgt_fault":False},
       "vehicle_status":{"pre_flight_checks_pass":True,"failsafe":False}}
    for name,flags in requirements.items():
        fields=topics.get(name,{})
        timestamp=fields.get("timestamp",float("nan"))
        limit=1000000 if age_limits is None else age_limits[name]
        if not np.isfinite(timestamp) or not 0<=time_usec-timestamp<=limit:return False
        if any(fields.get(key) is not value for key,value in flags.items()):return False
    local=topics["vehicle_local_position"];flags=topics["estimator_status_flags"];status=topics["vehicle_status"]
    return bool(np.isfinite(local.get("heading",float("nan")))
                and (flags.get("cs_mag_hdg") is True or flags.get("cs_mag_3d") is True)
                and status.get("arming_state")==2 and status.get("nav_state")==14)

def settling_time(times,errors,rates,step_time):
    t,e,w=map(lambda x:np.asarray(x,dtype=float),(times,errors,rates))
    if t.ndim!=1 or e.shape!=t.shape or w.shape!=t.shape or not np.isfinite(np.r_[t,e,w,step_time]).all():
        raise ValueError("finite aligned1D trajectory required")
    if len(t)<2 or np.any(np.diff(t)<=0):return None
    good=(np.abs(e)<np.deg2rad(2))&(np.abs(w)<.05)
    suffix=np.logical_and.accumulate(good[::-1])[::-1]
    for i in np.flatnonzero((t>=step_time)&suffix):
        if t[-1]-t[i]>=2 and np.max(np.diff(t[i:]))<=.15:
            return float(t[i]-step_time)
    return None
