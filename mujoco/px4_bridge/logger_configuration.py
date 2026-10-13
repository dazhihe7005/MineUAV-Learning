"""Pinned PX4 profile0 + explicit single-instance logger file in private etc."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from .compatibility_identity import tree_hash
from .diagnostics import sha256

REQUIRED_TOPICS=("vehicle_visual_odometry","estimator_status_flags",
                 "estimator_aid_src_ev_pos","estimator_aid_src_ev_vel",
                 "estimator_aid_src_gnss_pos","estimator_aid_src_gnss_vel",
                 "vehicle_local_position","vehicle_status","vehicle_gps_position")
LOGGER_TOPICS=tuple((name,0) for name in REQUIRED_TOPICS if name!="vehicle_local_position") + (
    ("vehicle_local_position",20),("actuator_outputs_sim",0),("actuator_motors",20),
    ("control_allocator_status",20),("vehicle_command",0),("vehicle_control_mode",0),
    ("failsafe_flags",0),("sensor_combined",20),("vehicle_attitude",20),
    ("logger_status",100),("cpuload",100))
MAX_SUBSCRIPTIONS=255
CONFIG_RELATIVE=Path("logging/logger_topics.txt")


def render_topics():
    return "".join(f"{name} {interval} 0\n" for name,interval in LOGGER_TOPICS)


def compiled_topic_names(binary):
    output=subprocess.check_output(["nm","-g","--defined-only",str(binary)],text=True)
    names=set(re.findall(r"\b__orb_([a-z][a-z0-9_]*)$",output,re.M))
    if not names:raise ValueError("no compiled uORB topic identities available")
    return names


def audit_topics(text, available):
    subscriptions=[]
    for line in text.splitlines():
        if not line or line.startswith("#"):continue
        fields=line.split()
        if len(line.encode())>=79 or len(fields)!=3:
            raise ValueError("explicit short topic/interval/instance line required")
        name,interval,instance=fields
        if (not re.fullmatch(r"[a-z][a-z0-9_]*",name) or name not in available
                or not interval.isdigit() or not 0<=int(interval)<=65535 or instance!="0"):
            raise ValueError("unknown topic or invalid explicit instance/interval")
        if name in [row[0] for row in subscriptions]:raise ValueError("duplicate topic instance")
        subscriptions.append((name,int(interval),0))
    if not 0<len(subscriptions)<=MAX_SUBSCRIPTIONS:raise ValueError("logger subscription limit exceeded")
    if not set(REQUIRED_TOPICS)<={row[0] for row in subscriptions}:raise ValueError("missing required evidence topic")
    return {"subscriptions":len(subscriptions),"instances":[0],
            "topics":[row[0] for row in subscriptions],
            "entries":[{"topic":name,"interval_ms":interval,"instance":instance}
                       for name,interval,instance in subscriptions],
            "text_sha256":hashlib.sha256(text.encode()).hexdigest(),
            "profile":0,"mission_profile":0,"limit":MAX_SUBSCRIPTIONS}


def startup_tree_hash(root):
    h=hashlib.sha256();root=Path(root)
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative=path.relative_to(root)
        if relative==CONFIG_RELATIVE:continue
        h.update(relative.as_posix().encode()+b"\0")
        h.update(sha256(path).encode()+b"\n")
    return h.hexdigest()


def prepare_private_rootfs(runtime, rootfs):
    runtime=Path(runtime);rootfs=Path(rootfs);private=runtime/"etc"
    if private.exists() or private.is_symlink():raise ValueError("runtime etc must be new and private")
    if (rootfs/CONFIG_RELATIVE).exists():raise ValueError("unexpected source logger file; no overwrite allowed")
    if any(path.is_symlink() and path.is_dir() for path in rootfs.rglob("*")):
        raise ValueError("directory symlink in source startup tree")
    original=tree_hash(rootfs)
    shutil.copytree(rootfs,private,symlinks=False)
    if tree_hash(private)!=original:raise ValueError("private startup copy differs from verified PX4 rootfs")
    destination=private/CONFIG_RELATIVE;destination.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temp=tempfile.mkstemp(prefix="logger_topics.",suffix=".tmp",dir=destination.parent)
    try:
        with os.fdopen(descriptor,"w") as stream:
            stream.write(render_topics());stream.flush();os.fsync(stream.fileno())
        os.replace(temp,destination)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    if startup_tree_hash(private)!=original or tree_hash(rootfs)!=original:
        raise ValueError("startup files changed while preparing logger")
    return {"startup_tree_sha256":original,"logger_file_sha256":sha256(destination),
            "relative_path":"etc/logging/logger_topics.txt","private_etc":True}
