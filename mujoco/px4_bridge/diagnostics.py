"""Finite clock, memory guard, hashes and recoverable atomic summaries."""
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile

class SimClock:
    def __init__(self):self.time_usec=1000000
    def advance(self,physics_steps):
        if not isinstance(physics_steps,int) or physics_steps<=0:raise ValueError("positive integer steps")
        self.time_usec+=physics_steps*2000
        return self.time_usec

class CommandGate:
    def __init__(self):self.last=-1
    def accept(self,timestamp):
        if not isinstance(timestamp,int) or timestamp<=self.last or timestamp<0:
            raise ValueError("nonmonotonic/stale actuator timestamp")
        self.last=timestamp

def available_memory():
    lines=Path("/proc/meminfo").read_text().splitlines()
    return int(next(line.split()[1] for line in lines if line.startswith("MemAvailable:")))*1024

def require_memory(available_bytes=None,reserve_bytes=int(1.5*1024**3)):
    available=available_memory() if available_bytes is None else available_bytes
    if available<reserve_bytes:raise MemoryError("less than 1.5GiB available; stop SITL")
    return available

def require_startup_memory(available_bytes=None):
    # Measured bridge~.55GiB + PX4~.02GiB; budget1GiB before compiling meshes.
    return require_memory(available_bytes,reserve_bytes=int(2.5*1024**3))

def rss_bytes(pid):
    try:
        lines=Path(f"/proc/{pid}/status").read_text().splitlines()
        return int(next(line.split()[1] for line in lines if line.startswith("VmRSS:")))*1024
    except (OSError,StopIteration):return 0

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()

def atomic_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=p.name+".",suffix=".tmp",dir=p.parent)
    try:
        with os.fdopen(fd,"w") as f:
            json.dump(value,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
        os.replace(temp,p)
    finally:
        if os.path.exists(temp):os.unlink(temp)

def own_peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
