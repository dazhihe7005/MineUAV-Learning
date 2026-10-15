"""Linux independent group lifetime guard, forked before importing Torch/MuJoCo."""
import os
import signal
import time
from research_ops.storage import OpsError,process_identity
from research_ops.environment import resource_status
from research_ops_v2.execution import process_live


def start_guard(parent,deadline,root):
    if os.getpgrp()!=os.getpid():raise OpsError('worker is not its own isolated process group')
    identity=process_identity()
    watcher=os.fork()
    if watcher==0:
        while True:
            try:
                if not process_live(parent) or not process_live(identity) or time.monotonic()>=deadline:
                    raise OpsError('authorized lifetime ended')
                r=resource_status(root)
                if r['mem_available_mib']<1536 or r['disk_free_bytes']<2*1024**3:raise OpsError('resource guard')
            except BaseException:
                os.killpg(os.getpgrp(),signal.SIGKILL);os._exit(2)
            time.sleep(.05)
    return watcher


def stop_guard(watcher):
    try:os.kill(watcher,signal.SIGTERM)
    except ProcessLookupError:pass
    os.waitpid(watcher,0)
