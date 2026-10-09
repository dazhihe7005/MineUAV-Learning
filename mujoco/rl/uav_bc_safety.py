"""Atomic artifact publication and sequential, memory-guarded phase supervisor."""
import argparse,fcntl,json,os,signal,subprocess,tempfile,time
from pathlib import Path

GIB=1024**3

def atomic_bytes(path,writer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:writer(f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        d=os.open(path.parent,os.O_DIRECTORY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def atomic_json(path,value):
    payload=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
    atomic_bytes(path,lambda f:f.write(payload))

def atomic_npz(path,**arrays):
    import numpy as np
    atomic_bytes(path,lambda f:np.savez_compressed(f,**arrays))

def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('cannot measure physical memory; refuse unguarded execution')

def oom_count():
    return int(next(x.split()[1] for x in Path('/proc/vmstat').read_text().splitlines() if x.startswith('oom_kill ')))

def check_budget(workers,available):
    if workers not in (1,2):raise ValueError('at most2 simulation workers, positive count required')
    if available<5.5*GIB:raise MemoryError('need4GiB phase budget plus1.5GiB system reserve')

def check_running(available,rss,oom_delta):
    if available<2*GIB or rss>4*GIB or oom_delta>0:
        raise MemoryError('stop phase: physical-memory headroom/RSS/OOM guard')

def tree_memory(pid):
    pending=[pid];seen=set();rss=hwm=0
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p)
        try:
            for line in Path(f'/proc/{p}/status').read_text().splitlines():
                if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
                if line.startswith('VmHWM:'):hwm+=int(line.split()[1])*1024
            pending.extend(map(int,Path(f'/proc/{p}/task/{p}/children').read_text().split()))
        except (FileNotFoundError,ProcessLookupError):continue
    return rss,hwm

def supervise(command,directory,phase):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'phase.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        check_budget(1,available_memory());start_oom=oom_count();start=time.monotonic()
        process=None;peak=peak_hwm=0;minimum=available_memory();failure=None
        def terminate_signal(signum,frame):raise InterruptedError(f'received signal {signum}')
        previous_term=signal.signal(signal.SIGTERM,terminate_signal)
        try:
            process=subprocess.Popen(command,start_new_session=True)
            while process.poll() is None:
                free=available_memory();rss,hwm=tree_memory(process.pid)
                minimum=min(minimum,free);peak=max(peak,rss);peak_hwm=max(peak_hwm,hwm)
                check_running(free,rss,oom_count()-start_oom);time.sleep(.1)
        except BaseException as e:
            # Never release the phase lock while an abnormal exit leaves its
            # owned child running. Includes SIGTERM and monitoring errors.
            failure=str(e) or type(e).__name__
        finally:
            signal.signal(signal.SIGTERM,signal.SIG_IGN)
            try:
                if process is not None and process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:os.killpg(process.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        process.wait()
            finally:signal.signal(signal.SIGTERM,previous_term)
        result=dict(phase=phase,workers=1,process_tree_peak_sampled_rss_bytes=peak,
            process_tree_peak_hwm_bytes=peak_hwm,minimum_available_bytes=minimum,
            oom_count_delta=oom_count()-start_oom,exit_code=process.wait() if process is not None else None,failure=failure,
            elapsed_seconds=time.monotonic()-start,poll_seconds=.1,phase_rss_limit_bytes=4*GIB,
            system_reserve_bytes=int(1.5*GIB),early_abort_available_bytes=2*GIB)
        atomic_json(directory/f'resource_{phase}.json',result)
        if failure or result['exit_code']:raise RuntimeError(f'phase incomplete, atomic completed records preserved: {result}')
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--phase',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('child command required')
    supervise(command,a.directory,a.phase)
