"""Phase2D offline-only benchmarks/log audit. Never imports a PX4 launch entry.

PYTHONPATH=mujoco python -m px4_bridge.analyze_query_scheduling
"""
import argparse
import json
import time
import tracemalloc
from pathlib import Path
import numpy as np
from .allocation_runner import allocation_status
from .streaming_diagnostics import StreamingDiagnostics
from .telemetry_freshness import TelemetryFreshness
from .yaw_telemetry_runner import TelemetryYawProtocol
from .query_scheduler import BoundedQueryScheduler
from .yaw_validation import latest_topics
from .diagnostics import sha256,atomic_json,own_peak_rss_bytes,require_memory,available_memory

ROOT=Path(__file__).resolve().parents[2]
PARTS=ROOT/'mujoco/reports'
ROW=[-.017207708,.017207708,-.017207708,.017207708]
DELTA='INFO  [control_allocator] MineUAV yaw axis: '+ ' '.join(map(str,ROW))+'\n'

def summary(values):
    a=np.asarray(values,dtype=float)
    if not a.size or not np.isfinite(a).all():raise ValueError('finite nonempty measurements required')
    return dict(n=len(a),mean=float(a.mean()),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),
                max=float(a.max()),median=float(np.median(a)))

def stopping_evidence(raw):
    fix=raw['phase2']['telemetry_fix'];q=[e for e in fix['events'] if e['kind']=='query'];d=fix['last_decision']
    return {'fault':fix['fault'],'query_service_deadline_wall_s':q[-1]['wall_s']+.65,
        'last_actuator_age_s':d['wall_s']-d['last_actuator'][1],
        'actuator_deadline_wall_s':d['last_actuator'][1]+.25,
        'last_five_query_intervals_s':np.diff([e['wall_s'] for e in q])[-5:].tolist(),
        'query_interval_s':summary(np.diff([e['wall_s'] for e in q])),
        'step_sent':raw['phase2']['actual_step_time_s'] is not None,
        'simulation_stop_s':raw['final_physical_state']['time_s'],
        'wall_end_s':raw['wall_seconds'],'last_check_wall_s':d['wall_s']}

def log_identity(path):
    before=sha256(path);raw=json.loads(path.read_text());p=StreamingDiagnostics();count=0;rows=0
    for i in range(0,len(raw['px4_console']),1024):
        for e in p.feed(raw['px4_console'][i:i+1024]):
            count+=e['kind']=='topic';rows+=e['kind']=='yaw_row'
    assert sha256(path)==before
    return {'path':str(path.relative_to(ROOT)),'sha256':before,'console_chars':len(raw['px4_console']),
        'console_sha256':__import__('hashlib').sha256(raw['px4_console'].encode()).hexdigest(),
        'primary_topic_records':count,'allocator_row_records':rows,'parser_peak_chars':p.peak_buffer_chars,
        'query_count':sum(e['kind']=='query' for e in raw['phase2']['telemetry_fix']['events'])}

def preservation_evidence():
    previous=json.loads((PARTS/'px4_mujoco_yaw_allocation_fix.json').read_text())
    changed={'mujoco/px4_bridge/telemetry_freshness.py','mujoco/px4_bridge/yaw_telemetry_runner.py'}
    out={}
    for name,item in previous['preservation']['protected_files'].items():
        if name in changed:continue # explicitly authorized monitoring edits, not historical model changes
        current=sha256(ROOT/name)
        assert current==item['current_sha256'],'protected artifact changed: '+name
        out[name]={'sha256':current,'unchanged_from_phase2c':True}
    runtime={name:sha256(ROOT/'mujoco/px4_bridge'/name) for name in
        ('simulation_runner.py','query_scheduler.py','streaming_diagnostics.py','telemetry_freshness.py',
         'yaw_telemetry_runner.py','allocation_runner.py','transport.py','yaw_validation.py')}
    return {'protected_artifacts':out,'new_runtime_sha256':runtime,'runtime_not_sitl_verified':True}

def fixture_text(topics,clock):
    # Explicit constructed timing fixtures, NOT reconstructed PTY arrival data.
    return ''.join('TOPIC: '+name+'\n'+''.join('    '+key+': '+str(clock if key=='timestamp' else min(value,clock) if key=='timestamp_sample' else value)+'\n'
        for key,value in fields.items())+'\n' for name,fields in topics.items() if name in TelemetryFreshness.SOURCE_LIMITS)

def schedule_simulation():
    p=TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]});s=BoundedQueryScheduler(p);starts=[]
    raw=json.loads((PARTS/'px4_yaw_telemetry_fix_parts/yaw_01/smoke_summary.json').read_text())
    topics=latest_topics(raw['px4_console']);fault=None
    for i in range(3001):
        wall=i*.004;clock=33000000+int(wall*.8*1e6)
        p.on_actuator(clock,wall)
        s.service(clock,wall,lambda b:starts.append(wall))
        if p.monitor.query:p.on_console_chunk(fixture_text(topics,clock),clock,wall)
        decision=p.monitor.decision(clock,wall)
        if decision['safe_stop']:fault=decision['reason'];break
    return {'kind':'constructed offline timing simulation','sim_rate_wall_ratio':.8,
        'queries':len(starts),'query_interval_s':summary(np.diff(starts)),
        'fault':fault,'max_scheduling_lateness_s':s.max_lateness_s,
        'pending_queue_max':1,'not_a_live_px4_measurement':True}

def fault_detection_simulation():
    """Causal 4ms-check fixtures, not original wire arrivals or live latency."""
    raw=json.loads((PARTS/'px4_yaw_telemetry_fix_parts/yaw_01/smoke_summary.json').read_text())
    topics=latest_topics(raw['px4_console']);healthy=fixture_text(topics,33000000);cases=[]
    for scenario,deadline in [('query_reply_stopped',.15),('actuator_stopped_sim_stalled',.25),
            ('query_service_stopped_sim_nearly_stalled',.65),('duplicate_publication',.65),('received_failsafe',.004)]:
        p=TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]})
        p.query_requested(33000000,0.,'');p.on_console_chunk(healthy,33000000,0.);p.on_actuator(33000000,0.)
        if scenario=='query_reply_stopped':p.query_requested(33000000,0.,'')
        for i in range(1,201):
            wall=i*.004
            clock=33000000 if scenario=='actuator_stopped_sim_stalled' else 33000000+i if 'nearly_stalled' in scenario else 33000000+i*4000
            if scenario!='actuator_stopped_sim_stalled':p.on_actuator(clock,wall)
            if scenario=='duplicate_publication' and i==125:
                p.query_requested(clock,wall,'');p.on_console_chunk(healthy,clock,wall)
            if scenario=='received_failsafe' and i==1:
                p.query_requested(clock,wall,'')
                p.on_console_chunk(fixture_text(topics,clock).replace('failsafe: False','failsafe: True'),clock,wall)
            d=p.monitor.decision(clock,wall)
            if d['safe_stop']:
                reason=p.monitor.fault
                # Only AFTER detection deliver late healthy data; it cannot
                # alter the recorded detection time or erase the latch.
                p.on_console_chunk(fixture_text(topics,clock),clock,wall)
                cases.append({'scenario':scenario,'deadline_s':deadline,'detected_wall_s':wall,
                    'overshoot_after_deadline_s':wall-deadline,'reason':reason,
                    'simulation_progress_usec_per_check':0 if scenario=='actuator_stopped_sim_stalled' else 1 if 'nearly_stalled' in scenario else 4000,
                    'detection_delay_after_received_record_s':0. if scenario=='received_failsafe' else None,
                    'late_health_cleared_fault':p.monitor.fault is None})
                break
        else:raise AssertionError('constructed fault was not detected: '+scenario)
    return {'kind':'constructed causal fault-injection fixtures','check_period_s':.004,
        'not_a_live_px4_measurement':True,'cases':cases,
        'host_descheduling_or_blocked_disk_not_bounded_by_this_fixture':True}

def measure(call,repetitions=100):
    out=[]
    for _ in range(repetitions):
        t=time.perf_counter_ns();call();out.append((time.perf_counter_ns()-t)/1000.)
    return out

def benchmark(console,batches=5,repetitions=100):
    results=[]
    for size in (1024,10240,102400,265803,1048576):
        # Real console pattern repeated/truncated to exact requested character
        # size; these are workload fixtures, not longer actual flight logs.
        history=(console*((size//len(console))+1))[:size]
        whole=history+DELTA;old=[];new=[];batch_results=[]
        p=StreamingDiagnostics()
        # Only complete real records are needed to precondition parser state;
        # arbitrary truncation in the scaling fixture is not treated as data.
        for i in range(0,len(console),4096):p.feed(console[i:i+4096])
        before=p.processed_chars
        for _ in range(batches):
            a=measure(lambda:allocation_status(whole,ROW),repetitions)
            b=measure(lambda:p.feed(DELTA),repetitions)
            old+=a;new+=b;batch_results.append({'old_us':summary(a),'incremental_us':summary(b)})
        tracemalloc.start();allocation_status(whole,ROW);old_peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
        tracemalloc.start();parsed=p.feed(DELTA);new_peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
        assert len(parsed)==1 and parsed[0]['kind']=='yaw_row'
        results.append({'history_chars':size,'delta_chars':len(DELTA),'old_us':summary(old),'incremental_us':summary(new),
            'batch_results':batch_results,'incremental_processed_chars':p.processed_chars-before,
            'expected_delta_work_chars':len(DELTA)*(batches*repetitions+1),'peak_parser_chars':p.peak_buffer_chars,
            'verified_fixed_delta_record_count':len(parsed),
            'tracemalloc_single_old_peak_bytes':old_peak,'tracemalloc_single_delta_peak_bytes':new_peak})
        require_memory()
    return results

def query_benchmark(topics,batches,repetitions):
    text=fixture_text(topics,33000000);old=[];new=[];message=[];old_repeated=[]
    for _ in range(batches):
        def legacy():
            m=TelemetryFreshness();m.begin_query(33000000,0.);m.observe(text,33000000,.005)
            assert m.query is None and m.fault is None
        def stream():
            p=TelemetryYawProtocol({'field_ned_gauss':[.2,.01,.4]});p.query_requested(33000000,0.,'')
            for i in range(0,len(text),1024):p.on_console_chunk(text[i:i+1024],33000000,.005)
            assert p.monitor.query is None and p.monitor.fault is None
        def legacy_repeated():
            m=TelemetryFreshness();m.begin_query(33000000,0.)
            for end in range(1024,len(text)+1024,1024):m.observe(text[:end],33000000,.005)
            assert m.query is None and m.fault is None
        old+=measure(legacy,repetitions);new+=measure(stream,repetitions)
        old_repeated+=measure(legacy_repeated,repetitions)
        p=StreamingDiagnostics();message+=measure(lambda:p.feed(DELTA),repetitions)
    return {'full_query_legacy_us':summary(old),'full_query_incremental_us':summary(new),
        'old_repeated_suffix_query_us':summary(old_repeated),
        'single_new_allocator_record_us':summary(message),'query_chars':len(text),
        'includes_parser_construction':True,'not_actual_console_query_response_time':True,
        'legacy_baseline_kind':'retained whole-response parsing path with current shared validation, not frozen base validator CPU cost'}

def figures(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=PARTS/'figures/px4_query_scheduling';folder.mkdir(parents=True,exist_ok=True)
    r=report['parsing_benchmark'];x=[v['history_chars']/1024 for v in r]
    fig,ax=plt.subplots(figsize=(7,4))
    for name,label in [('old_us','old accumulated scan'),('incremental_us','new fixed delta')]:
        ax.plot(x,[v[name]['mean'] for v in r],marker='o',label=label)
        ax.fill_between(x,[v[name]['mean'] for v in r],[v[name]['p95'] for v in r],alpha=.15)
    ax.set(xscale='log',yscale='log',xlabel='Historical text KiB (constructed workloads)',ylabel='µs / check (mean, band to P95)');ax.legend();fig.tight_layout()
    fig.savefig(folder/'diagnostic_parse_scaling.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    q=report['phase2c_stopping_evidence'];ax.plot(q['last_five_query_intervals_s'],marker='o',label='observed Phase2C final intervals')
    ax.axhline(.5,color='g',label='nominal period');ax.axhline(.65,color='r',label='unchanged service limit')
    ax.set(xlabel='Last five intervals',ylabel='wall seconds');ax.legend();fig.tight_layout();fig.savefig(folder/'observed_query_drift.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));names=['legacy parsing path','new chunked query','new single record']
    qb=report['query_benchmark'];values=[qb[k]['mean'] for k in ['full_query_legacy_us','full_query_incremental_us','single_new_allocator_record_us']]
    ax.bar(names,values);ax.set(ylabel='offline processing µs (not wire round trip)');fig.tight_layout();fig.savefig(folder/'query_processing_cost.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batches',type=int,default=5);p.add_argument('--repetitions',type=int,default=100)
    p.add_argument('--verification',type=Path,help='Optional separately recorded final test/review evidence JSON')
    args=p.parse_args()
    if not 3<=args.batches<=10 or not 20<=args.repetitions<=200:p.error('bounded repeated measurement budget')
    require_memory();identities=[]
    for phase in ('px4_yaw_telemetry_fix_parts','px4_yaw_allocation_parts'):
        for run in ('disarmed_01','yaw_01'):
            path=PARTS/phase/run/'smoke_summary.json'
            identities.append(log_identity(path))
            for log in sorted(path.parent.rglob('*.ulg')):identities.append({'path':str(log.relative_to(ROOT)),'sha256':sha256(log)})
    raw=json.loads((PARTS/'px4_yaw_allocation_parts/yaw_01/smoke_summary.json').read_text())
    reference=json.loads((PARTS/'px4_yaw_telemetry_fix_parts/yaw_01/smoke_summary.json').read_text())
    report={'phase':'2D offline only','base_commit':'26f3df6ed1417c9c9b90696809a2c80d99a43ff5',
        'new_px4_processes':0,'new_mujoco_episodes':0,'limits_unchanged':{'query_response_s':.15,'actuator_watchdog_s':.25,'query_service_s':.65,'source_limits_usec':TelemetryFreshness.SOURCE_LIMITS},
        'log_identities':identities,'phase2c_stopping_evidence':stopping_evidence(raw),
        'preservation':preservation_evidence(),
        'parsing_benchmark':benchmark(raw['px4_console'],args.batches,args.repetitions),
        'query_benchmark':query_benchmark(latest_topics(reference['px4_console']),args.batches,args.repetitions),
        'constructed_timing_simulation':schedule_simulation(),
        'constructed_fault_detection':fault_detection_simulation(),
        'bounds':{'line_chars':8192,'record_chars':32768,'slice_chars':4096,'recent_console_chars':65536,'final_archive_bytes':16777216,'event_count':8192},
        'limitations':['Original PTY read boundaries/arrival times and per-query CPU/I/O timings were not recorded; cannot uniquely attribute live timeout.',
            'Allocator console row has no PX4 source publication timestamp; it is a bounded host-received query snapshot, not a health heartbeat.',
            'Offline benchmarks and synthetic timing are not live PX4 or hard-real-time certification. Changed runtime requires new authorized disarmed and bounded flight verification.']}
    if args.verification:
        report['verification']=json.loads(args.verification.read_text())
    report['resources']={'workers':1,'analysis_peak_rss_bytes':own_peak_rss_bytes(),'available_memory_bytes_at_end':available_memory(),'oom_observed':False}
    figures(report)
    report['resources']['analysis_peak_rss_bytes']=own_peak_rss_bytes()
    for item in identities:assert sha256(ROOT/item['path'])==item['sha256']
    atomic_json(PARTS/'px4_mujoco_query_scheduling.json',report)
    print(json.dumps({'peak_rss_bytes':report['resources']['analysis_peak_rss_bytes'],'fixed_delta_means_us':[x['incremental_us']['mean'] for x in report['parsing_benchmark']]}))
    return 0

if __name__=='__main__':raise SystemExit(main())
