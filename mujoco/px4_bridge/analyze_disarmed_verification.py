"""Read-only Phase2E assessor of a single existing raw SITL result."""
import math
import numpy as np
from .simulation_runner import streaming_runtime_identity


def _finite(number):
    return isinstance(number,(int,float)) and not isinstance(number,bool) and math.isfinite(number)


def _stats(values):
    if not values:return None
    a=np.asarray(values,dtype=float)
    if not np.isfinite(a).all():return None
    return {'n':len(values),'mean':float(a.mean()),'p50':float(np.median(a)),
        'p95':float(np.percentile(a,95)),'p99':float(np.percentile(a,99)),
        'max':float(a.max()),'min':float(a.min())}


def assess_run(raw):
    """Every PASS derives from recorded evidence; absent evidence cannot pass."""
    checks={}
    def check(name,ok):checks[name]='PASS' if ok else 'FAIL'
    phase=raw.get('phase2') or {};fresh=phase.get('telemetry_fix') or {}
    decision=fresh.get('last_decision') or {}
    metrics=raw.get('runtime_measurements') or {};samples=metrics.get('samples') or {}
    queries=metrics.get('queries') or [];events=fresh.get('events') or []
    monitor_queries=[e for e in events if e.get('kind')=='query']
    sent=raw.get('messages_sent') or {};received=raw.get('messages_received') or {}
    sim=raw.get('simulation_seconds')
    final_sim_end_s=(raw.get('final_physical_state') or {}).get('time_s')
    check('duration',raw.get('complete') is True and _finite(sim) and 16-1e-8<=sim<=16.008
        and _finite(final_sim_end_s) and abs(sim-final_sim_end_s)<1e-8)
    check('bidirectional_communication',sent.get('HIL_SENSOR')==4000 and sent.get('HIL_GPS')==160
        and received.get('HIL_ACTUATOR_CONTROLS',0)>=3750)
    check('no_arm',raw.get('kind')=='disarmed_communication_smoke'
        and raw.get('flight_command_sent') is False and phase.get('active') is False
        and not any(sent.get(name,0) for name in ('COMMAND_LONG','SET_MODE','SET_POSITION_TARGET_LOCAL_NED')))
    check('zero_motors_on_exit',raw.get('zero_control_on_stop')==[0.,0.,0.,0.]
        and all(r.get('ctrl_omega_squared')==[0.,0.,0.,0.] for r in raw.get('records',[]))
        and bool(raw.get('records')))
    check('px4_cleanup',raw.get('px4_exit_code')==0)
    check('no_safety_fault',raw.get('exception') is None and 'fault' in fresh and fresh['fault'] is None
        and decision.get('safe_stop') is False)
    check('history_preserved',raw.get('history_unchanged') is True
        and bool(raw.get('history_hashes_before'))
        and raw.get('history_hashes_before')==raw.get('history_hashes_after'))
    check('new_streaming_runtime',raw.get('streaming_runtime_hashes')==streaming_runtime_identity()
        and bool(metrics.get('samples')))
    check('query_count',len(queries)>=12 and len(queries)==len(monitor_queries)
        and len(queries)==(metrics.get('scheduler') or {}).get('starts'))
    qdur=[];qinterval=[];valid_q=True
    for i,q in enumerate(queries):
        try:
            start_wall_s,end_wall_s=q['start_wall_s'],q['end_wall_s']
            delta_clock=q['end_clock_usec']-q['start_clock_usec']
            valid_q&=all(_finite(v) for v in (start_wall_s,end_wall_s,delta_clock)) and 0<=start_wall_s<=end_wall_s and end_wall_s-start_wall_s<=.15+1e-9 and 0<=delta_clock<=150000
            if i and _finite(start_wall_s):
                gap=start_wall_s-queries[i-1]['start_wall_s'];qinterval.append(gap)
                simgap=q['start_clock_usec']-queries[i-1]['start_clock_usec']
                valid_q&=0<gap<=.65+1e-9 and 0<simgap<=650000 and start_wall_s>=queries[i-1]['end_wall_s']
            qdur.append(end_wall_s-start_wall_s)
            valid_q&=abs(start_wall_s-monitor_queries[i]['wall_s'])<1e-6
            valid_q&=q['start_clock_usec']==monitor_queries[i]['clock_usec']
        except (KeyError,IndexError,TypeError):valid_q=False
    check('query_deadlines',bool(queries) and valid_q
        and 7000000<=queries[0]['start_clock_usec']<=7650000
        and _finite(final_sim_end_s) and queries[-1]['start_clock_usec']>=1000000+int(final_sim_end_s*1e6)-650000
        and _finite(decision.get('clock_usec')) and _finite(decision.get('wall_s'))
        and 0<=decision['clock_usec']-queries[-1]['start_clock_usec']<=650000
        and 0<=decision['wall_s']-queries[-1]['start_wall_s']<=.65
        and 0<=1000000+int(final_sim_end_s*1e6)-decision['clock_usec']<=8000
        and _finite(raw.get('wall_seconds')) and 0<=raw['wall_seconds']-decision['wall_s']<=.65
        and (samples.get('query_response_seconds') or {}).get('n')==len(queries)
        and _finite((samples.get('query_response_seconds') or {}).get('max'))
        and (samples['query_response_seconds']['max']<=.15+1e-9))
    check('allocator_yaw_row',len(queries)>0 and metrics.get('row_valid_at_completions')==len(queries)
        and bool((phase.get('allocation_fix') or {}).get('expected_yaw_row')))
    check('actuator_watchdog',received.get('HIL_ACTUATOR_CONTROLS',0)>=3750
        and metrics.get('actuator_receipts')==received.get('HIL_ACTUATOR_CONTROLS')-raw.get('stale_bootstrap_messages_discarded',0)
        and (samples.get('actuator_interval_wall_seconds') or {}).get('n')==metrics.get('actuator_receipts',0)-1
        and _finite((samples.get('actuator_interval_wall_seconds') or {}).get('max'))
        and samples['actuator_interval_wall_seconds']['max']<=.25+1e-9)
    check('incremental_parser',_finite(metrics.get('parser_processed_chars'))
        and _finite(metrics.get('parser_peak_buffer_chars'))
        and metrics['parser_peak_buffer_chars']<=32768
        and _finite(metrics.get('recent_peak_chars')) and metrics['recent_peak_chars']<=65536
        and (samples.get('parse_validate_seconds') or {}).get('n',0)>0)
    check('instrumentation_complete','dropped' in metrics and metrics['dropped']=={})
    check('simulation_boundaries',_finite(raw.get('system_min_available_bytes'))
        and raw['system_min_available_bytes']>1.5*1024**3)
    derived={'checks':checks,'overall':'PASS' if all(v=='PASS' for v in checks.values()) else 'FAIL',
        'query_interval_wall_s':_stats(qinterval),'query_response_recomputed_s':_stats(qdur),
        'query_count':len(queries),'monitor_query_count':len(monitor_queries),
        'first_failed_checks':[name for name,v in checks.items() if v=='FAIL']}
    return derived
