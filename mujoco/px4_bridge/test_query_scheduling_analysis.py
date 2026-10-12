import json
from pathlib import Path
import pytest
from .test_interfaces import module

def test_measured_percentiles_and_nonfinite_rejection():
    m=module('analyze_query_scheduling')
    assert m.summary([1.,2.,3.,4.,5.])=={'n':5,'mean':3.,'p95':4.8,'p99':4.96,'max':5.,'median':3.}
    with pytest.raises(ValueError):m.summary([float('nan')])

def test_phase2c_deadline_recomputed_from_original_timestamps():
    m=module('analyze_query_scheduling');raw=Path(__file__).resolve().parents[1]/'reports/px4_yaw_allocation_parts/yaw_01/smoke_summary.json'
    r=m.stopping_evidence(json.loads(raw.read_text()))
    assert r['query_service_deadline_wall_s']==pytest.approx(16.085843422999822)
    assert r['fault']=='safety query service stopped'
    assert r['last_actuator_age_s']==pytest.approx(.002077584999824)
    assert r['step_sent'] is False

def test_deterministic_simulated_schedule_not_live_claim():
    r=module('analyze_query_scheduling').schedule_simulation()
    assert r['kind']=='constructed offline timing simulation'
    assert r['fault'] is None and r['queries']>=20
    assert r['query_interval_s']['max']<=.504001

def test_saved_log_streaming_matches_baseline_count_and_identity():
    m=module('analyze_query_scheduling');path=Path(__file__).resolve().parents[1]/'reports/px4_yaw_allocation_parts/yaw_01/smoke_summary.json'
    r=m.log_identity(path)
    assert r['console_chars']==265803 and r['query_count']==19
    #19 scheduled batches×3 +3 diagnostic listeners =60 captured primaries.
    assert r['primary_topic_records']==60 and len(r['sha256'])==64

def test_constructed_fault_detection_uses_causal_deadlines_and_latches():
    r=module('analyze_query_scheduling').fault_detection_simulation()
    assert r['not_a_live_px4_measurement'] is True
    cases={x['scenario']:x for x in r['cases']}
    for name,budget in [('query_reply_stopped',.15),('actuator_stopped_sim_stalled',.25),
                        ('query_service_stopped_sim_nearly_stalled',.65),('duplicate_publication',.65)]:
        case=cases[name]
        assert case['deadline_s']==pytest.approx(budget)
        assert budget < case['detected_wall_s'] <= budget+.004001
        assert case['late_health_cleared_fault'] is False
    assert cases['received_failsafe']['detection_delay_after_received_record_s']==0.
