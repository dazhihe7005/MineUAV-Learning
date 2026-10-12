"""Offline Phase2F analysis tests; this module never starts PX4 or MuJoCo."""
import copy
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

from .test_interfaces import module


def audit():
    return module('analyze_yaw_closed_loop')


def saved_recording():
    root=Path(__file__).resolve().parents[2]
    rawpath=root/'mujoco/reports/px4_yaw_closed_loop_parts/yaw_01/smoke_summary.json'
    if not rawpath.exists():pytest.skip('single raw SITL recording is local-only')
    return json.loads(rawpath.read_text()),rawpath


def test_onset_requires_two_consecutive_poststep_samples():
    t=[31.98,32.,32.02,32.04,32.06]
    assert audit().onset_after_step(t,[.1,.005,.03,.04,.001],32.,.02)==pytest.approx(32.02)
    assert audit().onset_after_step(t,[.1,.005,.03,.001,.04],32.,.02) is None
    assert audit().onset_after_step(t,[-.1,-.005,-.03,-.04,-.001],32.,.02,direction=-1)==pytest.approx(32.02)


def test_onset_rejects_noncausal_or_nonfinite_samples():
    with pytest.raises(ValueError):audit().onset_after_step([32.,31.],[0,.1],32.,.02)
    with pytest.raises(ValueError):audit().onset_after_step([32.,32.02],[0,np.nan],32.,.02)


def test_motor_thrust_and_frd_reaction_torque_keep_physical_signs():
    out=audit().motor_model_response([[.2,.4,.2,.4]],10.,.2,[1,-1,1,-1])
    np.testing.assert_allclose(out['thrust_n'],[[2.,4.,2.,4.]])
    np.testing.assert_allclose(out['reaction_torque_frd_nm'],[[-.04,.08,-.04,.08]])
    assert out['total_reaction_yaw_frd_nm'].tolist()==pytest.approx([.08])


def test_motor_model_rejects_out_of_range_commands():
    with pytest.raises(ValueError):audit().motor_model_response([[0,0,0,1.01]],10.,.2,[1,-1,1,-1])


def test_status_does_not_turn_estimator_or_allocator_only_response_into_physical_success():
    evidence={'step':32.,'complete':True,'settling_time_s':None,'physical_change_deg':0.,
        'motor_yaw_torque_max_nm':.2,'allocator_achieved_count':65,'allocator_sample_count':65,
        'unallocated_yaw_max_abs':1e-8,'motor_saturation_count':0,'bounds_valid':True,
        'ekf_healthy':True,'query_healthy':True,'px4_setpoint_observed':True,
        'rate_request_observed':True,'torque_request_observed':True}
    result=audit().control_chain_status(evidence)
    assert result['allocator_torque']=='PASS'
    assert result['physical_yaw_direction']=='FAIL'
    assert result['yaw_settling']=='FAIL'
    assert result['control_chain_prerequisite']=='BLOCKED'
    evidence['step']=None
    assert audit().control_chain_status(evidence)['physical_yaw_direction']=='NOT TESTED'


def test_status_pass_requires_actual_positive_yaw_and_original_settling():
    evidence={'step':32.004,'complete':True,'settling_time_s':3.36,'physical_change_deg':9.9,
        'motor_yaw_torque_max_nm':.2,'allocator_achieved_count':65,'allocator_sample_count':65,
        'unallocated_yaw_max_abs':1e-8,'motor_saturation_count':0,'bounds_valid':True,
        'ekf_healthy':True,'query_healthy':True,'px4_setpoint_observed':True,
        'rate_request_observed':True,'torque_request_observed':True}
    result=audit().control_chain_status(evidence)
    assert result['physical_yaw_direction']=='PASS'
    assert result['yaw_settling']=='PASS'
    assert result['control_chain_prerequisite']=='PASS'
    assert result['phase3_entry']=='NOT TESTED'


def test_saved_run_safety_audit_rejects_truncated_query_tail():
    raw,_=saved_recording()
    assert audit().flight_safety_checks(raw)['overall']=='PASS'
    damaged=copy.deepcopy(raw)
    for _ in range(3):
        damaged['runtime_measurements']['queries'].pop()
        damaged['phase2']['telemetry_fix']['events']=[e for e in damaged['phase2']['telemetry_fix']['events']
            if not (e['kind']=='query' and e['clock_usec']>damaged['runtime_measurements']['queries'][-1]['start_clock_usec'])]
    damaged['runtime_measurements']['scheduler']['starts']-=3
    damaged['runtime_measurements']['row_valid_at_completions']-=3
    damaged['runtime_measurements']['samples']['query_response_seconds']['n']-=3
    assert audit().flight_safety_checks(damaged)['overall']=='FAIL'


def test_saved_run_safety_audit_rejects_physical_envelope_violation():
    raw,_=saved_recording()
    damaged=copy.deepcopy(raw)
    damaged['records'][1200]['position_enu_m'][2]=3.0
    assert audit().flight_safety_checks(damaged)['overall']=='FAIL'


@pytest.mark.parametrize('mutation', ['first_only','missing_tail','interior_gap','backwards_clock'])
def test_saved_run_safety_audit_rejects_incomplete_physical_trace(mutation):
    raw,_=saved_recording()
    damaged=copy.deepcopy(raw)
    records=damaged['records']
    if mutation=='first_only':damaged['records']=records[:1]
    elif mutation=='missing_tail':damaged['records']=records[:2000]
    elif mutation=='interior_gap':damaged['records']=records[:1000]+records[1010:]
    else:damaged['records'][1000]['clock_usec']-=20000
    result=audit().flight_safety_checks(damaged)
    assert result['checks']['physical_record_time_coverage']=='FAIL'
    assert result['overall']=='FAIL'


def test_report_rejects_evidence_not_matching_frozen_single_run_hash(tmp_path):
    raw,_=saved_recording()
    raw['records'][100]['position_enu_m'][0]+=.001
    altered=tmp_path/'smoke_summary.json'
    altered.write_text(json.dumps(raw))
    with pytest.raises(ValueError,match='raw summary SHA256'):
        audit().build_report(altered,write=False)


def test_report_recomputes_single_saved_run_without_running_sitl():
    raw,path=saved_recording()
    if os.environ.get('MINEUAV_PYULOG_SEARCH_PATH'):
        sys.path.append(os.environ['MINEUAV_PYULOG_SEARCH_PATH'])
    pytest.importorskip('pyulog')
    report=audit().build_report(path,write=False)
    assert report['raw_evidence']['summary_sha256']==module('diagnostics').sha256(path)
    assert report['experiment']['sitl_runs']==1
    assert report['experiment']['simulation_seconds']==pytest.approx(45.)
    assert report['yaw_response']['settling_time_s']==pytest.approx(3.36,abs=.03)
    assert report['status']['control_chain_prerequisite']=='PASS'
    assert report['status']['phase3_entry']=='NOT TESTED'
    assert report['raw_evidence']['runtime_source_hashes']==raw['streaming_runtime_hashes']


def test_offline_plots_emit_six_evidence_figures_without_simulator(tmp_path):
    t=np.array([31.98,32.,32.02,32.04,32.06,32.08,32.1])
    q=np.deg2rad(np.array([90,90,90,91,92,93,94],dtype=float))
    trace={'t':t,'true':q,'estimate':q+.001,'desired':np.deg2rad([90,90,100,100,100,100,100]),
        'rate':np.array([0,0,.02,.08,.1,.08,.04]),
        'motors':np.tile([.4,.4,.46,.46],(len(t),1)),
        'position':np.column_stack([np.zeros(len(t)),np.zeros(len(t)),np.full(len(t),2.051)]),
        'local':{'timestamp':np.array([33000000,33100000]),'heading_good_for_control':np.array([1,1])},
        'flags':{'timestamp':np.array([33000000,33100000]),'cs_mag_aligned_in_flight':np.array([1,1])}}
    raw={'phase2':{'actual_step_time_s':32.},'runtime_measurements':{'queries':[
        {'start_wall_s':32.,'end_wall_s':32.005},{'start_wall_s':32.5,'end_wall_s':32.504}]}}
    aux={'vehicle_rates_setpoint':{'timestamp':np.array([33000000,33020000]),'yaw':np.array([0,.2])},
        'vehicle_torque_setpoint':{'timestamp':np.array([33000000,33020000]),'xyz[2]':np.array([0,.03])},
        'control_allocator_status':{'timestamp':np.array([33000000,33100000]),'unallocated_torque[2]':np.array([0,0]),
            'torque_setpoint_achieved':np.array([1,1])}}
    files=module('yaw_closed_loop_figures').write_figures(tmp_path,trace,raw,aux,{'status':{'yaw_settling':'PASS'}})
    assert len(files)==6
    assert all((tmp_path/name).exists() for name in files)
