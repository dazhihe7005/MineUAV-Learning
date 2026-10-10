"""Exactly two old-manifest Scripted flights; fail-closed atomic telemetry cache."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

from uav_gust_recovery_audit import sha, digest
from uav_bc_safety import atomic_json, atomic_npz
from uav_pi_telemetry import PassiveTelemetry, verify_passive

BASE = 'a3e6ee4da8eb437f04d2e2c030f5be7b7b5d4737'
PARTS = 'uav_pi_internal_telemetry_parts'
OLD_PARTS = 'uav_bc_external_disturbance_seed0_parts'
OLD_MANIFEST = 'uav_bc_external_disturbance_manifest_seed0.json'
CONDITIONS = ('constant_medium', 'gust_medium')
CHECKPOINTS = {
    'uav_bc_mlp_seed0.pt': '7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c',
    'uav_bc_yaw_augmented_seed0.pt': 'd29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce',
    'joint_latent_world_model_v3_autonomous_consistency.pt': '42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9'}
SOURCES = ('rl/mine_uav_env.py','rl/ppo_pi_env.py','rl/audit_velocity_pi.py',
    'rl/test_env_scripted_policy.py','rl/uav_bc_external_disturbance.py','rl/uav_bc_robustness.py',
    'rl/decision_fidelity_snapshot.py','control/velocity_command_controller.py',
    'control/hover_controller.py','control/position_controller.py','control/control_allocator.py',
    'models/mine_uav_dynamics_v2.xml','models/rotor_actuators.xml','models/rotor_config.json')


def select_states(manifest):
    result=[]
    for c in CONDITIONS:
        matches=[s for s in manifest['states'] if s['condition']==c and s['index']==0]
        if len(matches)!=1: raise ValueError('fixed index000 state missing/duplicated')
        result.append(matches[0])
    if result[0]['target_id'] != result[1]['target_id']: raise ValueError('unpaired target')
    return result


def physical_fingerprint(snapshot):
    """Full state equality, excluding ONLY the intentionally different force schedule name."""
    from decision_fidelity_snapshot import fingerprint
    s=copy.copy(snapshot);s.environment=dict(snapshot.environment)
    s.environment.pop('force_condition')
    return fingerprint(s)


def identities(root):
    root=Path(root)
    models={n:sha(root/'mujoco/rl/models'/n) for n in CHECKPOINTS}
    if models!=CHECKPOINTS: raise ValueError('canonical checkpoint changed')
    return dict(checkpoints=models,unchanged_sources={n:sha(root/'mujoco'/n) for n in SOURCES})


def validate_completed(row, identity, task):
    if not row.get('completed') or row.get('identity')!=identity or row.get('task_id')!=task:
        raise ValueError('incomplete/stale/duplicate task identity')
    body={k:v for k,v in row.items() if k!='record_sha256'}
    if row.get('record_sha256')!=digest(body): raise ValueError('record hash mismatch')
    if sha(row['raw_path'])!=row['raw_sha256']: raise ValueError('telemetry raw hash mismatch')
    baseline=json.loads(Path(row['reference_json']).read_text())
    if sha(row['reference_json'])!=identity['references'][task]['record_sha256']:
        raise ValueError('original record changed')
    if sha(baseline['trace_path'])!=baseline['trace_sha256']:
        raise ValueError('original raw trajectory changed')
    with np.load(row['raw_path'],allow_pickle=False) as new, np.load(baseline['trace_path'],allow_pickle=False) as old:
        if verify_passive(old,new)!=row['passivity']: raise ValueError('cached passivity evidence mismatch')
        verify_telemetry(new,row['steps'])
    for field in ('steps','success','timeout','physical_failure','termination_reason','initial_snapshot_sha256'):
        if row[field]!=baseline[field]: raise ValueError('original episode accounting changed')
    return row


def verify_telemetry(z, steps):
    n=steps*4;t=z['control_time_s']
    if len(t)!=n: raise ValueError('100Hz control row missing/duplicated')
    np.testing.assert_allclose(t,np.arange(n)*.01,atol=1e-8,rtol=0)
    for key in z:
        if key.startswith('control_') and key!='control_update_counts' and (len(z[key])!=n or not np.isfinite(z[key]).all()):
            raise ValueError('unaligned/nonfinite internal telemetry: '+key)
    np.testing.assert_array_equal(z['control_policy_step'],np.arange(n)//4)
    np.testing.assert_array_equal(z['control_actual_ctrl_u'],z['control_rotor_command_u'])
    np.testing.assert_allclose(z['control_p_contribution_m_s2_derived']+z['control_i_contribution_m_s2'],
        z['control_output_before_limiting_m_s2'],atol=1e-12,rtol=0)
    np.testing.assert_array_equal(z['control_integral_before_m'][1:],z['control_integral_after_m'][:-1])
    ticks=np.rint(t/.002).astype(int)
    np.testing.assert_array_equal(z['control_external_force_world_n'],z['physics_forces'][ticks])
    boundary=np.arange(steps)*4
    np.testing.assert_allclose(z['control_position_m'][boundary],z['positions'][:-1],atol=0,rtol=0)
    np.testing.assert_allclose(z['control_commanded_velocity_m_s'][boundary],z['actions'][:,:3]*np.array([1.5,1.5,1.]),atol=0,rtol=0)


def run(root, verify_only=False, recover_verifier_error=False):
    from uav_bc_external_disturbance import DisturbanceEnv, initial_snapshot
    from uav_bc_external_evaluation import attach_physics,verify_arrays
    from uav_bc_robustness import rollout
    from test_env_scripted_policy import scripted_action
    from decision_fidelity_snapshot import capture,restore,fingerprint
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS
    m=json.loads((reports/OLD_MANIFEST).read_text())
    if digest({k:v for k,v in m.items() if k!='sha256'})!=m['sha256']:
        raise ValueError('old manifest content hash changed')
    selected=select_states(m);before=identities(root)
    # Existing code/config must still be the old acquisition implementation.
    for name,expected in m['identity']['source_sha256'].items():
        if name in SOURCES and before['unchanged_sources'][name]!=expected:
            raise ValueError('historical dynamics/interface source changed: '+name)
    refs={}
    for c in CONDITIONS:
        path=reports/OLD_PARTS/f'{c}_000_scripted.json';r=json.loads(path.read_text())
        if sha(r['trace_path'])!=r['trace_sha256']: raise ValueError('reference raw hash mismatch')
        refs[c]=dict(record_sha256=sha(path),raw_sha256=r['trace_sha256'])
    identity=dict(base_commit=BASE,reference_manifest_sha256=sha(reports/OLD_MANIFEST),
        reference_manifest_content_sha256=m['sha256'],references=refs,original=before,
        acquisition_source_sha256={n:sha(root/'mujoco/rl'/n) for n in ('uav_pi_telemetry.py','run_uav_pi_internal_telemetry.py')},
        selected_states=selected,workers=1,unique_experimental_episodes=2,comparison_atol=1e-12,comparison_rtol=0)
    frozen=parts/'selection.json'
    if frozen.exists():
        previous=json.loads(frozen.read_text())
        if previous!=identity:
            approved='21a38c12011f1a0da1a6b2ec2d4ebb5c8c259ea8b580e1548c3ea5212a11fc1b'
            a={k:v for k,v in previous.items() if k not in ('acquisition_source_sha256','maximum_new_episodes')}
            b={k:v for k,v in identity.items() if k not in ('acquisition_source_sha256','unique_experimental_episodes')}
            if (not recover_verifier_error or a!=b or
                previous['acquisition_source_sha256']['run_uav_pi_internal_telemetry.py']!=approved or
                previous['acquisition_source_sha256']['uav_pi_telemetry.py']!=identity['acquisition_source_sha256']['uav_pi_telemetry.py'] or
                any((parts/f'{c}.json').exists() for c in CONDITIONS)):
                raise ValueError('frozen selection/config changed')
            atomic_json(parts/'selection_attempt1.json',previous)
            atomic_json(parts/'verifier_recovery.json',dict(reason='Legacy25Hz control_update_counts mistaken for100Hz telemetry; first Constant flight completed but publication failed.',
                previous_selection_sha256=digest(previous),current_selection_sha256=digest(identity),
                previous_acquisition_source_sha256=previous['acquisition_source_sha256'],
                current_acquisition_source_sha256=identity['acquisition_source_sha256'],
                unchanged_experimental_conditions=True,technical_repeat_only='constant_medium',additional_execution_count=1))
            atomic_json(frozen,identity)
    else:
        if verify_only: raise ValueError('no acquisition selection')
        atomic_json(frozen,identity)
    env=DisturbanceEnv();rows=[];paired=[];acquired=0
    try:
        for s in selected:
            c=s['condition'];env.configure(c,s['direction_world']);snap,meta=initial_snapshot(env,s)
            if meta!=s['initial_state']: raise ValueError('old exact initial state cannot be reproduced')
            physical=physical_fingerprint(snap);paired.append(physical)
            prior=restore(env,snap)
            if fingerprint(capture(env,prior))!=fingerprint(snap): raise ValueError('exact reset failed')
            path=parts/f'{c}.json';lease=parts/f'{c}.started.json'
            if path.exists():
                row=validate_completed(json.loads(path.read_text()),identity,c)
                if row['paired_physical_controller_sha256']!=physical: raise ValueError('cached initial pair changed')
            else:
                if verify_only: raise ValueError('missing required completed episode')
                attempts=1
                if lease.exists():
                    if not recover_verifier_error or c!='constant_medium' or not (parts/'verifier_recovery.json').exists():
                        raise ValueError('prior incomplete attempt; no silent extra flight/retry')
                    previous_lease=json.loads(lease.read_text())
                    if (parts/f'{c}.attempt1.started.json').exists(): raise ValueError('technical retry budget exhausted')
                    atomic_json(parts/f'{c}.attempt1.started.json',previous_lease);attempts=2
                atomic_json(lease,dict(task_id=c,identity_sha256=digest(identity),started=True,execution_attempts=attempts))
                with PassiveTelemetry(env) as recorder:
                    row,z=rollout(env,snap,scripted_action)
                acquired+=1
                z=attach_physics(env,z)
                z.update(recorder.arrays(z['physics_force_times'],z['physics_forces']))
                raw=parts/f'{c}.npz';atomic_npz(raw,**z)
                verify_arrays(row,z,c,s['direction_world']);verify_telemetry(z,row['steps'])
                reference=reports/OLD_PARTS/f'{c}_000_scripted.json'
                oldrow=json.loads(reference.read_text())
                with np.load(oldrow['trace_path'],allow_pickle=False) as old:passivity=verify_passive(old,z)
                for f in ('steps','success','timeout','physical_failure','termination_reason','initial_snapshot_sha256'):
                    if row[f]!=oldrow[f]: raise ValueError('nontelemetry termination changed: '+f)
                row.update(task_id=c,identity=identity,raw_path=str(raw),raw_sha256=sha(raw),reference_json=str(reference),
                    passivity=passivity,paired_physical_controller_sha256=physical,completed=True,
                    control_samples=len(recorder.control_rows),physics_samples=len(z['physics_force_times']),execution_attempts=attempts)
                row['record_sha256']=digest(row);atomic_json(path,row)
                validate_completed(row,identity,c)
            rows.append(row)
            print(c,row['steps'],row['termination_reason'],'passive bitwise:',row['passivity']['bitwise_equal'],flush=True)
    finally:env.close()
    if len(rows)!=2 or len({r['task_id'] for r in rows})!=2 or len(set(paired))!=1:
        raise ValueError('incomplete/duplicate/unpaired two-episode cohort')
    after=identities(root)
    if after!=before: raise ValueError('controller/model/physics source mutated')
    result=dict(identity=identity,rows=rows,initial_physical_controller_sha256=paired[0],
        original_hashes_before=before,original_hashes_after=after,new_episodes_this_invocation=acquired,
        total_completed_new_episodes=2,total_flight_executions_including_technical_repeat=sum(r['execution_attempts'] for r in rows),
        bitwise_reproduced_all_original_traces=all(r['passivity']['bitwise_equal'] for r in rows))
    atomic_json(parts/('verification.json' if verify_only else 'acquisition.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify-only',action='store_true');p.add_argument('--recover-verifier-error',action='store_true');a=p.parse_args()
    run(Path(__file__).resolve().parents[2],a.verify_only,a.recover_verifier_error)
