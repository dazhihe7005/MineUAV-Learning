"""R29 fixed read-only counterfactual function probes; no training authority."""
import math
from pathlib import Path
from research_ops.storage import OpsError, read_json, sha256, digest
from research_ops.identity import git
from .runtime import ROOT, HOME
from .budget import BudgetStore

R28_COMMIT = '4a757f43e6564ff9f7d656ae59a2dafe03a4ff2d'
R28_REPORT_SHA = '54af04e2001f6612fea6a20d6af1837243a3a7b23c728ddf3d61d3b0921b35fc'
R28_PUBLICATION_SHA = '2fbbb6984d1b61842071d0d9bbced524a638271391a33b8bcb17767bcd879c1a'
R28_ID = 'ARL_R28_HOLD_GOAL_YAW_SEED0'


def probe_grid():
    rows = []
    for axis in range(3):
        for distance in (-1., -.5, -.2, -.05, 0., .05, .2, .5, 1.):
            for velocity in (-.2, 0., .2):
                if axis and distance == velocity == 0:
                    continue
                state = [0.] * 7
                state[axis], state[axis + 3] = distance, velocity
                rows.append({'axis': axis, 'distance_m': distance, 'velocity_m_s': velocity,
                    'state': state, 'source': 'COUNTERFACTUAL_POLICY_FUNCTION_PROBE'})
    return rows


def configuration():
    return {'experiment_id': 'AUTORL_R29_POLICY_FUNCTION_PROBE_ONLY',
        'kind': 'COUNTERFACTUAL_POLICY_FUNCTION_PROBE', 'checkpoint_steps': [0, 32768, 65536],
        'context': 'R28_STEP0_SAVED_LAST_OBS_LIDAR_HEADING_AGE_FIXED_ALL_MODELS',
        'model_input_rows': 237, 'grid_sha256': digest(probe_grid()),
        'state_convention': 'ENU position error,ENU world velocity,yaw error0;not an actual simultaneous physical state',
        'world_velocity_action_scale': [1.5, 1.5, 1., 1.],
        'assessment': 'axiswise signed toward commands at v0 near .05/.2 versus far .5/1m;zero-error velocity damping at +/-.2m/s',
        'hypothesis_threshold': 'far signed mean or zero-error braking mean <= half ownprior and ownprior>1e-6 => weakened function response in this context only',
        'wall_seconds': 180, 'workers': 1, 'device': 'cpu',
        'training_timesteps': 0, 'optimizer_updates': 0, 'physics_ticks': 0, 'new_episodes': 0, 'test_episodes': 0,
        'claim_limit': 'not a trajectory,success rate,causal explanation,LiDAR dependence or closed-loop stability test'}


def references(root=ROOT):
    root = Path(root)
    try:
        report_path = root / 'mujoco/reports/autonomous_rl_held_goal_r28.json'
        publication_path = root / 'mujoco/reports/autonomous_rl_held_goal_r28_publication.json'
        if sha256(report_path) != R28_REPORT_SHA or sha256(publication_path) != R28_PUBLICATION_SHA:
            raise OpsError('R29 released R28 input differs')
        report, publication = read_json(report_path), read_json(publication_path)
        if (report['scientific_result'] != 'FAIL' or report['actual_timesteps'] != 65536
                or report['result_review']['status'] != 'GO' or publication['result_review']['status'] != 'GO'):
            raise OpsError('R29 reviewed R28 result missing')
        archived = root / '.research_local/ops_v2/versions' / ('v2_' + publication['registration_sha256']) / 'state.json'
        envelope = read_json(archived); state = envelope['state']
        if (digest(state) != envelope['sha256'] or state['phase_complete'] is not True
                or state['latest_successful_push']['commit'] != R28_COMMIT
                or git(root, 'merge-base', '--is-ancestor', R28_COMMIT, 'HEAD', check=False).returncode):
            raise OpsError('R29 R28 publication not complete')
        run = root / '.research_local/authorized_rl_v1/runs' / R28_ID
        raw = {v['path']: v for v in report['raw_files']}
        if len(raw) != 23 or {p.name for p in run.iterdir()} != {Path(p).name for p in raw}:
            raise OpsError('R29 R28 raw evidence set changed')
        for name, ref in raw.items():
            p = root / name
            if not p.is_relative_to(run) or p.is_symlink() or sha256(p) != ref['sha256']:
                raise OpsError('R29 R28 immutable raw identity differs')
        budget = BudgetStore(root / '.research_local/authorized_rl_v1').status()
        if budget['stop_requested'] or any(v['status'] not in ('COMPLETE', 'FAIL') for v in budget['attempts']):
            raise OpsError('R29 stop or pending attempt blocks diagnostic')
        entry = next(v for v in budget['attempts'] if v['experiment_id'] == R28_ID)
        if (entry['status'] != 'COMPLETE' or entry['result']['summary_sha256'] != report['summary']['sha256']
                or entry['protocol'] != report['configuration']):
            raise OpsError('R29 actual ledger does not match completed R28')
        models = []
        for step in (0, 32768, 65536):
            run_id = f'{R28_ID}_STEP{step}'; archive = run / (run_id + '.zip'); side = run / (run_id + '.json')
            record = read_json(side)
            if (record['execution_protocol_sha256'] != report['execution_protocol_sha256']
                    or record['training_timesteps'] != step or record['checkpoint_sha256'] != sha256(archive)
                    or digest({k: v for k, v in record.items() if k != 'sidecar_payload_sha256'}) != record['sidecar_payload_sha256']):
                raise OpsError('R29 fixed checkpoint/sidecar mismatch')
            models.append({'step': step, 'run_id': run_id, 'checkpoint_path': str(archive.relative_to(root)),
                'checkpoint_sha256': sha256(archive), 'sidecar_sha256': sha256(side),
                'execution_protocol_sha256': report['execution_protocol_sha256']})
        return {'models': models, 'R28_scientific_result': 'FAIL', 'budget_snapshot_sha256': digest(budget),
            'R28_report_sha256': R28_REPORT_SHA, 'R28_publication_sha256': R28_PUBLICATION_SHA}
    except (KeyError, TypeError, ValueError, OSError, StopIteration) as error:
        raise OpsError('missing/invalid released R29 inputs') from error


def require_model(reference, trusted):
    if reference not in trusted['models']:
        raise ValueError('R29 untrusted checkpoint reference before deserialization')


def make_protocol(registration, root=ROOT):
    return {**configuration(), **references(root), 'registration_sha256': digest(registration),
        'source_identity_sha256': digest(registration['source_identity'])}


def assess(rows):
    grid = probe_grid()
    expected = [(s, i) for s in (0, 32768, 65536) for i in range(len(grid))]
    if [(v['step'], v['grid_index']) for v in rows] != expected:
        raise ValueError('R29 incomplete/duplicate/wrong ordered function probes')
    for row in rows:
        a, clipped = row['raw_mean'], row['clipped_action']
        if (len(a) != 4 or len(clipped) != 4 or not all(math.isfinite(v) for v in a + clipped)
                or any(abs(max(-1., min(1., x)) - y) > 1e-6 for x, y in zip(a, clipped))):
            raise ValueError('R29 invalid nonfinite/unclipped model outputs')
    output = []
    for model_index, step in enumerate((0, 32768, 65536)):
        local = rows[79 * model_index:79 * (model_index + 1)]
        for axis in range(3):
            value = {'step': step, 'axis': axis}
            for name, distances in (('near', (.05, .2)), ('far', (.5, 1.))):
                group = [(g, r) for g, r in zip(grid, local) if g['axis'] == axis
                    and abs(g['distance_m']) in distances and g['velocity_m_s'] == 0.]
                value[name + '_signed_velocity_command_mean'] = sum(math.copysign(1., g['distance_m'])
                    * r['clipped_action'][axis] * (1.5 if axis < 2 else 1.) for g, r in group) / len(group)
            group = [(g, r) for g, r in zip(grid, local) if g['axis'] == axis
                and g['distance_m'] == 0. and g['velocity_m_s'] != 0.]
            value['zero_error_braking_mean'] = sum(-math.copysign(1., g['velocity_m_s'])
                * r['clipped_action'][axis] * (1.5 if axis < 2 else 1.) for g, r in group) / len(group)
            output.append(value)
    for row in output:
        prior = output[row['axis']]
        for key in ('far_signed_velocity_command_mean', 'zero_error_braking_mean'):
            row[key + '_weakened_vs_prior'] = prior[key] > 1e-6 and row[key] <= .5 * prior[key]
    return {'axiswise': output, 'claim': 'FUNCTION_PROBE_ONLY_NOT_CLOSED_LOOP_OR_CAUSAL_STABILITY'}
