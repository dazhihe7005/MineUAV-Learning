"""Pure offline metrics; missing reward logs are never reconstructed."""
import hashlib
import json
from types import SimpleNamespace
import numpy as np

COMPONENTS = ('progress', 'action', 'brake', 'success', 'failure', 'total')
POLICY_DT = .04


def distribution(values):
    x = np.asarray(values, dtype=float)
    if not x.size:
        return None
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('finite scalar observations required')
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                sample_sd=float(x.std(ddof=1)) if len(x) > 1 else None,
                min=float(x.min()), max=float(x.max()),
                q25=float(np.quantile(x, .25)), q75=float(np.quantile(x, .75)))


def correlations(x, y):
    from decision_fidelity_metrics import rankdata
    a, b = np.asarray(x, float), np.asarray(y, float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(np.r_[a, b]).all():
        raise ValueError('aligned finite observations required')
    result = dict(n=len(a), pearson=None, spearman=None)
    if len(a) >= 2 and np.ptp(a) > 0 and np.ptp(b) > 0:
        result.update(pearson=float(np.corrcoef(a, b)[0, 1]),
                      spearman=float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]))
    return result


def reward_availability(trace):
    n = len(trace['actions'])
    recorded = 'rewards' in trace
    values = None
    if recorded:
        values = np.asarray(trace['rewards'])
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError('misaligned/nonfinite recorded rewards')
    return dict(recorded_per_step_reward=recorded,
                recorded_components='reward_breakdown' in trace,
                final_episode_return=float(values.sum()) if recorded else None,
                status='recorded' if recorded else 'not observable: acquisition discarded reward',
                reconstructed_reward=False)


def distance_bin(distance):
    return '<1m' if distance < 1 else ('1-2m' if distance < 2 else '>=2m')


def duration_bin(seconds):
    return '<=8s' if seconds <= 8 else ('8-12s' if seconds <= 12 else '>12s')


def state_metrics(trace, target):
    o, p, a = (np.asarray(trace[k]) for k in ('observations', 'positions', 'actions'))
    n = len(a)
    target = np.asarray(target, float)
    if n < 1 or o.shape != (n + 1, 7) or p.shape != (n + 1, 3) or a.shape != (n, 4):
        raise ValueError('invalid episode shapes')
    if target.shape != (3,) or not np.isfinite(np.r_[o.ravel(), p.ravel(), a.ravel(), target]).all():
        raise ValueError('nonfinite or wrong target')
    if not np.asarray(trace['physical_state_valid']).all():
        raise ValueError('invalid-state placeholder is not a physical observation')
    if not np.array_equal((target - p).astype(np.float32), o[:, :3]):
        raise ValueError('position/error frame or target mismatch')
    prior = np.asarray(trace['previous_actions'])
    if prior.shape != a.shape or not np.array_equal(prior[0], np.zeros(4)) or not np.array_equal(prior[1:], a[:-1]):
        raise ValueError('previous-action indexing corrupted')
    if not np.array_equal(trace['step_index'], np.arange(n)) or np.any(np.abs(a) > 1):
        raise ValueError('step or action bounds corrupted')
    streak = np.asarray(trace['success_streak'])
    if streak.shape != (n + 1,) or streak[0] != 0 or np.any((streak < 0) | (streak > 5)):
        raise ValueError('invalid success counter')
    if np.any((streak[1:] > 0) & (streak[1:] != streak[:-1] + 1)):
        raise ValueError('invalid dwell recursion')
    d = np.linalg.norm(target - p, axis=1)
    v = o[:, 3:6].astype(float)
    speed = np.linalg.norm(v, axis=1)
    command = a[:, :3].astype(float) * [1.5, 1.5, 1.]
    command_speed = np.linalg.norm(command, axis=1)
    near = d[:-1] < .1
    approach = (d[:-1] < .5) & (speed[:-1] > 1e-6)
    opposed = np.sum(command * v[:-1], axis=1) < 0
    start = max(0, n - 75)
    post = np.arange(start + 1, n + 1)
    return dict(steps=n, duration_s=n * POLICY_DT, initial_distance_m=float(d[0]),
                initial_distance_bin=distance_bin(d[0]), duration_bin=duration_bin(n * POLICY_DT),
                final_distance_m=float(d[-1]), final_speed_m_s=float(speed[-1]),
                distance_reduction_m=float(d[0] - d[-1]), peak_speed_m_s=float(speed.max()),
                joint_post_samples=int(np.count_nonzero(streak[1:])), maximum_streak=int(streak.max()),
                ever_distance_gate=bool(np.any(d[1:] < .1)),
                ever_joint_gate=bool(np.any(streak[1:] > 0)),
                near_pre_samples=int(near.sum()),
                near_speed_mean_m_s=float(speed[:-1][near].mean()) if near.any() else None,
                near_command_speed_mean_m_s=float(command_speed[near].mean()) if near.any() else None,
                opposing_command_samples=int(np.count_nonzero(opposed & approach)),
                approach_moving_samples=int(approach.sum()),
                command_tracking_error_mean_m_s=float(np.linalg.norm(command - v[:-1], axis=1).mean()),
                last3s_duration_s=(n - start) * POLICY_DT,
                last3s_distance_reduction_m=float(d[start] - d[-1]),
                last3s_speed_mean_m_s=float(speed[post].mean()),
                last3s_command_speed_mean_m_s=float(command_speed[start:].mean()),
                last3s_joint_fraction=float(np.mean(streak[post] > 0)),
                last3s_slow_outside_gate_fraction=float(np.mean((d[post] >= .1) & (speed[post] < .15))),
                first2s_distance_reduction_m=float(d[0] - d[50]) if n >= 50 else None,
                first2s_speed_mean_m_s=float(speed[1:51].mean()) if n >= 50 else None,
                last3s_reward_trend=None)


def verify_training_episode(row, monitor):
    terms = row['reward_components']
    if set(terms) != set(COMPONENTS) or not np.isfinite(list(terms.values())).all():
        raise ValueError('invalid recorded reward component schema')
    if abs(terms['total'] - sum(terms[k] for k in COMPONENTS[:-1])) > 1e-9:
        raise ValueError('component sum differs from recorded total')
    if row['steps'] != int(monitor['l']) or row['termination_reason'] != monitor['termination_reason']:
        raise ValueError('Monitor episode identity/length mismatch')
    if abs(terms['total'] - float(monitor['r'])) > 5.01e-7:
        raise ValueError('recorded component/Monitor return disagreement')
    d0 = float(np.linalg.norm(np.asarray(row['target']) - [0., 0., 1.]))
    dend = float(monitor['distance_m'])
    if abs(terms['progress'] - 10 * (d0 - dend)) > 1e-9:
        raise ValueError('recorded progress disagrees with independent terminal distance')
    success = row['termination_reason'] == 'success'
    failed = row['termination_reason'] not in ('success', 'time_limit')
    if terms['success'] != (10 if success else 0) or terms['failure'] != (-10 if failed else 0):
        raise ValueError('terminal bonus/penalty disagrees with outcome')
    if terms['action'] > 0 or terms['brake'] > 0:
        raise ValueError('invalid positive action/brake penalty')
    return dict(seed=row.get('seed'), rank=row['rank'], episode_index=row['episode_index'],
                success=success, timeout=row['termination_reason'] == 'time_limit',
                steps=row['steps'], duration_s=row['steps'] * POLICY_DT,
                initial_distance_m=d0, initial_distance_bin=distance_bin(d0),
                duration_bin=duration_bin(row['steps'] * POLICY_DT),
                final_distance_m=dend, recorded_return=terms['total'],
                difficulty_adjusted_return=terms['total'] - 10 * d0,
                components=terms, monitor_rounding_error=terms['total'] - float(monitor['r']))


def validate_cohort(actual_keys, expected_keys):
    if len(actual_keys) != len(expected_keys) or len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != set(expected_keys):
        raise ValueError('duplicate/omitted/unexpected tasks')


def verify_record_identity(row, manifest_sha, controller, model_identity):
    payload = {k: v for k, v in row.items() if k != 'record_sha256'}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    expected = dict(manifest_sha256=manifest_sha, controller=controller, model_identity=model_identity)
    if row.get('record_sha256') != digest or row.get('identity') != expected:
        raise ValueError('corrupt record or wrong model/manifest')


def stratify(rows, value_key):
    result = {}
    for distance in ('<1m', '1-2m', '>=2m'):
        for duration in ('<=8s', '8-12s', '>12s'):
            for outcome in ('success', 'failure'):
                selected = [r for r in rows if r['initial_distance_bin'] == distance and r['duration_bin'] == duration and r['success'] == (outcome == 'success')]
                values = [r[value_key] for r in selected if r.get(value_key) is not None]
                result[f'{distance}/{duration}/{outcome}'] = dict(n=len(selected), observable=len(values), value=distribution(values))
    return result


def source_examples():
    """Evaluate actual reward method on plain synthetic namespaces, NOT flights."""
    from mine_uav_env import MineUAVEnv
    examples = {}
    cases = [('stopped_at_020m', .2, .2, 0., False),
             ('moving_at_060m', .6, .6, .4, False),
             ('approaching_009m_at_020m_s', .1, .09, .2, False),
             ('moving_at_009m', .09, .09, .2, False),
             ('success_terminal_flag_given', .09, .09, .1, True)]
    for name, previous, current, speed, success in cases:
        obj = SimpleNamespace(reward_version='v2', data=SimpleNamespace(qvel=np.array([speed, 0., 0., 0., 0., 0.])))
        _, examples[name] = MineUAVEnv._compute_reward(obj, previous, current, np.zeros(4), success, False)
    return examples


def discounted_bonus(step, gamma=.99):
    if not isinstance(step, int) or step < 1 or not 0 < gamma <= 1:
        raise ValueError('positive terminal step and valid gamma required')
    return float(10 * gamma ** (step - 1))


def paired_early_states(rows):
    """Descriptive common-target/time contrasts; never independent seed counts."""
    from itertools import combinations
    by_seed = {}
    for r in rows:
        group = by_seed.setdefault(r['seed'], {})
        if r['target_id'] in group:
            raise ValueError('duplicate seed/target identity')
        group[r['target_id']] = r
    result = {}
    for a, b in combinations(sorted(by_seed), 2):
        if by_seed[a].keys() != by_seed[b].keys():
            raise ValueError('unpaired target cohort')
        discordant = observed = censored = 0
        distances, speeds = [], []
        for target in by_seed[a]:
            x, y = by_seed[a][target], by_seed[b][target]
            if x['success'] == y['success']:
                continue
            discordant += 1
            success, failure = (x, y) if x['success'] else (y, x)
            if any(r[k] is None for r in [success, failure] for k in ['first2s_distance_reduction_m', 'first2s_speed_mean_m_s']):
                censored += 1
                continue
            observed += 1
            distances.append(failure['first2s_distance_reduction_m'] - success['first2s_distance_reduction_m'])
            speeds.append(failure['first2s_speed_mean_m_s'] - success['first2s_speed_mean_m_s'])
        result[f'{a}_vs_{b}'] = dict(discordant_pairs=discordant, observed_2s_pairs=observed,
                                    censored_before_2s=censored,
                                    failed_minus_success_distance_reduction_m=distribution(distances),
                                    failed_minus_success_speed_m_s=distribution(speeds))
    return result
