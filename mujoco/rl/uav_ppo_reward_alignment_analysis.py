"""Authenticated, serial OFFLINE audit. No env construction or rollout calls."""
import csv
import json
import resource
import zipfile
from pathlib import Path
import numpy as np
from uav_ppo_reward_alignment import (
    COMPONENTS, correlations, distribution, discounted_bonus, paired_early_states, reward_availability,
    source_examples, state_metrics, stratify, validate_cohort, verify_record_identity,
    verify_training_episode)
from latent_dynamics_data import file_hash, json_hash
from uav_bc_safety import atomic_bytes, atomic_json
from uav_ppo_multiseed import read_manifest, verify_seal, verify_completion

ROOT = Path(__file__).resolve().parents[2]
BASE = 'b36d7539dbf53fa3f1c17a510ac3a12f8b3bcc62'
BRANCH = 'feat/uav-ppo-reward-alignment-audit'
PARTS = 'uav_ppo_reward_alignment_parts'
REPORT = 'uav_ppo_reward_alignment_audit.json'


def cohort_summary(rows):
    keys = ['initial_distance_m', 'duration_s', 'final_distance_m', 'final_speed_m_s',
            'distance_reduction_m', 'peak_speed_m_s', 'maximum_streak',
            'near_speed_mean_m_s', 'near_command_speed_mean_m_s',
            'last3s_distance_reduction_m', 'last3s_speed_mean_m_s',
            'last3s_joint_fraction', 'last3s_slow_outside_gate_fraction']
    return dict(n=len(rows), success=sum(r['success'] for r in rows),
                timeout=sum(r['timeout'] for r in rows),
                ever_distance_gate=sum(r['ever_distance_gate'] for r in rows),
                ever_joint_gate=sum(r['ever_joint_gate'] for r in rows),
                max_streak_histogram={str(i): sum(r['maximum_streak'] == i for r in rows) for i in range(6)},
                metrics={k: distribution([r[k] for r in rows if r[k] is not None]) for k in keys},
                metric_missing={k: sum(r[k] is None for r in rows) for k in keys})


def training_summary(rows):
    success = [r for r in rows if r['success']]
    failed = [r for r in rows if not r['success']]
    threshold = float(np.quantile([r['recorded_return'] for r in rows], .75))
    summary = dict(n=len(rows), successes=len(success), failures=len(failed),
                   high_return_definition='at/above own-seed Q75 of all recorded training returns; not a Final-test criterion',
                   return_q75=threshold,
                   high_return_failures=sum(r['recorded_return'] >= threshold for r in failed),
                   positive_return_failures=sum(r['recorded_return'] > 0 for r in failed),
                   return_initial_distance_correlation=correlations(
                       [r['recorded_return'] for r in rows], [r['initial_distance_m'] for r in rows]),
                   raw_return_strata=stratify(rows, 'recorded_return'),
                   adjusted_return_strata=stratify(rows, 'difficulty_adjusted_return'),
                   duration_strata_note='Failures all15s in these runs; many shorter-success cells have no comparable failures. No unmatched causal comparison.')
    for name, group in [('success', success), ('failure', failed)]:
        summary[name] = dict(n=len(group),
                            return_=distribution([r['recorded_return'] for r in group]),
                            per_step_return=distribution([r['recorded_return'] / r['steps'] for r in group]),
                            adjusted_return=distribution([r['difficulty_adjusted_return'] for r in group]),
                            mean_components={k: float(np.mean([r['components'][k] for r in group])) if group else None for k in COMPONENTS})
    best = max(failed, key=lambda r: r['recorded_return']) if failed else None
    summary['maximum_return_failure'] = best
    summary['failed_above_success_q25'] = (int(sum(r['recorded_return'] >= np.quantile(
        [s['recorded_return'] for s in success], .25) for r in failed)) if success else None)
    return summary


def load_verified(root):
    root = Path(root)
    old = root / 'mujoco/reports/uav_ppo_multiseed_replication_parts'
    m = read_manifest(root)
    hashes = {}
    def remember(path):
        p = Path(path); hashes[str(p)] = file_hash(p)
        return p
    remember(root / 'mujoco/reports/uav_ppo_multiseed_preregistration.json')
    remember(root / 'mujoco/reports/uav_ppo_multiseed_replication.json')
    e = json.loads(remember(old / 'evaluation.json').read_text()); verify_seal(e)
    if e['manifest_sha256'] != m['sha256']:
        raise ValueError('wrong evaluation manifest')
    import stable_baselines3 as sb3
    sbroot = Path(sb3.__file__).parent
    for name in ['common/on_policy_algorithm.py', 'common/vec_env/dummy_vec_env.py',
                 'common/buffers.py', 'common/monitor.py', 'ppo/ppo.py']:
        remember(sbroot / name)
    helpers = ['uav_ppo_bc_nominal.py', 'uav_ppo_bc_nominal_run.py',
               'uav_ppo_bc_nominal_analysis.py', 'uav_bc_robustness.py',
               'latent_dynamics_data.py', 'decision_fidelity_metrics.py', 'uav_bc_safety.py',
               'uav_ppo_reward_alignment.py', 'uav_ppo_reward_alignment_analysis.py',
               'test_uav_ppo_reward_alignment.py']
    for path in list(m['source_sha256']) + ['mujoco/rl/' + n for n in helpers]:
        remember(root / path)
    for meta in m['frozen_models'].values():
        remember(root / meta['path'])
    completions, checkpoints = {}, {}
    for seed in range(5):
        t = json.loads(remember(old / f'seed{seed}.json').read_text())
        verify_completion(t, m, seed); completions[seed] = t
        for path in [t['checkpoint_path'], t['published_checkpoint_path'],
                     t['training_episodes_path'], t['actual_train_targets_path']]:
            remember(path)
        if file_hash(t['published_checkpoint_path']) != t['checkpoint_sha256']:
            raise ValueError('published model identity changed')
        with zipfile.ZipFile(t['published_checkpoint_path']) as z:
            data = json.loads(z.read('data'))
        if (data['gamma'], data['gae_lambda'], data['seed'], data['num_timesteps']) != (.99, .95, seed, 100352):
            raise ValueError('checkpoint protocol drift')
        checkpoints[str(seed)] = dict(path=t['published_checkpoint_path'],
                                     sha256=t['checkpoint_sha256'], gamma=data['gamma'],
                                     parameter_sha256=t['final_parameter_sha256'])
    states = {s['key']: s for s in m['states']}
    records = [r for r in e['records'] if r['controller'] in [f'ppo_seed{s}' for s in range(5)]]
    validate_cohort([r['task_key'] for r in records], [s['key'] + f'/ppo_seed{seed}' for s in m['states'] for seed in range(5)])
    final = []
    schemas = set()
    for r in records:
        seed = int(r['controller'][-1]); state = states[r['key']]
        model = e['identities'][r['controller']]
        if model['checkpoint_sha256'] != checkpoints[str(seed)]['sha256'] or model['parameter_sha256'] != checkpoints[str(seed)]['parameter_sha256']:
            raise ValueError('evaluation used wrong checkpoint')
        verify_record_identity(r, m['sha256'], r['controller'], model)
        if r['target'] != state['target'] or r['target_id'] != state['target_id'] or r['initial_snapshot_sha256'] != state['initial_state']['snapshot_sha256']:
            raise ValueError('target/snapshot identity mismatch')
        tracepath = remember(r['trace_path'])
        if file_hash(tracepath) != r['trace_sha256']:
            raise ValueError('trace hash mismatch')
        if json.loads(remember(tracepath.with_suffix('.json')).read_text()) != r:
            raise ValueError('sealed evaluation/original episode record mismatch')
        with np.load(tracepath, allow_pickle=False) as a:
            schemas.add(tuple(sorted(a.files)))
            availability = reward_availability(a)
            if availability['recorded_per_step_reward'] or availability['recorded_components']:
                raise ValueError('schema changed: audit missingness assumptions must be reconsidered')
            values = state_metrics(a, r['target'])
            for key, expected in [('steps', r['steps']), ('maximum_streak', r['maximum_success_streak']),
                                  ('ever_joint_gate', r['ever_joint_threshold']),
                                  ('ever_distance_gate', r['reached_target_region'])]:
                if values[key] != expected:
                    raise ValueError('trajectory/summary gate mismatch: ' + key)
            if not np.array_equal(a['observations'][0], state['initial_state']['observation']):
                raise ValueError('wrong initial observation')
            if not np.isclose(values['final_distance_m'], r['final_distance_m'], rtol=0, atol=1e-12):
                raise ValueError('final physical distance mismatch')
        success = values['maximum_streak'] == 5
        if success != r['success'] or (not success and (not r['timeout'] or r['steps'] != 375)):
            raise ValueError('outcome/time-limit inconsistency')
        final.append(dict(seed=seed, task_key=r['task_key'], target_id=r['target_id'],
                          success=r['success'], timeout=r['timeout'],
                          record_sha256=r['record_sha256'], trace_sha256=r['trace_sha256'],
                          reward_availability=availability, **values))
    if len(final) != 1000 or sum(r['success'] for r in final) != 303 or sum(r['timeout'] for r in final) != 697:
        raise ValueError('wrong complete Final cohort')
    training = []
    for seed, t in completions.items():
        episodes = json.loads(Path(t['training_episodes_path']).read_text())
        visits = {(r['rank'], r['episode_index']): r for r in t['actual_train_targets']}
        keys = [(r['rank'], r['episode_index']) for r in episodes]
        if len(keys) != len(set(keys)) or len(episodes) != t['completed_training_episodes']:
            raise ValueError('training episode omissions/duplicates')
        for rank in range(8):
            path = remember(Path(t['checkpoint_path']).parent / f'monitor{rank}.monitor.csv')
            with path.open() as f:
                next(f); monitor = list(csv.DictReader(f))
            group = sorted((r for r in episodes if r['rank'] == rank), key=lambda r: r['episode_index'])
            if len(group) != len(monitor) or [r['episode_index'] for r in group] != list(range(len(group))):
                raise ValueError('Monitor episode/index mismatch')
            for row, logged in zip(group, monitor):
                if row['target'] != visits[(rank, row['episode_index'])]['target']:
                    raise ValueError('training target indexing mismatch')
                training.append(verify_training_episode(dict(row, seed=seed), logged))
    return m, final, training, completions, checkpoints, hashes, sorted(schemas)


def plots(root, final, training, source, learning):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(root) / 'mujoco/reports/uav_ppo_reward_alignment_figures'
    out.mkdir(exist_ok=True); hashes = {}
    def save(fig, name):
        fig.tight_layout(); path = out / name
        atomic_bytes(path, lambda f: fig.savefig(f, format='png', dpi=140))
        plt.close(fig); hashes[str(path.relative_to(root))] = file_hash(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cases = source['synthetic_one_step_examples']
    axes[0].bar(range(len(cases)), [c['total'] for c in cases.values()])
    axes[0].set_xticks(range(len(cases)), ['stopped\n0.20m', 'moving\n0.60m', 'approach\n0.09m', 'moving\n0.09m', 'terminal\nsuccess flag'])
    axes[0].set_ylabel('One-step source reward\n(synthetic, NOT flights)')
    axes[1].plot([5, 10, 15], [discounted_bonus(n) for n in [125, 250, 375]], marker='o')
    axes[1].set_xlabel('Hypothetical terminal time (s)'); axes[1].set_ylabel('10 x gamma^(T-1), gamma=.99')
    save(fig, 'reward_source_and_discount.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in zip(axes, ['recorded_return', 'difficulty_adjusted_return'],
                              ['Recorded TRAIN episode return', 'TRAIN R - 10*d_initial (diagnostic only)']):
        values = [[r[key] for r in training if r['seed'] == s and r['success'] == success] for s in range(5) for success in [True, False]]
        ax.boxplot(values, tick_labels=[f'{s}\n{"S" if success else "F"}' for s in range(5) for success in [True, False]], showfliers=False)
        ax.set_title(title); ax.set_xlabel('Training seed / outcome; evolving stochastic policy')
    save(fig, 'recorded_train_reward_distributions.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, k, title in zip(axes, ['final_distance_m', 'final_speed_m_s'], ['Final distance (m)', 'Final speed (m/s)']):
        values = [[r[k] for r in final if r['seed'] == s and r['success'] == success] for s in range(5) for success in [True, False]]
        ax.boxplot(values, tick_labels=[f'{s}\n{"S" if success else "F"}' for s in range(5) for success in [True, False]], showfliers=False)
        ax.set_ylabel(title); ax.set_xlabel('Final deterministic cohort: seed / outcome')
    save(fig, 'final_state_by_success_and_seed.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    values = [[r['near_speed_mean_m_s'] for r in final if r['seed'] == s and r['success'] == success and r['near_speed_mean_m_s'] is not None] for s in range(5) for success in [True, False]]
    axes[0].boxplot(values, tick_labels=[f'{s}\n{"S" if success else "F"}' for s in range(5) for success in [True, False]], showfliers=False)
    axes[0].axhline(.15, ls='--', color='gray'); axes[0].set_ylabel('Episode mean speed in pre-action d<.10m (m/s)')
    failed = [r for r in final if not r['success']]
    axes[1].bar(range(5), [sum(r['maximum_streak'] == i for r in failed) for i in range(5)])
    axes[1].set_xlabel('Maximum recorded success streak, timeout only'); axes[1].set_ylabel('Failure count /697')
    save(fig, 'near_target_braking_and_dwell.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in range(5):
        group = [r for r in failed if r['seed'] == s]
        ax.scatter([r['last3s_distance_reduction_m'] for r in group], [r['last3s_speed_mean_m_s'] for r in group], s=8, alpha=.5, label=str(s))
    ax.axvline(0, color='gray'); ax.axhline(.15, ls='--', color='gray'); ax.legend(title='Seed')
    ax.set_xlabel('Distance reduction in12-15s window (m)'); ax.set_ylabel('Actual mean speed in12-15s (m/s)')
    ax.set_title('697timeouts: physical state trends; reward trend NOT recorded')
    save(fig, 'timeout_terminal_state_trends.png')
    fig, ax = plt.subplots(figsize=(7, 5))
    for s, item in learning.items():
        ax.scatter(item['recorded_final_rolling_train_return'], item['final_success_rate'] * 100)
        ax.annotate(s, (item['recorded_final_rolling_train_return'], item['final_success_rate'] * 100))
    ax.set_xlabel('Logged final rolling100 stochastic TRAIN episode return'); ax.set_ylabel('Frozen deterministic Final success (%)')
    ax.set_title('5training seeds; not Final-test reward correlation')
    save(fig, 'train_return_vs_final_success.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ['progress', 'action', 'brake', 'success', 'failure']
    for ax, outcome in zip(axes, [True, False]):
        for s in range(5):
            group = [r for r in training if r['seed'] == s and r['success'] == outcome]
            y = [np.mean([r['components'][k] for r in group]) if group else np.nan for k in labels]
            ax.plot(labels, y, marker='o', label=str(s))
        ax.set_title('TRAIN ' + ('success' if outcome else 'failure')); ax.set_ylabel('Mean recorded cumulative component'); ax.legend(title='Seed')
    save(fig, 'recorded_train_reward_components.png')
    return hashes


def analyze(root=ROOT):
    root = Path(root)
    m, final, training, completions, checkpoints, hashes, schemas = load_verified(root)
    source = dict(reward_version='v2', gamma=.99, gae_lambda=.95, reward_clip=False,
                  reward_normalization=False, action_normalization='[-1,1]^4; squared norm includes yaw-rate dimension',
                  observation_normalization='none; raw7D', reward_evaluation_hz=25, policy_dt_s=.04,
                  terms=dict(progress='10*(d_previous-d_current), native meters',
                             action='-.005*sum(normalized_action**2), all4dimensions',
                             brake='-.5*clip((.5-d_current)/.4,0,1)*world_speed_squared',
                             success='+10 only after distance<.10 AND speed<.15 for5steps',
                             failure='-10 physical/nonfinite failure only',
                             absolute_distance='absent inV2; present inV3 only',
                             yaw='no dedicated yaw term or yaw success gate', timeout='no timeout reward penalty'),
                  units='reward terms unnormalized, no dt multiplier; brake-weight clip is not reward clipping',
                  termination='success/physical failure -> terminated; otherwise375policy steps -> truncated',
                  discount_half_life_s=float(.04*np.log(.5)/np.log(.99)),
                  synthetic_one_step_examples=source_examples(),
                  hypothetical_terminal_bonus_at_s={str(seconds): discounted_bonus(seconds*25) for seconds in [5,10,15]},
                  progress_identity='Undiscounted progress=10*(d0-dT); discounted=10*(d0-(1-gamma)*sum_t=1..T-1 gamma^(t-1)*d_t-gamma^(T-1)*dT)',
                  bootstrap=dict(dummy_vec='done=terminated|truncated; TimeLimit.truncated=truncated&&!terminated; saves pre-reset terminal_observation',
                                 rollout='only time-limit done adds gamma*V(terminal_observation) to float32 buffer reward; Monitor records env reward before this correction',
                                 gae='delta=r_buffer+gamma*next_value*nonterminal-V; A=delta+gamma*lambda*nonterminal*A_next; return=A+V',
                                 truncation_value='not observable: historical per-step values/advantages/buffer rewards not persisted',
                                 interpretation='Implementation handles time limits as critic continuation, whereas evaluation counts timeout as failure; causal impact unresolved'))
    by_seed = {str(s): dict(all=cohort_summary([r for r in final if r['seed']==s]),
                           success=cohort_summary([r for r in final if r['seed']==s and r['success']]),
                           failure=cohort_summary([r for r in final if r['seed']==s and not r['success']]),
                           distance_duration_strata=stratify([r for r in final if r['seed']==s], 'final_distance_m')) for s in range(5)}
    train_summary = {str(s): training_summary([r for r in training if r['seed']==s]) for s in range(5)}
    learning = {str(s): dict(recorded_final_rolling_train_return=t['history'][-1]['mean_episode_return'],
                            recorded_final_rolling_train_success=t['history'][-1]['recent_training_success_rate'],
                            final_success_rate=by_seed[str(s)]['all']['success']/200) for s,t in completions.items()}
    figures = plots(root, final, training, source, learning)
    for path, sha in hashes.items():
        if file_hash(path) != sha:
            raise ValueError('read-only input mutated during audit: ' + path)
    value = dict(experiment='Offline RewardV2 and Task-Success Alignment Audit', base_commit=BASE, branch=BRANCH,
                 prior_manifest_sha256=m['sha256'], checkpoint_identities=checkpoints,
                 identity=dict(input_files_sha256=hashes, aggregate_input_identity_sha256=json_hash(hashes),
                               raw_schemas=schemas, before_after_equal=True),
                 accounting=dict(final_ppo_episodes=1000, success=303, failure=697, timeout=697,
                                 recorded_train_episodes=len(training), new_simulation_episodes=0, optimizer_steps=0),
                 source_contract=source,
                 observability=dict(final_cumulative_reward=None, final_discounted_return=None,
                                    final_reward_components=None, final_reward_vs_success=None,
                                    final_timeout_reward_trends=None, high_reward_final_failure=None,
                                    status='not observable: all1000Final acquisition traces discard reward/breakdown; no same-flight original reward logs for consistency verification',
                                    final_reward_reconstruction_performed=False,
                                    velocity_precision='float32 observation only; internal float64 qvel not saved',
                                    recorded_training_components='independently checked against40MonitorCSV files and target/terminal-distance progress identity; supplemental evolving-policy data only'),
                 final_by_seed=by_seed, final_success=cohort_summary([r for r in final if r['success']]),
                 final_failure=cohort_summary([r for r in final if not r['success']]),
                 common_target_2s_contrasts=paired_early_states(final),
                 training_supplement=train_summary, learning_cross_seed=learning,
                 final_rolling_train_return_vs_final_success=correlations(
                     [r['recorded_final_rolling_train_return'] for r in learning.values()], [r['final_success_rate'] for r in learning.values()]),
                 statistics=dict(unit='5training seeds; episodes are repeated fixed-target observations, not independent training seeds',
                                 groups='within seed, initial distance bins<1/1-2/>=2m and duration<=8/8-12/>12s; empty cells null',
                                 adjusted_train_return='R-10*d0 removes known undiscounted progress initial-distance offset; diagnostic, not new reward',
                                 common_time='first2s contrasts use identical target and2s horizon, success/failure-discordant policies; short completed episodes censored explicitly',
                                 limits='target/policy observations recur across contrasts; not independent training seeds. Training-age confounding remains; no matched duration overlap for short successes vs15s failures; no causal inference'),
                 final_episode_metrics=final, figures=figures,
                 evidence=dict(Confirmed=[
                     'Five seeds actually use RewardV2, gamma.99 and unchanged termination/PI; raw identities preserved.',
                     'V2 has near-target speed penalty and success bonus, but no timeout, absolute-distance, yaw or intermediate dwell term.',
                     'The source admits zero instantaneous reward for stopped outside-threshold states and positive approach reward despite a speed-gate miss; synthetic examples only.',
                     'Final reward/Value histories are missing; actual high-reward Final failures cannot be confirmed.',
                     'Recorded training component totals agree with independently logged Monitor rewards and progress/end-distance relation.'],
                               Supported_hypothesis=[
                     'Early progress versus delayed success discounting and critic continuation at time limit merit alignment attention.',
                     'Observed residual-distance and braking/dwell limitations are compatible with imperfect reward/task alignment, but not uniquely explained by it.'],
                               Unresolved=[
                     'Whether original Final failing flights earned high cumulative or discounted reward; timeout reward trend.',
                     'Historical terminal bootstrap magnitudes, advantages and their causal effect on training.',
                     'Whether changing reward would improve PPO; PI/POMDP/finite-budget/numerical alternatives remain unisolated.']),
                 resource=dict(analysis_processes=1, simulation_workers=0, sequential_npz_loading=True,
                               peak_self_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),
                 next='Only if separately authorized: instrument existing acquisition to preserve per-step raw reward/breakdown, termination flags and training terminal bootstrap/values. Do not automatically rerun or train.')
    atomic_json(root/'mujoco/reports'/REPORT, value)
    print(json.dumps(dict(accounting=value['accounting'], training={s: dict(n=t['n'],success=t['successes'],high_return_failures=t['high_return_failures']) for s,t in train_summary.items()},
                          correlation=value['final_rolling_train_return_vs_final_success'],rss_bytes=value['resource']['peak_self_rss_bytes']),indent=2))
    return value


if __name__=='__main__': analyze()
