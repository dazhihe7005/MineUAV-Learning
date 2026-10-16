"""Unique fixed R13 fresh-network preflight. No train/physics/arbitrary CLI."""
import argparse
import copy
import os
import time
from research_ops.storage import FileLock, OpsError, atomic_json, read_json, digest, sha256, process_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT, HOME, verify_current
from .budget import BudgetStore
from .centered_preflight import configuration, make_protocol, verify_checks, load_background

OUT = ROOT/'.research_local/autorl_r13_centered_preflight/run_01'


def run():
    if os.getpgrp() != os.getpid():
        raise OpsError('R13 requires isolated owned process group')
    from research_ops_v2.manager import OperationsV2
    from .lifetime import start_guard, stop_guard
    manager = OperationsV2(ROOT)
    parent = process_identity(os.getppid())
    with FileLock(manager.lock_path):
        _, reg = verify_current()
        if reg['phase_id'] != 'AUTORL-R13-CENTERED-PREFLIGHT':
            raise OpsError('wrong registered R13 phase')
        verify_checks(reg)
        enforce_resources(ROOT)
        spec = read_json(ROOT/'mujoco/reports/autonomous_rl_centered_preflight_r13_prereg.json')
        if spec != make_protocol(reg):
            raise OpsError('R13 protocol/budget/source/history changed')
        watcher = start_guard(parent, time.monotonic()+120, ROOT)
        try:
            OUT.mkdir(parents=True, exist_ok=False)
            atomic_json(OUT/'attempt.json', {'experiment_id': spec['experiment_id'], 'protocol_sha256': digest(spec),
                'source_identity_sha256': digest(reg['source_identity']), 'owner': process_identity(), 'parent': parent}, exclusive=True)
            try:
                return probe(reg, spec)
            except BaseException as error:
                atomic_json(OUT/'failure.json', {'status': 'FAIL', 'exception_type': type(error).__name__,
                    'message': str(error)[:2000], 'context': 'zero-update preflight; no complete result presumed; no retry'}, exclusive=True)
                raise
        finally:
            stop_guard(watcher)


def probe(reg, spec):
    import resource
    import gymnasium as gym
    import numpy as np
    import torch
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from research_ops.identity import source_identity
    from lidar_integration.ppo_preflight import make_ppo
    from lidar_integration.ppo_training_runner import _weights_sha
    from lidar_integration.ppo_formal_validation import _optimizer_states_equal
    from .state_centered_policy import make_centered_ppo, initialize_centered_prior, verify_centered_policy
    from .actor_prior import initialize_actor_prior
    from .feedback_probe import infer_panel
    def forbidden(*a, **kw):
        raise AssertionError('R13_NO_OPTIMIZER_ROLLOUT_OR_PHYSICS')
    PPO.learn = forbidden
    PPO.train = forbidden
    torch.optim.Adam.step = forbidden
    class NoPhysicsObservationContract(gym.Env):
        observation_space = spaces.Dict({'lidar': spaces.Box(0, 1, (3, 12, 24), dtype=np.float32),
            'state': spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32), 'heading': spaces.Box(-1, 1, (2,), dtype=np.float32),
            'lidar_age': spaces.Box(0, .15, (1,), dtype=np.float32)})
        action_space = spaces.Box(-1, 1, (4,), dtype=np.float32)
        reset = forbidden
        step = forbidden
    torch.set_num_threads(1)
    begun = time.monotonic()
    cpu = time.process_time()
    before = {str(p.relative_to(ROOT)): sha256(p) for p in HOME.rglob('*') if p.is_file()}
    budget = BudgetStore(HOME).status()
    resources = enforce_resources(ROOT)
    base, seal, _, scripted = load_background()
    model = make_centered_ppo(NoPhysicsObservationContract(), seed=0)
    initialization = initialize_centered_prior(model)
    baseline = make_ppo(NoPhysicsObservationContract(), seed=0)
    initialize_actor_prior(baseline)
    with torch.no_grad():
        baseline.policy.log_std.fill_(np.log(.3))
    panel = infer_panel(model, base, spec['cases'])
    old_panel = infer_panel(baseline, base, spec['cases'])
    mean_error = max(abs(x-y) for a,b in zip(panel['rows'],old_panel['rows']) for x,y in zip(a['raw_mean'],b['raw_mean']))
    value_error = max(abs(a['critic_value']-b['critic_value']) for a,b in zip(panel['rows'],old_panel['rows']))
    optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
    saved = OUT/'R13_NEW_INITIAL_POLICY_ONLY.zip'
    model.save(saved)
    loaded = PPO.load(saved, device='cpu')
    verified = verify_centered_policy(loaded)
    replay = infer_panel(loaded, base, spec['cases'])
    if (mean_error > 1e-6 or value_error > 1e-6 or panel['rows'] != replay['rows']
            or not _optimizer_states_equal(optimizer, loaded.policy.optimizer.state_dict())
            or _weights_sha(model.policy) != _weights_sha(loaded.policy)):
        raise OpsError('centered initial equivalence or reload differs')
    after = {str(p.relative_to(ROOT)): sha256(p) for p in HOME.rglob('*') if p.is_file()}
    if (before != after or BudgetStore(HOME).status() != budget or spec != make_protocol(reg)
            or source_identity(ROOT, reg['source_paths']) != reg['source_identity']):
        raise OpsError('R13 changed grant/history/budget/source')
    from research_ops_v2.manager import OperationsV2
    history = OperationsV2(ROOT).verify()['history']
    result = {'status': 'PASS', 'experiment_id': spec['experiment_id'], 'registration_sha256': digest(reg),
        'source_identity_sha256': digest(reg['source_identity']), 'protocol_sha256': digest(spec),
        'initialization': initialization, 'verified_loaded_architecture': verified,
        'new_policy_sha256': panel['policy_sha256'], 'old_fresh_prior_sha256': old_panel['policy_sha256'],
        'initial_mean_max_absolute_difference': mean_error, 'initial_value_max_absolute_difference': value_error,
        'new_policy_panel': panel, 'old_class_fresh_initial_reference_panel': old_panel,
        'scripted_reference': scripted, 'historical_sensor_snapshot': seal,
        'save_reload_exact_rows': True, 'empty_optimizer_state_roundtrip_verified': True,
        'nonempty_optimizer_state_roundtrip': 'NOT TESTED', 'checkpoint_sha256': sha256(saved),
        'training_timesteps': 0, 'optimizer_updates': 0, 'physics_steps': 0, 'new_scans': 0,
        'episodes': 0, 'validation_episodes': 0, 'test_episodes': 0,
        'budget_snapshot_sha256': digest(budget), 'authorization_files_unchanged': before == after,
        'authorization_file_count': len(before), 'authorization_files_sha256': before, 'history': history,
        'wall_seconds': time.monotonic()-begun, 'cpu_seconds': time.process_time()-cpu,
        'worker_hwm_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'pre_model_mem_available_mib': resources['mem_available_mib'],
        'claim_limit': configuration()['claim_limit']}
    atomic_json(OUT/'result.json', result, exclusive=True)
    atomic_json(OUT/'complete.json', {'result_sha256': sha256(OUT/'result.json')}, exclusive=True)
    return {'status': 'PASS', 'result_sha256': sha256(OUT/'result.json'), 'optimizer_updates': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    import json
    print(json.dumps(run(), allow_nan=False))


if __name__ == '__main__':
    main()
