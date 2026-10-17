"""R29 fixed foreground inference only. No environment/rollout/learning entry."""
import argparse
import copy
import hashlib
import json
import os
import resource
import time
from pathlib import Path
import numpy as np
import torch
from research_ops.storage import OpsError, FileLock, read_json, atomic_json, digest, sha256, process_identity
from research_ops.identity import source_identity
from research_ops.environment import enforce_resources
from .runtime import ROOT, HOME, verify_current
from .budget import BudgetStore
from .policy_function_probe import require_model, references, probe_grid, make_protocol, assess

OUT = ROOT / '.research_local/autorl_r29_policy_function/run_01'


def load_model(reference, prereg, trusted):
    require_model(reference, trusted)
    # A fresh production reference check precedes pickle deserialization.
    require_model(reference, references())
    from lidar_integration.ppo_formal_checkpoint import CheckpointManager
    from .state_centered_policy import verify_centered_policy
    model = CheckpointManager(ROOT / Path(reference['checkpoint_path']).parent, prereg,
        execution_protocol_sha256=reference['execution_protocol_sha256']).load_verified(None, run_id=reference['run_id'])
    verify_centered_policy(model)
    if model.num_timesteps != reference['step']:
        raise OpsError('R29 model step differs')
    if not all(torch.isfinite(p).all() for p in model.policy.parameters()):
        raise OpsError('R29 nonfinite checkpoint parameters')
    model.policy.set_training_mode(False)
    return model


def make_observation(context, states):
    shapes = {'lidar': (1, 3, 12, 24), 'state': (1, 7), 'heading': (1, 2), 'lidar_age': (1, 1)}
    if (not isinstance(context, dict) or set(context) != set(shapes)
            or any(not isinstance(context[k], np.ndarray) or context[k].shape != shape
                or context[k].dtype != np.float32 or not np.isfinite(context[k]).all() for k, shape in shapes.items())):
        raise ValueError('invalid saved observation dimensions/dtype/finite values')
    lidar, heading, age = context['lidar'], context['heading'], context['lidar_age']
    if (np.any((lidar < 0) | (lidar > 1)) or not np.isin(lidar[:, 1:], (0., 1.)).all()
            or np.any(lidar[:, 2] > lidar[:, 1]) or abs(float(np.sum(heading ** 2)) - 1.) > 1e-5
            or np.any((age < 0.) | (age > .15))):
        raise ValueError('invalid saved lidar masks/range/heading/age')
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 7 or not 1 <= len(states) <= 79 or not np.isfinite(states).all():
        raise ValueError('invalid bounded finite counterfactual states')
    output = {k: np.repeat(v, len(states), axis=0) for k, v in context.items()}
    output['state'] = states.astype(np.float32, copy=True)
    if not np.isfinite(output['state']).all():
        raise ValueError('counterfactual float32 conversion overflow')
    return output


def observation_sha(context):
    h = hashlib.sha256()
    for key, value in sorted(context.items()):
        h.update(key.encode()); h.update(str(value.shape).encode()); h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def forward(model, observation):
    tensor, _ = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(tensor).distribution
    raw = distribution.mean.detach().cpu().numpy()
    action, _ = model.predict(observation, deterministic=True)
    value = model.policy.predict_values(tensor).detach().cpu().numpy()
    std = distribution.stddev.detach().cpu().numpy()
    if any(not np.isfinite(x).all() for x in (raw, action, value, std)):
        raise OpsError('R29 nonfinite policy inference')
    return {'raw_mean': raw.tolist(), 'clipped_action': action.tolist(), 'value': value[:, 0].tolist(),
        'std': std.tolist()}


def verify_checks(reg):
    p = ROOT / '.research_local/autorl_r29_checks_01.json'; v = read_json(p)
    if (v['status'] != 'PASS' or v['registration_sha256'] != digest(reg)
            or v['source_identity_sha256'] != digest(reg['source_identity']) or v['optimizer_updates']):
        raise OpsError('R29 incomplete/stale regression identity')
    actual = {}
    for k in ('operations', 'tensor', 'historical', 'runtime_binding'):
        ref = v['evidence'][k]; path = ROOT / ref['path']
        if ref['path'] != f'.research_local/autorl_r29_{k}_final_01.json' or sha256(path) != ref['sha256']:
            raise OpsError('R29 changed regression evidence')
        actual[k] = read_json(path)
    op, te, hi, bi = (actual[k] for k in ('operations', 'tensor', 'historical', 'runtime_binding'))
    if (not op['complete'] or (op['tests_run'], op['failures'], op['errors'], op['optimizer_updates']) != (258, 0, 0, 0)
            or len(op['results']) != 258 or any(r['outcome'] != 'PASS' for r in op['results'])
            or op['source_identity_sha256'] != digest(reg['source_identity'])
            or op['test_source_identity'] != source_identity(ROOT, ['tests/research_ops_v2'])
            or te['returncode'] or te['tests_run'] != 85 or te['optimizer_updates'] or 'Ran 85 tests' not in te['stderr']
            or te['source_identity_sha256'] != digest(reg['source_identity']) or te['registration_sha256'] != digest(reg)
            or not hi['complete'] or hi['exit_code'] or hi['actual_optimizer_updates'] or hi['forbidden_calls']
            or len(hi['results']) != 885 or any(r['outcome'] != 'passed' for r in hi['results'])
            or bi['returncode'] or bi['registration_sha256'] != digest(reg)
            or bi['source_before_sha256'] != digest(reg['source_identity'])
            or bi['source_after_sha256'] != digest(reg['source_identity']) or bi['final'] != v['evidence']['historical']):
        raise OpsError('R29 current full selected regression incomplete')
    return sha256(p)


def probe(reg, spec):
    import mujoco
    from stable_baselines3 import PPO
    from unittest.mock import patch
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from lidar_integration.ppo_formal_validation import _optimizer_states_equal
    from lidar_integration.ppo_training_runner import _weights_sha
    def forbidden(*args, **kwargs):
        raise AssertionError('R29_FORBIDS_TRAINING_OPTIMIZER_AND_PHYSICS')
    started = time.monotonic(); cpu = time.process_time(); minimum = float('inf')
    frozen = {str(p.relative_to(ROOT)): sha256(p) for p in HOME.rglob('*') if p.is_file()}
    budget = BudgetStore(HOME).status(); rows = []; models = []; context = None
    def guard():
        nonlocal minimum
        minimum = min(minimum, enforce_resources(ROOT)['mem_available_mib'])
        if (time.monotonic() - started > spec['wall_seconds'] - 5 or BudgetStore(HOME).status()['stop_requested']
                or source_identity(ROOT, reg['source_paths']) != reg['source_identity']):
            raise OpsError('R29 duration/stop/source protection')
    torch.set_num_threads(1); prereg = load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    with patch.object(PPO, 'learn', forbidden), patch.object(PPO, 'train', forbidden), \
            patch.object(torch.optim.Adam, 'step', forbidden), patch.object(mujoco, 'mj_step', forbidden):
        for reference in spec['models']:
            guard(); model = load_model(reference, prereg, spec)
            before = _weights_sha(model.policy); optimizer = copy.deepcopy(model.policy.optimizer.state_dict())
            if context is None:
                context = copy.deepcopy(model._last_obs)
                # This is a saved real sensor context, not a recreated live sample.
                make_observation(context, np.zeros((1, 7), dtype=np.float32))
                atomic_json(OUT/'context.json', {'checkpoint': reference, 'observation_sha256': observation_sha(context),
                    'observation': {k: v.tolist() for k, v in context.items()},
                    'source_sample_timestamp': 'NOT_AVAILABLE_FROM_SAVED_OBSERVATION',
                    'scope': 'stored step0 observation;not current or future physical sample'}, exclusive=True)
            immutable_context = observation_sha(context)
            observation = make_observation(context, np.asarray([v['state'] for v in probe_grid()], dtype=np.float32))
            infer_start = time.perf_counter()
            with torch.inference_mode():
                output = forward(model, observation); repeated = forward(model, observation)
            elapsed = (time.perf_counter() - infer_start) * 1000.
            if (output != repeated or observation_sha(context) != immutable_context or _weights_sha(model.policy) != before
                    or model.num_timesteps != reference['step'] or not _optimizer_states_equal(optimizer, model.policy.optimizer.state_dict())
                    or any(p.grad is not None for p in model.policy.parameters())):
                raise OpsError('R29 nondeterministic or mutated model/context')
            if reference['step'] == 0 and before != '650a5841c96a054bf43db9a54a6c5c6da6af8b4b362e4c934a2121d0a16849a4':
                raise OpsError('R29 original prior policy differs')
            rows.extend({'step': reference['step'], 'grid_index': i, 'raw_mean': output['raw_mean'][i],
                'clipped_action': output['clipped_action'][i], 'value': output['value'][i], 'std': output['std'][i]}
                for i in range(len(probe_grid())))
            models.append({'reference': reference, 'policy_sha256_before_after': before, 'optimizer_unchanged': True,
                'training_timesteps_before_after': model.num_timesteps, 'paired_batch_forward_wall_ms': elapsed,
                'legacy_loader_mode': 'WARM_START_NAME_DESERIALIZATION_ONLY_NO_TRAINING'})
            del model, optimizer
    guard()
    if (BudgetStore(HOME).status() != budget or spec != make_protocol(reg)
            or {str(p.relative_to(ROOT)): sha256(p) for p in HOME.rglob('*') if p.is_file()} != frozen):
        raise OpsError('R29 authorization/raw/protocol changed')
    result = {'status': 'PASS', 'registration_sha256': digest(reg), 'source_identity_sha256': digest(reg['source_identity']),
        'protocol_sha256': digest(spec), 'models': models, 'rows': rows, 'assessment': assess(rows),
        'context_file': {'path': str((OUT/'context.json').relative_to(ROOT)), 'sha256': sha256(OUT/'context.json')},
        'context_sha256': observation_sha(context), 'grid_sha256': digest(probe_grid()),
        'model_input_rows': 237, 'exact_repeated_inference_rows': 237,
        'training_timesteps': 0, 'optimizer_updates': 0, 'physics_ticks': 0, 'new_episodes': 0, 'test_episodes': 0,
        'authorization_files_unchanged': True, 'authorization_file_count': len(frozen), 'budget_snapshot_sha256': digest(budget),
        'wall_seconds': time.monotonic() - started, 'cpu_seconds': time.process_time() - cpu,
        'worker_hwm_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        'guard_sampled_min_MemAvailable_mib': minimum, 'claim_limit': spec['claim_limit']}
    atomic_json(OUT/'result.json', result, exclusive=True)
    atomic_json(OUT/'complete.json', {'result_sha256': sha256(OUT/'result.json')}, exclusive=True)
    return {'status': 'PASS', 'result_sha256': sha256(OUT/'result.json'), 'training_timesteps': 0}


def run(expected_sha):
    if os.getpgrp() != os.getpid():
        raise OpsError('R29 requires an owned isolated foreground process group')
    from research_ops_v2.manager import OperationsV2
    from .lifetime import start_guard, stop_guard
    m = OperationsV2(ROOT); parent = process_identity(os.getppid())
    with FileLock(m.lock_path):
        _, reg = verify_current()
        if reg['phase_id'] != 'AUTORL-R29-POLICY-FUNCTION':
            raise OpsError('R29 wrong fixed reviewed phase')
        verify_checks(reg); enforce_resources(ROOT)
        spec = read_json(ROOT/'mujoco/reports/autonomous_rl_policy_function_r29_prereg.json')
        if digest(spec) != expected_sha or spec != make_protocol(reg):
            raise OpsError('R29 frozen protocol/explicit identity differs')
        watcher = start_guard(parent, time.monotonic() + spec['wall_seconds'], ROOT)
        try:
            OUT.mkdir(parents=True, exist_ok=False)
            atomic_json(OUT/'attempt.json', {'owner': process_identity(), 'parent': parent,
                'protocol_sha256': digest(spec), 'kind': spec['kind']}, exclusive=True)
            try:
                return probe(reg, spec)
            except BaseException as error:
                import traceback
                atomic_json(OUT/'failure.json', {'exception_type': type(error).__name__, 'message': str(error)[:2000],
                    'traceback': traceback.format_exc()[-12000:], 'context': 'unique no-learning diagnostic;no retry'}, exclusive=True)
                raise
        finally:
            stop_guard(watcher)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha', required=True)
    print(json.dumps(run(parser.parse_args().expected_sha), allow_nan=False))


if __name__ == '__main__':
    main()
