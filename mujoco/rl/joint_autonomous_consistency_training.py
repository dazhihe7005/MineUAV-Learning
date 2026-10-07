"""v3: unchanged v1 observation path plus autonomous (NOT local) consistency."""
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from latent_dynamics_data import normalize
from joint_latent_world_model import JointLatentWorldModel, component_hashes
from joint_latent_training import predict_windows, observation_loss, diagnostics
from joint_consistency_training import consistency_mse, selection_epoch

LAMBDA = .1
KIND = 'joint_latent_world_model_v3_autonomous_consistency'


def autonomous_mse(prediction, target, scale):
    if prediction.ndim != 3 or prediction.shape[1:] != (10, 64) or target.shape != prediction.shape:
        raise ValueError('fixed autonomous horizon10/latent64 required')
    return consistency_mse(prediction, target, scale)


def total_loss(obs, auto_cons):
    return obs + LAMBDA * auto_cons


@torch.no_grad()
def reference_targets(model, episodes, stats, horizon=10, starts=None):
    """Training targets only: causal real histories, including terminal frame N."""
    if horizon != 10:
        raise ValueError('fixed autonomous horizon10 only')
    if starts is None:
        starts = [np.arange(max(0, len(e['action']) - horizon + 1)) for e in episodes]
    if len(starts) != len(episodes):
        raise ValueError('window episode mismatch')
    selected = []
    for ep, ids in zip(episodes, starts):
        n = len(ep['action']); ids = np.asarray(ids)
        if not (len(ep['obs']) == len(ep['previous_action']) == len(ep['next_obs']) == n):
            raise ValueError('episode boundary length mismatch')
        if not len(ids):
            continue
        if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or (ids < 0).any() or (ids + horizon > n).any():
            raise ValueError('window crosses episode boundary')
        selected.append((ep, ids))
    if not selected:
        raise ValueError('no legal reference windows')
    x = np.zeros((len(selected), max(len(ep['obs']) + 1 for ep, _ in selected), 11), np.float32)
    for i, (ep, _) in enumerate(selected):
        obs = np.vstack([ep['obs'], ep['next_obs'][-1]])
        previous = np.vstack([ep['previous_action'], ep['action'][-1]])
        x[i, :len(obs)] = np.concatenate([normalize(obs, stats, 'obs'), normalize(previous, stats, 'previous_action')], -1)
    # No gradient on any reference-branch output; source E gradient comes only
    # through the original autonomous rollout's initial prefix state.
    encoded, _ = model.encoder(torch.from_numpy(x), None)
    return torch.cat([encoded[i, ids[:, None] + np.arange(1, horizon + 1)]
                      for i, (_, ids) in enumerate(selected)]).detach()


def loss_batch(model, episodes, stats, scale, horizon=10, starts=None):
    out = predict_windows(model, episodes, stats, horizon, starts)
    obs = observation_loss(out['normalized_observations'], out['truth'], stats)
    target = reference_targets(model, episodes, stats, horizon, starts)
    cons = autonomous_mse(out['latents'][:, 1:], target, scale)
    return dict(out, loss=total_loss(obs, cons), observation_loss=obs,
                consistency_loss=cons, reference_targets=target)


@torch.no_grad()
def validation_losses(model, episodes, stats, scale, horizon=10, batch_size=16):
    model.eval(); totals = np.zeros(3); count = 0
    for begin in range(0, len(episodes), batch_size):
        batch = episodes[begin:begin + batch_size]
        if not any(len(e['action']) >= horizon for e in batch):
            continue
        row = loss_batch(model, batch, stats, scale, horizon); diagnostics(row)
        values = np.array([row['observation_loss'].item(), row['consistency_loss'].item(), row['loss'].item()])
        if not np.isfinite(values).all():
            raise FloatingPointError('non-finite v3 validation objective')
        totals += values * row['window_count']; count += row['window_count']
    if not count:
        raise ValueError('no validation windows')
    a = totals / count
    return dict(observation=float(a[0]), consistency=float(a[1]), weighted_consistency=float(LAMBDA * a[1]), total=float(a[2]))


def save_autonomous(path, model, stats, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=KIND, state_dict=model.state_dict(), statistics=stats, metadata=metadata), path)


def load_autonomous(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['kind'] != KIND:
        raise ValueError('invalid v3 checkpoint')
    model = JointLatentWorldModel(); model.load_state_dict(saved['state_dict'])
    return model.eval(), saved['statistics'], saved['metadata']


def train_autonomous(model, train, val, stats, scale, config, path):
    if (config['horizon'] != 10 or type(config['seed']) is not int or
            config['seed'] not in range(5) or config['learning_rate'] != .0003):
        raise ValueError('fixed K10 seeds0..4 lr.0003 only')
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model.requires_grad_(True); before = component_hashes(model)
    initial = validation_losses(model, val, stats, scale, 10, config['batch_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(seed); history = []; best = float('inf'); started = time.monotonic()
    parameters = list(model.parameters()); slices = {}; offset = 0
    for key in before:
        length = len(list(getattr(model, key).parameters())); slices[key] = slice(offset, offset + length); offset += length
    for epoch in range(1, config['epochs'] + 1):
        model.train(); total = np.zeros(3); count = 0
        norms = {k: [] for k in before}; objective = {k: [] for k in before}
        maxima = dict(normalized_prediction_abs_max=0., latent_norm_max=0.)
        order = rng.permutation(len(train))
        for begin in range(0, len(train), config['batch_size']):
            batch = [train[j] for j in order[begin:begin + config['batch_size']]]
            if not any(len(e['action']) >= 10 for e in batch):
                continue
            optimizer.zero_grad(set_to_none=True)
            row = loss_batch(model, batch, stats, scale)
            for k, value in diagnostics(row).items():
                maxima[k] = max(maxima[k], value)
            if not torch.isfinite(row['loss']):
                raise FloatingPointError('non-finite v3 training loss; no clipping')
            cg = torch.autograd.grad(LAMBDA * row['consistency_loss'], parameters, retain_graph=True, allow_unused=True)
            row['loss'].backward()
            for key in before:
                ps, cs = parameters[slices[key]], cg[slices[key]]
                if any(p.grad is None for p in ps):
                    raise AssertionError(f'{key} missing total gradient')
                cn = torch.sqrt(sum(c.square().sum() for c in cs if c is not None)) if any(c is not None for c in cs) else torch.tensor(0.)
                tn = torch.sqrt(sum(p.grad.square().sum() for p in ps))
                on = torch.sqrt(sum((p.grad - (c if c is not None else 0)).square().sum() for p, c in zip(ps, cs)))
                if not torch.isfinite(torch.stack([cn, tn, on])).all():
                    raise FloatingPointError('non-finite v3 gradient; no clipping')
                norms[key].append(tn.item())
                objective[key].append((cn.item(), on.item(), cn.item()/max(on.item(), 1e-12), cn.item()/max(tn.item(), 1e-12)))
            optimizer.step()
            total += np.array([row['observation_loss'].item(), row['consistency_loss'].item(), row['loss'].item()]) * row['window_count']
            count += row['window_count']
        if not count:
            raise ValueError('no training windows')
        valid = validation_losses(model, val, stats, scale, 10, config['batch_size']); a = total / count
        row = dict(epoch=epoch,
            train=dict(observation=float(a[0]), consistency=float(a[1]), weighted_consistency=float(LAMBDA*a[1]), total=float(a[2])),
            validation=valid, windows_used=count,
            episode_order_sha256=hashlib.sha256(order.astype('<i8').tobytes()).hexdigest(),
            gradient_norms={k: dict(mean=float(np.mean(v)), max=max(v)) for k, v in norms.items()},
            objective_gradient_diagnostics={k: dict(weighted_consistency_norm=float(np.mean([t[0] for t in v])),
                observation_norm=float(np.mean([t[1] for t in v])), mean_consistency_to_observation_ratio=float(np.mean([t[2] for t in v])),
                max_consistency_to_observation_ratio=float(max(t[2] for t in v)), mean_consistency_to_total_ratio=float(np.mean([t[3] for t in v])),
                fraction_batches_consistency_norm_exceeds_observation=float(np.mean([t[0] > t[1] for t in v]))) for k, v in objective.items()}, **maxima)
        history.append(row)
        if valid['observation'] < best:
            best = valid['observation']
            save_autonomous(path, model, stats, dict(config=config, best_epoch=epoch, validation_observation_loss=best,
                parameter_hashes_before=before, initial_latent_std=list(scale), lambda_autonomous_consistency=LAMBDA,
                local_consistency_enabled=False, selection_metric='Validation observation objective only, uniform all-window k0..10 normalized MSE'))
        print(f'Autonomous epoch{epoch:02d}/{config["epochs"]} train_obs={a[0]:.8f} train_auto={a[1]:.8f} '
              f'val_obs={valid["observation"]:.8f} val_auto={valid["consistency"]:.8f} '
              f'grad_max E/T/D={"/".join(f"{max(norms[k]):.5f}" for k in before)} latent_max={maxima["latent_norm_max"]:.3f}', flush=True)
    selected, _, meta = load_autonomous(path); after = component_hashes(selected)
    if any(before[k] == after[k] for k in before):
        raise AssertionError('component did not update')
    return dict(best_epoch=meta['best_epoch'], final_epoch=config['epochs'], initial_validation=initial,
        best_validation=validation_losses(selected, val, stats, scale, 10, config['batch_size']),
        best_checkpoint_train=validation_losses(selected, train, stats, scale, 10, config['batch_size']),
        final_train=history[-1]['train'], final_validation=history[-1]['validation'], history=history,
        parameter_hashes_before=before, parameter_hashes_best=after, parameter_count=sum(p.numel() for p in model.parameters()),
        architecture={k: repr(getattr(model, k)) for k in before}, elapsed_seconds=time.monotonic()-started,
        optimizer='same v1 Adam defaults lr=.0003 betas=.9/.999 eps1e-8; no clipping',
        order_rule=f'same default_rng(seed={seed}) complete-episode permutations/all legalK10 starts/per episode batch',
        objective_gradient_rule='all minibatches: weighted autonomous-consistency via autograd.grad; total via loss.backward; observation=total-consistency (FP roundoff); diagnostic only',
        consistency_kind='autonomous multi-step only; no local term; fixed detached current-Encoder reference targets')
