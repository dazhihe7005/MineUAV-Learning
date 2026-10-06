"""Differentiable fixed-action K-step objective, without horizon teacher forcing.

Complete episodes are shuffled/minibatched as in the one-step trainer. All legal
windows inside each batch have equal weight. Prefix encoding is differentiable:
there is no loss on the prefix, and no detach anywhere in the prediction graph.
The unidirectional prefix cache at t includes x_t exactly once and no future.
"""
import random
import time

import numpy as np
import torch

from latent_dynamics_data import normalize
from latent_dynamics_models import make_model, save_model, load_model
from run_latent_memory_ablation import parameter_hash


def predict_windows(model, episodes, statistics, horizon, starts_by_episode=None):
    if not isinstance(horizon, int) or horizon < 1 or not episodes:
        raise ValueError('invalid window horizon/episodes')
    if starts_by_episode is None:
        starts_by_episode = [np.arange(max(0, len(ep['obs'])-horizon+1)) for ep in episodes]
    if len(starts_by_episode) != len(episodes):
        raise ValueError('invalid window start lists')
    selected = []
    for ep, starts in zip(episodes, starts_by_episode):
        starts = np.asarray(starts)
        if not len(starts):
            continue
        if (starts.ndim != 1 or not np.issubdtype(starts.dtype, np.integer)
                or np.any(starts < 0) or np.any(starts+horizon > len(ep['obs']))):
            raise ValueError('invalid window: crosses episode boundary')
        selected.append((ep, starts))
    if not selected:
        raise ValueError('no legal windows')
    # Every GRU batch lane is a new complete episode with zero initial hidden.
    length = max(len(ep['obs']) for ep, _ in selected)
    prefix = np.zeros((len(selected), length, 11), np.float32)
    for i, (ep, _) in enumerate(selected):
        prefix[i, :len(ep['obs'])] = np.concatenate([
            normalize(ep['obs'], statistics, 'obs'),
            normalize(ep['previous_action'], statistics, 'previous_action')], -1)
    z, _ = model.encoder(torch.from_numpy(prefix), None)
    state = torch.cat([z[i, starts] for i, (_, starts) in enumerate(selected)]).unsqueeze(0)
    current = torch.from_numpy(np.concatenate([ep['obs'][ids] for ep, ids in selected]).astype(np.float64))
    # Ground truth is separate and used only by the caller's loss, never forward.
    truth = torch.from_numpy(np.concatenate([
        ep['next_obs'][ids[:, None]+np.arange(horizon)[None, :]] for ep, ids in selected
    ]).astype(np.float64))
    obs_mean = torch.tensor(statistics['obs']['mean'], dtype=torch.float64)
    obs_std = torch.tensor(statistics['obs']['std'], dtype=torch.float64)
    delta_mean = torch.tensor(statistics['delta']['mean'], dtype=torch.float64)
    delta_std = torch.tensor(statistics['delta']['std'], dtype=torch.float64)
    steps, states = [], []
    for k in range(horizon):
        action = torch.from_numpy(np.concatenate([
            normalize(ep['action'][ids+k], statistics, 'action') for ep, ids in selected]))
        if k == 0:
            delta = model.head(torch.cat([state[0], action], -1))
        else:
            previous = torch.from_numpy(np.concatenate([
                normalize(ep['previous_action'][ids+k], statistics, 'previous_action') for ep, ids in selected]))
            x = torch.cat([((current-obs_mean)/obs_std).float(), previous], -1)
            delta, state = model.predict_step(x, action, state)
        current = current + delta.double()*delta_std + delta_mean
        steps.append(current); states.append(state)
    return dict(predictions=torch.stack(steps, 1), truth=truth,
                steps=steps, states=states, window_count=len(current))


def observation_loss(prediction, truth, statistics):
    scale = torch.tensor(statistics['obs']['std'], dtype=prediction.dtype)
    return ((prediction-truth)/scale).square().mean()


def diagnostics(out):
    pred = out['predictions'].detach()
    norms = torch.stack([s.detach().norm(dim=-1).max() for s in out['states']])
    if not torch.isfinite(pred).all() or not torch.isfinite(norms).all():
        raise FloatingPointError('non-finite prediction/GRU hidden: stop, do not clip or detach')
    return dict(prediction_abs_max=pred.abs().max().item(), hidden_norm_max=norms.max().item())


@torch.no_grad()
def validation_loss(model, episodes, stats, horizon, batch_size=16):
    model.eval(); total, count = 0., 0
    for start in range(0, len(episodes), batch_size):
        out = predict_windows(model, episodes[start:start+batch_size], stats, horizon)
        diagnostics(out)
        loss = observation_loss(out['predictions'], out['truth'], stats)
        if not torch.isfinite(loss):
            raise FloatingPointError('non-finite validation loss')
        total += loss.item()*out['window_count']; count += out['window_count']
    return total/count


def train_multistep(train, val, stats, config, path):
    """Test is intentionally not an argument. Save validation-best model only."""
    seed = config['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model = make_model('history'); initial = parameter_hash(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(seed); history = []; best = float('inf'); best_epoch = None
    started = time.monotonic()
    initial_validation = validation_loss(model, val, stats, config['horizon'], config['batch_size'])
    for epoch in range(1, config['epochs']+1):
        model.train(); total, count = 0., 0; gradient_norms = []
        maxima = dict(prediction_abs_max=0., hidden_norm_max=0.)
        order = rng.permutation(len(train))
        for start in range(0, len(train), config['batch_size']):
            out = predict_windows(model, [train[i] for i in order[start:start+config['batch_size']]],
                                  stats, config['horizon'])
            for key, value in diagnostics(out).items():
                maxima[key] = max(maxima[key], value)
            optimizer.zero_grad(set_to_none=True)
            loss = observation_loss(out['predictions'], out['truth'], stats)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss epoch {epoch}: stop')
            loss.backward()
            norm = torch.sqrt(sum(p.grad.detach().square().sum() for p in model.parameters() if p.grad is not None))
            if not torch.isfinite(norm):
                raise FloatingPointError(f'non-finite gradient epoch {epoch}: stop, no new clipping')
            gradient_norms.append(norm.item()); optimizer.step()
            total += loss.item()*out['window_count']; count += out['window_count']
        validation = validation_loss(model, val, stats, config['horizon'], config['batch_size'])
        row = dict(epoch=epoch, train_loss=total/count, validation_loss=validation,
                   gradient_norm_mean=float(np.mean(gradient_norms)), gradient_norm_max=max(gradient_norms),
                   windows_used=count, **maxima)
        history.append(row)
        if validation < best:
            best, best_epoch = validation, epoch
            save_model(path, model, 'history', stats, dict(best_epoch=epoch, validation_loss=best,
                train_loss_online_average=total/count, initial_parameter_sha256=initial, config=config,
                selection_metric='Validation all-window uniform K-step observation-normalized MSE'))
        print(f'multistep: epoch {epoch:02d}/{config["epochs"]} train={total/count:.8f} val={validation:.8f} '
              f'grad_max={max(gradient_norms):.5f} hidden_max={maxima["hidden_norm_max"]:.3f}', flush=True)
    best_model, _, _ = load_model(path)
    return dict(best_epoch=best_epoch, final_epoch=config['epochs'], initial_parameter_sha256=initial,
        best_validation_loss=best, initial_validation_loss=initial_validation,
        best_checkpoint_train_loss=validation_loss(best_model, train, stats, config['horizon'], config['batch_size']),
        final_train_loss=history[-1]['train_loss'], final_validation_loss=history[-1]['validation_loss'],
        elapsed_seconds=time.monotonic()-started, history=history, parameter_count=sum(p.numel() for p in model.parameters()),
        optimizer='Adam defaults (betas=.9,.999 eps=1e-8), no gradient clipping',
        epoch_order_rule='default_rng(seed=0).permutation(train_episode_count); 16 complete-episode minibatches',
        warmup_gradient_rule='differentiable true prefix, no prefix loss; all K prediction steps fully connected')
