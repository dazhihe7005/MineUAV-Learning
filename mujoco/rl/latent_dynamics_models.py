"""Three fixed-capacity delta predictors and one shared supervised trainer."""
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from latent_dynamics_data import pad_episodes


def mlp(input_dim):
    return nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64),
                         nn.Tanh(), nn.Linear(64, 7))


class MarkovDynamics(nn.Module):
    def __init__(self, oracle=False):
        super().__init__()
        self.oracle = oracle
        self.head = mlp(14 if oracle else 11)

    def forward(self, batch):
        parts = [batch['obs']]
        if self.oracle:
            parts.append(batch['pi'])
        parts.append(batch['action'])
        return self.head(torch.cat(parts, -1))


class HistoryLatentDynamics(nn.Module):
    """No mutable episode memory. Caller explicitly resets streaming state=None.

    z_t sees current obs plus PREVIOUS action; current action goes to head ONLY.
    No PI state is read here, nor does the trainer construct an auxiliary loss.
    """
    def __init__(self):
        super().__init__()
        self.encoder = nn.GRU(11, 64, num_layers=1, batch_first=True)
        self.head = mlp(68)

    def predict_sequence(self, history_input, action):
        latent, _ = self.encoder(history_input, None)
        return self.head(torch.cat([latent, action], -1)), latent

    def predict_step(self, history_input, action, state=None):
        latent, state = self.encoder(history_input[:, None, :], state)
        return self.head(torch.cat([latent[:, 0], action], -1)), state

    def forward(self, batch):
        pred, _ = self.predict_sequence(torch.cat([batch['obs'], batch['previous_action']], -1),
                                        batch['action'])
        return pred


class NoMemoryLatentDynamics(HistoryLatentDynamics):
    """Identical parameters/head, but each timestep is a separate length-1 lane.

    Reshape B,T into B*T independent GRU batch lanes, all with hidden=None (=0).
    This is NOT sequence detachment: no previous hidden value enters any step.
    Previous executed action still remains an explicit feature of current x_t.
    """
    def predict_sequence(self, history_input, action):
        batch_size, length, _ = history_input.shape
        latent, _ = self.encoder(history_input.reshape(batch_size * length, 1, 11), None)
        latent = latent.reshape(batch_size, length, 64)
        return self.head(torch.cat([latent, action], -1)), latent

    def predict_step(self, history_input, action, state=None):
        return super().predict_step(history_input, action, None)


def make_model(kind):
    if kind == 'history':
        return HistoryLatentDynamics()
    if kind == 'no_memory':
        return NoMemoryLatentDynamics()
    if kind in ('markov', 'oracle'):
        return MarkovDynamics(oracle=(kind == 'oracle'))
    raise ValueError(f'unknown model {kind}')


def masked_mse(pred, target, mask):
    return torch.mean((pred[mask] - target[mask]).square())


def save_model(path, model, kind, statistics, metadata):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(kind=kind, state_dict=model.state_dict(), statistics=statistics,
                    metadata=metadata, architecture=dict(hidden=[64, 64], activation='Tanh',
                    gru_hidden=64 if kind in ('history', 'no_memory') else None, output=7)), path)


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(checkpoint['kind'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval(), checkpoint['statistics'], checkpoint['metadata']


@torch.no_grad()
def evaluate_loss(model, episodes, stats, batch_size=16):
    total, n = 0., 0
    model.eval()
    for start in range(0, len(episodes), batch_size):
        batch = pad_episodes(episodes[start:start + batch_size], stats)
        count = int(batch['mask'].sum()) * 7
        total += float(masked_mse(model(batch), batch['delta'], batch['mask'])) * count
        n += count
    return total / n


def train_model(kind, train, val, stats, config, path):
    """Equal complete-episode schedule for all models; Test is not an argument."""
    seed = config['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    model = make_model(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    rng = np.random.default_rng(seed)
    history = []; best = float('inf'); best_epoch = None
    started = time.monotonic()
    initial_validation = evaluate_loss(model, val, stats, config['batch_size'])
    for epoch in range(1, config['epochs'] + 1):
        model.train(); total, n = 0., 0
        order = rng.permutation(len(train))
        for start in range(0, len(train), config['batch_size']):
            batch = pad_episodes([train[i] for i in order[start:start + config['batch_size']]], stats)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_mse(model(batch), batch['delta'], batch['mask'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'non-finite loss: {kind} epoch {epoch}')
            loss.backward(); optimizer.step()
            count = int(batch['mask'].sum()) * 7
            total += loss.item() * count; n += count
        validation = evaluate_loss(model, val, stats, config['batch_size'])
        history.append(dict(epoch=epoch, train_loss=total / n, validation_loss=validation))
        if validation < best:
            best, best_epoch = validation, epoch
            save_model(path, model, kind, stats, dict(best_epoch=epoch, validation_loss=validation,
                                                     train_loss_online_average=total / n, config=config))
        if epoch == 1 or epoch % 10 == 0:
            print(f'{kind}: epoch {epoch:02d}/{config["epochs"]} train={total/n:.6f} val={validation:.6f}', flush=True)
    best_model, _, _ = load_model(path)
    return dict(best_epoch=best_epoch, final_epoch=config['epochs'], best_validation_loss=best,
                initial_validation_loss=initial_validation,
                best_checkpoint_train_loss=evaluate_loss(best_model, train, stats, config['batch_size']),
                final_train_loss=history[-1]['train_loss'], final_validation_loss=history[-1]['validation_loss'],
                elapsed_seconds=time.monotonic() - started, history=history,
                parameter_count=sum(p.numel() for p in model.parameters()),
                epoch_order_rule='default_rng(seed=0).permutation(train_episode_count), reset per model')
