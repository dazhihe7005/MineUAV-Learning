"""Read-only fixed-action autoregressive dynamics, with exact prefix warm-up.

Metrics are endpoint errors, not time-averaged errors inside each window.
The PI baseline's hidden input is teacher-forced; predicted obs are never reset.
"""
import numpy as np
import torch

from latent_dynamics_data import DIMENSIONS, json_hash, normalize

HORIZONS = (1, 5, 10, 25, 50)
KINDS = ('markov', 'no_memory', 'history', 'pi_state')


def start_manifest(episodes, horizons, dataset_hash):
    if not horizons or any(int(h) != h or h < 1 for h in horizons):
        raise ValueError('invalid horizon')
    rows = []
    for i, ep in enumerate(episodes):
        n = len(ep['obs'])
        rows.append(dict(episode_id=i, target_id=ep['metadata']['target_id'],
                         source=ep['metadata']['source'], transition_count=n,
                         source_rows=ep['source_rows'],
                         valid_start_count={str(h): max(0, n-h+1) for h in horizons}))
    # Each entry's valid starts are exactly range(valid_start_count[h]); compact,
    # lossless specification of every sample identity, not a random subsample.
    manifest = dict(protocol='all starts t in range(max(0,N-H+1)), per episode',
                    dataset_test_sha256=dataset_hash, horizons=list(horizons),
                    episodes=rows, window_counts={str(h): sum(r['valid_start_count'][str(h)] for r in rows)
                                                  for h in horizons})
    manifest['manifest_sha256'] = json_hash(manifest)
    return manifest


@torch.inference_mode()
def teacher_latents(model, episode, statistics):
    x = np.concatenate([normalize(episode['obs'], statistics, 'obs'),
                        normalize(episode['previous_action'], statistics, 'previous_action')], -1)
    latent, _ = model.encoder(torch.from_numpy(x)[None], None)
    return latent[0].clone()


@torch.inference_mode()
def rollout_windows(model, kind, episode, statistics, starts, horizon, prefix_latents=None):
    starts = np.asarray(starts)
    n = len(episode['obs'])
    if (kind not in KINDS or not isinstance(horizon, (int, np.integer)) or horizon < 1
            or starts.ndim != 1 or len(starts) == 0 or not np.issubdtype(starts.dtype, np.integer)
            or np.any(starts < 0) or np.any(starts + horizon > n)):
        raise ValueError('invalid rollout window')
    model.eval()
    if kind == 'history':
        prefix_latents = (teacher_latents(model, episode, statistics)
                          if prefix_latents is None else prefix_latents)
        # cache[t] already includes x_t exactly once. First head evaluation uses
        # z_t directly, then only predictions drive the encoder at t+1 onward.
        state = prefix_latents[starts].unsqueeze(0).clone()
    else:
        state = None
    current = episode['obs'][starts].astype(np.float64).copy()
    predictions, autoregressive_latents, true_latents = [], [], []
    delta_std = np.asarray(statistics['delta']['std'])
    delta_mean = np.asarray(statistics['delta']['mean'])
    for k in range(horizon):
        index = starts + k
        action = torch.from_numpy(normalize(episode['action'][index], statistics, 'action'))
        obs = torch.from_numpy(normalize(current, statistics, 'obs'))
        if kind in ('history', 'no_memory'):
            previous = torch.from_numpy(normalize(episode['previous_action'][index], statistics, 'previous_action'))
            if kind == 'history' and k == 0:
                delta = model.head(torch.cat([state[0], action], -1))
            else:
                x = torch.cat([obs, previous], -1)
                delta, state = model.predict_step(x, action, state if kind == 'history' else None)
            if kind == 'history':
                autoregressive_latents.append(state[0].numpy().copy())
                true_latents.append(prefix_latents[index].numpy().copy())
        else:
            batch = dict(obs=obs, action=action)
            if kind == 'pi_state':
                batch['pi'] = torch.from_numpy(normalize(episode['pi'][index], statistics, 'pi'))
            delta = model(batch)
        current = current + delta.numpy().astype(np.float64) * delta_std + delta_mean
        predictions.append(current.copy())
    result = dict(predictions=np.stack(predictions, axis=1))
    if kind == 'history':
        result['latent_autoregressive'] = np.stack(autoregressive_latents, axis=1)
        result['latent_teacher_forced'] = np.stack(true_latents, axis=1)
    return result


class ErrorAccumulator:
    """Streaming all-window endpoint errors; no raw rollout arrays persisted."""
    def __init__(self, observation_std):
        self.std = np.asarray(observation_std, np.float64)
        self.count = 0
        self.squared = np.zeros(7, np.float64)
        self.absolute = np.zeros(7, np.float64)
        self.nonfinite_windows = 0

    def add(self, error):
        error = np.asarray(error, np.float64)
        if len(error) == 0:
            return
        good = np.isfinite(error).all(axis=1)
        self.nonfinite_windows += int((~good).sum())
        error = error[good]
        self.count += len(error)
        self.squared += np.sum(error**2, axis=0)
        self.absolute += np.sum(np.abs(error), axis=0)

    def result(self):
        if self.count == 0:
            return dict(window_count=0, nonfinite_windows=self.nonfinite_windows)
        mse = self.squared / self.count
        return dict(window_count=self.count, nonfinite_windows=self.nonfinite_windows,
                    normalized_observation_rmse=float(np.sqrt(np.mean(mse / self.std**2))),
                    physical_overall_rmse_mixed_units=float(np.sqrt(mse.mean())),
                    physical={name: dict(rmse=float(np.sqrt(mse[i])), mae=float(self.absolute[i]/self.count))
                              for i, name in enumerate(DIMENSIONS)},
                    horizontal_velocity_rmse=float(np.sqrt(mse[3:5].mean())),
                    vertical_velocity_rmse=float(np.sqrt(mse[5])))


class LatentAccumulator:
    def __init__(self):
        self.count = 0; self.l2_sum = 0.; self.cosine_sum = 0.
        self.auto_norm_sum = 0.; self.true_norm_sum = 0.; self.max_norm = 0.

    def add(self, predicted, reference):
        good = np.isfinite(predicted).all(1) & np.isfinite(reference).all(1)
        predicted, reference = predicted[good].astype(np.float64), reference[good].astype(np.float64)
        auto, true = np.linalg.norm(predicted, axis=1), np.linalg.norm(reference, axis=1)
        self.count += len(predicted)
        self.l2_sum += float(np.linalg.norm(predicted-reference, axis=1).sum())
        self.cosine_sum += float((np.sum(predicted*reference, axis=1) / np.maximum(auto*true, 1e-12)).sum())
        self.auto_norm_sum += float(auto.sum()); self.true_norm_sum += float(true.sum())
        if len(auto):
            self.max_norm = max(self.max_norm, float(auto.max()))

    def result(self):
        if not self.count:
            return dict(window_count=0)
        return dict(window_count=self.count, mean_l2=self.l2_sum/self.count,
                    mean_cosine=self.cosine_sum/self.count,
                    autoregressive_mean_norm=self.auto_norm_sum/self.count,
                    teacher_forced_mean_norm=self.true_norm_sum/self.count,
                    autoregressive_max_norm=self.max_norm,
                    indexing='encoder z_(t+H-1) used to predict endpoint o_(t+H); H=1 is identical warmed z_t')
