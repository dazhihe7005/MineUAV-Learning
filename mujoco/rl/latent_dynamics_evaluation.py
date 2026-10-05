"""Physical and train-standardized delta metrics, plus frozen post-hoc probe."""
import numpy as np
import torch

from latent_dynamics_data import DIMENSIONS, pad_episodes


def metrics(prediction, truth, names=DIMENSIONS):
    pred, true = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    if pred.shape != true.shape or pred.ndim != 2 or len(pred) == 0:
        raise ValueError('invalid prediction/target shapes')
    if not (np.isfinite(pred).all() and np.isfinite(true).all()):
        raise ValueError('non-finite predictions')
    error = pred - true
    sse = np.sum(error ** 2, axis=0)
    sst = np.sum((true - true.mean(0)) ** 2, axis=0)
    per = {name: dict(rmse=float(np.sqrt(np.mean(error[:, i] ** 2))),
                      mae=float(np.mean(np.abs(error[:, i]))),
                      r2=float(1 - sse[i] / sst[i]) if sst[i] > 1e-14 else None)
           for i, name in enumerate(names)}
    return dict(count=len(true), overall=dict(rmse=float(np.sqrt(np.mean(error ** 2))),
                                              mae=float(np.mean(np.abs(error))),
                                              r2=float(1 - sse.sum() / sst.sum()) if sst.sum() > 1e-14 else None),
                per_dimension=per)


def gap_closure(markov, history, oracle):
    gap = markov - oracle
    return float((markov - history) / gap) if gap > max(1e-9, abs(markov) * 1e-6) else None


def distance_masks(distance):
    return {'<0.1': distance < .1, '0.1-0.2': (distance >= .1) & (distance < .2),
            '0.2-0.5': (distance >= .2) & (distance < .5), '>=0.5': distance >= .5}


def magnitude_masks(magnitude, thresholds):
    low, high = thresholds
    return {'small': magnitude < low, 'medium': (magnitude >= low) & (magnitude < high),
            'large': magnitude >= high}


@torch.no_grad()
def predict_episodes(model, episodes, stats, history=False):
    predictions, latents = [], []
    for episode in episodes:
        batch = pad_episodes([episode], stats)  # each new episode starts hidden=None
        if history:
            p, z = model.predict_sequence(torch.cat([batch['obs'], batch['previous_action']], -1),
                                           batch['action'])
            latents.append(z[0].numpy())
        else:
            p = model(batch)
        physical = p[0].numpy() * np.asarray(stats['delta']['std']) + np.asarray(stats['delta']['mean'])
        predictions.append(physical)
    return predictions, latents


def fit_linear_probe(train_latent, train_pi):
    """Post-hoc unregularized OLS with intercept; encoder is never passed here."""
    x = np.column_stack([np.asarray(train_latent, np.float64), np.ones(len(train_latent))])
    coefficient, _, rank, singular_values = np.linalg.lstsq(x, np.asarray(train_pi, np.float64), rcond=None)
    return dict(coefficient=coefficient, rank=int(rank), singular_values=singular_values)


def apply_linear_probe(latent, probe):
    return np.column_stack([np.asarray(latent, np.float64), np.ones(len(latent))]) @ probe['coefficient']


def grouped_metrics(pred, truth, masks):
    return {name: metrics(pred[mask], truth[mask]) if np.any(mask) else dict(count=0, overall=None)
            for name, mask in masks.items()}
