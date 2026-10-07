"""Thirteen requested static v1/v2/v3 comparisons; no raw trajectory export."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from joint_latent_world_model import load_joint, predict_window
from joint_consistency_training import load_consistency


def decoder_sensitivity_figure(datasets, x):
    """Summary-only figure, also usable for read-only label-layout regeneration."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, data, style in datasets:
        horizons = [str(h) for h in (1, 5, 10, 25, 50)]
        axes[0].plot(x, [data['autonomous'][h]['decoder_sensitivity']['amplification_ratio']['mean'] for h in horizons], style, label=name)
        axes[1].plot(x, [data['autonomous'][h]['decoder_sensitivity']['decoded_normalized_l2']['mean'] for h in horizons], style, label=name)
    for ax, label in zip(axes, ['Empirical amplification\n(NOT Jacobian; native coordinates)', 'Mean normalized D(pred)-D(reference) L2']):
        ax.legend(); ax.set_xlabel('Horizon (s)'); ax.set_ylabel(label); ax.grid(alpha=.2)
    return fig


def render_figures(r, plot, context, test, directory):
    paths = []; directory = Path(directory); hs = r['horizons_steps']; x = np.asarray(hs)/25
    datasets = [('Joint v1', r['baselines']['v1']['audit'], 'o--'),
                ('Local v2', r['baselines']['v2']['audit'], 's:'), ('Autonomous v3', r['audit'], '^-')]
    def save(name):
        p = directory/(name+'.png'); plt.tight_layout(); plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close(); paths.append(str(p))
    def labels(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=.2)
    def curves(ax, field, key, ylabel):
        for name, data, style in datasets:
            ax.plot(x, [data['autonomous'][str(h)][field][key] for h in hs], style, label=name)
        ax.legend(); labels(ax, 'Recorded-action horizon (s)', ylabel)
    rows = r['training']['history']; epochs = [a['epoch'] for a in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for part, key, style in [('train', 'observation', '-'), ('validation', 'observation', '--'),
                             ('train', 'weighted_consistency', ':'), ('train', 'total', '-.')]:
        axes[0].plot(epochs, [a[part][key] for a in rows], style, label=f'{part} {key}')
    axes[0].legend(); labels(axes[0], 'Epoch', 'MSE objectives (different scales)')
    for module in ('encoder', 'transition', 'decoder'):
        axes[1].plot(epochs, [a['objective_gradient_diagnostics'][module]['mean_consistency_to_observation_ratio'] for a in rows], label=module)
    axes[1].axhline(1, color='gray', ls='--'); axes[1].legend(); labels(axes[1], 'Epoch', 'Mean weighted-auto / observation gradient norm')
    save('autonomous_consistency_training_curve')
    fig, ax = plt.subplots(figsize=(8, 4)); curves(ax, 'observation', 'normalized_observation_rmse', 'Normalized observation endpoint RMSE'); save('v1_v2_v3_rmse_horizon')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    curves(axes[0], 'latent', 'normalized_rmse', 'Own-Train-scale latent RMSE (diagnostic)')
    curves(axes[1], 'latent', 'mean_cosine', 'Within-model latent cosine (different geometries)'); save('v1_v2_v3_latent_consistency')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); cs = ['1', '5', '10', '25', 'never']; xx = np.arange(5)
    for name, data, style in datasets:
        for ax, field, key in [(axes[0], 'observation', 'normalized_observation_rmse'), (axes[1], 'latent', 'normalized_rmse')]:
            ax.plot(xx, [data['periodic_correction'][c][field][key] for c in cs], style, label=name)
    for ax, label in zip(axes, ['H50 observation RMSE', 'H50 own-Train-scale latent RMSE']):
        ax.set_xticks(xx, cs); ax.legend(); labels(ax, 'Diagnostic correction interval; no endpoint reset', label)
    save('v1_v2_v3_periodic_correction')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, label in [(axes[0], 'mahalanobis_rms', 'Own-Train regularized Mahalanobis RMS'),
                            (axes[1], 'pca_residual_l2', 'Own-Train PCA residual L2 (coordinate-dependent)')]:
        for name, data, style in datasets:
            for state, suffix in [('predicted', style), ('reference', '-')]:
                ax.plot(x, [data['autonomous'][str(h)]['distribution'][state][key]['mean'] for h in hs], suffix,
                        alpha=1 if state == 'predicted' else .5, label=f'{name} {state}')
        ax.legend(fontsize=8); labels(ax, 'Horizon (s)', label)
    save('v1_v2_v3_off_manifold')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4)); names = ['v1', 'v2', 'v3']
    axes[0].bar(names, [data['encoder_reconstruction']['normalized_observation_rmse'] for _, data, _ in datasets])
    labels(axes[0], '', 'Current reconstruction normalized RMSE')
    dims = list(r['audit']['encoder_reconstruction']['physical']['per_dimension'])
    for i, (name, data, _) in enumerate(datasets):
        axes[1].bar(np.arange(7)+(i-1)*.25, [data['encoder_reconstruction']['physical']['per_dimension'][d]['rmse'] for d in dims], .25, label=name)
    axes[1].set_xticks(np.arange(7), dims, rotation=25); axes[1].legend(); labels(axes[1], 'Physical dimensions (mixed units)', 'Current reconstruction physical RMSE'); save('v1_v2_v3_reconstruction')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, label in zip(axes, ['horizontal_velocity_rmse', 'vertical_velocity_rmse', None], ['Horizontal velocity RMSE (m/s)', 'vz RMSE (m/s)', 'yaw error RMSE (rad)']):
        if key:
            curves(ax, 'observation', key, label)
        else:
            for name, data, style in datasets:
                ax.plot(x, [data['autonomous'][str(h)]['observation']['physical']['yaw_error']['rmse'] for h in hs], style, label=name)
            ax.legend(); labels(ax, 'Horizon (s)', label)
    save('v3_velocity_rmse')
    fig = decoder_sensitivity_figure(datasets, x)
    save('v3_decoder_sensitivity')
    for field, filename, xlabel in [('distance_regions', 'v3_error_vs_distance', 'True start distance'),
        ('pi_magnitude_regions', 'v3_error_vs_pi', 'True start PI magnitude (Train bins)'),
        ('action_magnitude_regions', 'v3_error_vs_action', 'Start executed4D action norm (Train bins)')]:
        bins = list(r['audit']['autonomous']['50'][field]); xx = np.arange(len(bins)); fig, ax = plt.subplots(figsize=(8, 4))
        for i, (name, data, _) in enumerate(datasets):
            ax.bar(xx+(i-1)*.25, [data['autonomous']['50'][field][n]['observation'].get('normalized_observation_rmse', np.nan) for n in bins], .25, label=name)
        ax.set_xticks(xx, bins); ax.legend(); labels(ax, xlabel, 'H50 normalized observation RMSE'); save(filename)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)); variances = [('v1', r['baselines']['v1']['latent_variance']), ('v2', r['baselines']['v2']['latent_variance']), ('v3', r['latent_variance'])]
    for i, (name, data) in enumerate(variances):
        for split, style in [('train', '-'), ('test', '--')]:
            axes[0].plot(np.sort(data[split]['per_dimension_std']), style, label=f'{name} {split}')
        axes[1].bar(np.arange(2)+(i-1)*.25, [data[s]['effective_rank'] for s in ('train', 'test')], .25, label=name)
    axes[0].legend(fontsize=8); labels(axes[0], 'Sorted dimensions; no semantic interpretation', 'Native latent std')
    axes[1].set_xticks(np.arange(2), ['Train', 'Test']); axes[1].legend(); labels(axes[1], '', 'Covariance-entropy effective rank'); save('v3_latent_variance')
    selected = r['selected_window']; ep = test[selected['episode_id']]; s = plot['selected']; stats = r['normalization']
    mean = np.asarray(stats['obs']['mean']); std = np.asarray(stats['obs']['std']); time = np.arange(len(s['truth']))/25
    predictions = {'Autonomous v3': s['observations']['never']}
    for name, loader, path in [('Joint v1', load_joint, context['baseline']['model']['path']), ('Local v2', load_consistency, context['v2']['model']['path'])]:
        model, _, _ = loader(path); model.eval().requires_grad_(False)
        with torch.inference_mode():
            out = predict_window(model, ep, selected['start'], selected['horizon'], stats)
        predictions[name] = out['normalized_observations'][0].numpy()*std+mean
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, dim, label in [(axes[0], 3, 'vx (m/s)'), (axes[1], 4, 'vy (m/s)')]:
        ax.plot(time, s['truth'][:, dim], 'k--', label='Recorded')
        for name, p in predictions.items(): ax.plot(time, p[:, dim], label=name)
        ax.legend(); labels(ax, 'Horizon (s)', label)
    for name, p in predictions.items(): axes[2].plot(time, np.sqrt(np.mean(((p-s['truth'])/std)**2, axis=1)), label=name)
    axes[2].legend(); labels(axes[2], 'Horizon (s)', 'Selected-window normalized observation RMSE')
    fig.suptitle('Original fixed Test start; autonomous T-only recorded-action rollout'); save('selected_v3_rollout')
    return paths
