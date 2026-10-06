"""Compact comparison figures; selection is complete before Test is plotted."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from latent_dynamics_multistep import rollout_windows


def render_figures(report, evaluation, trained, directory):
    paths = []; horizons = report['horizons_steps']; seconds = np.asarray(horizons)/25
    colors = {'history':'#EE7733', 'multistep_history':'#4477AA'}
    labels = {'history':'One-step trained', 'multistep_history':'Multi-step trained (K=10)'}

    def save(fig, filename):
        fig.tight_layout(); path = directory/filename; fig.savefig(path, dpi=160)
        plt.close(fig); paths.append(str(path))

    rows = report['training']['history']; fig, ax = plt.subplots(figsize=(8, 4.5))
    for key, label in [('train_loss','Train online average'), ('validation_loss','Validation')]:
        ax.plot([r['epoch'] for r in rows], [r[key] for r in rows], label=label)
    ax.axvline(report['training']['best_epoch'], color='k', ls=':', label='Validation best')
    ax.set(xlabel='Epoch', ylabel='Uniform K-step observation-normalized MSE', yscale='log')
    ax.legend(); ax.grid(alpha=.2); save(fig, 'multistep_training_loss_curve.png')

    fig, ax = plt.subplots(figsize=(8, 4.5))
    reference = {'markov':('Markov','#888888'), 'no_memory':('No-Memory','#CCBB44'),
                 'pi_state':('PI-State (TF PI)','#228833')}
    for kind, (label,color) in reference.items():
        ax.plot(seconds, [report['models'][kind]['horizons'][str(h)]['normalized_observation_rmse'] for h in horizons],
                'o--', color=color, alpha=.7, label=label)
    for kind in colors:
        ax.plot(seconds, [report['models'][kind]['horizons'][str(h)]['normalized_observation_rmse'] for h in horizons],
                'o-', color=colors[kind], label=labels[kind])
    ax.axvline(.4, color='k', ls=':', label='Training horizon')
    ax.set(xlabel='Horizon (s)', ylabel='Endpoint observation-normalized RMSE'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    save(fig, 'one_step_vs_multistep_rmse_vs_horizon.png')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, field, title in zip(axes, ['horizontal_velocity_rmse','vertical_velocity_rmse'], ['vx/vy','vz']):
        for kind in colors:
            axis.plot(seconds, [report['models'][kind]['horizons'][str(h)][field] for h in horizons],
                      'o-', color=colors[kind], label=labels[kind])
        axis.axvline(.4,color='k',ls=':'); axis.set(xlabel='Horizon (s)', ylabel='RMSE (m/s)',title=title); axis.grid(alpha=.2)
    axes[0].legend(fontsize=8); save(fig, 'one_step_vs_multistep_velocity_rmse.png')

    for field, filename in [('distance_regions','one_step_vs_multistep_error_vs_distance.png'),
                            ('pi_magnitude_regions','one_step_vs_multistep_error_vs_pi_magnitude.png')]:
        names = list(report['models']['history']['horizons'][str(horizons[0])][field])
        fig, axes = plt.subplots(1, len(names), figsize=(4*len(names), 4), squeeze=False)
        for axis, name in zip(axes[0], names):
            for kind in colors:
                axis.plot(seconds, [report['models'][kind]['horizons'][str(h)][field][name].get('normalized_observation_rmse', np.nan)
                                    for h in horizons], 'o-', color=colors[kind], label=labels[kind])
            axis.axvline(.4,color='k',ls=':'); axis.set(xlabel='Horizon (s)',ylabel='Normalized RMSE',title=name); axis.grid(alpha=.2)
        axes[0,0].legend(fontsize=7); save(fig, filename)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, field, title in zip(axes, ['mean_l2','mean_cosine','autoregressive_mean_norm'],
                                 ['Latent L2 (model-specific coordinates)','Latent cosine similarity','Autoregressive latent norm']):
        for kind in colors:
            axis.plot(seconds, [report['models'][kind]['latent_drift'][str(h)][field] for h in horizons],
                      'o-', color=colors[kind], label=labels[kind])
        axis.set(xlabel='Horizon (s)',ylabel=title); axis.grid(alpha=.2)
    axes[0].legend(fontsize=7); save(fig, 'one_step_vs_multistep_latent_drift.png')

    selected = report['selected_window']; ep = evaluation['episodes'][selected['episode_id']]
    start, horizon = selected['start'], selected['horizon']; clock = np.arange(horizon+1)/25
    truth = np.vstack([ep['obs'][start], ep['next_obs'][start:start+horizon]])
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for axis, dim, label in zip(axes.flat, range(6), ['error_x (m)','error_y (m)','error_z (m)','vx (m/s)','vy (m/s)','vz (m/s)']):
        axis.plot(clock, truth[:,dim], 'k--',label='Recorded truth'); axis.set(xlabel='Horizon (s)',ylabel=label); axis.grid(alpha=.2)
    for kind, net in [('history', evaluation['models']['history']), ('multistep_history', trained)]:
        out = rollout_windows(net, 'history', ep, report['normalization'], [start], horizon)
        trace = np.vstack([ep['obs'][start],out['predictions'][0]])
        for axis, dim in zip(axes.flat, range(6)):
            axis.plot(clock, trace[:,dim], color=colors[kind],label=labels[kind])
    axes[0,0].legend(fontsize=8); save(fig, 'selected_rollout_one_step_vs_multistep.png')
    return paths
