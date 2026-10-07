"""Seven necessary static scientific figures; no raw traces or extra audits."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'v1': '#4477AA', 'v3': '#CC6677'}


def render_figures(branches, summary, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    saved = []
    def save(name):
        plt.tight_layout()
        path = output/(name+'.png'); plt.savefig(path, dpi=150); plt.close()
        saved.append(str(path))
    def paired_panel(values, title, ylabel):
        for seed, (a, b) in enumerate(zip(values['v1_values'], values['v3_values'])):
            if a is None or b is None:
                for pos, value in enumerate((a, b)):
                    if value is not None:
                        plt.plot(pos, value, 'o', label=f'seed {seed}: counterpart failed')
                    else:
                        plt.text(pos, .02 + seed*.045, f's{seed} failed', transform=plt.gca().get_xaxis_transform(),
                                 ha='center', fontsize=7)
                continue
            plt.plot([0, 1], [a, b], 'o-', alpha=.75, label=f'seed {seed}')
        plt.xticks([0, 1], ['Joint v1', 'Autonomous v3'])
        plt.ylabel(ylabel); plt.title(title); plt.grid(alpha=.2)
        plt.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1))

    plt.figure(figsize=(6.5, 4))
    paired_panel(summary['horizons']['50'], 'H50 paired joint-training seeds (fixed pretrained init)', 'Normalized observation RMSE')
    save('v1_v3_multiseed_h50_paired')

    horizons = [1, 5, 10, 25, 50]; x = np.asarray(horizons)/25
    plt.figure(figsize=(7, 4))
    for condition in ('v1', 'v3'):
        values = summary['horizons']
        for seed in range(5):
            curve = [values[str(h)][condition+'_values'][seed] for h in horizons]
            plt.plot(x, [np.nan if v is None else v for v in curve], color=COLORS[condition], alpha=.18)
        mean = np.array([values[str(h)][condition]['mean'] for h in horizons], dtype=float)
        std = np.array([values[str(h)][condition]['std'] for h in horizons], dtype=float)
        plt.plot(x, mean, 'o-', color=COLORS[condition], label=f'{condition}: mean ± sample SD')
        plt.fill_between(x, mean-std, mean+std, color=COLORS[condition], alpha=.13)
    plt.axvline(.4, color='gray', linestyle=':', label='K10 training horizon')
    plt.xlabel('Horizon (s @25 Hz)'); plt.ylabel('Normalized observation RMSE'); plt.grid(alpha=.2); plt.legend(fontsize=8)
    save('v1_v3_multiseed_rmse_horizon')

    plt.figure(figsize=(7, 4)); seeds = np.arange(5)
    for h, offset in [(25, -.18), (50, .18)]:
        values = summary['horizons'][str(h)]['improvement_percent']
        plt.bar(seeds+offset, [np.nan if v is None else v for v in values], .36, label=f'H{h}')
    plt.axhline(0, color='black', linewidth=.8); plt.xticks(seeds); plt.xlabel('Paired training seed')
    plt.ylabel('100 × (v1 − v3) / v1 (%)'); plt.title('Positive means v3 better; no seed selection')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1)); plt.grid(axis='y', alpha=.2)
    save('v1_v3_multiseed_paired_improvement')

    for field, title, ylabel, filename in [
        ('h50_horizontal_velocity', 'H50 horizontal velocity', 'Physical RMSE (m/s)', 'v1_v3_multiseed_velocity'),
        ('h50_yaw', 'H50 yaw error', 'Physical RMSE (rad)', 'v1_v3_multiseed_yaw'),
        ('current_reconstruction', 'Current observation reconstruction', 'Normalized observation RMSE', 'v1_v3_multiseed_reconstruction')]:
        plt.figure(figsize=(6.5, 4)); paired_panel(summary[field], title, ylabel); save(filename)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, interval in zip(axes, (1, 10)):
        plt.sca(ax)
        paired_panel(summary['correction_gaps'][f'never_minus_C{interval}'], f'H50: never − C{interval}', 'Normalized observation RMSE gap')
    save('v1_v3_multiseed_correction_gap')
    return saved
