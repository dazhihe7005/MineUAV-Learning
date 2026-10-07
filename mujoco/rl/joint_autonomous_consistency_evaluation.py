"""Same v1/v2 frozen protocols, with v3's own Train latent coordinate statistics."""
import numpy as np

from joint_latent_world_model import component_hashes, latent_summary
from joint_latent_evaluation import encoded_sequences
from joint_composition_core import freeze_joint, fit_manifold, distribution_scores, describe
from joint_composition_evaluation import audit
from run_joint_latent_composition_audit import leakage_checks
from joint_consistency_evaluation import local_relative
from joint_autonomous_consistency_training import load_autonomous


def baseline_records(context):
    from run_joint_consistency_v2 import BASELINE_FIELDS
    return dict(v1=dict(audit={k: context['baseline_audit'][k] for k in BASELINE_FIELDS},
                       relative=context['v2']['relative_local_residual']['v1'],
                       latent_variance=context['baseline']['joint']['latent_variance']),
                v2=dict(audit={k: context['v2']['audit'][k] for k in BASELINE_FIELDS},
                       relative=context['v2']['relative_local_residual']['v2'],
                       latent_variance=context['v2']['latent_variance']))


def evaluate(model_path, context, test):
    model, stats, _ = load_autonomous(model_path); freeze_joint(model)
    before = component_hashes(model)
    sequences = encoded_sequences(model, context['splits']['train'], stats)
    train = np.concatenate(sequences)
    manifold = fit_manifold(sequences, [e['action'] for e in context['splits']['train']])
    scores = distribution_scores(train, manifold)
    manifold['train_score_summary'] = {k: describe(v) for k, v in scores.items()}
    manifold['train_score_quantiles'] = {k: np.quantile(v, [1/3, 2/3]).tolist() for k, v in scores.items()}
    manifold['provenance'].update(dataset_sha256=context['dataset']['source_dataset_sha256']['train'], encoder_parameter_sha256=before['encoder'])
    b = context['baseline']
    metrics, plot = audit(model, context['splits']['train'], test, stats,
                          b['pi_magnitude_bins']['thresholds_m_s2'], b['horizons_steps'], b['selected_window'], manifold)
    checks = leakage_checks(dict(source=b, model=model, stats=stats, test=test))
    if component_hashes(model) != before:
        raise AssertionError('evaluation changed v3 parameters')
    return dict(audit=metrics, manifold=manifold, relative=local_relative(model, test, stats),
        latent_variance=dict(train=latent_summary(train), test=latent_summary(np.concatenate(encoded_sequences(model, test, stats)))),
        verification=dict(checks, no_parameter_mutation=True, reference_branch_absent_at_inference=True)), plot


def comparison(baselines, new, relative, horizons):
    audits = {k: v['audit'] for k, v in baselines.items()}; audits['v3'] = new
    rows = {}
    for h in horizons:
        values = {name: data['autonomous'][str(h)]['observation']['normalized_observation_rmse'] for name, data in audits.items()}
        rows[str(h)] = dict(values, v3_reduction_vs_v1_percent=100*(values['v1']-values['v3'])/values['v1'],
                           v3_reduction_vs_v2_percent=100*(values['v2']-values['v3'])/values['v2'],
                           regime='in_training_horizon' if h <= 10 else 'out_of_training_horizon')
    corrections = {}
    for name, data in audits.items():
        a = data['periodic_correction']['never']['observation']['normalized_observation_rmse']
        b = data['periodic_correction']['1']['observation']['normalized_observation_rmse']
        corrections[name] = dict(never=a, C1=b, absolute_gap=a-b, never_to_C1_ratio=a/b)
    return dict(observation=rows, correction_sensitivity=corrections,
                relative_local_residual=dict({k: v['relative'] for k, v in baselines.items()}, v3=relative),
                latent_comparison_limit='Each model uses its own post-training Train coordinates; raw latent errors, norms and decoder ratios are not cross-model physical invariants.')
