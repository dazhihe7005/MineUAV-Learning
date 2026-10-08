"""Read-only scientific postprocessing; never rerun or modify MPC control."""
import argparse
import json
from pathlib import Path
import numpy as np
from decision_fidelity_metrics import summary
from decision_fidelity_verification import verify_report
from decision_fidelity_plots import render_figures
from run_world_model_decision_fidelity import REPORT,CONDITIONS,write_json
from latent_dynamics_data import file_hash


def supplement(report):
    rows=report['states']; out=dict(paired_support={},additional_reference_diagnostics={})
    for key in ('spearman','kendall','regret','normalized_regret','model_selected_true_rank_percentile'):
        pairs=[(r['conditions']['unconstrained'][key],r['conditions']['train_supported'][key]) for r in rows]
        delta=[None if a is None or b is None else b-a for a,b in pairs]
        out['paired_support'][key]=dict(supported_minus_unconstrained=summary(delta),
            positive_count=sum(v is not None and v>0 for v in delta),negative_count=sum(v is not None and v<0 for v in delta))
    for c in CONDITIONS:
        rows_c=[r['conditions'][c] for r in rows]; bias=[]; near_gap=[]
        for r,m in zip(rows,rows_c):
            with np.load(r['local_arrays']['path'],allow_pickle=False) as data:
                bias.append(float(np.mean(data[c+'_pred_cost']-data[c+'_true_cost'])))
            near_gap.append((r['scripted']['cost']-m['oracle']['true_cost'])/(m['true_cost_spread']+1e-12))
        out['additional_reference_diagnostics'][c]=dict(model_selected_in_worst_half_count=sum(m['model_selected_true_rank_percentile']>.5 for m in rows_c),
            scripted_beats_model_selected_count=sum(r['scripted']['cost']<m['model_selected']['true_cost'] for r,m in zip(rows,rows_c)),
            true_oracle_beats_scripted_count=sum(r['scripted']['cost']>m['oracle']['true_cost'] for r,m in zip(rows,rows_c)),
            scripted_minus_true_best_cost=summary([r['scripted']['cost']-m['oracle']['true_cost'] for r,m in zip(rows,rows_c)]),
            scripted_minus_true_best_range_fraction=summary(near_gap),predicted_minus_true_cost_bias=summary(bias))
    out['conclusion']=dict(
        q1='Partial first-decision ranking utility, not random: rho~0.60 and lower expected regret in93%+states; best-tail fidelity poor (Top1=7/200), scripted quality systematically under-ranked.',
        q2='True-cost oracle candidates improve distance in193/200 states, zero failures. Gains are small at0.4s (~1.1cm). This disproves blanket useless-candidate/cost claims, NOT proof of oracle closed-loop success.',
        q3='Scripted true sequence beats99.5%+candidates on average, better than model selected in189/200, but is only near predicted top12%. Scripted lies outside the attained best cost in78/200; candidate coverage is incomplete, not uniformly poor.',
        most_supported='Incomplete decision-cost/best-tail ranking fidelity plus a candidate-coverage limitation; central95% support does not repair ranking. First-decision evidence alone cannot uniquely explain later closed-loop failure or prove cost/horizon/replanning misalignment.',
        next='Reuse saved failed MPC trajectories for one offline exact-state later-decision ranking audit with unchanged candidates/cost, to test whether ranking deteriorates under receding-horizon state-distribution shift. Do not execute now.')
    return out


def finalize(root):
    directory=Path(root)/'mujoco/reports'; report=json.loads((directory/REPORT).read_text())
    # Rebuild postprocessing from immutable execution data; standalone verifier
    # checks the final postprocessing source/figure hashes and derived statistics.
    report['independent_verification']=verify_report(root,check_postprocessing=False)
    report['analysis']=supplement(report)
    report['figures']=render_figures(report,directory)
    report['postprocessing_sources']={str(Path(__file__).with_name(n)):file_hash(Path(__file__).with_name(n))
        for n in ('decision_fidelity_verification.py','decision_fidelity_plots.py',Path(__file__).name)}
    write_json(directory/REPORT,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args(); report=finalize(args.root)
    print(json.dumps(dict(verification=report['independent_verification'],figures=len(report['figures']),analysis=report['analysis']),indent=2))
