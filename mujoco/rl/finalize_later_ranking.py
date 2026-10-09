"""Deterministic analysis refresh, without touching expensive rollout identity."""
import argparse
import json
from pathlib import Path
from latent_dynamics_data import file_hash
from run_latent_random_shooting_mpc import write_json
from run_later_decision_ranking import REPORT
from later_ranking_plots import render

def interpretation(report):
    evidence={}
    for c in ('unconstrained','train_supported'):
        a=report['results'][c]['all']; paired=a['paired']['individual']
        evidence[c]=dict(paired_episodes=a['paired']['episodes'],
            spearman_decreased_episodes=sum(p['spearman_delta'] is not None and p['spearman_delta']<0 for p in paired),
            regret_increased_episodes=sum(p['regret_delta']>0 for p in paired),
            S3_minus_S0=a['paired']['S3_minus_S0'],failed_only=a['failed_only_paired']['S3_minus_S0'])
    return dict(paired_evidence=evidence,
        main_finding='Later actual MPC states show materially worse decision fidelity, not merely the same mediocre first-decision ranking. Whole-set rho becomes negative and selected regret exceeds analytical random expectation on average at S2/S3.',
        best_tail='True best26 discrimination was already weak at S0 and becomes anti-correlated later. Median true best-second margins and tail spreads grow, so worsening is not explained solely by increasingly indistinguishable good candidates.',
        state_shift='Actual-state observation standardized distance and encoder-latent Mahalanobis RMS rise; within-episode degradation and descriptive OOD correlations support closed-loop distribution shift as an additional failure mechanism, not unique causal proof.',
        recovery='Oracle H10 positive-distance-progress falls sharply, and scripted H10 mean progress turns negative at later exact states. This also implicates short-horizon recovery limitations; negative immediate progress does not prove longer-horizon scripted recovery impossible.',
        support='Marginal central95% action support does not restore later ranking; S3 supported regret/rank are worse despite lower observation prediction error. Stop this marginal-support explanation as a sufficient remedy, not as a proof that all joint-support/OOD issues are absent.',
        limitations='Condition trajectories and relative-stage physical states differ. Pooled800-state correlations include repeated episodes and progress effects; they are descriptive, not independent-sample significance claims.',
        next_only='One controlled on-policy model-data robustness experiment using new training targets/seeds disjoint from this audit cohort, with architecture/planner/cost/H/N held fixed; do not execute now.')

def finalize(root,collect=False,workers=8):
    directory=Path(root).resolve()/'mujoco/reports'; p=directory/REPORT
    if collect:
        from later_ranking_reporting import collect_and_report
        collect_and_report(root,workers)
    report=json.loads(p.read_text())
    report['analysis']=interpretation(report)
    report['model']['checkpoint_sha256_after']=file_hash(report['model']['checkpoint'])
    report['figures']=render(report,directory)
    report['postprocessing_sources']={str(Path(__file__).with_name(name)):file_hash(Path(__file__).with_name(name))
        for name in ('later_ranking_plots.py','later_ranking_verification.py','later_ranking_reporting.py',Path(__file__).name)}
    write_json(p,report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]); p.add_argument('--collect',action='store_true'); p.add_argument('--workers',type=int,default=8)
    a=p.parse_args(); finalize(a.root,a.collect,a.workers)
