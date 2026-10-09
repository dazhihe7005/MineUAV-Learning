"""Report-only categorical aggregation, preserving frozen execution sources."""
from collections import Counter
import copy
from later_ranking_analysis import aggregate,CONDITIONS,SPLITS
from later_ranking_core import STAGES

def aggregate_for_report(records):
    # Numeric measurements retain the original summary implementation. Reason
    # maps are categorical counts, NOT numeric features with identical schemas.
    numeric=copy.deepcopy(records)
    for r in numeric: r['metrics'].pop('termination_reason_counts',None)
    result=aggregate(numeric)
    for c in CONDITIONS:
        for split in (*SPLITS,'all'):
            for stage in STAGES:
                counts=Counter()
                for r in records:
                    if r['condition']==c and r['stage']==stage and (split=='all' or r['split']==split):
                        counts.update(r['metrics'].get('termination_reason_counts',{}))
                result[c][split]['stages'][stage]['candidate_termination_reason_totals']=dict(counts)
    return result

def collect_and_report(root,workers):
    """Swap only report assembly; simulation source identity stays unchanged.

    The runner's aggregate alias is used solely AFTER all episode computations.
    Spawned workers import its unchanged source; no numerical rollout function,
    candidate, archived observation or cache identity is replaced/relabelled.
    """
    import run_later_decision_ranking as runner
    original=runner.aggregate
    try:
        runner.aggregate=aggregate_for_report
        return runner.run(root,workers)
    finally:
        runner.aggregate=original
