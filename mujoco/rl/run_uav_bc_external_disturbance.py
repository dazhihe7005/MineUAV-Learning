"""Verified publication only; no model fitting and no controller modifications."""
import argparse,json
from pathlib import Path
from latent_dynamics_data import file_hash
from uav_bc_safety import atomic_json
from uav_bc_external_disturbance import REPORT,MANIFEST,PARTS,MODELS,BASE,conditions
from uav_bc_external_evaluation import verify,CONTROLLERS,evaluation_provenance

def resource_summary(resources):
    required={'baseline','related','smoke','evaluation','resume','verification','tests'}
    if not required.issubset(resources):raise ValueError('missing completed resource phase evidence')
    if any(r['exit_code'] or r['failure'] or r['oom_count_delta'] or r['workers']!=1 for r in resources.values()):
        raise ValueError('incomplete/unsafe/OOM evaluation, no publication')
    return dict(workers=1,oom_occurred=False,peak_sampled_process_tree_rss_bytes=max(r['process_tree_peak_sampled_rss_bytes'] for r in resources.values()),
        maximum_sum_process_hwm_bytes=max(r['process_tree_peak_hwm_bytes'] for r in resources.values()),
        minimum_available_bytes=min(r['minimum_available_bytes'] for r in resources.values()),
        hwm_note='Sum of separate historical VmHWM is not a simultaneous physical-memory peak; sampled tree RSS is reported separately.',
        phases=resources,swap_modified=False,heavy_phases_serialized=True)

def public_rows(rows):return [{k:v for k,v in r.items() if k not in ('identity','trace_path')} for r in rows]

def publication_provenance(root,ev):
    result=evaluation_provenance(root,ev['identity'])
    result['publication_source_sha256']={n:file_hash(Path(root)/'mujoco/rl'/n) for n in
        ['run_uav_bc_external_disturbance.py','uav_bc_external_plots.py','uav_bc_yaw_data.py',
         'uav_bc_yaw_plots.py','latent_dynamics_data.py']}
    result['note']='Acquisition cache and trace identities are preserved; revised metric verification and publication are separately identified, not newly acquired episodes.'
    return result

def triple_outcomes(rows):
    from collections import Counter
    grouped={}
    for r in rows:
        key=(r['condition'],r['key']);pair=grouped.setdefault(key,{})
        if r['controller'] in pair:raise ValueError('duplicate three-way episode')
        pair[r['controller']]=bool(r['success'])
    result={}
    for (condition,key),pair in grouped.items():
        if set(pair)!=set(CONTROLLERS):raise ValueError('incomplete three-way episode')
        result.setdefault(condition,Counter())['/'.join(str(int(pair[n])) for n in CONTROLLERS)]+=1
    return {c:dict(episodes=sum(counts.values()),all_failed=counts['0/0/0'],all_success=counts['1/1/1'],
        outcome_counts=dict(counts),ordering=list(CONTROLLERS)) for c,counts in result.items()}

def interpret(metrics):
    expert_only={c:metrics[c]['paired']['yaw_augmented_minus_scripted']['discordance']['first_only_success'] for c in conditions()}
    shared=[c for c in conditions() if all(metrics[c][n]['successes']<100 for n in CONTROLLERS)]
    return dict(expert_only_success_count=expert_only,shared_failed_conditions=shared,
        scope='Only fixed simulated lateral COM resultant forces; no wind-speed inference or real mine certification. BC observes7D feedback, not force/wind magnitude.',
        holding_limit='Original task stops on5step success. Constant last1s proxy is not proof of indefinite target holding; gust early termination is censored.',
        next_step=['One read-only fixed-controller PI-integral/braking-response audit after gust removal, with paired Scripted/BC traces and no parameter tuning; not executed.'])

def publish(root):
    from uav_bc_external_plots import plots
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS
    verified=verify(root);m=json.loads((reports/MANIFEST).read_text());ev=json.loads((parts/'evaluation.json').read_text())
    resources={p.stem.removeprefix('resource_'):json.loads(p.read_text()) for p in parts.glob('resource_*.json') if p.stem!='resource_publish'}
    safe=resource_summary(resources);tests=json.loads((parts/'test_evidence.json').read_text())
    if tests['related']['exit_code'] or tests['readonly_fixture_full_suite']['exit_code'] or tests['bare_full_suite']['failures']:
        raise ValueError('algorithmic test failure; do not publish/commit')
    figures=plots(root,ev)
    value=dict(experiment='FROZEN BC EXTERNAL DISTURBANCE ROBUSTNESS',date='2026-10-10',branch='feat/uav-bc-external-disturbance',base_commit=BASE,
        design=dict(conditions=m['conditions'],targets=100,controller_episodes=1800,no_training=True,unchanged_original_task=True,
            architecture='7→128ReLU→128ReLU→4linear;18052parameters',force_seed=m['seed'],physical_preflight=m['preflight'],
            recovery_definition=m['recovery_definition'],steady_definition=m['steady_definition']),
        manifest=dict(path='mujoco/reports/'+MANIFEST,file_sha256=file_hash(reports/MANIFEST),content_sha256=m['sha256']),
        evaluation_provenance=dict(publication_provenance(root,ev),acquisition_cache_sha256=file_hash(parts/'evaluation.json')),
        checkpoint_hashes_before=ev['checkpoint_hashes_before'],checkpoint_hashes_after=ev['checkpoint_hashes_after'],
        parameter_hashes_before=ev['parameter_hashes_before'],parameter_hashes_after=ev['parameter_hashes_after'],all_frozen=ev['all_frozen'],
        metrics=ev['metrics'],three_way_paired_outcomes=triple_outcomes(ev['rows']),interpretation=interpret(ev['metrics']),verification=verified,
        resources=safe,tests=tests,episode_metrics=public_rows(ev['rows']),
        figures={n:file_hash(reports/n) for n in figures},
        limitations=['single fixed100-target cohort','nominal estimated MuJoCo mass/inertia and simplified collision',
            'force stress tests, not aerodynamic wind model','no external disturbance observation','unchanged15s timeout and success termination',
            'finite terminal window, not asymptotic steady state','gust early success may censor exposure','no training or controller tuning'])
    atomic_json(reports/REPORT,value);return value

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[2];verify(root) if a.verify else publish(root)
