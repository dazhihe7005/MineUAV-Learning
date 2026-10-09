"""Read-only verification/publication after separately guarded frozen evaluations."""
import argparse
import json
from pathlib import Path
import numpy as np
from latent_dynamics_data import file_hash,json_hash
from uav_bc_safety import atomic_json
from uav_bc_robustness import (BASE,MODEL,MODEL_SHA,PARTS,MANIFEST,REPORT,conditions,
    validate_record,validate_cohort,controller_summary,paired_summary,manifest_identity)


def representative_pairs(rows):
    mapping={(r['key'],r['controller']):r for r in rows}
    successes=[r for r in rows if r['controller']=='bc' and r['success'] and r['condition']=='combined_large']
    if not successes:successes=[r for r in rows if r['controller']=='bc' and r['success'] and r['condition']!='nominal']
    gaps=[r for r in rows if r['controller']=='bc' and not r['success'] and mapping[(r['key'],'scripted')]['success']]
    return {name:(r,mapping[(r['key'],'scripted')]) if r else None
        for name,r in [('success',successes[0] if successes else None),('bc_gap',gaps[0] if gaps else None)]}


def aggregate(rows,states):
    state_map={s['key']:s for s in states};out={}
    names=list(dict.fromkeys(s['condition'] for s in states))
    for name in names:
        out[name]={}
        for split in ['benchmark','holdout','pooled']:
            selected=[r for r in rows if r['condition']==name and (split=='pooled' or r['split']==split)]
            if not selected:continue
            by={c:[r for r in selected if r['controller']==c] for c in ['scripted','bc']}
            result={c:controller_summary(values) for c,values in by.items()}
            result['paired']=paired_summary(by['scripted'],by['bc'])
            groups={}
            for component in ['position_m','velocity_m_s','yaw_sign']:
                values={}
                for r in by['bc']:
                    p=state_map[r['key']]['perturbation']
                    if component=='yaw_sign':label='positive' if p['yaw_rad']>0 else 'negative' if p['yaw_rad']<0 else 'zero'
                    else:
                        v=np.asarray(p[component]);i=int(np.argmax(abs(v)))
                        label=('xyz'[i]+('+' if v[i]>0 else '-')) if np.linalg.norm(v)>0 else 'zero'
                    values.setdefault(label,[]).append(r['key'])
                a={r['key']:r for r in by['scripted']};b={r['key']:r for r in by['bc']}
                groups[component]={label:dict(episodes=len(keys),bc_successes=sum(b[k]['success'] for k in keys),
                    scripted_successes=sum(a[k]['success'] for k in keys),bc_mean_final_distance_m=float(np.mean([b[k]['final_distance_m'] for k in keys if b[k]['final_distance_m'] is not None])) if any(b[k]['final_distance_m'] is not None for k in keys) else None,
                    bc_mean_max_speed_m_s=float(np.mean([b[k]['maximum_speed_m_s'] for k in keys]))) for label,keys in values.items()}
            result['direction_groups']=groups;out[name][split]=result
    return out


def assert_nominal_trace(new,old):
    if file_hash(new['trace_path'])!=new['trace_sha256'] or file_hash(old['path'])!=old['sha256']:
        raise ValueError('nominal comparison trace corrupted')
    with np.load(new['trace_path'],allow_pickle=False) as a,np.load(old['path'],allow_pickle=False) as b:
        np.testing.assert_array_equal(a['observations'],b['observations'])
        np.testing.assert_array_equal(a['actions'],b['actions'])


def verify(root):
    from uav_bc_evaluation import trace_metrics
    from uav_bc_policy import load,parameter_hash
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS
    manifest=json.loads((reports/MANIFEST).read_text());payload={k:v for k,v in manifest.items() if k!='sha256'}
    if json_hash(payload)!=manifest['sha256'] or manifest_identity(root)!=manifest['identity']:raise ValueError('immutable manifest/source/runtime mismatch')
    result=json.loads((parts/'evaluation.json').read_text());rows=result['records'];states=manifest['states']
    validate_cohort(rows,[s['key']+'/'+c for s in states for c in ('scripted','bc')])
    state_map={s['key']:s for s in states};pairs={};nominal=0
    baseline=json.loads((reports/'uav_behavior_cloning_seed0_parts/evaluation.json').read_text())['records']
    for r in rows:
        s=state_map[r['key']];validate_record(r,dict(manifest_sha256=manifest['sha256'],checkpoint_sha256=MODEL_SHA,
            policy_parameter_hash=result['frozen_bc']['parameter_hash_before']),s,r['controller'])
        pairs.setdefault(r['key'],[]).append(r['initial_snapshot_sha256'])
        with np.load(r['trace_path'],allow_pickle=False) as a:
            valid=a['physical_state_valid']
            recomputed=trace_metrics(a)
            for k,v in recomputed.items():
                if valid.all() or k not in ('distance_reduction_m','mean_distance_m','minimum_distance_m','mean_speed_m_s','crossings'):
                    np.testing.assert_allclose(r[k],v,rtol=0,atol=0)
            np.testing.assert_array_equal(a['positions'][0],s['initial_state']['qpos'][:3])
            np.testing.assert_allclose(np.asarray(r['target'])-a['positions'][valid],a['observations'][valid,:3],atol=1e-6,rtol=0)
            if r['success']:
                if a['success_streak'][-1]!=5:raise ValueError('invalid success hold accounting')
                if not np.all(np.linalg.norm(a['observations'][-5:,:3],axis=1)<.1) or not np.all(np.linalg.norm(a['observations'][-5:,3:6],axis=1)<.15):
                    raise ValueError('success violates unchanged distance/speed hold')
            speed=np.linalg.norm(a['observations'][:,3:6],axis=1);yaw=np.abs(a['observations'][:,6])
            if r['maximum_speed_m_s']!=float(speed[valid].max()) or r['final_abs_yaw_error_rad']!=(float(yaw[-1]) if valid[-1] else None):raise ValueError('recovery metric mismatch')
        if r['condition']=='nominal':
            old=next(x for x in baseline[r['controller']][r['split']] if x['target_id']==r['target_id'])
            if old['env_seed']!=r['env_seed'] or old['target']!=r['target']:raise ValueError('nominal targets/seeds changed')
            assert_nominal_trace(r,old);nominal+=1
    if any(len(v)!=2 or v[0]!=v[1] for v in pairs.values()):raise ValueError('unequal paired initial snapshots')
    if nominal!=400:raise ValueError('missing nominal comparison')
    policy,_=load(root/'mujoco/rl/models'/MODEL)
    if file_hash(root/'mujoco/rl/models'/MODEL)!=MODEL_SHA or parameter_hash(policy.actor)!=result['frozen_bc']['parameter_hash_before']:
        raise ValueError('frozen BC mutated')
    summary=dict(passed=True,verified_controller_episodes=len(rows),paired_snapshots=len(pairs),nominal_bitwise_reproduced_episodes=nominal,
        checkpoint_unchanged=True,parameter_hash_unchanged=True,all_cache_hashes_valid=True,no_duplicate_or_missing_tasks=True)
    atomic_json(parts/'verification.json',summary);return summary


def publish(root):
    from uav_bc_robustness_plots import plots
    from uav_bc_policy import load
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS
    verification=verify(root);manifest=json.loads((reports/MANIFEST).read_text());evaluation=json.loads((parts/'evaluation.json').read_text())
    rows=evaluation['records'];summary=aggregate(rows,manifest['states']);representatives=representative_pairs(rows)
    figures=plots(reports,rows,summary,representatives)
    resources={p.stem.removeprefix('resource_'):json.loads(p.read_text()) for p in sorted(parts.glob('resource_*.json')) if p.stem!='resource_publish'}
    archived=reports/'uav_bc_robustness_pre_review_parts'
    resources.update({'pre_review_'+p.stem.removeprefix('resource_'):json.loads(p.read_text()) for p in sorted(archived.glob('resource_*.json'))})
    for required in ['smoke','evaluate','resume','tests']:
        r=resources[required]
        if r['exit_code']!=0 or r['failure'] or r['oom_count_delta']:raise ValueError('incomplete/unsafe phase')
    policy,_=load(root/'mujoco/rl/models'/MODEL)
    compact=[{k:v for k,v in r.items() if k not in ('identity','trace_path')} for r in rows]
    raw=[dict(key=r['task_key'],sha256=r['trace_sha256']) for r in rows]
    gaps=[r for r in rows if r['controller']=='bc' and not r['success']]
    scientific=dict(nominal_reproduced=True,bc_successes=sum(r['success'] for r in rows if r['controller']=='bc'),
        scripted_successes=sum(r['success'] for r in rows if r['controller']=='scripted'),episodes_per_controller=1800,
        bc_specific_gap_conditions=[c for c in conditions() if summary[c]['pooled']['paired']['discordance']['scripted_only_success']],
        next_step='A single separate yaw-coverage-supervised BC data-control experiment, with frozen architecture and strict target isolation; not executed.' if gaps
            else 'A single separate fixed-policy external-disturbance recovery evaluation; not executed.')
    output=dict(experiment='BC INITIAL-STATE PERTURBATION ROBUSTNESS',date='2026-10-09',seed=0,
        git=dict(branch='feat/uav-bc-initial-state-robustness',base_branch='feat/uav-behavior-cloning-baseline',base_commit=BASE),
        architecture='7 -> 128 ReLU -> 128 ReLU -> 4 linear, 18052 parameters',training=False,
        frozen_bc=evaluation['frozen_bc'],expert='Existing scripted_action uses position error and yaw error only; same 7D observation, no privileged state.',
        environment=dict(name='unchanged MineUAVPIEnv RewardV2',physics_hz=500,control_hz=100,policy_hz=25,timeout_s=15.,
            success='distance <0.10m AND speed <0.15m/s for 5 consecutive policy steps; yaw is NOT a success criterion',
            failures=['nonfinite_state','z_below_zero','outside_flight_area','excessive_tilt'],
            physical_model_limitations='estimated inertia/COM, coarse central collision box, instantaneous rotor response'),
        perturbation=dict(conditions=conditions(),seed=manifest['identity']['perturbation_seed'],sampling=manifest['sampling'],
            feasibility=manifest['engineering_feasibility'],actual_initial_minimum_clearance_m=min(s['initial_state']['ground_clearance_m'] for s in manifest['states']),
            min_initial_height_m=min(s['initial_state']['qpos'][2] for s in manifest['states']),
            action_observation_normalization='unchanged checkpoint Train-only statistics',initial_yaw_train_std_rad=policy.stats['obs']['std'][6]),
        sample_counts=dict(valid_initial_states=1800,invalid_initial_states=0,controller_episodes=len(rows),conditions=9,benchmark_per_condition=100,holdout_per_condition=100),
        metrics=summary,episodes=compact,verification=verification,
        post_review_fix_reproduction=json.loads((parts/'post_fix_reproduction.json').read_text()),
        resume=dict(completed_tasks=3600,unique_task_keys=3600,task_record_sha256=json_hash([dict(key=r['task_key'],sha256=r['record_sha256']) for r in rows]),
            protocol='atomic fsync/replace NPZ then hash-bound JSON; completed tasks validate record/source/runtime/manifest/snapshot/trace before reuse; incomplete orphan NPZ can be regenerated'),
        resource_safety=dict(workers=1,oom_count_delta=sum(r['oom_count_delta'] for r in resources.values()),
            peak_sampled_process_tree_rss_bytes=max(r['process_tree_peak_sampled_rss_bytes'] for r in resources.values()),
            peak_process_tree_hwm_bytes=max(r['process_tree_peak_hwm_bytes'] for r in resources.values()),
            minimum_available_bytes=min(r['minimum_available_bytes'] for r in resources.values()),swap_modified=False,
            rss_abort_gib=4,system_reserve_gib=1.5,early_abort_available_gib=2,poll_seconds=.1,
            note='sampled process-tree RSS / summed per-process HWM; polling protection is not a hard OS guarantee',phases=resources),
        tests=json.loads((parts/'test_evidence.json').read_text()),
        manifest=dict(path='mujoco/reports/'+MANIFEST,file_sha256=file_hash(reports/MANIFEST),semantic_sha256=manifest['sha256']),
        local_only=dict(directory='mujoco/reports/'+PARTS,trace_count=len(rows),trace_hash_manifest_sha256=json_hash(raw),
            includes='raw per-episode NPZ, atomic resume records, runtime logs, review/plan scratch; no model copies'),
        figures={name:file_hash(reports/name) for name in figures},scientific_summary=scientific,
        limitations=['one frozen training seed','fixed 200 targets and one direction per target/component, no repeated-direction sampling',
            'initial-state offsets only, no persistent disturbance or dynamics randomization','zero PI memory and task-yaw target at reset retained',
            'task success does not require yaw stabilization','no causal attribution from distribution shift or direction groups',
            'zero observed failures do not imply zero population failure risk','no training or recovery policy improvement'],
        next_step=scientific['next_step'])
    atomic_json(reports/REPORT,output);return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2]
    print(json.dumps(verify(root) if args.verify_only else publish(root)['scientific_summary'],indent=2))
