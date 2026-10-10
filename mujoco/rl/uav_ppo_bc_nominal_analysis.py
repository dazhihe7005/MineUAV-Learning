"""Recompute matched metrics from hashed immutable records; no simulation."""
import json
from pathlib import Path
import numpy as np
from uav_ppo_bc_nominal import BASE,BRANCH,PARTS,MANIFEST,REPORT,CONTROLLERS,SOURCE_NAMES
from uav_bc_robustness import controller_summary,paired_summary,validate_record,validate_cohort
from uav_bc_safety import atomic_json,atomic_bytes
from latent_mpc_evaluation import distribution
from latent_dynamics_data import file_hash,json_hash
from uav_ppo_bc_nominal_run import motion_metrics,validate_execution

LABELS={'scripted':'Scripted','original_bc':'Original BC','yaw_bc':'Yaw-Augmented BC','ppo7d':'PPO 7D PI (seed0)'}

def summary(rows):
    if not rows:raise ValueError('nonempty cohort required')
    out=controller_summary(rows)
    out['trajectory_length_m']=distribution([r['trajectory_length_m'] for r in rows])
    out['successful_trajectory_length_m']=distribution([r['trajectory_length_m'] for r in rows if r['success']]) if out['successes'] else None
    out['failed_target_entry_without_success']=sum(not r['success'] and r['reached_target_region'] for r in rows)
    out['failures_ever_joint_threshold']=sum(not r['success'] and r['ever_joint_threshold'] for r in rows)
    out['failed_maximum_success_streak']=distribution([r['maximum_success_streak'] for r in rows if not r['success']]) if out['successes']<len(rows) else None
    return out

def paired(baseline,alternative):
    a={r['key']:r for r in baseline};b={r['key']:r for r in alternative}
    if not a or a.keys()!=b.keys() or len(a)!=len(baseline) or len(b)!=len(alternative):raise ValueError('paired cohort mismatch')
    for k in a:
        for field in ['target','initial_snapshot_sha256']:
            if a[k][field]!=b[k][field]:raise ValueError('not same physical task')
    out=paired_summary(baseline,alternative)
    out['discordance']['alternative_only_success']=out['discordance'].pop('bc_only_success')
    out['discordance']['baseline_only_success']=out['discordance'].pop('scripted_only_success')
    out['sign']='alternative minus baseline; negative distance/time/length/speed favours alternative'
    for field in ['trajectory_length_m','maximum_speed_m_s','final_speed_m_s']:
        vals=[b[k][field]-a[k][field] for k in a if a[k].get(field) is not None and b[k].get(field) is not None]
        out[field+'_delta']=distribution(vals) if vals else None
    both=[k for k in a if a[k]['success'] and b[k]['success']]
    out['successful_trajectory_length_delta_m']=distribution([b[k]['trajectory_length_m']-a[k]['trajectory_length_m'] for k in both]) if both else None
    return out

def load_verified(root):
    root=Path(root);r=root/'mujoco/reports';parts=r/PARTS
    manifest=json.loads((r/MANIFEST).read_text());e=json.loads((parts/'evaluation.json').read_text())
    payload={k:v for k,v in manifest.items() if k!='sha256'}
    if json_hash(payload)!=manifest['sha256'] or e['manifest_sha256']!=manifest['sha256']:raise ValueError('manifest mismatch')
    states={s['key']:s for s in manifest['states']}
    identity=dict(manifest_sha256=manifest['sha256'],model_identity=manifest['identity']['models'])
    rows=e['records'];expected=[s['key']+'/'+n for s in manifest['states'] for n in (CONTROLLERS if s['split']=='fresh' else ['ppo7d'])]
    validate_cohort(rows,expected)
    for row in rows:
        validate_record(row,identity,states[row['key']],row['controller'])
        with np.load(row['trace_path'],allow_pickle=False) as t:
            if any(row.get(k)!=v for k,v in motion_metrics(t).items()):raise ValueError('metric/trace mismatch')
    for repeat in e['deterministic_repeats'].values():validate_execution(repeat,identity)
    for n,sha in manifest['identity']['source_sha256'].items():
        if file_hash(root/'mujoco'/n)!=sha:raise ValueError('immutable source changed')
    for model in manifest['identity']['models'].values():
        if model.get('path') and file_hash(root/model['path'])!=model['checkpoint_sha256']:raise ValueError('checkpoint changed')
    return manifest,e

def figures(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(root)/'mujoco/reports/uav_ppo_bc_nominal_figures';out.mkdir(exist_ok=True)
    cohort={n:[r for r in rows if r['controller']==n and r['split']=='fresh'] for n in CONTROLLERS}
    labels=[LABELS[n] for n in CONTROLLERS];hashes={}
    def save(fig,name):
        fig.tight_layout();path=out/name;atomic_bytes(path,lambda f:fig.savefig(f,format='png',dpi=145));plt.close(fig)
        hashes[str(path.relative_to(root))]=file_hash(path)
    fig,ax=plt.subplots(figsize=(8,4));s=[summary(cohort[n]) for n in CONTROLLERS]
    ax.bar(labels,[x['success_rate']*100 for x in s],color=['#64748b','#16a34a','#059669','#2563eb'])
    for i,x in enumerate(s):ax.text(i,x['success_rate']*100+1,f"{x['successes']}/200",ha='center')
    ax.set_ylim(0,112);ax.set_ylabel('Success (%)');ax.set_title('Matched fresh nominal targets; deterministic policies')
    ax.tick_params(axis='x',labelsize=8);save(fig,'ppo_bc_success.png')
    for field,ylabel,name,successful in [
        ('final_distance_m','Final target distance (m)','ppo_bc_final_distance.png',False),
        ('completion_time_s','Completion time (s); successful episodes only','ppo_bc_completion_time.png',True),
        ('trajectory_length_m','Travelled distance (m); successful episodes only','ppo_bc_trajectory_length.png',True)]:
        fig,ax=plt.subplots(figsize=(8,4))
        values=[[r[field] for r in cohort[n] if r.get(field) is not None and (r['success'] or not successful)] for n in CONTROLLERS]
        ax.boxplot(values,tick_labels=[f'{LABELS[n]}\nn={len(v)}' for n,v in zip(CONTROLLERS,values)],showfliers=True)
        ax.set_ylabel(ylabel);ax.tick_params(axis='x',labelsize=8);save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(labels,[x['near_target_actual_speed_m_s'] for x in s]);axes[0].set_ylabel('Near-target actual speed (m/s)')
    axes[1].bar(labels,[100*x['action_saturation_fraction'] for x in s]);axes[1].set_ylabel('Command elements |a|>=0.95 (%)')
    for ax in axes:ax.tick_params(axis='x',labelrotation=15,labelsize=7)
    save(fig,'ppo_bc_braking_saturation.png')
    # Selection fixed by first manifest index, never strongest visual effect.
    ppo=cohort['ppo7d'];success=next((x for x in ppo if x['success']),None);failed=next((x for x in ppo if not x['success']),None)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for j,chosen in enumerate([success,failed]):
        if chosen is None:
            for ax in axes[:,j]:ax.text(.5,.5,'No such outcome',ha='center');ax.set_axis_off()
            continue
        for name in CONTROLLERS:
            row=next(x for x in cohort[name] if x['key']==chosen['key'])
            with np.load(row['trace_path'],allow_pickle=False) as a:
                t=np.arange(len(a['observations']))*.04;d=np.linalg.norm(a['observations'][:,:3],axis=1);v=np.linalg.norm(a['observations'][:,3:6],axis=1)
                axes[0,j].plot(t,d,label=LABELS[name]);axes[1,j].plot(t,v,label=LABELS[name])
        axes[0,j].axhline(.1,ls='--',c='gray');axes[1,j].axhline(.15,ls='--',c='gray')
        axes[0,j].set_title(('First PPO success: ' if j==0 else 'First PPO failure: ')+chosen['key'])
        axes[0,j].set_ylabel('Distance (m)');axes[1,j].set_ylabel('Speed (m/s)');axes[1,j].set_xlabel('Simulation time (s)')
    axes[0,0].legend(fontsize=7);save(fig,'ppo_bc_paired_trajectories.png')
    return hashes

def analyze(root):
    root=Path(root);manifest,e=load_verified(root);rows=e['records'];r=root/'mujoco/reports';parts=r/PARTS
    fresh={n:[x for x in rows if x['split']=='fresh' and x['controller']==n] for n in CONTROLLERS}
    main={n:summary(v) for n,v in fresh.items()}
    pairs={a+'_vs_'+b:dict(baseline=a,alternative=b,**paired(fresh[a],fresh[b]))
        for a,b in [('scripted','original_bc'),('scripted','yaw_bc'),('original_bc','yaw_bc'),('original_bc','ppo7d'),('yaw_bc','ppo7d')]}
    history={s:summary([x for x in rows if x['split']==s]) for s in ['benchmark','holdout']}
    original=manifest['provenance']['historical_pi7d']['final_evaluation']
    reproduced={s:dict(previous_successes=original[s]['successes'],current_successes=history[s]['successes'],
        success_count_matches=original[s]['successes']==history[s]['successes'],
        previous_mean_final_distance_m=original[s].get('mean_final_distance_m'),
        current_mean_final_distance_m=history[s]['final_distance_m']['mean']) for s in history}
    resources=[json.loads(p.read_text()) for p in sorted(parts.glob('resource_*.json'))]
    if not resources or any(v['exit_code'] or v['failure'] or v['oom_count_delta'] for v in resources):raise ValueError('incomplete/unsafe execution')
    compact=['key','controller','target_id','target','initial_snapshot_sha256','success','termination_reason','steps',
        'completion_time_s','final_distance_m','final_speed_m_s','final_velocity_xyz_m_s','maximum_speed_m_s',
        'trajectory_length_m','near_steps','near_actual_sum','action_saturation_elements','crossings','reached_target_region',
        'ever_joint_threshold','maximum_success_streak','trace_sha256','record_sha256']
    value=dict(experiment='PPO vs BC matched nominal evaluation',base_commit=BASE,branch=BRANCH,
        manifest=dict(path='mujoco/reports/'+MANIFEST,sha256=manifest['sha256'],file_sha256=file_hash(r/MANIFEST)),
        compatibility=manifest['compatibility'],identities=manifest['identity'],checkpoint_provenance=manifest['provenance'],
        inventory=manifest['inventory'],protocol=manifest['protocol'],target_isolation=manifest['isolation'],
        main=main,paired=pairs,historical_pi7d_reproduction=reproduced,historical_pi7d_metrics=history,
        ppo_failure_diagnostics=dict(timeout_count=main['ppo7d']['timeouts'],physical_failures=main['ppo7d']['physical_failures'],
            timeout_mean_final_distance_m=float(np.mean([x['final_distance_m'] for x in fresh['ppo7d'] if x['timeout']])),
            timeout_mean_final_speed_m_s=float(np.mean([x['final_speed_m_s'] for x in fresh['ppo7d'] if x['timeout']])),
            ever_entered_distance_threshold=main['ppo7d']['failed_target_entry_without_success'],
            ever_joint_threshold_without5step_hold=main['ppo7d']['failures_ever_joint_threshold'],
            interpretation='Residual position error and/or insufficient simultaneous5-step settling within15s; crossings occur in some failures, not universal high-speed divergence/saturation. No single causal root established.'),
        eligible_pi7d_training_seeds=[0],across_seed_mean_sd='N/A: exactly one independently trained compatible PI7D seed',
        frozen=e['frozen'],deterministic_repeats=e['deterministic_repeats'],
        accounting=dict(main_episodes=800,historical_episodes=200,formal_unique=1000,technical_repeats=4,
            actual_unique_experiment_executions=e['unique_experiment_executions'],resume_added_executions=e['new_episode_executions_this_invocation']),
        metrics_definitions=dict(completion='successful episodes only; timeout is N/A; paired time only common successes',
            near_target='pre-action policy samples with distance<0.10m; sample weighted',
            saturation='normalized command element abs(a)>=0.95; NOT allocator/PI saturation',
            trajectory_length='sum physical xyz chord lengths at25Hz, not continuous high-frequency path length',
            uncertainty='Wilson95% descriptive fixed-target intervals; target dispersion std is population SD, not training-seed SD'),
        resources=resources,resource_summary=dict(workers=1,oom=False,
            peak_process_tree_rss_bytes=max(v['process_tree_peak_sampled_rss_bytes'] for v in resources),
            peak_process_tree_hwm_bytes=max(v['process_tree_peak_hwm_bytes'] for v in resources)),
        records=[{k:x[k] for k in compact} for x in rows],figures=figures(root,rows),
        limitations=['single compatible PI7D seed','PPO complete training target stream unavailable','historical PI7D report lacks original checkpoint SHA',
            'different architectures and training budgets; no sample-efficiency inference','nominal simulation only; no PPO disturbance/initial-yaw evaluation',
            'deterministic PPO only; no stochastic evaluation','no training or controller changes'],
        verification=dict(evaluation_summary_sha256=file_hash(parts/'evaluation.json'),all1000_formal_records_verified=True,
            all400_initial_states_paired=True,technical_repeat_exact_arrays=4,resume_additional_flights=e['new_episode_executions_this_invocation'],
            source_tutorial='mujoco/rl/PPO_FROM_SOURCE.md',test_suite='38 relevant evaluation-only unittest tests; excludes training and8-env factory fixtures'),
        conclusion='Existing PI7D seed0 PPO reproduces historical74/82 success and achieves165/200 on new nominal targets; bothBC policies andScripted200/200. PPO is slower and longer-path on165 common successes, with35timeouts,0physical failures,0command saturation. Fair closed-loop7D PI comparison is supported, not equal-budget algorithm/sample-efficiency superiority. Historical95.6% is a different P-controller5-seed protocol;10D PI-PPO has privileged integral inputs and is not the fair main comparison.',
        next='Pre-register a matched7D PI-PPO independent-seed training protocol with complete target provenance and separate validation/test before any retraining; do not execute in this stage.')
    atomic_json(r/REPORT,value)
    print(json.dumps({n:{k:v[k] for k in ['successes','timeouts','physical_failures','completion_time_s','trajectory_length_m']} for n,v in main.items()},indent=2))
    return value

if __name__=='__main__':analyze(Path(__file__).resolve().parents[2])
