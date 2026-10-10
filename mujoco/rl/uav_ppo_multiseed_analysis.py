"""Verified fixed-target paired metrics; training seeds are statistical units."""
import json
from pathlib import Path
import numpy as np
from uav_ppo_multiseed import ROOT,BASE,BRANCH,PARTS,MANIFEST,REPORT,read_manifest,verify_seal,verify_completion,seal
from uav_ppo_bc_nominal_analysis import summary,paired
from uav_ppo_bc_nominal_run import motion_metrics,validate_execution
from uav_bc_robustness import validate_cohort,validate_record
from uav_bc_evaluation import trace_metrics
from uav_bc_safety import atomic_json,atomic_bytes
from latent_dynamics_data import file_hash,json_hash

def seed_statistics(values):
    x=np.asarray(values,float)
    if x.shape!=(5,) or not np.isfinite(x).all():raise ValueError('five finite independent training-run summaries required')
    mean=float(x.mean());sd=float(x.std(ddof=1));half=2.7764451051977987*sd/np.sqrt(5)
    return dict(n_seeds=5,mean=mean,sample_sd=sd,median=float(np.median(x)),min=float(x.min()),max=float(x.max()),
        exploratory_t95_ci=[float(mean-half),float(mean+half)],statistical_unit='training run; same fixed200targets',
        caveat='n5 exploratory t interval; shared target set and partially shared environment RNG streams; not population guarantee')

def failure_diagnosis(rows):
    failed=[r for r in rows if not r['success']]
    return dict(failures=len(failed),timeouts=sum(r['timeout'] for r in failed),physical_failures=sum(r['physical_failure'] for r in failed),
        failures_entered_distance_threshold=sum(r['reached_target_region'] for r in failed),
        failures_ever_joint_threshold=sum(r['ever_joint_threshold'] for r in failed),
        failed_max_streak4=sum(r['maximum_success_streak']==4 for r in failed),
        failed_crossing=sum(r['crossings']>0 for r in failed),
        failed_final_distance_pass_only=sum(r['final_distance_m'] is not None and r['final_speed_m_s'] is not None and r['final_distance_m']<.1 and r['final_speed_m_s']>=.15 for r in failed),
        failed_final_speed_pass_only=sum(r['final_distance_m'] is not None and r['final_speed_m_s'] is not None and r['final_distance_m']>=.1 and r['final_speed_m_s']<.15 for r in failed),
        failed_final_neither=sum(r['final_distance_m'] is not None and r['final_speed_m_s'] is not None and r['final_distance_m']>=.1 and r['final_speed_m_s']>=.15 for r in failed),
        interpretation='Threshold entry, terminal state and5-step dwell are distinct; no unique causal failure mechanism inferred.')

def verify_all(root):
    root=Path(root);parts=root/'mujoco/reports'/PARTS;m=read_manifest(root)
    train={str(s):json.loads((parts/f'seed{s}.json').read_text()) for s in range(5)}
    for s,v in train.items():
        verify_completion(v,m,int(s))
        if file_hash(v['published_checkpoint_path'])!=v['checkpoint_sha256']:raise ValueError('published checkpoint changed')
        from stable_baselines3 import PPO
        from uav_bc_policy import parameter_hash
        from uav_ppo_multiseed import assert_model_protocol
        model=PPO.load(v['published_checkpoint_path'],device='cpu');assert_model_protocol(model)
        if (model.seed,model.num_timesteps,model._n_updates)!=(int(s),100352,490):raise ValueError('ZIP seed/budget drift')
        if {int(state['step'].item()) for state in model.policy.optimizer.state.values()}!={3920}:raise ValueError('optimizer step budget drift')
        if parameter_hash(model.policy)!=v['final_parameter_sha256']:raise ValueError('ZIP parameter identity drift')
        del model
        if json.loads(Path(v['actual_train_targets_path']).read_text())!=v['actual_train_targets']:raise ValueError('target audit mismatch')
        forbidden=np.asarray([t['target'] for rows in m['splits'].values() for t in rows])
        for t in v['actual_train_targets']:
            if np.any(np.linalg.norm(forbidden-t['target'],axis=1)<1e-8):raise ValueError('Train/Test target leakage')
        if [r['timesteps'] for r in v['history']]!=list(range(2048,100353,2048)):raise ValueError('omitted optimization history')
    if len({v['initial_parameter_sha256'] for v in train.values()})!=5:raise ValueError('random seed initialization duplicated')
    e=json.loads((parts/'evaluation.json').read_text());verify_seal(e)
    if e['manifest_sha256']!=m['sha256']:raise ValueError('evaluation wrong protocol')
    states={s['key']:s for s in m['states']};names=list(e['identities'])
    validate_cohort(e['records'],[s['key']+'/'+n for s in m['states'] for n in names])
    if len(e['records'])!=1600:raise ValueError('missing formal episodes')
    for r in e['records']:
        identity=dict(manifest_sha256=m['sha256'],controller=r['controller'],model_identity=e['identities'][r['controller']])
        validate_record(r,identity,states[r['key']],r['controller'])
        with np.load(r['trace_path'],allow_pickle=False) as a:
            for k,v in {**trace_metrics(a),**motion_metrics(a)}.items():
                if r[k]!=v:raise ValueError('raw trajectory/summary mismatch: '+k)
            if r['success'] and int(a['success_streak'][-1])<5:raise ValueError('false success')
    for name,r in e['repeats'].items():
        validate_execution(r,dict(manifest_sha256=m['sha256'],controller=name,model_identity=e['identities'][name]))
    return m,train,e

def plot_all(root,cohorts,train):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(root)/'mujoco/reports/uav_ppo_multiseed_figures';out.mkdir(exist_ok=True);hashes={}
    names=list(cohorts);labels=['Scripted','Original BC','Yaw BC']+[f'PPO seed{s}' for s in range(5)]
    def save(fig,name):
        fig.tight_layout();p=out/name;atomic_bytes(p,lambda f:fig.savefig(f,format='png',dpi=140));plt.close(fig);hashes[str(p.relative_to(root))]=file_hash(p)
    fig,ax=plt.subplots(figsize=(10,4));counts=[sum(r['success'] for r in cohorts[n]) for n in names]
    ax.bar(labels,counts,color=['gray','green','teal']+['royalblue']*5)
    for i,v in enumerate(counts):ax.text(i,v+2,f'{v}/200',ha='center',fontsize=8)
    ax.set_ylim(0,225);ax.set_ylabel('Success count /200 fixed targets');save(fig,'success_by_seed.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for seed,v in train.items():
        h=v['history'];t=np.array([x['timesteps'] for x in h])/1000
        axes[0].plot(t,[x['mean_episode_return'] for x in h],label=seed)
        axes[1].plot(t,[x['recent_training_success_rate'] for x in h])
        points=v['validation'];axes[2].plot([int(k)/1000 for k in points],[x['successes']/40 for x in points.values()],marker='o')
    axes[0].set_ylabel('Train rolling100 episode return');axes[1].set_ylabel('Train rolling100 success');axes[2].set_ylabel('Fixed Val40 success, diagnostic only')
    for a in axes:a.set_xlabel('Training transitions (thousands)')
    axes[0].legend(title='Training seed');save(fig,'learning_curves.png')
    metrics=['policy_gradient_loss','value_loss','entropy_loss','approx_kl','clip_fraction','explained_variance']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,k in zip(axes.flat,metrics):
        for seed,v in train.items():ax.plot([x['timesteps']/1000 for x in v['history']],[x['train'][k] for x in v['history']],label=seed)
        ax.set_title(k);ax.set_xlabel('Transitions (thousands)')
    axes[0,0].legend(title='Seed');save(fig,'ppo_learning_metrics.png')
    for field,name,title,onlysuccess in [('final_distance_m','final_distance.png','Final distance (m); all outcomes',False),
        ('completion_time_s','completion_time.png','Completion time (s); successes only',True),
        ('trajectory_length_m','trajectory_length.png','Path length (m); successes only',True)]:
        fig,ax=plt.subplots(figsize=(11,4));vals=[[r[field] for r in cohorts[n] if r.get(field) is not None and (r['success'] or not onlysuccess)] for n in names]
        ax.boxplot(vals,tick_labels=[f'{l}\nn={len(v)}' for l,v in zip(labels,vals)],showfliers=True);ax.set_ylabel(title);ax.tick_params(axis='x',labelsize=8);save(fig,name)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for s in range(5):
        d=failure_diagnosis(cohorts[f'ppo_seed{s}']);axes[0].bar(s,d['timeouts'],color='royalblue',label='Timeout' if s==0 else None)
        axes[0].bar(s,d['physical_failures'],bottom=d['timeouts'],color='darkorange',label='Physical failure' if s==0 else None)
        axes[1].bar(s,d['failures_entered_distance_threshold'],color='teal')
    axes[0].set_ylabel('Failure count');axes[0].legend();axes[1].set_ylabel('Failures ever distance<0.10m')
    for ax in axes:ax.set_xlabel('Training seed');ax.set_xticks(range(5))
    save(fig,'failure_modes.png')
    chosen=next((r for s in range(5) for r in cohorts[f'ppo_seed{s}'] if not r['success']),None)
    fig,axes=plt.subplots(2,1,figsize=(9,6))
    if chosen:
        for n in ['scripted','original_bc','yaw_bc',chosen['controller']]:
            r=next(x for x in cohorts[n] if x['key']==chosen['key'])
            with np.load(r['trace_path'],allow_pickle=False) as a:
                t=np.arange(len(a['observations']))*.04
                axes[0].plot(t,np.linalg.norm(a['observations'][:,:3],axis=1),label=n)
                axes[1].plot(t,np.linalg.norm(a['observations'][:,3:6],axis=1),label=n)
        axes[0].set_title('First timeout in fixed seed/manifest order: '+chosen['controller']+'/'+chosen['key'])
        axes[0].legend();axes[0].axhline(.1,ls='--',color='gray');axes[1].axhline(.15,ls='--',color='gray')
    else:axes[0].text(.2,.5,'No PPO failure exists; no failure example manufactured')
    axes[0].set_ylabel('Target distance (m)');axes[1].set_ylabel('Speed (m/s)');axes[1].set_xlabel('Simulation time (s)');save(fig,'paired_timeout_example.png')
    return hashes

def analyze(root=ROOT):
    root=Path(root);parts=root/'mujoco/reports'/PARTS;m,train,e=verify_all(root)
    names=['scripted','original_bc','yaw_bc']+[f'ppo_seed{s}' for s in range(5)]
    cohorts={n:[r for r in e['records'] if r['controller']==n] for n in names}
    metrics={n:summary(v) for n,v in cohorts.items()}
    pairs={f'{bc}_vs_seed{s}':paired(cohorts[bc],cohorts[f'ppo_seed{s}']) for bc in ['scripted','original_bc','yaw_bc'] for s in range(5)}
    aggregate={k:seed_statistics([metrics[f'ppo_seed{s}'][k] for s in range(5)]) for k in ['success_rate','successes','timeouts','physical_failures','action_saturation_fraction']}
    for k in ['final_distance_m','final_speed_m_s','maximum_speed_m_s','trajectory_length_m','completion_time_s']:
        vals=[metrics[f'ppo_seed{s}'][k]['mean'] if metrics[f'ppo_seed{s}'][k] else None for s in range(5)]
        aggregate[k]=seed_statistics(vals) if all(v is not None for v in vals) else dict(values=vals,note='no-success metrics missing; never replaced with timeout')
    for k in ['near_target_actual_speed_m_s','near_target_command_speed_m_s']:
        vals=[metrics[f'ppo_seed{s}'][k] for s in range(5)]
        aggregate[k]=seed_statistics(vals) if all(v is not None for v in vals) else dict(values=vals,note='no near-target sample is missing, not zero')
    paired_seed_aggregate={bc:dict(success_delta=seed_statistics([pairs[f'{bc}_vs_seed{s}']['success_delta'] for s in range(5)]),
        ppo_success_wins=sum(pairs[f'{bc}_vs_seed{s}']['success_delta']>0 for s in range(5)),
        bc_success_wins=sum(pairs[f'{bc}_vs_seed{s}']['success_delta']<0 for s in range(5))) for bc in ['scripted','original_bc','yaw_bc']}
    resources=[json.loads(p.read_text()) for p in sorted(parts.glob('resource_*.json'))]
    if not resources or any(r['exit_code'] or r['failure'] or r['oom_count_delta'] for r in resources):raise ValueError('incomplete or unsafe resource phase')
    audit=seal(dict(manifest_sha256=m['sha256'],targets_by_training_seed={s:v['actual_train_targets'] for s,v in train.items()},
        exact_Val_Final_overlap=0,coordinate_tolerance_m=1e-8))
    auditpath=root/'mujoco/reports/uav_ppo_multiseed_training_target_audit.json';atomic_json(auditpath,audit)
    compact=['key','controller','target_id','target','initial_snapshot_sha256','success','termination_reason','steps',
        'completion_time_s','final_distance_m','final_speed_m_s','maximum_speed_m_s','trajectory_length_m',
        'near_steps','near_actual_sum','action_saturation_elements','crossings','reached_target_region','ever_joint_threshold','maximum_success_streak','trace_sha256','record_sha256']
    value=dict(experiment='Fixed-protocol7D PI PPO multi-seed replication',base_commit=BASE,branch=BRANCH,
        config=m['config'],config_sha256=m['config_sha256'],preregistration_sha256=m['sha256'],
        manifest_path='mujoco/reports/'+MANIFEST,manifest_file_sha256=file_hash(root/'mujoco/reports'/MANIFEST),
        target_audit=dict(path=str(auditpath.relative_to(root)),sha256=audit['sha256'],file_sha256=file_hash(auditpath),
            actual_target_visits={s:len(v['actual_train_targets']) for s,v in train.items()},actual_overlap=0,
            stream_overlap=m['train_specification']['cross_seed_target_stream_overlap']),
        training={s:{k:v for k,v in t.items() if k!='actual_train_targets'} for s,t in train.items()},
        individual=metrics,ppo_seed_aggregate=aggregate,paired=pairs,paired_seed_aggregate=paired_seed_aggregate,
        failures={str(s):failure_diagnosis(cohorts[f'ppo_seed{s}']) for s in range(5)},
        all_checkpoints_and_sources_verified=True,actual_adam_step_counters_verified=5,
        runtime_controls=dict(torch_threads=1,torch_deterministic_algorithms=True,
            note='set in preregistered training-source hash; historical BLAS/thread details unavailable'),
        frozen=e['frozen'],technical_repeats=e['repeats'],
        accounting=dict(training_seeds=5,total_training_transitions=501760,optimizer_steps=19600,
            formal_final_episodes=1600,diagnostic_validation_episodes=600,technical_repeat_episodes=8,
            final_unique_executions=e['unique_executions'],resume_new_executions=e['new_executions'],
            seed_attempts={s:v['attempt'] for s,v in train.items()}),
        learning_diagnostics={s:dict(validation_successes={k:v['successes'] for k,v in t['validation'].items()},
            final_train_metrics=t['history'][-1]['train'],final_recent_training_success_rate=t['history'][-1]['recent_training_success_rate'],
            final_return=t['history'][-1]['mean_episode_return'],
            final_differential_entropy_nats=-t['history'][-1]['train']['entropy_loss'],
            value_loss_range=[min(h['train']['value_loss'] for h in t['history']),max(h['train']['value_loss'] for h in t['history'])],
            approximate_kl_range=[min(h['train']['approx_kl'] for h in t['history']),max(h['train']['approx_kl'] for h in t['history'])],
            interpretation='Milestone regression is diagnostic, never checkpoint selection. Value fit/KL/clip/returns alone do not identify a causal root.') for s,t in train.items()},
        resources=resources,resource_summary=dict(workers=1,logical_envs=8,oom=False,
            peak_process_tree_rss_bytes=max(r['process_tree_peak_sampled_rss_bytes'] for r in resources)),
        metrics_definitions=dict(time='success only; paired time/path common successes',
            saturation='normalized command elements abs(a)>=.95, not PI/allocator saturation',
            trajectory='sum25Hz physical position chord lengths',near='pre-action samples distance<.10m, sample-weighted',
            statistical_unit='five training runs; fixed target200 not200seeds; n5 uncertainty exploratory'),
        records=[{k:r[k] for k in compact} for r in e['records']],figures=plot_all(root,cohorts,train),
        conclusion=dict(ppo_success_mean=aggregate['success_rate']['mean'],ppo_success_sample_sd=aggregate['success_rate']['sample_sd'],
            ppo_vs_original_bc_success_wins=paired_seed_aggregate['original_bc']['ppo_success_wins'],
            ppo_vs_yaw_bc_success_wins=paired_seed_aggregate['yaw_bc']['ppo_success_wins'],
            validation_regression_seeds=[s for s,t in train.items() if t['validation']['100352']['successes']<max(v['successes'] for v in t['validation'].values())],
            historical_reference='82.5% was a different trained checkpoint and different200targets; current fixed-protocol replication is not exact historical optimizer replay',
            interpretation='Judgment applies to this fixed7D PI PPO100k protocol, not PPO algorithm generally or sample efficiency. Finite learning and settling limitations are observations, reward/value/hidden-state causality unresolved.',
            reward_design='Worth an offline reward-versus-settling alignment audit before tuning; no reward change performed.',
            sac='Not required by present evidence; a new algorithm cannot isolate current protocol instability.',
            next='One preregistered offline RewardV2/task-settling alignment audit, no training or automatic execution.'),
        limitations=['five training seeds only','same fixed200Final targets for all policies','partially shared Gym target RNG streams',
            'historical full training targets unavailable, no exact historic training replay claimed',
            'different PPO/BC training data/budgets and network size; not sample-efficiency comparison',
            'nominal deterministic evaluation only','no new hyperparameters/rewards/algorithms tried'])
    atomic_json(root/'mujoco/reports'/REPORT,value)
    print(json.dumps({'successes':{n:x['successes'] for n,x in metrics.items()},'aggregate':aggregate['success_rate']},indent=2))
    return value

if __name__=='__main__':analyze()
