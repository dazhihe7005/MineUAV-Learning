"""Sequential guarded reproduction and compact aggregate publication."""
import argparse,json,sys
from pathlib import Path
from uav_bc_safety import atomic_json,supervise

REPORT='uav_behavior_cloning_seed0.json'

def outcome_prose(summaries):
    n=sum(r['episodes'] for r in summaries['bc'].values());s=sum(r['successes'] for r in summaries['bc'].values())
    en=sum(r['episodes'] for r in summaries['scripted'].values());es=sum(r['successes'] for r in summaries['scripted'].values())
    conclusion=f'BC: {s}/{n} successes; Scripted: {es}/{en}. '
    if s==n and es==en:
        conclusion+='An independent learned policy baseline is established for this nominal waypoint cohort, matching scripted success.'
    else:conclusion+='Not an all-success result; failures/timeouts must be retained and the independent baseline assessed without blanket success claims.'
    conclusion+=' No privileged expert information is used; not a robustness/general-flight guarantee.'
    failures=n-s
    failure_note=('No failed BC episodes; the failed-trajectory figure is an explicit no-failure notice.' if failures==0
        else f'{failures} unsuccessful BC episodes retained, including failures/timeouts; representative failure comes from actual records.')
    return dict(conclusion=conclusion,failure_note=failure_note)

def publish(root):
    import numpy as np
    from latent_dynamics_data import file_hash
    from uav_bc_data import PARTS,MANIFEST,SPLITS,load_split
    from uav_bc_evaluation import EVALUATION,CANONICAL
    from uav_bc_policy import load
    from uav_bc_training import MODEL
    from uav_bc_plots import plots
    root=Path(root);reports=root/'mujoco/reports';parts=reports/PARTS
    training=json.loads((parts/'training.json').read_text());evaluation=json.loads((parts/'evaluation.json').read_text())
    data=json.loads((reports/MANIFEST).read_text());verification=json.loads((parts/'verification.json').read_text())
    if not verification['passed']:raise ValueError('failed verification cannot be published as complete')
    policy,_=load(root/'mujoco/rl/models'/MODEL);test=load_split(data,'test');pred=policy.raw_actions(test['observations'])
    figures=plots(reports,training,evaluation,test,pred)
    resources={p.stem.removeprefix('resource_'):json.loads(p.read_text()) for p in sorted(parts.glob('resource_*.json'))
        if p.stem!='resource_publish'}
    for required in ('collect','train','evaluate','verify'):
        if required not in resources or resources[required]['exit_code']!=0 or resources[required]['failure'] or resources[required]['oom_count_delta']:
            raise ValueError('incomplete/unsafe experiment phase')
    compact={condition:{task:[{k:v for k,v in r.items() if k not in ('identity','path')} for r in rows]
        for task,rows in tasks.items()} for condition,tasks in evaluation['records'].items()}
    outcomes=outcome_prose(evaluation['closed_loop'])
    normalized_action_error=evaluation['offline']['raw']['per_dimension']
    physical_error=dict(names=['vx_cmd_m_s','vy_cmd_m_s','vz_cmd_m_s','yaw_rate_cmd_rad_s'],
        rmse=(np.asarray(normalized_action_error['rmse'])*[1.5,1.5,1,1]).tolist(),
        mae=(np.asarray(normalized_action_error['mae'])*[1.5,1.5,1,1]).tolist())
    result=dict(experiment='BEHAVIOR CLONING UAV POLICY BASELINE',date='2026-10-09',seed=0,
        git=dict(branch='feat/uav-behavior-cloning-baseline',base_branch='feat/world-model-onpolicy-adaptation',
            base_commit='4858ce65a29806f37a813f0225b6aa6adfa6cd15'),
        expert=dict(source='mujoco/rl/test_env_scripted_policy.py:scripted_action',
            inputs_used=['target_error_xyz','yaw_error'],privileged_information=False,
            formula='velocity=clip(error_xyz/3, physical velocity bounds); env action xyz=velocity/[1.5,1.5,1]; yaw=clip(yaw_error,-1,1)',
            ignores_velocity=True,labels='actual 4D high-level commands, not PWM or world-model predictions'),
        policy=dict(architecture='7 -> 128 ReLU -> 128 ReLU -> 4 linear',parameters=18052,
            observation='target-position xyz, world-frame velocity xyz, wrapped yaw error',
            input_normalization='Train observation mean/std',output='Train-standardized action -> inverse -> legal env clipping [-1,1]',
            physical_scale=[1.5,1.5,1,1],execution='independent neural MLP only; no teacher/policy/world model/history fallback'),
        dataset=dict(counts=data['counts'],episodes=120,total_action_samples=sum(r['steps'] for r in data['counts'].values()),
            split='original PI84/18/18 target grouping; independent newly collected full expert episodes',
            episode_seed_formula='2026100900 + original target_id',target_overlap_with_benchmark_holdout=0,
            original_dataset_reused_for='targets/split only, not dynamics labels',
            manifests={n:dict(path='mujoco/reports/'+n,sha256=file_hash(reports/n)) for n in (MANIFEST,SPLITS,EVALUATION)},
            raw='local-only ignored parts; each episode hash and generation identity retained in manifests'),
        environment=dict(name='unchanged MineUAVPIEnv RewardV2',physics_hz=500,control_hz=100,policy_hz=25,
            success='distance <0.10m AND speed <0.15m/s for 5 consecutive steps',failure_timeout='unchanged environment rules'),
        training=dict(training,optimizer='Adam',learning_rate=.001,batch_size=512,max_epochs=100,
            objective='mean over Train samples and four action dims of ((predicted_command - expert_command)/Train_action_std)^2',
            selection='Validation action normalized MSE only',gradient_clipping=False,early_stopping=False,hyperparameter_search=False),
        offline=dict(evaluation['offline'],physical_command_error=physical_error),closed_loop=evaluation['closed_loop'],
        episodes=compact,state_distribution=evaluation['state_distribution'],frozen_bc=evaluation['frozen_bc'],
        canonical_world_model=evaluation['canonical_world_model'],verification=verification,
        tests=json.loads((parts/'test_evidence.json').read_text()) if (parts/'test_evidence.json').exists() else None,
        resource_safety=dict(workers=1,max_workers_allowed=2,sequential_phases=True,system_reserve_gib=1.5,
            rss_abort_gib=4,available_memory_abort_gib=2,poll_seconds=.1,swap_modified=False,
            oom_count_delta=sum(r['oom_count_delta'] for r in resources.values()),
            peak_sampled_process_tree_rss_bytes=max(r['process_tree_peak_sampled_rss_bytes'] for r in resources.values()),
            peak_process_tree_hwm_bytes=max(r['process_tree_peak_hwm_bytes'] for r in resources.values()),
            minimum_available_bytes=min(r['minimum_available_bytes'] for r in resources.values()),
            guarantee='polling fail-safe and sampled process-tree telemetry, not hard OS memory/real-time guarantee',phases=resources),
        figures={n:file_hash(reports/n) for n in figures},
        conclusion=outcomes['conclusion'],
        diagnostics='Visited-state posthoc imitation errors, distribution shifts, completion time, crossings and saturation are reported separately; offline imitation error alone does not establish closed-loop success or failure causality.',
        limitations=['single training seed','120 target expert trajectories','nominal simulation, fixed reset/target manifests',
            'simple directly representable expert rule; no hidden-state parity problem for this expert',
            'narrow nominal yaw/action coverage; no noise, perturbation or saturation/recovery generalization claim',
            outcomes['failure_note'],
            'historical unrelated test dependencies handled read-only; no old file restoration'],
        next_step='One separate fixed-manifest perturbed-initial-state BC robustness evaluation; no retraining or DAgger in this stage.')
    atomic_json(reports/REPORT,result);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish-only',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[2];parts=root/'mujoco/reports/uav_behavior_cloning_seed0_parts'
    if args.publish_only:publish(root);return
    # Cached episodes are hash/source/runtime checked; interrupted training restarts
    # the identical complete budget, not a silently shortened/continued budget.
    for name,module in [('collect','uav_bc_data.py'),('train','uav_bc_training.py'),('evaluate','uav_bc_evaluation.py'),('verify','verify_uav_behavior_cloning.py')]:
        if name=='train' and (parts/'training.json').exists():
            from uav_bc_policy import load
            from latent_dynamics_data import file_hash
            from uav_bc_data import MANIFEST
            t=json.loads((parts/'training.json').read_text());path=root/'mujoco/rl/models/uav_bc_mlp_seed0.pt';_,m=load(path)
            if t['epochs_completed']!=100 or file_hash(path)!=t['checkpoint']['sha256'] or m['provenance']['dataset_manifest_sha256']!=file_hash(root/'mujoco/reports'/MANIFEST):
                raise ValueError('stale/incomplete trained model; do not silently resume')
            continue
        supervise([sys.executable,str(root/'mujoco/rl'/module)],parts,name)
    supervise([sys.executable,__file__,'--publish-only'],parts,'publish')

if __name__=='__main__':main()
