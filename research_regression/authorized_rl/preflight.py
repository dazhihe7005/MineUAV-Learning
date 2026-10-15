"""Bounded physical/forward/checkpoint tests; optimizer/learn/train tripwires active."""
import argparse
import time
from pathlib import Path
from research_ops.storage import atomic_json,read_json,digest,sha256,OpsError
from .runtime import ROOT


def run(output):
    if output.exists():raise OpsError('preflight output already used')
    work=output.with_suffix('');work.mkdir(exist_ok=False,parents=True)
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    def forbidden(*args,**kwargs):raise AssertionError('NO_OPTIMIZER_PREFLIGHT: forbidden update')
    PPO.learn=forbidden;PPO.train=forbidden;torch.optim.Adam.step=forbidden
    from lidar_integration.ppo_formal_scheduler import load_l6_prereg
    from lidar_integration.ppo_preflight import make_ppo
    from lidar_integration.ppo_formal_checkpoint import CheckpointManager
    from lidar_integration.ppo_formal_validation import _optimizer_states_equal
    from lidar_integration.ppo_training_runner import _weights_sha
    from test_env_scripted_policy import scripted_action
    from .worker import save_checkpoint_probe,runtime_sources
    from .curriculum import make_scheduler
    torch.set_num_threads(1)
    prereg=load_l6_prereg(ROOT/'mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json')
    scheduler=make_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR','CURRICULUM',0)
    checks=[];begun=time.monotonic();before=runtime_sources(prereg)
    try:
        obs,info=scheduler.reset(seed=0)
        assert scheduler.by_identity[scheduler.current_identity]['layout']=='empty'
        assert obs['lidar'].shape==(3,12,24) and obs['state'].shape==(7,) and obs['heading'].shape==(2,)
        checks.append('real_physics_dict_and_empty_eligibility')
        before_time=float(scheduler.active.unwrapped.data.time)
        for _ in range(2):obs,_,_,_,info=scheduler.step(scripted_action(obs['state']))
        assert abs(float(scheduler.active.unwrapped.data.time)-before_time-.08)<1e-9
        assert scheduler.completed_policy_steps==2
        checks.append('actual_mj_step_two_policy_steps_500hz')
        scheduler.completed_policy_steps=32768  # FORCED_TEST_STATE, not a trained counter.
        obs,info=scheduler.reset()
        assert scheduler.phase_switch_step==32768
        assert scheduler.current_identity in scheduler.by_identity
        assert float(scheduler.active.unwrapped.data.time)==0.
        checks.append('forced_test_state_curriculum_next_natural_reset_and_cache_clear')
    finally:scheduler.close()
    scheduler=make_scheduler(prereg,ROOT/'.research_local/dependencies/MuJoCo-LiDAR','UNIFORM',0)
    try:
        model=make_ppo(scheduler,n_steps=128,seed=0)
        model._last_obs=model.get_env().reset()
        weights=_weights_sha(model.policy)
        model.train=forbidden  # Same instance-method override hazard as runtime instrumentation.
        manager=CheckpointManager(work,prereg,execution_protocol_sha256='a'*64)
        record,(probe,action,optimizer)=save_checkpoint_probe(model,scheduler,manager,run_id='PREFLIGHT_UNTRAINED',seed=0)
        checks.append('actual_sb3_save_excludes_passive_train_closure')
        restored=manager.load_verified(None,run_id='PREFLIGHT_UNTRAINED')
        assert 'train' not in vars(restored) and restored.num_timesteps==0
        assert _weights_sha(restored.policy)==weights
        assert _optimizer_states_equal(optimizer,restored.policy.optimizer.state_dict())
        assert np.array_equal(action,restored.predict(probe,deterministic=True)[0])
        assert not model.policy.optimizer.state and model.num_timesteps==0
        checks.append('actual_checkpoint_prediction_optimizer_weights_equal_without_learning')
        assert before==runtime_sources(prereg)
        checks.append('historical_runtime_identity_unchanged')
    finally:scheduler.close()
    result={'status':'PASS','checks':checks,'tests_run':len(checks),'optimizer_updates':0,
        'training_timesteps':0,'forced_test_state':'scheduler threshold only; no trained progress claimed',
        'test_episodes':0,'elapsed_seconds':time.monotonic()-begun,'runtime_sources':before,
        'checkpoint':{'path':str(work/'PREFLIGHT_UNTRAINED.zip'),'sha256':record['checkpoint_sha256'],
        'purpose':'UNTRAINED_FORWARD_CHECKPOINT_TEST_ONLY; forbidden as training initialization'}}
    atomic_json(output,result,exclusive=True);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args()
    if not args.evidence.resolve().is_relative_to(ROOT/'.research_local'):raise OpsError('preflight evidence must stay local')
    print(run(args.evidence))
