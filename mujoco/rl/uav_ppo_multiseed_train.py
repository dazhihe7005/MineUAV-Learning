"""Sequential fresh PPO training; passive logs, diagnostic-only validation."""
import argparse
import gc
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from train_ppo_waypoint import make_ppo,TRAIN_METRICS
from ppo_pi_env import MineUAVPIEnv
from uav_bc_policy import parameter_hash
from uav_bc_safety import atomic_json,atomic_bytes
from latent_dynamics_data import file_hash,json_hash
from uav_ppo_multiseed import (ROOT,PARTS,read_manifest,AuditedPIEnv,assert_model_protocol,seal,verify_completion)

@contextmanager
def preserve_rng(module):
    py=random.getstate();npstate=np.random.get_state();ts=torch.get_rng_state();mode=module.training
    try:yield
    finally:
        random.setstate(py);np.random.set_state(npstate);torch.set_rng_state(ts);module.set_training_mode(mode) if hasattr(module,'set_training_mode') else module.train(mode)

def atomic_model(model,path):atomic_bytes(path,lambda f:model.save(f))

def validate(model,targets):
    env=MineUAVPIEnv(reward_version='v2',target_distribution='full');rows=[]
    try:
        with preserve_rng(model.policy):
            model.policy.set_training_mode(False)
            for t in targets:
                o,_=env.reset(seed=t['env_seed'],options={'target_position':t['target']});ret=0.
                while True:
                    a,_=model.predict(o,deterministic=True);o,r,done,timeout,info=env.step(a);ret+=r
                    if done or timeout:break
                rows.append(dict(target_id=t['target_id'],success=info['termination_reason']=='success',
                    termination_reason=info['termination_reason'],return_=ret,final_distance_m=info['distance_m']))
    finally:env.close()
    return dict(episodes=len(rows),successes=sum(r['success'] for r in rows),
        mean_return=float(np.mean([r['return_'] for r in rows])),
        mean_final_distance_m=float(np.mean([r['final_distance_m'] for r in rows])),records=rows)

class ReplicationCallback(BaseCallback):
    def __init__(self,manifest,directory,seed):
        super().__init__();self.manifest=manifest;self.directory=directory;self.seed=seed
        self.history=[];self.validation={};self.last=0;self.started=time.monotonic()
    def _on_step(self):return True
    def _on_rollout_start(self):self.record_update()
    def _on_training_end(self):self.record_update()
    def record_update(self):
        step=self.model.num_timesteps
        if not step or step==self.last:return
        if step%2048 or self.model._n_updates!=step//2048*10:raise ValueError('checkpoint not post complete update')
        metrics={k:float(self.model.logger.name_to_value['train/'+k]) for k in TRAIN_METRICS}
        if not all(np.isfinite(v) for v in metrics.values()):raise ValueError('nonfinite optimization metrics')
        recent=list(self.model.ep_info_buffer or [])
        row=dict(timesteps=step,optimization_epochs=self.model._n_updates,train=metrics,
            mean_episode_return=float(np.mean([r['r'] for r in recent])) if recent else None,
            mean_episode_length=float(np.mean([r['l'] for r in recent])) if recent else None,
            recent_episodes=len(recent),recent_training_success_rate=float(np.mean([r['termination_reason']=='success' for r in recent])) if recent else None,
            policy_std=self.model.policy.log_std.detach().exp().tolist(),elapsed_s=time.monotonic()-self.started)
        self.history.append(row);self.last=step
        atomic_json(self.directory/'history.json',self.history)
        if step in self.manifest['config']['validation_milestones']:
            self.validation[str(step)]=validate(self.model,self.manifest['splits']['validation'])
            atomic_model(self.model,self.directory/f'diagnostic_{step}.zip')
            atomic_json(self.directory/'validation.json',self.validation)
            print('seed',self.seed,'post-update',step,'Val',self.validation[str(step)]['successes'],'/40',flush=True)
        elif step//2048%10==0:
            print('seed',self.seed,'post-update',step,'recent Train success',row['recent_training_success_rate'],flush=True)

def publish(record):
    path=Path(record['published_checkpoint_path'])
    if path.exists():
        if file_hash(path)!=record['checkpoint_sha256']:raise ValueError('refuse overwrite historical/unmatched model')
    else:atomic_bytes(path,lambda f:f.write(Path(record['checkpoint_path']).read_bytes()))

def train(seed,root=ROOT):
    root=Path(root);m=read_manifest(root)
    if seed not in m['config']['seeds']:raise ValueError('unregistered seed')
    parts=root/'mujoco/reports'/PARTS;complete=parts/f'seed{seed}.json'
    if complete.exists():
        record=json.loads(complete.read_text());verify_completion(record,m,seed);publish(record)
        print('verified complete seed',seed,'no training repeated',flush=True);return record
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    attempt=1
    while (parts/f'seed{seed}_attempt{attempt:03d}').exists():attempt+=1
    directory=parts/f'seed{seed}_attempt{attempt:03d}';directory.mkdir(parents=True)
    atomic_json(directory/'task.json',dict(seed=seed,attempt=attempt,status='started',manifest_sha256=m['sha256']))
    forbidden=[t['target'] for v in m['splits'].values() for t in v]
    env=DummyVecEnv([lambda rank=rank:Monitor(AuditedPIEnv(rank,forbidden),
        filename=str(directory/f'monitor{rank}'),info_keywords=('distance_m','termination_reason')) for rank in range(8)])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);start=time.monotonic()
    try:
        env.seed(seed)
        model=make_ppo(env,directory/'tensorboard',log_std_init=-2.,seed=seed)
        model.tensorboard_log=None;model.verbose=0;assert_model_protocol(model)
        initial=parameter_hash(model.policy)
        atomic_json(directory/'initial.json',dict(seed=seed,parameter_sha256=initial,
            initial_log_std=model.policy.log_std.detach().tolist(),random_initialization=True,manifest_sha256=m['sha256']))
        callback=ReplicationCallback(m,directory,seed)
        model.learn(total_timesteps=100352,callback=callback,log_interval=1)
        if model.num_timesteps!=100352 or model._n_updates!=490 or len(callback.history)!=49:
            raise ValueError('incomplete training')
        visits=[v for e in env.envs for v in e.unwrapped.visits]
        episodes=[v for e in env.envs for v in e.unwrapped.completed]
        if len({(v['rank'],v['episode_index']) for v in visits})!=len(visits):raise ValueError('duplicate target visit')
        atomic_json(directory/'actual_train_targets.json',visits)
        atomic_json(directory/'training_episodes.json',episodes)
        final=directory/'final.zip';atomic_model(model,final)
        loaded=PPO.load(final,device='cpu');assert_model_protocol(loaded)
        if (loaded.seed,loaded.num_timesteps,loaded._n_updates)!=(seed,100352,490):raise ValueError('reload budget/seed identity')
        final_hash=parameter_hash(model.policy)
        if parameter_hash(loaded.policy)!=final_hash:raise ValueError('checkpoint reload parameter mismatch')
        if final_hash==initial:raise ValueError('no actual model update')
        record=seal(dict(status='complete',seed=seed,manifest_sha256=m['sha256'],attempt=attempt,
            actual_timesteps=100352,optimization_epochs=490,optimizer_steps=3920,
            initial_parameter_sha256=initial,final_parameter_sha256=final_hash,
            checkpoint_path=str(final),published_checkpoint_path=str(root/f'mujoco/rl/models/uav_ppo_fixed_protocol_seed{seed}.zip'),
            checkpoint_sha256=file_hash(final),history=callback.history,validation=callback.validation,
            actual_train_targets_path=str(directory/'actual_train_targets.json'),actual_train_targets_sha256=file_hash(directory/'actual_train_targets.json'),
            actual_train_targets=visits,training_episodes_path=str(directory/'training_episodes.json'),
            training_episodes_sha256=file_hash(directory/'training_episodes.json'),completed_training_episodes=len(episodes),
            wall_seconds=time.monotonic()-start,selection='final budget irrespective of Val success',
            parameters=sum(p.numel() for p in model.policy.parameters()),checkpoint_reload_verified=True))
        verify_completion(record,m,seed);atomic_json(complete,record);publish(record)
        print('TRAIN COMPLETE',seed,'timesteps',model.num_timesteps,'hash',final_hash,flush=True)
        return record
    finally:env.close();gc.collect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);a=p.parse_args();train(a.seed)
