"""One bounded, headless 20,000-sample PPO smoke run for MineUAV waypoints.

With eight environments and n_steps=256, each complete PPO update uses
2,048 samples. A hard callback cap stops at exactly 20,000 collected samples;
the final partial rollout is not used for an update by upstream SB3.
"""

import json
import math
import sys
import time
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from mine_uav_env import MineUAVEnv


RL_DIR = Path(__file__).resolve().parent
ROOT = RL_DIR.parent
MODEL_PATH = RL_DIR / "models" / "ppo_waypoint_smoke.zip"
TENSORBOARD_DIR = RL_DIR / "tensorboard" / "ppo_waypoint_smoke"
MONITOR_DIR = RL_DIR / "logs" / "ppo_waypoint_smoke"
REPORT_PATH = ROOT / "reports" / "ppo_waypoint_smoke.json"
EVAL_HISTORY_PATH = MONITOR_DIR / "periodic_eval.jsonl"
N_ENVS = 8
N_STEPS = 256
TOTAL_TIMESTEPS = 20_000
SEED = 20260929
TRAIN_METRICS = (
    "policy_gradient_loss", "value_loss", "entropy_loss", "approx_kl",
    "clip_fraction", "explained_variance",
)


def make_training_envs(monitor_dir: Path, n_envs: int = N_ENVS,
                       reward_version: str = "v1",
                       target_distribution: str = "full") -> DummyVecEnv:
    """Create independent, read-only-model, headless envs in this process."""
    if n_envs <= 0:
        raise ValueError("n_envs must be positive")
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def factory(rank: int):
        def create():
            return Monitor(
                MineUAVEnv(render_mode=None, reward_version=reward_version,
                           target_distribution=target_distribution),
                filename=str(monitor_dir / f"train_{rank}"),
                info_keywords=("distance_m", "termination_reason"),
            )
        return create

    return DummyVecEnv([factory(rank) for rank in range(n_envs)])


def make_ppo(train_env: DummyVecEnv, tensorboard_dir: Path,
             log_std_init: float = 0.0, seed: int = SEED) -> PPO:
    """Use the requested fixed PPO settings and separate 64x64 actor/critic."""
    policy_kwargs = {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
    if log_std_init != 0.0:
        policy_kwargs["log_std_init"] = float(log_std_init)
    return PPO(
        "MlpPolicy", train_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
        n_steps=N_STEPS, batch_size=256, n_epochs=10,
        tensorboard_log=str(tensorboard_dir), device="cpu", seed=seed,
        verbose=1,
    )


def evaluate_waypoints(model, env: MineUAVEnv, episodes: int,
                       seed: int = SEED + 10_000) -> dict:
    """Deterministic rollout on an independent env; never changes training state."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    successes = 0
    distances = []
    lengths = []
    completion_times = []
    reasons = {}
    base_env = env.unwrapped  # Monitor does not proxy environment-specific fields.
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        if base_env.render_mode == "human" and not base_env.viewer_is_running:
            break
        for step in range(base_env.max_episode_steps):
            if base_env.render_mode == "human" and not base_env.viewer_is_running:
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if base_env.render_mode == "human" and (step + 1) % 5 == 0:
                position = base_env.data.qpos[:3]
                print(f"t={base_env.data.time:.2f}s pos={np.round(position, 3)} "
                      f"distance={info['distance_m']:.3f}m", flush=True)
            if terminated or truncated:
                reason = info["termination_reason"]
                reasons[reason] = reasons.get(reason, 0) + 1
                lengths.append(step + 1)
                distances.append(info["distance_m"])
                if reason == "success":
                    successes += 1
                    completion_times.append((step + 1) * base_env.policy_dt)
                break
        else:
            raise RuntimeError("Evaluation episode exceeded the environment time limit")
        if base_env.render_mode == "human" and not base_env.viewer_is_running:
            break
    completed = len(lengths)
    return {
        "episodes": completed,
        "successes": successes,
        "success_rate": successes / completed if completed else None,
        "mean_final_distance_m": float(np.mean(distances)) if completed else None,
        "mean_episode_length": float(np.mean(lengths)) if completed else None,
        "mean_completion_time_s": (float(np.mean(completion_times))
                                   if completion_times else None),
        "termination_reasons": reasons,
    }


class SmokeCallback(BaseCallback):
    """Collect rollout metrics, evaluate periodically, and enforce a hard cap."""

    def __init__(self, eval_env: MineUAVEnv, eval_history_path: Path,
                 total_timesteps: int = TOTAL_TIMESTEPS):
        super().__init__()
        self.eval_env = eval_env
        self.eval_history_path = eval_history_path
        self.total_timesteps = total_timesteps
        self.completed_rollouts = 0
        self.rollout_history = []
        self.evaluation_history = []

    def _on_step(self) -> bool:
        return self.num_timesteps < self.total_timesteps

    def _on_rollout_end(self) -> None:
        self.completed_rollouts += 1
        episodes = list(self.model.ep_info_buffer or [])
        self.rollout_history.append({
            "timesteps": self.num_timesteps,
            "ep_rew_mean": (float(np.mean([episode["r"] for episode in episodes]))
                            if episodes else None),
            "ep_len_mean": (float(np.mean([episode["l"] for episode in episodes]))
                            if episodes else None),
            "episode_count_in_window": len(episodes),
        })
        if self.completed_rollouts % 2 == 0:
            stats = evaluate_waypoints(self.model, self.eval_env, episodes=5,
                                       seed=SEED + 10_000)
            stats["training_timesteps"] = self.num_timesteps
            self.evaluation_history.append(stats)
            with self.eval_history_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(stats) + "\n")
            self.logger.record("eval/success_rate", stats["success_rate"])
            self.logger.record("eval/mean_final_distance_m", stats["mean_final_distance_m"])
            self.logger.record("eval/mean_episode_length", stats["mean_episode_length"])


def run_smoke_training() -> dict:
    """Run exactly one capped 20k-sample smoke test and save its artifacts."""
    if MODEL_PATH.exists():
        raise FileExistsError(f"Preserving existing checkpoint: {MODEL_PATH}")
    # Model and mesh resources were prepared in earlier phases. Env creation
    # only reads them; no mesh converter or concurrent asset writer runs here.
    from control_allocator import MODEL
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_env = make_training_envs(MONITOR_DIR)
    eval_env = Monitor(MineUAVEnv(render_mode=None),
                       filename=str(MONITOR_DIR / "evaluation"),
                       info_keywords=("distance_m", "termination_reason"))
    start = time.perf_counter()
    try:
        model = make_ppo(train_env, TENSORBOARD_DIR)
        callback = SmokeCallback(eval_env, EVAL_HISTORY_PATH)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback,
                    tb_log_name="PPO", log_interval=1)
        duration_s = time.perf_counter() - start
        if model.num_timesteps != TOTAL_TIMESTEPS:
            raise RuntimeError(f"Training budget mismatch: {model.num_timesteps}")
        train_metrics = {key: float(model.logger.name_to_value[f"train/{key}"])
                         for key in TRAIN_METRICS}
        if not all(math.isfinite(value) for value in train_metrics.values()):
            raise RuntimeError(f"Non-finite PPO metrics: {train_metrics}")
        model.dump_logs(iteration=callback.completed_rollouts)
        model.save(MODEL_PATH)
        final_eval = evaluate_waypoints(model, eval_env, episodes=20)
        result = {
            "python": sys.version.split()[0], "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "n_envs": train_env.num_envs, "n_steps": N_STEPS,
            "rollout_buffer_size": N_STEPS * train_env.num_envs,
            "requested_timesteps": TOTAL_TIMESTEPS,
            "sampled_timesteps": model.num_timesteps,
            "completed_rollouts": callback.completed_rollouts,
            "updated_timesteps": callback.completed_rollouts * N_STEPS * train_env.num_envs,
            "training_duration_s": duration_s,
            "rollout_history": callback.rollout_history,
            "periodic_evaluations": callback.evaluation_history,
            "final_train_metrics": train_metrics,
            "final_evaluation": final_eval,
            "model_path": str(MODEL_PATH),
            "tensorboard_dir": str(TENSORBOARD_DIR),
            "monitor_dir": str(MONITOR_DIR),
        }
        REPORT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    print(json.dumps(run_smoke_training(), indent=2))
