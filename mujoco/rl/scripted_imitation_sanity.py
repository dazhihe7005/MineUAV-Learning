"""Supervised 7→64→64→4 scripted-policy representation sanity check.

This script never optimizes reward and never trains PPO. Dataset episodes
are split before training, and evaluation uses unseen fixed waypoints.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from curriculum_diagnostics import evaluate_curriculum
from mine_uav_env import MineUAVEnv
from reward_v2_diagnostics import EVAL_SEED
from test_env_scripted_policy import scripted_action
from train_ppo_curriculum import LOCAL_EVAL_SEED


RL_DIR = Path(__file__).resolve().parent
REPORT_PATH = RL_DIR.parent / "reports" / "scripted_imitation_sanity.json"
MODEL_PATH = RL_DIR / "models" / "scripted_imitation_mlp_64x64.pt"
TRAIN_DATA_PATH = RL_DIR / "datasets" / "scripted_imitation_train.npz"
VALIDATION_DATA_PATH = RL_DIR / "datasets" / "scripted_imitation_validation.npz"
SUPERVISED_SEED = 20263002
TRAIN_EPISODES_PER_TASK = 200
VALIDATION_EPISODES_PER_TASK = 50
MAX_EPOCHS = 100
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
PATIENCE_EPOCHS = 12


class ActorMeanMLP(nn.Module):
    """SB3's default tanh actor hidden layers and linear Gaussian-mean head."""

    def __init__(self):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(7, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.action_net = nn.Linear(64, 4)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.action_net(self.policy_net(observations))


class ImitationPolicy:
    """Deterministic mean action, clipped exactly at the existing Box boundary."""

    def __init__(self, actor: ActorMeanMLP):
        self.actor = actor.cpu().eval()

    def predict(self, observation, deterministic: bool = True):
        if not deterministic:
            raise ValueError("Supervised imitation policy is deterministic only")
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (7,) or not np.isfinite(obs).all():
            raise ValueError("Expected finite seven-dimensional observation")
        with torch.no_grad():
            mean = self.actor(torch.from_numpy(obs)).numpy()
        return np.clip(mean, -1.0, 1.0).astype(np.float32), None


def collect_success_episodes(task: str, seed_start: int,
                             required_successes: int, max_attempts: int) -> dict:
    """Keep whole successful scripted trajectories; never split their steps."""
    if (task not in ("local", "full") or required_successes <= 0
            or max_attempts < required_successes):
        raise ValueError("Invalid task or scripted collection budget")
    distribution = "local_a" if task == "local" else "full"
    env = MineUAVEnv(render_mode=None, reward_version="v2",
                     target_distribution=distribution)
    observations, actions, episode_ids, episode_seeds = [], [], [], []
    failures = []
    attempts = 0
    try:
        for offset in range(max_attempts):
            if len(episode_seeds) >= required_successes:
                break
            seed = seed_start + offset
            observation, _ = env.reset(seed=seed)
            episode_observations, episode_actions = [], []
            for _ in range(env.max_episode_steps):
                action = scripted_action(observation)
                episode_observations.append(observation.copy())
                episode_actions.append(action.copy())
                observation, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Scripted dataset episode exceeded the time limit")
            attempts += 1
            if info["termination_reason"] == "success":
                observations.extend(episode_observations)
                actions.extend(episode_actions)
                episode_ids.extend([seed] * len(episode_actions))
                episode_seeds.append(seed)
            else:
                failures.append({"seed": seed, "reason": info["termination_reason"]})
        if len(episode_seeds) != required_successes:
            raise RuntimeError(f"Only {len(episode_seeds)} successful {task} trajectories "
                               f"from {attempts} attempts")
        return {
            "task": task, "seed_start": seed_start,
            "attempts": attempts, "successes": len(episode_seeds),
            "failed_attempts": failures, "episode_seeds": episode_seeds,
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        }
    finally:
        env.close()


def train_imitation_model(train_x: np.ndarray, train_y: np.ndarray,
                          val_x: np.ndarray, val_y: np.ndarray,
                          max_epochs: int = MAX_EPOCHS,
                          batch_size: int = BATCH_SIZE) -> tuple[ActorMeanMLP, list[dict], dict]:
    if (train_x.ndim != 2 or train_x.shape[1] != 7 or train_y.shape != (len(train_x), 4)
            or val_x.ndim != 2 or val_x.shape[1] != 7 or val_y.shape != (len(val_x), 4)
            or min(len(train_x), len(val_x), max_epochs, batch_size) <= 0
            or not all(np.isfinite(array).all() for array in (train_x, train_y, val_x, val_y))):
        raise ValueError("Need finite, nonempty 7D observation / 4D action arrays")
    torch.manual_seed(SUPERVISED_SEED)
    rng = np.random.default_rng(SUPERVISED_SEED)
    actor = ActorMeanMLP().cpu()
    optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    train_obs = torch.from_numpy(np.asarray(train_x, dtype=np.float32))
    train_act = torch.from_numpy(np.asarray(train_y, dtype=np.float32))
    val_obs = torch.from_numpy(np.asarray(val_x, dtype=np.float32))
    val_act = torch.from_numpy(np.asarray(val_y, dtype=np.float32))
    history = []
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    unimproved = 0
    for epoch in range(1, max_epochs + 1):
        actor.train()
        for indices in np.array_split(rng.permutation(len(train_obs)),
                                      math.ceil(len(train_obs) / batch_size)):
            optimizer.zero_grad()
            prediction = actor(train_obs[indices])
            loss = torch.mean((prediction - train_act[indices]) ** 2)
            loss.backward()
            optimizer.step()
        actor.eval()
        with torch.no_grad():
            train_mse = float(torch.mean((actor(train_obs) - train_act) ** 2))
            validation_mse = float(torch.mean((actor(val_obs) - val_act) ** 2))
        if not math.isfinite(train_mse) or not math.isfinite(validation_mse):
            raise RuntimeError("Non-finite supervised MSE")
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "validation_mse": validation_mse})
        if validation_mse < best_loss - 1e-7:
            best_loss = validation_mse
            best_epoch = epoch
            best_state = copy.deepcopy(actor.state_dict())
            unimproved = 0
        else:
            unimproved += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"Imitation epoch {epoch}: train={train_mse:.6g}, "
                  f"validation={validation_mse:.6g}", flush=True)
        if unimproved >= PATIENCE_EPOCHS:
            break
    actor.load_state_dict(best_state)
    actor.eval()
    return actor, history, {"epoch": best_epoch, "validation_mse": best_loss}


def offline_action_metrics(predicted: np.ndarray, scripted: np.ndarray) -> dict:
    predicted = np.asarray(predicted, dtype=float)
    scripted = np.asarray(scripted, dtype=float)
    if (predicted.shape != scripted.shape or predicted.ndim != 2
            or predicted.shape[1] != 4 or len(predicted) == 0
            or not np.isfinite(predicted).all() or not np.isfinite(scripted).all()):
        raise ValueError("Need matched finite N×4 action arrays")
    error = predicted - scripted
    correlations = []
    for axis in range(4):
        a, b = predicted[:, axis], scripted[:, axis]
        correlations.append(float(np.corrcoef(a, b)[0, 1])
                            if np.std(a) > 1e-12 and np.std(b) > 1e-12 else None)
    return {
        "samples": len(predicted),
        "overall_mse": float(np.mean(error ** 2)),
        "mse_per_axis": np.mean(error ** 2, axis=0).tolist(),
        "correlation_per_axis": correlations,
        "max_absolute_error": float(np.max(np.abs(error))),
        "max_absolute_error_per_axis": np.max(np.abs(error), axis=0).tolist(),
        "raw_output_outside_action_box_fraction": float(np.mean(np.abs(predicted) > 1)),
    }


def _combine_datasets(records: list[dict]) -> dict:
    return {
        "observations": np.concatenate([row["observations"] for row in records]),
        "actions": np.concatenate([row["actions"] for row in records]),
        "episode_ids": np.concatenate([row["episode_ids"] for row in records]),
        "task_labels": np.concatenate([
            np.full(len(row["episode_ids"]), row["task"], dtype="U5")
            for row in records]),
    }


def run_imitation_sanity() -> dict:
    for path in (REPORT_PATH, MODEL_PATH, TRAIN_DATA_PATH, VALIDATION_DATA_PATH):
        if path.exists():
            raise FileExistsError(f"Preserving existing imitation artifact: {path}")
    from reward_v2_alignment_audit import fixed_waypoints
    train_records = [
        collect_success_episodes("local", 20270001, TRAIN_EPISODES_PER_TASK,
                                 TRAIN_EPISODES_PER_TASK * 2),
        collect_success_episodes("full", 20280001, TRAIN_EPISODES_PER_TASK,
                                 TRAIN_EPISODES_PER_TASK * 2),
    ]
    val_records = [
        collect_success_episodes("local", 20290001, VALIDATION_EPISODES_PER_TASK,
                                 VALIDATION_EPISODES_PER_TASK * 2),
        collect_success_episodes("full", 20300001, VALIDATION_EPISODES_PER_TASK,
                                 VALIDATION_EPISODES_PER_TASK * 2),
    ]
    train = _combine_datasets(train_records)
    validation = _combine_datasets(val_records)
    train_ids = set(train["episode_ids"].tolist())
    val_ids = set(validation["episode_ids"].tolist())
    fixed_ids = {seed for task in ("local", "full") for seed, _ in fixed_waypoints(task)}
    if train_ids & val_ids or train_ids & fixed_ids or val_ids & fixed_ids:
        raise RuntimeError("Scripted dataset episode leakage")
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(TRAIN_DATA_PATH, **train)
    np.savez_compressed(VALIDATION_DATA_PATH, **validation)
    print(f"Scripted dataset: train {len(train['observations'])} steps / "
          f"{len(train_ids)} episodes; validation {len(validation['observations'])} "
          f"steps / {len(val_ids)} episodes", flush=True)
    actor, history, selection = train_imitation_model(
        train["observations"], train["actions"],
        validation["observations"], validation["actions"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(actor.state_dict(), MODEL_PATH)
    with torch.no_grad():
        predicted = actor(torch.from_numpy(validation["observations"])).numpy()
    offline = offline_action_metrics(predicted, validation["actions"])
    policy = ImitationPolicy(actor)
    local_env = MineUAVEnv(render_mode=None, reward_version="v2",
                           target_distribution="local_a")
    full_env = MineUAVEnv(render_mode=None, reward_version="v2",
                          target_distribution="full")
    try:
        print("Evaluating imitation MLP on 100 unseen local waypoints", flush=True)
        local = evaluate_curriculum(policy, local_env, episodes=100,
                                    seed=LOCAL_EVAL_SEED)
        print("Evaluating imitation MLP on 100 unseen full waypoints", flush=True)
        full = evaluate_curriculum(policy, full_env, episodes=100,
                                   seed=EVAL_SEED)
        for task, result in (("local", local), ("full", full)):
            expected = fixed_waypoints(task)
            actual = [(ep["seed"], ep["target_position_m"])
                      for ep in result["episode_summaries"]]
            if actual != expected:
                raise RuntimeError(f"Imitation {task} evaluation targets differ")
        report = {
            "experiment": "Scripted policy representation sanity check; supervised MSE only",
            "environment_reward_version": "v2",
            "model_path": str(MODEL_PATH),
            "train_dataset_path": str(TRAIN_DATA_PATH),
            "validation_dataset_path": str(VALIDATION_DATA_PATH),
            "network": "7→64 Tanh→64 Tanh→4 Linear Gaussian-mean analogue",
            "supervised_seed": SUPERVISED_SEED,
            "training_config": {"loss": "MSE", "optimizer": "Adam", "learning_rate": LEARNING_RATE,
                                "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                                "early_stopping_patience": PATIENCE_EPOCHS},
            "dataset": {
                "train_steps": len(train["observations"]),
                "validation_steps": len(validation["observations"]),
                "train_episodes": len(train_ids), "validation_episodes": len(val_ids),
                "train_episode_ids": sorted(train_ids),
                "validation_episode_ids": sorted(val_ids),
                "collections": {
                    "train": [{key: value for key, value in row.items()
                               if key not in ("observations", "actions", "episode_ids")}
                              for row in train_records],
                    "validation": [{key: value for key, value in row.items()
                                    if key not in ("observations", "actions", "episode_ids")}
                                   for row in val_records],
                },
            },
            "loss_history": history,
            "selected_checkpoint": selection,
            "lowest_observed_validation_epoch": min(
                history, key=lambda row: row["validation_mse"])["epoch"],
            "offline_validation": offline,
            "closed_loop": {"local": local, "full": full},
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        print(f"Imitation closed-loop: local {local['successes']}/100, "
              f"full {full['successes']}/100 success", flush=True)
        return report
    finally:
        local_env.close()
        full_env.close()


if __name__ == "__main__":
    result = run_imitation_sanity()
    print(f"Saved {REPORT_PATH}; validation MSE={result['offline_validation']['overall_mse']:.6g}")
