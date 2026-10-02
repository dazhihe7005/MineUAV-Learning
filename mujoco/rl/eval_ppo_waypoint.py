"""Deterministic checkpoint evaluation; add --human for the desktop viewer."""

import argparse
import json
import time
from pathlib import Path

from stable_baselines3 import PPO

from mine_uav_env import MineUAVEnv
from train_ppo_waypoint import MODEL_PATH, evaluate_waypoints


def run_ppo_evaluation(model_path: Path = MODEL_PATH, episodes: int = 1,
                       render_mode: str | None = None,
                       seed: int = 20270929,
                       reward_version: str = "v1",
                       target_distribution: str = "full") -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    model = PPO.load(model_path, device="cpu")
    env = MineUAVEnv(render_mode=render_mode, reward_version=reward_version,
                     target_distribution=target_distribution)
    try:
        result = evaluate_waypoints(model, env, episodes=episodes, seed=seed)
        result["reward_version"] = reward_version
        result["target_distribution"] = target_distribution
        if render_mode == "human" and env.viewer_is_running:
            time.sleep(2.0)
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the MineUAV PPO smoke checkpoint")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20270929)
    parser.add_argument("--human", action="store_true", help="Open MuJoCo desktop viewer")
    parser.add_argument("--reward-version", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--target-distribution", choices=("full", "local_a", "local_b"),
                        default="full")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    result = run_ppo_evaluation(args.model, episodes=args.episodes,
                                render_mode="human" if args.human else None,
                                seed=args.seed, reward_version=args.reward_version,
                                target_distribution=args.target_distribution)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
