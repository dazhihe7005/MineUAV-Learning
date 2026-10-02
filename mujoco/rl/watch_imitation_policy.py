"""Watch the supervised actor fly an original full waypoint in MuJoCo.

No scripted action is called in this runner. The viewer uses the same
MjModel/MjData as the Gymnasium environment and follows its real-time pace.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from mine_uav_env import MineUAVEnv
from scripted_imitation_sanity import ActorMeanMLP, ImitationPolicy, MODEL_PATH


def load_imitation_policy(path: Path) -> ImitationPolicy:
    actor = ActorMeanMLP()
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return ImitationPolicy(actor)


def run_episode(env: MineUAVEnv, policy: ImitationPolicy, seed: int,
                status_sink=print) -> dict:
    observation, _ = env.reset(seed=seed)
    steps = 0
    last_info = None
    reason = None
    for _ in range(env.max_episode_steps):
        if env.render_mode == "human" and not env.viewer_is_running:
            reason = "viewer_closed"
            break
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        last_info = info
        if steps % 5 == 0 or terminated or truncated:
            command = info["velocity_command_m_s"]
            status_sink(
                f"t={env.data.time:.2f}s pos={np.round(env.data.qpos[:3], 3)} "
                f"target={np.round(env.target_position, 3)} "
                f"distance={info['distance_m']:.3f}m "
                f"speed={info['speed_m_s']:.3f}m/s "
                f"cmd={np.round(command, 3)} status={info['termination_reason'] or 'flying'}")
        if terminated or truncated:
            reason = info["termination_reason"]
            break
    if reason is None:
        reason = "time_limit"
    return {
        "seed": seed, "reason": reason, "policy_steps": steps,
        "target_position_m": env.target_position.tolist(),
        "final_position_m": env.data.qpos[:3].tolist(),
        "final_speed_m_s": float(np.linalg.norm(env.data.qvel[:3])),
        "final_distance_m": float(np.linalg.norm(env.target_position - env.data.qpos[:3])),
        "last_command_m_s": (last_info["velocity_command_m_s"].tolist()
                              if last_info is not None else [0.0, 0.0, 0.0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--pause-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    if not math.isfinite(args.pause_seconds) or args.pause_seconds < 0:
        parser.error("--pause-seconds must be finite and nonnegative")
    policy = load_imitation_policy(args.model)
    env = MineUAVEnv(render_mode="human", reward_version="v2",
                     target_distribution="full")
    try:
        result = run_episode(env, policy, args.seed)
        print(json.dumps(result, indent=2), flush=True)
        if env.viewer_is_running:
            time.sleep(args.pause_seconds)
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
