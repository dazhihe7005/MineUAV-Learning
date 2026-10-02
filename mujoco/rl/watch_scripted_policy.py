"""Watch the already-validated waypoint heuristic in the live MuJoCo viewer.

Run from the project root with `.venv/bin/python mujoco/rl/watch_scripted_policy.py`.
The default repeats until the viewer is closed or Ctrl+C; --episodes 1 runs
one complete episode for a finite demonstration.
"""

import argparse
import math
import time

import numpy as np

from mine_uav_env import MineUAVEnv
from test_env_scripted_policy import scripted_action


WAYPOINT = (2.0, 2.0, 1.5)
STATUS_EVERY_POLICY_STEPS = 5  # 25 Hz / 5 = 5 Hz terminal output.


def _tilt_degrees(quaternion: np.ndarray) -> float:
    cosine = float(np.clip(1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2), -1, 1))
    return math.degrees(math.acos(cosine))


def _status_line(env: MineUAVEnv, observation: np.ndarray, info: dict,
                 episode_status: str) -> str:
    p = env.data.qpos[:3]
    v = env.data.qvel[:3]
    target = env.target_position
    command = info["velocity_command_m_s"]
    return (
        f"t={env.data.time:5.2f}s "
        f"pos=[{p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f}] "
        f"target=[{target[0]:+.2f},{target[1]:+.2f},{target[2]:+.2f}] "
        f"vel=[{v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}] "
        f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f},"
        f"{info['yaw_rate_command_rad_s']:+.2f}] "
        f"distance={info['distance_m']:.3f}m yaw_error={observation[6]:+.3f}rad "
        f"status={episode_status}"
    )


def run_episode(env: MineUAVEnv, status_sink=print) -> dict:
    """Run one fixed-target Gym episode; return measurements, not a new policy."""
    observation, _ = env.reset(seed=0, options={"target_position": WAYPOINT})
    start_position = env.data.qpos[:3].copy()
    peak_speed = 0.0
    peak_tilt = 0.0
    reason = "viewer_closed" if env.render_mode == "human" and not env.viewer_is_running else None
    steps = 0
    last_info = None
    while reason is None and steps < env.max_episode_steps:
        if env.render_mode == "human" and not env.viewer_is_running:
            reason = "viewer_closed"
            break
        action = scripted_action(observation)
        observation, _, terminated, truncated, info = env.step(action)
        steps += 1
        last_info = info
        peak_speed = max(peak_speed, float(np.linalg.norm(env.data.qvel[:3])))
        peak_tilt = max(peak_tilt, _tilt_degrees(env.data.qpos[3:7]))
        status = info["termination_reason"] or "flying"
        if steps % STATUS_EVERY_POLICY_STEPS == 0 or terminated or truncated:
            status_sink(_status_line(env, observation, info, status))
        if env.render_mode == "human" and not env.viewer_is_running:
            reason = "viewer_closed"
        elif terminated or truncated:
            reason = info["termination_reason"]
    if reason is None:
        reason = "time_limit"
    final_position = env.data.qpos[:3].copy()
    return {
        "reason": reason,
        "start_position_m": start_position.tolist(),
        "target_position_m": env.target_position.tolist(),
        "final_position_m": final_position.tolist(),
        "final_velocity_m_s": env.data.qvel[:3].tolist(),
        "final_distance_m": float(np.linalg.norm(env.target_position - final_position)),
        "sim_time_s": float(env.data.time),
        "policy_steps": steps,
        "peak_speed_m_s": peak_speed,
        "peak_tilt_deg": peak_tilt,
        "last_command_m_s": (last_info["velocity_command_m_s"].tolist()
                              if last_info is not None else [0.0, 0.0, 0.0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch MineUAV scripted waypoint flight")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes; 0 repeats until viewer closes (default)")
    parser.add_argument("--pause-seconds", type=float, default=2.0,
                        help="Pause after each completed episode (default: 2)")
    args = parser.parse_args()
    if args.episodes < 0 or not math.isfinite(args.pause_seconds) or args.pause_seconds < 0:
        parser.error("episodes must be nonnegative and pause-seconds finite/nonnegative")
    env = MineUAVEnv(render_mode="human")
    try:
        episode_number = 0
        while args.episodes == 0 or episode_number < args.episodes:
            episode_number += 1
            print(f"Episode {episode_number}: start=(0,0,1), target={WAYPOINT}", flush=True)
            result = run_episode(env, status_sink=lambda line: print(line, flush=True))
            print(f"Episode {episode_number}: {result['reason']}; "
                  f"final={np.round(result['final_position_m'], 3).tolist()}, "
                  f"distance={result['final_distance_m']:.3f} m, "
                  f"speed={np.linalg.norm(result['final_velocity_m_s']):.3f} m/s, "
                  f"peak_tilt={result['peak_tilt_deg']:.1f} deg", flush=True)
            if result["reason"] == "viewer_closed" or not env.viewer_is_running:
                break
            deadline = time.monotonic() + args.pause_seconds
            while env.viewer_is_running and time.monotonic() < deadline:
                time.sleep(min(0.1, deadline - time.monotonic()))
            if not env.viewer_is_running:
                break
    except KeyboardInterrupt:
        print("Stopped by Ctrl+C", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
