"""Read-only, paired Reward V2 return audit for four fixed policies."""

import json
import math
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from mine_uav_env import MineUAVEnv
from test_env_scripted_policy import scripted_action


RL_DIR = Path(__file__).resolve().parent
REPORT_PATH = RL_DIR.parent / "reports" / "reward_v2_alignment_audit.json"
CURRICULUM_REPORT = RL_DIR.parent / "reports" / "ppo_waypoint_curriculum.json"
PPO_CHECKPOINT = RL_DIR / "models" / "ppo_waypoint_reward_v2_100k.zip"
COMPONENTS = ("progress", "action", "brake", "success", "failure", "total")
GAMMA = 0.99
RANDOM_SEED = 20263001


def fixed_waypoints(task: str) -> list[tuple[int, list[float]]]:
    if task not in ("local", "full"):
        raise ValueError("task must be 'local' or 'full'")
    report = json.loads(CURRICULUM_REPORT.read_text(encoding="utf-8"))
    episodes = report["baseline_v2_100k"][task]["episode_summaries"]
    result = [(int(ep["seed"]), ep["target_position_m"]) for ep in episodes]
    expected_first = 20261101 if task == "local" else 20261001
    if len(result) != 100 or [seed for seed, _ in result] != list(range(expected_first, expected_first + 100)):
        raise ValueError(f"{task} is not the fixed 100-waypoint evaluation set")
    for stage in "abc":
        other = [(int(ep["seed"]), ep["target_position_m"])
                 for ep in report["stages"][stage][task]["episode_summaries"]]
        if other != result:
            raise ValueError(f"Stage {stage} {task} waypoints differ")
    return result


def accumulate_reward_terms(rows: list[dict], gamma: float = GAMMA) -> dict:
    if not rows or not 0 < gamma <= 1:
        raise ValueError("Need reward rows and 0 < gamma <= 1")
    undiscounted = dict.fromkeys(COMPONENTS, 0.0)
    discounted = dict.fromkeys(COMPONENTS, 0.0)
    factor = 1.0
    for row in rows:
        if set(row) != set(COMPONENTS) or not all(math.isfinite(float(row[k])) for k in COMPONENTS):
            raise ValueError("Invalid Reward V2 breakdown")
        if not math.isclose(float(row["total"]),
                            sum(float(row[k]) for k in COMPONENTS[:-1]),
                            rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Reward V2 total differs from its components")
        for key in COMPONENTS:
            undiscounted[key] += float(row[key])
            discounted[key] += factor * float(row[key])
        factor *= gamma
    return {"undiscounted": undiscounted, "discounted": discounted}


def evaluate_episode_return(env: MineUAVEnv, policy, target, seed: int,
                            gamma: float = GAMMA) -> dict:
    if env.reward_version != "v2" or env.render_mode is not None:
        raise ValueError("Return audit requires a headless Reward V2 environment")
    observation, _ = env.reset(seed=seed, options={"target_position": target})
    rows = []
    for index in range(env.max_episode_steps):
        action = np.asarray(policy(observation), dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise ValueError("Policy action must be a normalized finite four-vector")
        observation, reward, terminated, truncated, info = env.step(action)
        row = info["reward_breakdown"]
        if not math.isclose(float(reward), float(row["total"]), rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("Environment reward differs from breakdown total")
        rows.append(row)
        if terminated or truncated:
            sums = accumulate_reward_terms(rows, gamma)
            return {
                "seed": seed, "target_position_m": np.asarray(target, dtype=float).tolist(),
                "success": info["termination_reason"] == "success",
                "termination_reason": info["termination_reason"],
                "episode_length": index + 1,
                "final_distance_m": float(info["distance_m"]),
                "final_speed_m_s": float(info["speed_m_s"]),
                **sums,
            }
    raise RuntimeError("Episode exceeded environment time limit")


def summarize_returns(episodes: list[dict]) -> dict:
    if not episodes:
        raise ValueError("Need evaluated episodes")
    return {
        "episodes": len(episodes),
        "successes": sum(ep["success"] for ep in episodes),
        "mean_episode_length": float(np.mean([ep["episode_length"] for ep in episodes])),
        "mean_final_distance_m": float(np.mean([ep["final_distance_m"] for ep in episodes])),
        "returns": {
            kind: {
                key: {"mean": float(np.mean([ep[kind][key] for ep in episodes])),
                      "median": float(np.median([ep[kind][key] for ep in episodes]))}
                for key in COMPONENTS}
            for kind in ("undiscounted", "discounted")
        },
        "episode_results": episodes,
    }


def paired_comparison(reference: list[dict], candidate: list[dict], kind: str,
                      bootstrap_seed: int) -> dict:
    if (kind not in ("undiscounted", "discounted") or len(reference) != len(candidate)
            or any((a["seed"], a["target_position_m"]) !=
                   (b["seed"], b["target_position_m"])
                   for a, b in zip(reference, candidate))):
        raise ValueError("Paired returns require identical seed/target order")
    differences = np.asarray([a[kind]["total"] - b[kind]["total"]
                              for a, b in zip(reference, candidate)])
    rng = np.random.default_rng(bootstrap_seed)
    draws = rng.integers(0, len(differences), size=(10_000, len(differences)))
    bootstrap_means = differences[draws].mean(axis=1)
    return {
        "mean_scripted_minus_candidate": float(differences.mean()),
        "median_scripted_minus_candidate": float(np.median(differences)),
        "scripted_higher_episode_fraction": float(np.mean(differences > 0)),
        "paired_bootstrap_95pct_mean_interval": np.quantile(
            bootstrap_means, [0.025, 0.975]).tolist(),
    }


def audit_alignment() -> dict:
    if REPORT_PATH.exists():
        raise FileExistsError(f"Preserving existing audit: {REPORT_PATH}")
    if not PPO_CHECKPOINT.is_file():
        raise FileNotFoundError(PPO_CHECKPOINT)
    model = PPO.load(PPO_CHECKPOINT, device="cpu")
    env = MineUAVEnv(render_mode=None, reward_version="v2")
    report = {
        "experiment": "Reward V2 policy-return alignment audit; no training",
        "reward_version": "v2", "gamma": GAMMA,
        "ppo_checkpoint": str(PPO_CHECKPOINT), "random_seed_base": RANDOM_SEED,
        "tasks": {},
    }
    try:
        for task in ("local", "full"):
            waypoints = fixed_waypoints(task)
            task_result = {"fixed_waypoints": [{"seed": seed, "target_position_m": target}
                                                for seed, target in waypoints],
                           "policies": {}, "paired_comparisons": {}}
            for policy_name in ("scripted", "ppo_v2", "zero", "random"):
                episodes = []
                for index, (seed, target) in enumerate(waypoints):
                    if policy_name == "scripted":
                        policy = scripted_action
                    elif policy_name == "ppo_v2":
                        policy = lambda obs: model.predict(obs, deterministic=True)[0]
                    elif policy_name == "zero":
                        policy = lambda obs: np.zeros(4, dtype=np.float32)
                    else:
                        rng = np.random.default_rng(RANDOM_SEED + seed)
                        policy = lambda obs: rng.uniform(-1, 1, size=4).astype(np.float32)
                    episodes.append(evaluate_episode_return(env, policy, target, seed, GAMMA))
                task_result["policies"][policy_name] = summarize_returns(episodes)
                print(f"Part A {task} {policy_name}: "
                      f"{task_result['policies'][policy_name]['successes']}/100 success",
                      flush=True)
            reference = task_result["policies"]["scripted"]["episode_results"]
            for name in ("ppo_v2", "zero", "random"):
                candidate = task_result["policies"][name]["episode_results"]
                task_result["paired_comparisons"][name] = {
                    kind: paired_comparison(reference, candidate, kind,
                                            bootstrap_seed=RANDOM_SEED + index)
                    for index, kind in enumerate(("undiscounted", "discounted"))
                }
            report["tasks"][task] = task_result
        REPORT_PATH.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        return report
    finally:
        env.close()


if __name__ == "__main__":
    result = audit_alignment()
    print(f"Saved {REPORT_PATH}; policies={list(result['tasks']['local']['policies'])}")
