"""One-seed PPO/PI observability diagnostic: only seven inputs become ten."""

import json
from pathlib import Path

from audit_ppo_robustness import MUJOCO_DIR
from ppo_pi_observable_env import MineUAVPIObservableEnv
from train_ppo_pi_lowstd import (output_paths as _base_output_paths,
                                 require_fresh_outputs,
                                 run_training as _base_run_training)


MODEL_PREFIX = "ppo_waypoint_pi_observable_lowstd"
TRACE_SCOPE = "ppo_pi_observability"


def output_paths(root: Path = MUJOCO_DIR) -> dict[str, Path]:
    return _base_output_paths(root, model_prefix=MODEL_PREFIX,
                              trace_scope=TRACE_SCOPE)


def require_fresh_observable_outputs(root: Path = MUJOCO_DIR) -> None:
    paths = output_paths(root)
    require_fresh_outputs(paths["model_dir"], paths["log_dir"],
                          paths["tensorboard_dir"], paths["trace_dir"],
                          paths["report"], model_prefix=MODEL_PREFIX)


def run_training(output_root: Path = MUJOCO_DIR) -> dict:
    return _base_run_training(
        output_root, env_type=MineUAVPIObservableEnv,
        model_prefix=MODEL_PREFIX, trace_scope=TRACE_SCOPE,
        expected_observation_shape=(10,),
        experiment_description="Reward V2 PPO with fixed PI and observed integral acceleration",
    )


if __name__ == "__main__":
    report = run_training()
    print(json.dumps({
        "actual_timesteps": report["actual_timesteps"],
        "benchmark_successes": report["milestones"]["100k"]
            ["evaluation"]["benchmark"]["successes"],
        "holdout_successes": report["milestones"]["100k"]
            ["evaluation"]["holdout"]["successes"],
        "report": str(output_paths()["report"]),
    }, indent=2), flush=True)
