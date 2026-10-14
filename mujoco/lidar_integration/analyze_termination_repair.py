"""Read-only audit of interrupted L9; no model loading, training or rollout."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_COMMIT = "c1c8d4f288a75a539b8721d4f991c613b830605f"
ORIGINAL_REPORT = ROOT / "mujoco/reports/mujoco_lidar_ppo_formal_seed0.json"
ORIGINAL_RAW = ROOT / "mujoco/reports/mujoco_lidar_ppo_formal_seed0_parts/run_01"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_sha(relative):
    result = subprocess.run(["git", "show", f"{BASE_COMMIT}:{relative}"], cwd=ROOT,
                            check=True, capture_output=True)
    return hashlib.sha256(result.stdout).hexdigest()


def audit_original_l9(*, raw_directory=ORIGINAL_RAW) -> dict:
    raw_directory = Path(raw_directory)
    if any((raw_directory / name).exists() for name in
           ("formal.complete.json", "formal_summary.json")):
        raise RuntimeError("original failed L9 must not acquire a completion marker")
    for relative in ("mujoco/reports/mujoco_lidar_ppo_formal_seed0.json",
                     "docs/MUJOCO_LIDAR_PPO_FORMAL_SEED0.md"):
        if _sha(ROOT / relative) != _base_sha(relative):
            raise RuntimeError("original L9 report/document identity changed")
    old = json.loads(ORIGINAL_REPORT.read_text())
    if old["overall_result"] != "FAIL" or old["test_episode_count"] != 0:
        raise RuntimeError("original failed L9 status or Test accounting changed")
    expected = old["raw_evidence_sha256"]
    actual = {path.name: _sha(path) for path in raw_directory.iterdir() if path.is_file()}
    if actual != expected:
        raise RuntimeError("original L9 evidence identity changed or incomplete")
    for section in old["runtime_source_sha256"].values():
        for name, digest in section.items():
            if _base_sha(f"mujoco/lidar_integration/{name}") != digest:
                raise RuntimeError("original runtime identity unavailable in base Git objects")
    prereg = json.loads((ROOT / "mujoco/reports/mujoco_lidar_ppo_training_prereg_seed0.json").read_text())
    checkpoints = {name: _sha(ROOT / "mujoco/rl/models" / name)
                   for name in prereg["historical_checkpoint_sha256"]}
    if checkpoints != prereg["historical_checkpoint_sha256"]:
        raise RuntimeError("historical checkpoint identity changed")
    if _sha(ROOT / "mujoco/models/mine_uav_dynamics_v2.xml") != old["original_mjcf_sha256"]:
        raise RuntimeError("historical physics model identity changed")
    episodes = []
    with (raw_directory / "train_episodes.jsonl").open() as handle:
        for line in handle:
            episodes.append(json.loads(line))
    counts = {key: Counter(row["outcome"] for row in episodes)[key] for key in
              ("success", "collision", "time_limit", "physical_failure", "sensor_failure")}
    if len(episodes) != 706 or counts != old["recorded_train_terminal_counts"]:
        raise RuntimeError("original completed-episode accounting mismatch")
    identities = {row["identity_sha256"] for row in prereg["scenario_manifest"]["train"]}
    if any(row["scenario_identity_sha256"] not in identities for row in episodes):
        raise RuntimeError("non-Train identity in original episodes")
    health = json.loads((raw_directory / "L9_HEALTH_STEP0100096.json").read_text())
    if health["actual_execution_step"] != 100096 or health["test_episode_count"] != 0:
        raise RuntimeError("original milestone identity mismatch")
    def mean(key):
        values = [row[key] for row in episodes]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("nonfinite original episode metric")
        return sum(values) / len(values)
    return {
        "overall_result": "FAIL", "completed_train_episodes": len(episodes),
        "terminal_counts": counts, "test_episode_count": 0,
        "original_raw_file_count": len(actual), "raw_evidence_sha256": actual,
        "original_sources_verified_against_base_git_objects": True,
        "original_source_sha256": old["runtime_source_sha256"],
        "historical_checkpoint_sha256": checkpoints,
        "original_mjcf_sha256": old["original_mjcf_sha256"],
        "specific_original_physical_trigger": "UNRESOLVED_NOT_RECORDED",
        "failed_transition_in_episode_counts": False,
        "mean_episode_return": mean("episode_return"),
        "mean_trajectory_length_m": mean("trajectory_length_m"),
        "mean_final_target_distance_m": mean("final_target_distance_m"),
        "mean_action_norm": mean("mean_action_norm"),
        "milestone": {key: health[key] for key in ("actual_execution_step", "completed_episodes",
            "terminal_counts", "ppo_diagnostics", "mean_action_norm", "mean_trajectory_length_m",
            "mean_final_target_distance_m", "motionless_episode_fraction")},
        "interpretation": "One partial seed, zero recorded successes; neither learned obstacle avoidance "
            "nor permanent PPO incapability is established. Single diagnostic sample is not a learning curve.",
        "next_diagnostic_priorities": ["canonical/raw terminal evidence including failed transitions",
            "success, collision, timeout and no-progress rates separately",
            "action norm, target progress, trajectory length and near-target braking",
            "value loss/explained variance and policy loss/KL/clip fraction across milestones"],
    }


if __name__ == "__main__":
    print(json.dumps(audit_original_l9(), indent=2, allow_nan=False))
