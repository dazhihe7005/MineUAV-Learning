"""Explicit one-shot seed0 formal PPO entry; import never starts training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
from pathlib import Path

from .ppo_formal_scheduler import load_l6_prereg, require_memory
from .ppo_training_orchestration import FormalPPOTrainingDriver
from .ppo_formal_seed0_health import L9FormalPPOTrainingDriver
from .ppo_training_runner import protocol_hash, _historical_model_hashes
from .scanner import PATTERN_SHA256, UPSTREAM_COMMIT, PATTERN_REL, validate_upstream
from .scene import BASE_MODEL, BASE_SHA256


FROZEN_L8_PROTOCOL_SHA256 = "fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6"
FROZEN_L8_REPORT_SHA256 = "a08df1131aaac650482f64e302c5717fad3cf599062fa5356376e02ac6f69c78"
FORMAL_ACTUAL_TIMESTEPS = 1_000_064
FORMAL_OUTPUT_DIR = Path(__file__).parents[1] / "reports/mujoco_lidar_ppo_formal_seed0_parts/run_01"
REPORTS = Path(__file__).parents[1] / "reports"
MIN_DISK_FREE_BYTES = 5 * 1024**3
EXPECTED_SOFTWARE = {"mujoco": "3.14.0", "stable-baselines3": "2.9.0",
                     "torch": "2.14.0+cpu", "gymnasium": "1.3.0", "numpy": "2.2.6"}


def preflight_formal_seed0(*, upstream_root: Path, output_dir: Path,
                           expected_protocol_sha256: str, seed: int,
                           actual_timesteps: int, authorized: bool) -> dict:
    if authorized is not True:
        raise PermissionError("formal seed0 training requires explicit authorization flag")
    if type(seed) is not int or seed != 0:
        raise ValueError("only seed0 is authorized")
    if type(actual_timesteps) is not int or actual_timesteps != FORMAL_ACTUAL_TIMESTEPS:
        raise ValueError("formal actual timestep budget must be exactly 1000064")
    if expected_protocol_sha256 != FROZEN_L8_PROTOCOL_SHA256:
        raise ValueError("requested formal protocol hash differs from frozen L8")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("existing formal output or attempt cannot be overwritten/retried")
    require_memory()
    disk_probe = output_dir.parent
    while not disk_probe.exists():
        disk_probe = disk_probe.parent
    disk_free = shutil.disk_usage(disk_probe).free
    if disk_free < MIN_DISK_FREE_BYTES:
        raise OSError("insufficient disk space for one formal seed0 run")
    prereg = load_l6_prereg(REPORTS / "mujoco_lidar_ppo_training_prereg_seed0.json")
    report_path = REPORTS / "mujoco_lidar_ppo_orchestration.json"
    if hashlib.sha256(report_path.read_bytes()).hexdigest() != FROZEN_L8_REPORT_SHA256:
        raise ValueError("L8 acceptance report identity mismatch")
    report = json.loads(report_path.read_text())
    if (report.get("TRAINING_DRIVER_READY") != "YES" or
            report.get("FORMAL_SEED0_READY") != "YES" or
            report.get("l8_formal_execution_protocol_sha256") != FROZEN_L8_PROTOCOL_SHA256 or
            report.get("scenario_split_counts") != {"train": 12, "validation": 6, "test": 6}):
        raise ValueError("L8 formal training readiness evidence is not valid")
    driver = FormalPPOTrainingDriver(REPORTS / "mujoco_lidar_ppo_training_prereg_seed0.json",
                                     upstream_root, output_dir)
    if (driver.protocol_sha256 != FROZEN_L8_PROTOCOL_SHA256 or
            prereg["historical_checkpoint_sha256"] != _historical_model_hashes()):
        raise ValueError("frozen protocol or historical model identity changed")
    if hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() != BASE_SHA256:
        raise ValueError("original 7kg MJCF identity mismatch")
    pinned = validate_upstream(upstream_root)
    if hashlib.sha256((pinned / PATTERN_REL).read_bytes()).hexdigest() != PATTERN_SHA256:
        raise ValueError("MID360 scan pattern identity mismatch")
    versions = {name: importlib.metadata.version(name) for name in EXPECTED_SOFTWARE}
    if versions != EXPECTED_SOFTWARE:
        raise ValueError("runtime dependency versions differ from verified L8 environment")
    return {
        "seed": 0, "workers": 1, "actual_timesteps": FORMAL_ACTUAL_TIMESTEPS,
        "l8_protocol_sha256": protocol_hash(driver.protocol),
        "l8_acceptance_report_sha256": FROZEN_L8_REPORT_SHA256,
        "scenario_manifest_sha256": prereg["scenario_manifest_sha256"],
        "train_scenes": 12, "validation_scenes": 6, "test_episode_count": 0,
        "reward_version": driver.protocol["reward_version"],
        "network_implementation_sha256": prereg["network_implementation_sha256"],
        "original_mjcf_sha256": BASE_SHA256,
        "historical_checkpoint_sha256": _historical_model_hashes(),
        "upstream_commit": UPSTREAM_COMMIT, "upstream_pattern_sha256": PATTERN_SHA256,
        "software_versions": versions, "disk_free_bytes": disk_free,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One-shot frozen LiDAR PPO formal seed0 training")
    parser.add_argument("--formal-seed0", action="store_true", required=True)
    parser.add_argument("--authorize-formal-training", action="store_true", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--actual-timesteps", type=int, required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args(argv)
    preflight_formal_seed0(upstream_root=args.upstream, output_dir=FORMAL_OUTPUT_DIR,
                           expected_protocol_sha256=args.expected_protocol_sha256,
                           seed=args.seed, actual_timesteps=args.actual_timesteps,
                           authorized=args.authorize_formal_training)
    driver = L9FormalPPOTrainingDriver(REPORTS / "mujoco_lidar_ppo_training_prereg_seed0.json",
                                       args.upstream, FORMAL_OUTPUT_DIR)
    result = driver.run_formal(allow_formal_training=True)
    print(json.dumps({"actual_timesteps": result["training_actual_timesteps"],
                      "event_count": result["event_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
