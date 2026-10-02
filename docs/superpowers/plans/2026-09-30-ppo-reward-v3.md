# PPO Reward V3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exactly one distance cost to Reward V2 and compare a separate, from-scratch 100k PPO run with the existing 100k baseline and V2 runs.

**Architecture:** Add an opt-in `v3` reward branch to the existing Gymnasium environment, preserving the V1/V2 branches. Reuse the fixed PPO factory and deterministic diagnostic evaluator, but write all V3 checkpoints, Monitor logs, TensorBoard events, and report under new names. Reuse the previously generated baseline/V2 100-waypoint evaluations only after checking seeds and target coordinates against V3.

**Tech Stack:** Python, MuJoCo, Gymnasium, Stable-Baselines3, NumPy, TensorBoard.

**Spec:** User's Reward V3 request in this conversation, 2026-09-30.

## Global Constraints

- Reward V3 = V2 plus only `r_distance = -0.02 * current_distance`; `lambda_brake = 0.5` stays fixed.
- No changes to observation/action, mapping, termination, limit, controller, physics/control/policy timing, PPO network/hyperparameters, or vector env settings.
- Train from scratch on CPU with 8 × 256 transitions per full rollout; stop at 100,352 timesteps and save 20k/50k/100k updated checkpoints.
- Evaluate each checkpoint on seeds 20261001–20261100 using deterministic actions; compare performance metrics, not raw reward, across versions.
- Preserve all original and V2 artifacts. This project is not a Git repository, so commits and worktrees do not apply.

## Review Focus

- Reward V1 and V2 remain unchanged after adding V3: Task 1 regression tests.
- V3 breakdown contains exactly the six requested terms plus total and uses post-step distance: Task 1 arithmetic tests.
- Undefined near-target speed is not logged as zero: Task 2 callback test.
- Checkpoints are after complete PPO updates and cannot overwrite other runs: Task 2 tests and artifact inspection.
- Baseline/V2/V3 waypoint coordinates are identical by seed, and raw returns are not used for cross-reward comparison: Task 3 report verification.

---

### Task 1: Opt-in Reward V3

**Files:** Modify `mujoco/rl/mine_uav_env.py` and `mujoco/rl/test_reward_v2.py`; create `mujoco/rl/test_reward_v3.py`.

**Interfaces:** `MineUAVEnv(reward_version="v3")`; breakdown keys `progress`, `distance`, `action`, `brake`, `success`, `failure`, `total`.

- [x] Write tests for V3 arithmetic, breakdown, boundaries and V1/V2 regression; change invalid-version test from `v3` to `v4`.
- [x] Run tests and observe missing V3 behavior.
- [x] Add only the distance term and V3 option.
- [x] Run tests and Gymnasium check_env.

### Task 2: Separate V3 training flow

**Files:** Create `mujoco/rl/train_ppo_reward_v3.py` and `mujoco/rl/test_train_ppo_reward_v3.py`; modify `mujoco/rl/eval_ppo_waypoint.py`.

**Interfaces:** V3 train script reuses `make_ppo`, `make_training_envs`, and `evaluate_diagnostics`; Viewer CLI accepts `--reward-version v3`.

- [x] Write failing schedule, artifact-protection, reward selection and CLI tests.
- [x] Run tests to confirm missing V3 flow.
- [x] Implement V3 callback and run function with independent paths, fixed schedule and comparison provenance checks.
- [x] Run tests and full RL regression suite.

### Task 3: Train, evaluate, inspect

**Files:** New V3 model ZIPs, report JSON, Monitor/TensorBoard files; update `mujoco/rl/README.md`.

- [x] Verify no existing V3 artifacts and both prior reports/checkpoints exist.
- [x] Train one fresh PPO curve through 100,352 timesteps; evaluate 100 fixed waypoints at all three milestones.
- [x] Verify checkpoint timesteps, PPO update counts, finite metrics, seed/target identity and preserved prior artifacts.
- [x] Run a V3 100k human Viewer episode and inspect its flight.
- [x] Run final tests and report comparative behavior without tuning anything.
