# PPO Reward V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test one near-waypoint braking reward at fixed PPO settings through the last complete rollout near 100k, preserving the baseline.

**Architecture:** Keep `MineUAVEnv` default reward V1 and add explicit V2 selection. Reuse the existing PPO factory and vector environment settings; write a common deterministic diagnostic evaluator for both rewards, then run a separate V2 training pipeline and compare the 100k checkpoints.

**Tech Stack:** Python, MuJoCo, Gymnasium, Stable-Baselines3 PPO, NumPy.

**Spec:** User's current Reward V2 request in this conversation.

## Global Constraints

- No changes to observation, action, success/failure/time limit, controller, frequencies, PPO settings or vector configuration.
- Preserve original progress/action/success/failure terms. Add only `-0.5 * clip((0.5-d)/0.4,0,1) * ||v_world||²`.
- Train from scratch through complete 8 × 256 rollouts, no more than about 100k; preserve baseline artifacts.
- Compare deterministic evaluations on seeds 20261001–20261100. Do not compare raw rewards across reward definitions.
- Project is not a Git repository; no commits or worktrees are available.

## Review Focus

- Boundary distances 0.5 and 0.1: weight must equal 0 and 1 respectively; Task 1 tests both.
- Numerical failure with NaN qvel: reward must remain finite; Task 1 tests it.
- Default environment must retain V1 reward and interface; Task 1 regression tests it.
- Same 100 waypoints for baseline and V2; Task 2 tests seed reproducibility and Task 3 verifies report seeds.
- Checkpoints must be saved after completed PPO updates without overwrites; Task 3 tests schedule and fresh-artifact guard.

---

### Task 1: Optional Reward V2

**Files:** Modify `mujoco/rl/mine_uav_env.py`; create `mujoco/rl/test_reward_v2.py`.

**Interfaces:** `MineUAVEnv(..., reward_version="v1" | "v2")`, default V1; V2 `info["reward_breakdown"]` has `progress`, `action`, `brake`, `success`, `failure`, `total`.

- [x] Write tests for V1 unchanged, V2 boundary/interpolated penalty and breakdown, invalid version, finite failure reward.
- [x] Run tests; confirm failure caused by absent V2.
- [x] Implement V2 selector and reward term using post-step `qvel[:3]`.
- [x] Run new and existing environment tests.

### Task 2: Deterministic diagnostic evaluator

**Files:** Create `mujoco/rl/reward_v2_diagnostics.py` and `mujoco/rl/test_reward_v2_diagnostics.py`.

**Interfaces:** `evaluate_diagnostics(model, env, episodes=100, seed=20261001) -> dict`, including threshold episode ratios, step-weighted near-target speeds, max hold streak, action boundary fractions, standard milestone metrics.

- [x] Write failing tests for metric aggregation on hand-checked episode traces and fixed seeds.
- [x] Run tests; confirm absent evaluator failure.
- [x] Implement bounded deterministic evaluation and aggregation.
- [x] Run diagnostics tests and an integration check.

### Task 3: Separate V2 training/evaluation artifacts

**Files:** Modify `mujoco/rl/train_ppo_waypoint.py` and `mujoco/rl/eval_ppo_waypoint.py`; create `mujoco/rl/train_ppo_reward_v2.py` and `mujoco/rl/test_train_ppo_reward_v2.py`.

**Interfaces:** Existing PPO factory unchanged; training-env factory accepts V2 selection defaulting V1. V2 checkpoint schedule 20480/51200/100352. Viewer CLI accepts `--reward-version v2`.

- [x] Write failing tests for schedule, V2 train-env selection, checkpoint protection, Viewer evaluation selector.
- [x] Run tests; confirm missing implementation.
- [x] Implement separate training callback, report, logs, TensorBoard, and baseline100k comparison.
- [x] Run tests and full RL regression suite.
- [x] Run the 100k training and verify all three updated checkpoints, evaluations, metrics and baseline comparison.
- [x] Run V2 100k desktop Viewer episode and inspect behavior.
