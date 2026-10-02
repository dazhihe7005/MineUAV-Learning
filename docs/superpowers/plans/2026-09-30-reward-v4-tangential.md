# Reward V4 Tangential Penalty Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans task by task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an independent 100k Reward V4 PPO baseline that differs from V2 only by near-target tangential-speed penalty, then compare it with V2 on the fixed 100 waypoints.

**Architecture:** Add a V4 branch to the existing environment reward. Reuse the unchanged PPO factory and a dedicated V4 training callback with separate artifact paths. Reuse the braking diagnostic's pre-command vector projection and crossing definition; use existing V2-style post-step evaluation for task metrics. Run one human-viewer episode after training.

**Tech Stack:** MineUAV Gymnasium/MuJoCo, Stable-Baselines3 PPO, NumPy, unittest.

**Spec:** User request in this conversation, 2026-09-30, Reward V4.

## Global Constraints

- V4 = V2 + `-0.5 * w(distance) * tangential_speed_squared`; no V3 distance penalty.
- Keep observation/action/controller/timing/task conditions/PPO factory unchanged.
- Use eight headless DummyVecEnv instances; complete rollouts to 100,352 steps only.
- Never overwrite baseline, V2, or V3 artifacts; evaluations use seeds 20261001–20261100.

## Review Focus

- Zero/near-zero target distance and non-finite terminal velocity keep reward finite.
- V2 reward and state transitions remain unchanged for the same action.
- First-entry/crossing metrics match the existing diagnostic's pre-command sampling and target-plane rule.
- Checkpoints are saved only after PPO updates, in V4-specific paths.
- Viewer mode uses the trained V4 checkpoint and never runs during training.

---

### Task 1: V4 reward

**Files:** Modify `mujoco/rl/mine_uav_env.py`; create `mujoco/rl/test_reward_v4.py`.

- [x] Write tests for V2 equality plus tangent term, gating, exact-target safety, failure finiteness, and unchanged transition.
- [x] Run red, implement the minimum V4 branch, run green.

### Task 2: Comparable diagnostics

**Files:** Extend `mujoco/rl/diagnose_ppo_braking.py`; create `mujoco/rl/reward_v4_diagnostics.py` and focused tests.

- [x] Test fixed-seed evaluation, near-target radial/tangential/command summaries, crossing counts, and V2 target equality.
- [x] Run red, implement, run green.

### Task 3: Independent PPO training and viewer

**Files:** Create `mujoco/rl/train_ppo_reward_v4.py` and tests; extend `mujoco/rl/eval_ppo_waypoint.py` reward-version choices.

- [x] Test schedule, artifact preservation, fixed PPO configuration, post-update checkpointing, and viewer CLI V4 selection.
- [x] Run red, implement, run green.
- [x] Train headless from scratch through 100,352 steps; evaluate 20k/50k/100k checkpoints.
- [x] Open a human-viewer episode, verify trajectories, run full regression tests, and report V2 vs V4 without auto-tuning.
