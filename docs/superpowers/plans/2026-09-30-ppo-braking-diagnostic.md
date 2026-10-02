# PPO Braking Diagnostic Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the existing scripted policy with Reward V2 100k deterministic PPO on the same 100 fixed waypoints, without training or changing the environment.

**Architecture:** A read-only evaluation script runs both policies at seeds 20261001–20261100 in fresh headless V2 environments. It records pre-command state and mapped commands per policy step, derives braking/crossing statistics, writes a compact JSON summary plus compressed step log, and plots one reproducibly selected paired episode.

**Tech Stack:** Existing MineUAV Gymnasium/MuJoCo stack, SB3 checkpoint loading, NumPy, Matplotlib, standard-library CSV/gzip/JSON.

**Spec:** User request in this conversation, dated 2026-09-30 (pure scripted vs Reward V2 braking diagnosis).

## Global Constraints

- No PPO training, reward/PPO/controller changes, or Reward V4.
- Same 100 fixed seeds and targets; PPO `deterministic=True`.
- Preserve every policy-step observation/action/velocity command in the log.
- Crossing must depend on target-relative direction, not distance alone.

## Review Focus

- Near-zero target distance: radial/tangential quantities must remain finite.
- Steps outside a distance bin must not leak into that bin's pooled mean.
- Crossing requires a signed target-plane transition, not just a local distance minimum.
- Half-second lookahead must not read beyond episode termination.
- Both policies must get the same target for each seed.

---

### Task 1: Diagnostic math and tests

**Files:** Create `mujoco/rl/test_ppo_braking_diagnostic.py`; create `mujoco/rl/diagnose_ppo_braking.py`.

**Interfaces:** `make_step_record(...) -> dict`, `crosses_target_plane(steps, start, window_s) -> bool`, `summarize_distance_bins(steps) -> dict`, `first_entry(steps, threshold) -> dict | None`.

- [x] Write tests for vector projections, active braking, bins, crossing with hysteresis, 0.5 s lookahead, and missing entries.
- [x] Run tests and observe the expected missing-implementation failure.
- [x] Implement the math and summaries without modifying the environment.
- [x] Run the focused tests and confirm pass.

### Task 2: Paired evaluator and report

**Files:** Complete `mujoco/rl/diagnose_ppo_braking.py`; output `mujoco/reports/ppo_braking_diagnostic.json` and `mujoco/reports/ppo_braking_policy_steps.csv.gz`.

**Interfaces:** Reuse `scripted_action` and `MineUAVEnv(reward_version="v2")`; load `ppo_waypoint_reward_v2_100k.zip` only for deterministic inference.

- [x] Add a small paired-seed evaluation test.
- [x] Run it and observe the missing-evaluator failure.
- [x] Implement evaluation, same-target assertion, per-step logging, first-entry statistics and report serialization.
- [x] Run focused tests and evaluate 100 paired waypoints.

### Task 3: Visuals and verification

**Files:** Complete plotting in `mujoco/rl/diagnose_ppo_braking.py`; output five `diagnostic_*.png` files in `mujoco/reports/`.

- [x] Test representative episode selection and output filenames.
- [x] Run it red, implement plots, run it green.
- [x] Run all relevant RL tests, inspect report/CSV/image dimensions, and interpret the observed data without speculation.
