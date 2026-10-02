# PPO + Velocity PI Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one PPO policy from scratch against fixed PI and compare nominal/conditional disturbed compatibility with historical P and direct-PI policies.

**Architecture:** Add a separate PI experiment script and tracked evaluation environments, reuse unchanged PPO construction and waypoint samplers, preserve all old artifacts, and gate disturbances on 100k nominal success.

**Tech Stack:** Python 3.10, MuJoCo, Gymnasium, SB3 PPO, NumPy, PyTorch CPU, unittest.

**Spec:** `docs/superpowers/specs/2026-10-02-ppo-pi-compatibility-design.md`

## Global Constraints

- One new training seed 0, 100,352 transitions, 8 env × 256 steps = 2,048 per full rollout. Save at 20,480, 51,200, 100,352.
- Exact Reward V2, 7D/4D interface, 500/100/25 Hz, frozen PI gains/caps, unchanged PPO hyperparameters and network. No domain randomization or other algorithm changes.
- Same fixed100 benchmark and holdout waypoint definitions as prior reports. Deterministic evaluation. No viewer in training/evaluation.
- Never overwrite source XML, historical reports/checkpoints, or existing new-output paths.
- No Git repository at project root; work in place and use distinct artifacts. Existing `.venv` only.

## Review Focus

- `env.reset()` must clear hidden integral memory; test a nonzero integral followed by reset.
- Every one of eight train envs must actually use PI and preserve Reward V2/observations/actions/timing; test real instantiated envs.
- Checkpoints must follow completed PPO updates; test exact 20k/50k/100k rounded schedule.
- Integral traces must align one-to-one with each deterministic episode and target, and never enter observation; test on real env episode.
- If either 100k nominal set has fewer than 90 successes, reject disturbance evaluation; test gate both ways.

### Task 1: PI environment and integral trace

**Files:** Create `mujoco/rl/ppo_pi_env.py`; create `mujoco/rl/test_ppo_pi_env.py`.

**Interfaces:** `MineUAVPIEnv` for monitored training; `PIRobustnessEnv(Scenario)` for disturbed evaluation; `begin_integral_capture()` / `take_integral_capture()` for evaluation; `make_pi_training_envs(monitor_dir, n_envs=8)` returns a DummyVecEnv.

- [ ] Write real-env tests for fixed PI gains/caps, identical interface/timing/reward, all eight ranks, reset clears a nonzero integral, and integral capture aligns with policy steps.
- [ ] Run tests and observe missing-feature RED.
- [ ] Implement minimal wrappers and headless monitored vector factory.
- [ ] Run focused and legacy environment/controller tests green.

### Task 2: Deterministic evaluation and nominal gate

**Files:** Create `mujoco/rl/evaluate_ppo_pi.py`; create `mujoco/rl/test_evaluate_ppo_pi.py`.

**Interfaces:** `evaluate_pi_policy(model, env, waypoints, trace_path)` reuses existing PPO diagnostics and saves per-episode PI trace; `nominal_gate(benchmark, holdout) -> bool`; `compact_pi_evaluation()` returns task/crossing/near-target/saturation/integral summaries.

- [ ] Write tests for fixed targets, exact trace alignment, finite/bounded integral fields, success/failure linkage, and the 90/100 gate on both sets.
- [ ] Run tests RED.
- [ ] Implement evaluator/compact metrics with separate exclusive trace outputs.
- [ ] Run tests GREEN and a 1-waypoint real PPO/PI evaluation smoke without training.

### Task 3: Fresh one-seed training

**Files:** Create `mujoco/rl/train_ppo_pi_lowstd.py`; create `mujoco/rl/test_train_ppo_pi_lowstd.py`; new checkpoint/log/TensorBoard/report paths only.

**Interfaces:** `checkpoint_schedule()`, `PIMilestoneCallback`, `run_training()` with seed 0 and exact 100,352-transition cap.

- [ ] Write tests for full-rollout schedule, output collision guard, PPO configuration/initialization, and complete milestone handling.
- [ ] Run tests RED.
- [ ] Implement minimal training wrapper and callback; keep upstream `make_ppo` unchanged.
- [ ] Run tests GREEN; run one full headless seed-0 training, save and evaluate 20k/50k/100k on both sets.

### Task 4: Conditional robustness and A/B/C report

**Files:** Create `mujoco/rl/audit_ppo_pi_compatibility.py`; create `mujoco/rl/test_audit_ppo_pi_compatibility.py`; independent part/report files under `mujoco/reports/ppo_pi_compatibility/`.

**Interfaces:** `run_conditional_robustness(report)` performs no work if gate fails; otherwise tests 13 unique conditions on holdout; `compare_abc(...)` pulls historical seed-0 paired metrics and PI-trained metrics.

- [ ] Write tests for gate stopping, exact ±5/±2/0 grid, checkpoint identity, paired target hashes, old artifact preservation and A/B/C numeric extraction.
- [ ] Run tests RED.
- [ ] Implement audit/report without touching training or historical scripts.
- [ ] Run tests GREEN; conditionally run disturbance audit, verify all saved data, summarize final results, stop.
