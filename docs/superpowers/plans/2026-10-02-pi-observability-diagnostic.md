# PI Hidden-State Observability Diagnostic Implementation Plan

> **For agentic workers:** Execute inline with test-driven development. This project is not a Git repository; preserve all existing experiment files and do not run Git commit/worktree steps.

**Goal:** Isolate whether exposing the existing velocity PI integral acceleration as three additional PPO observations improves nominal waypoint stopping at the same 100,352-step seed-0 budget.

**Architecture:** Keep `MineUAVEnv` and the existing 7D `MineUAVPIEnv` unchanged for old experiments. Add an opt-in 10D subclass that appends `VelocityCommandController.integral_acceleration_world` at each policy boundary, then use the existing PPO/evaluation helpers with new output names. Pair the new benchmark/holdout results with the same historical A/B/C nominal targets; do not run robustness sweeps.

**Tech Stack:** Python 3.10, Gymnasium, MuJoCo, Stable-Baselines3 PPO, NumPy, unittest.

**Spec:** User request in the current conversation, “PI HIDDEN-STATE / OBSERVABILITY SANITY CHECK”.

## Global Constraints

- Fixed PI: `Ki_xy=0.5`, `Ki_z=0.8`, XY/Z integral acceleration caps `1.5 m/s²`; existing anti-windup and reset remain unchanged.
- Reward V2, 4D velocity/yaw-rate action, controller/physics/policy frequencies, 64×64 actor/critic, all PPO hyperparameters and `log_std_init=-2.0` remain unchanged.
- Seed 0, fresh initialization, 8 `DummyVecEnv` environments, 256 rollout steps, one 100,352-step training run; save 20k/50k/100k full-rollout checkpoints.
- Evaluate deterministic on the identical fixed-100 benchmark and fixed-100 holdout; compare A/B/C/D nominal only. No additional training, disturbances or Domain Randomization.
- New artifacts must never overwrite previous 7D runs.

## Review Focus

- Reset must return the last three observations as exactly zero, including on repeated episodes.
- At every policy boundary, appended values must equal the actual PI contribution after the four controller updates, not a predicted or pre-step value.
- Finite state failures must return a finite 10D observation and preserve existing termination/reward behavior.
- Training seed and fixed target digests must match the previous seed-0 experiment; model/trace/log filenames must be disjoint.
- Comparison must not conflate one-seed D results with the earlier five-seed aggregate or imply a causal conclusion from a single seed alone.

## Task 1: Opt-in 10D PI observation

**Files:** Create `mujoco/rl/ppo_pi_observable_env.py`, `mujoco/rl/test_ppo_pi_observable_env.py`; optionally parameterize `make_pi_training_envs` in `mujoco/rl/ppo_pi_env.py` without changing its default.

**Interfaces:** `MineUAVPIObservableEnv` produces the original seven float32 entries plus three world-frame integral-acceleration entries, with a 10D Box. `make_pi_training_envs(..., env_type=MineUAVPIObservableEnv)` produces eight monitored headless environments.

- [ ] Write failing real-environment tests for reset zeros, exact post-step PI contribution, old 7D preservation, finite terminal observation, action/reward/step parity, and `gymnasium` environment checking.
- [ ] Run focused tests and confirm failure is the missing 10D feature.
- [ ] Implement only the subclass and minimal env-factory extension.
- [ ] Run focused and pre-existing environment tests; verify no behavior drift.

## Task 2: Fresh isolated PPO/PI/10D run

**Files:** Create `mujoco/rl/train_ppo_pi_observable.py`, `mujoco/rl/test_train_ppo_pi_observable.py`; minimally allow `PIMilestoneCallback` in `mujoco/rl/train_ppo_pi_lowstd.py` to accept a distinct model prefix while preserving its default.

**Interfaces:** `run_training()` trains seed 0 from scratch using the unchanged `make_ppo(..., log_std_init=-2, seed=0)`, fixed benchmark/holdout helpers, checkpoint schedule and `evaluate_pi_policy`; emits separate model/log/TensorBoard/trace/report paths with explicit 10D shape and checkpoint hashes.

- [ ] Write failing tests for output isolation, 8-env/10D PPO construction, checkpoint schedule, fixed target digests, and pre-existing output rejection.
- [ ] Run tests to observe the intended failure.
- [ ] Implement the smallest isolated training runner and callback prefix parameter.
- [ ] Run tests and a short real environment/evaluation integration check without PPO training.

## Task 3: Execute once, validate and compare

**Files:** Create `mujoco/rl/report_ppo_pi_observability.py`, `mujoco/rl/test_report_ppo_pi_observability.py`; publish one JSON report in `mujoco/reports/` and new model/log/trace artifacts.

**Interfaces:** Verify all checkpoints and digests, compare A/B/C/D on paired holdout, report benchmark/holdout 20k/50k/100k metrics, and state whether the data support the hidden-state hypothesis. No disturbance evaluations.

- [ ] Write failing tests for paired target/seed comparison and tamper/missing-artifact rejection.
- [ ] Implement read-only comparison from existing reports and the new result.
- [ ] Run the one authorized full training to 100,352 steps and publish its report/checkpoints.
- [ ] Verify model reload, finite PPO metrics, integral-observation alignment, all milestone evaluations, and the A/B/C/D comparison.
- [ ] Run full relevant regression suite and obtain one read-only code review before final reporting.
