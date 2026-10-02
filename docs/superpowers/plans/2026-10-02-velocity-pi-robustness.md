# Velocity PI Robustness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether an opt-in velocity PI loop fixes static-disturbance rejection and improves scripted/frozen-PPO waypoint robustness without retraining.

**Architecture:** Keep nominal P behavior as the default and add PI gains/state/anti-windup solely to the existing velocity controller. A new inference-only runner constructs PI-configured private robustness environments and writes independent hold and paired waypoint outputs; a summarizer compares them to the frozen P attribution report.

**Tech Stack:** Python 3.10, NumPy, MuJoCo, Gymnasium, Stable-Baselines3 CPU, unittest, matplotlib Agg.

**Spec:** `docs/superpowers/specs/2026-10-02-velocity-pi-robustness-design.md`

## Global Constraints

- No PPO training, reward/action/observation/allocator/attitude/XML modifications.
- Same five frozen checkpoints and exact fixed 100 holdout waypoints.
- Kp remains `[1.5,1.5,2.0] s^-1`; initial Ki is XY `0.5 s^-2`, Z `0.8 s^-2`; no parameter search.
- Existing 3 m/s² acceleration limits and 500/100/25 Hz timing remain unchanged.
- Historical P reports and checkpoint files must not be overwritten.
- The project has no Git repository; work in place without Git/worktree/commit steps.

## Review Focus

- Episode reset must clear PI state even after a failed/safety-terminated episode (Task 1 test).
- XY and Z integrators must stay bounded when acceleration saturates (Task 1 tests).
- A zero-velocity hold may stop drifting at a displaced position; do not report that as position recovery (Task 2 test).
- A safety-terminated hold has less than 15 s of observation and must be labeled as such (Task 2 test).
- All six policy comparisons at each disturbance level must use identical targets and independent state (Task 3 test).

### Task 1: PI controller with reset and anti-windup

**Files:** Modify `mujoco/control/velocity_command_controller.py`; extend `mujoco/rl/test_velocity_command_controller.py`; add a reset integration test under `mujoco/rl/` if needed.

**Interfaces:** `VelocityCommandController(..., ki_xy=0.0, ki_z=0.0, integral_accel_limit_xy=1.5, integral_accel_limit_z=1.5)`; read-only `integral_error`, `integral_acceleration_world`, `last_desired_acceleration_world`, and cumulative anti-windup freeze counters. Existing constructor calls remain P-equivalent.

- [ ] Add literal, behavior-based tests for integral cancellation of constant error, separate XY/Z gains, reset, horizontal/vertical contribution caps, saturation freeze/unwind, and invalid gains/dt.
- [ ] Run tests and confirm failures are due to missing PI behavior/API.
- [ ] Implement minimum PI state and anti-windup in the existing velocity command path; preserve default P behavior and downstream interfaces.
- [ ] Run the controller and environment regression tests and confirm green.

### Task 2: Zero-command PI hold probes

**Files:** Create `mujoco/rl/audit_velocity_pi.py`; create `mujoco/rl/test_audit_velocity_pi.py`; output independent hold JSON parts under `mujoco/reports/velocity_pi_parts/hold/`.

**Interfaces:** `configure_pi_env(env: RobustnessEnv) -> None`; `evaluate_pi_hold(scenario: Scenario, warmup_s=2.0, observe_s=15.0) -> dict`; generated records contain time, position/velocity, integral state/contribution, desired acceleration, motor RPM, terminal reason and actual observation duration.

- [ ] Add tests for nominal settle/injection, zero command, complete diagnostics, actual elapsed duration, reset isolation, and no overwrite.
- [ ] Run tests red, implement runner minimally, then run tests green.
- [ ] Run seven holds (nominal, mass ±5%, kf ±5%, X force ±5 N), inspect finite/bounded integrator and oscillation/drift indicators; if unstable, record a single minimal manual stability correction before Task 3.

### Task 3: Paired scripted/frozen-PPO PI evaluation

**Files:** Extend `mujoco/rl/audit_velocity_pi.py` and its tests; write independent 19-condition waypoint part files under `mujoco/reports/velocity_pi_parts/waypoint/`.

**Interfaces:** `evaluate_pi_waypoint_condition(scenario, policies, waypoints) -> dict` uses existing scripted policy, `load_frozen_policies`, `evaluate_episode`, and attribution episode summaries; `run_factor(factor)` resumes valid finished parts.

- [ ] Add tests for exact 100-target equality, deterministic five-checkpoint order, distinct P/PI outputs at disturbed conditions, and resume/no-overwrite behavior.
- [ ] Run tests red, implement, then run tests green.
- [ ] Run PI nominal first; then mass, thrust, force levels exactly as specified. Save each completed condition atomically without touching historical P artifacts.

### Task 4: Comparison report and sensitivity figures

**Files:** Create `mujoco/rl/summarize_velocity_pi.py`, `mujoco/rl/test_summarize_velocity_pi.py`; output `mujoco/reports/velocity_pi_robustness.json` and three separate PNG curves.

**Interfaces:** Validate 19 paired PI conditions (one shared nominal plus six nonnominal levels per factor) and seven hold probes. Compare corresponding P report values from `ppo_baseline_v1_attribution.json`. Report scripted and five-seed PPO success, nominal change, ±5 levels, late-stage error/velocity/commands, integral boundedness/anti-windup, RPM, oscillation and safety terminations.

- [ ] Add tests for pairing/part count, P report preservation, five-seed mean/SD/min/max, hold interpretation and curve output.
- [ ] Run tests red, implement, then run tests green.
- [ ] Generate report/figures and independently validate all parts, finite metrics, source XML hash, and final regression suite.
