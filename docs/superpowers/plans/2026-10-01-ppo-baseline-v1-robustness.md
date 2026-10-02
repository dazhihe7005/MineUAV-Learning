# MineUAV RL Baseline v1 Robustness Audit Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. This project is not a Git repository, and the user previously authorized execution without repeated approval; preserve all existing outputs and work in place. Track steps with checkboxes.

**Goal:** Measure five frozen PPO policies on the same holdout waypoints under isolated mass, inertia, k_f, motor-lag and constant-world-X-force perturbations.

**Architecture:** Add minimal physics hooks to the existing environment with no-hook nominal behavior. Put all runtime overrides and physical motor state in a focused robustness module; reuse existing trace/metric definitions and run one factor scenario per process. Aggregate all parts into a single audited JSON plus five success-rate figures.

**Tech Stack:** MuJoCo 3.14.0, Gymnasium, NumPy, PyTorch/SB3 PPO (inference only), matplotlib, unittest.

**Spec:** `docs/superpowers/specs/2026-10-01-ppo-baseline-v1-robustness-design.md`

## Global Constraints

- No training, reward, observation, action, controller-gain, timing, PPO-network/parameter or nominal-XML modification.
- Five checkpoints from `mujoco/reports/ppo_waypoint_v2_lowstd_multiseed.json`; same holdout seeds 20271001–20271100, deterministic prediction everywhere.
- Mass, inertia and k_f single-variable sweeps: factors 0.8,0.9,0.95,1.0,1.05,1.1,1.2. Lag: 0,20,40,60,80,100 ms. External ±X force: 0,±5,±10,±15,±20 N.
- Each scenario reports five per-policy evaluations, with null for undefined means. Final report path `mujoco/reports/ppo_baseline_v1_robustness.json` must not overwrite any preexisting file.

## Review Focus

- Runtime mass changes must not silently change inertia or controller mass. Test exact model arrays and a known-force acceleration probe.
- Inertia scaling must preserve the off-diagonal full-tensor structure. Test reconstruction from `body_iquat` and principal inertias.
- k_f scaling must not scale yaw reaction torque gear or nominal allocator. Test all six gear entries and controller B.
- Motor lag must act on omega, not on omega squared; reset must clear actual and commanded motor states. Test exact exponential response and first-step dynamics.
- External force must remain world +X/-X at COM regardless of yaw, not a body-axis force or added torque. Test accelerations at yaw=0 and yaw=90°.
- Every scenario must use the exact same 100 targets and all five checkpoints. Test target hashes/seed order and reject missing/duplicated parts.

## Task 1: Nominal-preserving physics hooks and scenario overrides

**Files:** Modify `mujoco/rl/mine_uav_env.py` minimally; create `mujoco/rl/robustness_dynamics.py`; create `mujoco/rl/test_robustness_dynamics.py`.

**Interfaces:** `RobustnessEnv(MineUAVEnv, scenario: Scenario)`; `Scenario(factor: str, value: float)`; default hooks preserve existing env behavior. Runtime model override uses `mujoco.mj_setConst` where needed.

- [ ] Write failing tests for nominal equivalence, isolated mass/inertia/k_f mutations, omega lag update/reset, world-axis force and invalid scenarios.
- [ ] Run targeted tests; confirm each fails because required hooks/overrides are missing.
- [ ] Implement minimal hooks and robustness class; document exact motor command/state/actual thrust flow.
- [ ] Run targeted tests and existing environment/physics tests. Run small physical acceleration probes.

## Task 2: Fixed holdout evaluation and isolated scenario parts

**Files:** Create `mujoco/rl/audit_ppo_robustness.py`; create `mujoco/rl/test_audit_ppo_robustness.py`.

**Interfaces:** `evaluate_scenario(factor: str, value: float) -> dict` loads all five policies, evaluates 100 holdout seeds each, and records per-policy/episode metrics; `run_part(factor: str)` writes one protected JSON part. Shared nominal is run once.

- [ ] Write failing tests for five checkpoint/target identity, metric equality with existing diagnostics, terminal failure/timeout accounting and null means.
- [ ] Implement evaluator with `evaluate_episode`, `summarize_policy`, `summarize_episodes`, `crossing_counts`; output precise per-policy data and scenario metadata.
- [ ] Run fresh nominal evaluation and compare with prior five-seed holdout results before continuing. Reject mismatch.
- [ ] Run four independent nonnominal factor parts, each with progress and protected output paths; no training.

## Task 3: Aggregate, plots, robustness boundaries

**Files:** Create `mujoco/rl/summarize_ppo_robustness.py`; create `mujoco/rl/test_summarize_ppo_robustness.py`; outputs in `mujoco/reports/`.

**Interfaces:** `summarize_parts() -> dict` verifies the exact scenario grid, five seeds and holdout target identity, computes mean/sample SD/min/max; plotting function writes five requested PNGs.

- [ ] Write failing tests for missing/duplicate scenario rejection, sample SD, null values, threshold classification and measured factor ranking.
- [ ] Implement aggregation and five uncluttered sensitivity plots.
- [ ] Run aggregate on completed parts, verify all expected counts and physical invariants; run full RL regression suite.
- [ ] Report nominal and each sensitivity, boundary ranges, NaN/unstable count and data-based next-priority recommendations; stop before Domain Randomization.
