# Robustness Attribution Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute the frozen PPO dynamics cliff by comparing scripted/PPO waypoint behavior with a zero-command low-level hold probe, without training or controller changes.

**Architecture:** Reuse the previous `RobustnessEnv` plant mismatch, `evaluate_episode` trace format, holdout sampler, saved PPO loader, and scripted action. Add an inference-only evaluator and report/plot summarizer; factor-condition files are exclusive-created, while all baseline outputs remain untouched.

**Tech Stack:** Python 3, MuJoCo, NumPy, matplotlib, Stable-Baselines3 checkpoint loading (no `learn`).

**Spec:** `docs/superpowers/specs/2026-10-01-robustness-attribution-audit-design.md`

## Global Constraints

- No PPO training, Domain Randomization, reward/controller/PPO-parameter/observation/action changes.
- Same 100 holdout targets, same Reward V2 environment and nominal controller, all five deterministic low-std 100k checkpoints.
- One factor changed per condition; source XML and prior audit artifacts preserved.
- Mass/k_f factors: 0.90, 0.95, 0.98, 0.99, 1.00, 1.01, 1.02, 1.05, 1.10. World-X force: -10, -5, -2, -1, 0, +1, +2, +5, +10 N.
- Primary terminal window: last 5 seconds available; supplementary last 1 second; expose span/sample count and null missing metrics.
- Project has no Git repository; use project-local design/plan/progress documents and no commit or worktree operations.

## Review Focus

- A successful episode shorter than 5 seconds: tail mean uses available data and reports its actual span.
- Mass/k_f sweep affects plant but not controller mass, allocator B, or other plant parameter.
- Constant X force stays world-fixed at yawed attitudes, without an unintended torque.
- Exact waypoint/seed pairing across scripted and all five PPO policies cannot silently drift.
- A disturbed hold probe must change plant after nominal settling without resetting MuJoCo/controller state.

---

### Task 1: Runtime injection and terminal-window diagnostics

**Files:**
- Modify: `mujoco/rl/robustness_dynamics.py`
- Create: `mujoco/rl/test_attribution_diagnostics.py`
- Create: `mujoco/rl/attribution_diagnostics.py`

**Interfaces:**
- Consumes: `RobustnessEnv`, `Scenario`, `diagnose_ppo_braking.evaluate_episode`.
- Produces: `RobustnessEnv.activate_scenario(scenario: Scenario) -> None`; `terminal_window_stats(steps, target, window_s) -> dict`; `episode_attribution_summary(trace) -> dict`.

- [ ] Write tests that fail for post-warmup one-factor injection, 5/1-second tail selection, command/velocity/error signs, and short-episode span reporting.
- [ ] Run `python -m unittest mujoco/rl/test_attribution_diagnostics.py -q`; expect missing API failures.
- [ ] Implement minimal helpers without changing nominal controller behavior.
- [ ] Re-run the targeted tests; expect pass.

### Task 2: Paired waypoint and low-level hold evaluations

**Files:**
- Create: `mujoco/rl/attribution_audit.py`
- Create: `mujoco/rl/test_attribution_audit.py`

**Interfaces:**
- Consumes: Task 1 summaries and activation; existing scripted policy, five-checkpoint loader, exact holdout sampler.
- Produces: `evaluate_waypoint_condition(scenario, policies, waypoints) -> dict`; `evaluate_hold_condition(scenario) -> dict`; exclusive condition-file writer and CLI factor runner.

- [ ] Write tests that fail for target mismatch detection, one-waypoint nominal parity, post-settling zero-command hold, and protected result paths.
- [ ] Run `python -m unittest mujoco/rl/test_attribution_audit.py -q`; expect missing API failures.
- [ ] Implement 100-target inference-only condition runners, shared nominal, and seven hold conditions; save compact per-episode summaries, not 500 Hz logs.
- [ ] Re-run targeted and existing robustness tests; expect pass.
- [ ] Run shared nominal; require scripted near 100/100 and each PPO seed to match previous nominal metrics before nonnominal factors.
- [ ] Run mass, thrust and world-X force condition grids; each file is exclusive-created and progress is reported per condition.

### Task 3: Aggregation, curves, and attribution decision

**Files:**
- Create: `mujoco/rl/summarize_attribution_audit.py`
- Create: `mujoco/rl/test_summarize_attribution_audit.py`
- Create: `mujoco/reports/ppo_baseline_v1_attribution.json` and three sensitivity PNGs through the script.

**Interfaces:**
- Consumes: Task 2 per-condition files, original robustness nominal report, Task 1 tail metrics.
- Produces: validated 25-condition paired report, seven-condition hold report, plots, measured A/B/C judgment.

- [ ] Write failing tests for exact grid, complete 100-target/6-policy pairing, five-seed mean/SD/min/max, and headless PNG output.
- [ ] Run `python -m unittest mujoco/rl/test_summarize_attribution_audit.py -q`; expect missing API failures.
- [ ] Implement validation, summaries, plots, and evidence-based attribution wording; preserve previous reports.
- [ ] Run target tests and relevant inference-only RL tests; expect pass.
- [ ] Run real aggregation; check all 25 waypoint conditions, 15,000 episode summaries, seven low-level holds, nominal parity, no nonfinite results, and three valid PNGs.

## Execution ruling

The user's explicit instruction to conduct this bounded audit plus earlier no-repeated-consent authority selects inline execution. The absence of Git makes skill-provided Git worktree/commit scripts inapplicable; this plan's progress ledger is the local recovery record. Cost if wrong: user may prefer a different warm-up or terminal window, but both are recorded and cannot alter frozen policies or controllers.
