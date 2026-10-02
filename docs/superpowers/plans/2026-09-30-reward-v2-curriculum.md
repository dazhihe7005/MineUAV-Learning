# Reward V2 Curriculum Implementation Plan

**Goal:** Train one continuous Reward V2 PPO policy through local A, local B, and original full waypoint distributions, then compare stopping behavior on fixed local and full evaluations.

**Architecture:** Add a reset-only target distribution selector to `MineUAVEnv`, defaulting to the unchanged original sampler. Reuse the exact PPO factory, controller, reward, and braking/crossing diagnostics. Switch eight headless training environments between stages with `PPO.set_env`, preserving weights, optimizer state, and cumulative timesteps; evaluate with separate environments.

**Tech Stack:** Python, MuJoCo, Gymnasium, Stable-Baselines3 PPO, unittest.

**Spec:** User's Reward V2 curriculum request in this conversation.

## Global Constraints

- Reward V2 only. No V3 distance term, V4 tangent term, or other reward changes.
- Unchanged seven-dimensional observation, four-dimensional action and mapping, controller, success/failure/time-limit rules, and 500/100/25 Hz timing.
- Fixed PPO 8 environments, 256 steps, 256 batch, 10 epochs, [64,64] actor/critic, CPU, and existing other hyperparameters.
- A: radius 0.25–0.50 m; B: radius 0.50–1.00 m; C: original full sampler; original start `(0,0,1)`, level attitude, zero velocity, yaw target zero.
- Stage budgets are complete 2,048-transition rollouts: 51,200 + 51,200 + 100,352 = 202,752 transitions.
- Every stage: deterministic fixed 100 local A waypoints and exact original 100 full waypoints. Train headless; show one corresponding human-viewer episode per stage.
- Preserve all preexisting models/logs/reports; separate curriculum artifacts.

## Review Focus

- Local sampler must honor radius bounds and z >= 0.7 m without perturbing the full sampler's seed-to-target mapping.
- `set_env` and repeated `learn(reset_num_timesteps=False)` must preserve weights, optimizer state, and cumulative timesteps.
- Each checkpoint must represent the model after the final PPO update, not merely collected rollout samples.
- Evaluation local/full seed and target coordinates must match across all stages and the old V2 full evaluation.
- Crossing definition must remain the existing signed target-plane rule, with conditional counts reported alongside all-episode counts.

## Tasks

### Task 1: Reset-only local target distributions

**Files:** `mujoco/rl/mine_uav_env.py`, `mujoco/rl/test_curriculum_sampler.py`.

**Interfaces:** `MineUAVEnv(..., target_distribution="full"|"local_a"|"local_b")`; default is original full sampler. Explicit `reset(options={"target_position": ...})` retains original validation and precedence.

- [x] Write tests for A/B targets, original full seed target, unchanged start/reward/interfaces, and invalid distribution (300 seeds per distribution; see execution note).
- [x] Run tests and observe feature-missing failures.
- [x] Implement only target selection in reset; draw uniform radius, random normalized 3D direction, reject directions with z outside [0.7,1.5].
- [x] Run focused and full RL suite.

### Task 2: Evaluation and viewer entry point

**Files:** `mujoco/rl/curriculum_diagnostics.py`, `mujoco/rl/eval_ppo_waypoint.py`, `mujoco/rl/test_curriculum_diagnostics.py`.

**Interfaces:** `evaluate_curriculum(model, env, episodes, seed)` returns V2-style post-step task summary plus existing pre-command braking/crossing summary. `--target-distribution` selects the viewer/evaluation task.

- [x] Write tests for deterministic local/full evaluations and CLI distribution selection; crossing is delegated to the existing tested helper.
- [x] Run tests and observe feature-missing failures.
- [x] Implement via `evaluate_episode`, `summarize_policy`, `summarize_episodes`, and existing `crossing_counts`.
- [x] Run focused and full RL suite.

### Task 3: Continuous staged trainer

**Files:** `mujoco/rl/train_ppo_waypoint.py` (factory optional distribution), `mujoco/rl/train_ppo_curriculum.py`, `mujoco/rl/test_train_ppo_curriculum.py`.

**Interfaces:** reuse `make_ppo`, create eight `Monitor` environments per stage, call `set_env` on same model; save A/B/C after 25/25/49 complete rollouts; separately evaluate fixed local/full and old V2 checkpoint; save report/monitor/TensorBoard under curriculum names.

- [x] Write tests for fixed PPO config, schedule, artifact isolation, stage continuity, and non-overwrite; target consistency checked on the full generated report.
- [x] Run tests and observe feature-missing failures.
- [x] Implement minimal trainer, preflight, report, and full-update checks.
- [x] Run focused and full RL suite.

### Task 4: Execute and verify experiment

**Files:** New model/log/report outputs only.

- [x] Train exactly the three complete-rollout stages, preserving checkpoints and optimizer continuity.
- [x] Verify all six stage/task evaluations use identical fixed local/full target sets; compare to old V2 full set.
- [x] Run a corresponding human Viewer episode per stage; report observed trajectory without tuning.
- [x] Run fresh full RL regression suite and inspect report/checkpoint metadata.

## Execution record

- No Git repository exists in this project; all changes were made in place and older experiment artifacts were preserved. No worktree or commits were possible.
- Sampler and diagnostics tests were seen failing for absent features before implementation; then passed. Small two-stage PPO integration test confirmed identical policy and optimizer objects with cumulative steps and updates.
- Full RL regression: 88 tests passed, one opt-in GUI test skipped. Real human-viewer episodes were run separately for A, B, and C.
- Complete rollout budgets: A 51,200 (250 updates), B cumulative 102,400 (500 updates), C cumulative 202,752 (990 updates). The same in-memory PPO model was used throughout.
- Fixed100 local targets and fixed100 original full targets were verified identical across baseline and all stages. Original V2 full evaluation target/metric records agree with the prior report.
- Stage B local success was only 1/100; C local returned to 0/100, so forgetting is possible but evidence is weak for robust stopping ever having been learned.
- Deferred minor: previous training environment is closed only after successful stage transition; on a stage B/C exception it remains open until process exit.
- Deferred minor: sampler test covers 300 seeds per local distribution rather than the initial plan's 1,000; observed bounds and 3D variation passed.
