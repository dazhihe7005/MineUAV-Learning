# MineUAV PPO 200k Baseline Implementation Plan

**Goal:** Train one new, continuous, fixed-configuration PPO baseline to the last complete rollout not exceeding 200,000 timesteps, with four evaluated checkpoints and plots.

**Architecture:** Reuse the existing headless `MineUAVEnv`, `make_training_envs()` and `make_ppo()` unchanged. A new callback records metrics after completed PPO updates, saves milestone checkpoints, and evaluates each on the same 100 seeded waypoints. A separate report and plotting script preserve the prior 20k smoke artifacts.

**Tech Stack:** Python 3.10, MuJoCo, Gymnasium, SB3 2.9.0, PyTorch CPU, Matplotlib.

**Spec:** User's current PPO baseline request; no task-definition or PPO-parameter edits.

## Global constraints

- Keep observation/action/reward/success/failure/time limit/controller/timing unchanged.
- Keep PPO architecture and hyperparameters exactly as in `train_ppo_waypoint.py`.
- Use 8 headless Monitor envs via DummyVecEnv, independent evaluation env, CPU.
- Never load `ppo_waypoint_smoke.zip` for training; never overwrite it.
- At 2,048 transitions per rollout: checkpoint steps 20,480, 51,200, 100,352; final 198,656. Never sample beyond 200,000.
- Evaluate the same 100 reset seeds at each milestone; run a separate final 100-seed evaluation with action telemetry.
- The workspace is not a Git repository; no worktree or commits are possible.

## Review focus

- Saving at `on_rollout_end` captures pre-update weights; use the next `on_rollout_start` or post-`learn` for updated weights.
- A 200,000 `learn()` request rounds up to 200,704 samples; request exactly 198,656 instead.
- Monitor wrappers do not proxy environment-specific attributes; use `env.unwrapped`.
- Checkpoint labels differ from actual timesteps; record both.
- Interrupted runs must not overwrite existing model files; persist milestones as they complete.

### Task 1: Schedule and fixed evaluation

**Files:** Create `mujoco/rl/train_ppo_baseline.py`, `mujoco/rl/test_ppo_baseline.py`.

**Interfaces:** `checkpoint_schedule()` returns label/target/actual-step mappings; `evaluate_baseline(model, env, episodes, seed)` returns reward, distance, length, completion time, success and behavior metrics.

- [x] Write tests with literal 2,048-step schedule and scripted-model evaluation; verify RED.
- [x] Implement schedule and evaluator; verify GREEN and existing RL suite.

### Task 2: One continuous training run and metrics

**Files:** Extend `train_ppo_baseline.py`, `test_ppo_baseline.py`.

**Interfaces:** Callback saves/evaluates after PPO updates, recording milestone and per-update metrics; `run_baseline_training()` creates a new model and writes JSON report, TensorBoard and Monitor logs.

- [x] Write tests for after-update checkpoint timing and artifact protection; verify RED.
- [x] Implement callback, config assertions and report persistence; verify GREEN.
- [x] Run one new training session through 198,656 timesteps; verify four checkpoints, identical evaluation seeds, 100 episodes at each milestone and final 100 episodes.

### Task 3: Curves and viewer validation

**Files:** Create `mujoco/rl/plot_ppo_baseline.py`; extend `mujoco/rl/README.md` only for commands.

**Interfaces:** `plot_report(report_path, output_dir)` creates the four requested PNGs from recorded metrics; existing `eval_ppo_waypoint.py --model ... --human` loads each checkpoint.

- [x] Write plot-output test and verify RED.
- [x] Implement plots and verify GREEN.
- [x] Generate four figures; check TensorBoard scalars for non-finite values; run RL regressions and one actual 200k Viewer episode.
