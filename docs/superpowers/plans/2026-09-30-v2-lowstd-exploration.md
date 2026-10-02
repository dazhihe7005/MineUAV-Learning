# Reward V2 Low-Std Exploration Implementation Plan

**Goal:** Audit the saved Reward V2 Gaussian policy and run one from-scratch Reward V2 PPO experiment differing only in `log_std_init=-2.0`.

**Architecture:** Keep the existing Reward V2 environment, fixed waypoint samplers, PPO factory and checkpoint schedule. Add a read-only stochastic-vs-deterministic evaluator, then minimally parameterize the PPO factory's initialization and add a separate low-std runner with independent files, full-rollout checkpoints, per-axis standard-deviation history, fixed local/full evaluation, and Viewer verification.

**Tech Stack:** Python, NumPy, PyTorch, installed Stable-Baselines3, Gymnasium, MuJoCo, unittest.

**Spec:** User's current continuous-action exploration request.

## Global constraints

- Reward V2 only; no reward, observation, action, controller, timing, curriculum, network or PPO hyperparameter changes other than `log_std_init`.
- PPO: seed 20260929, 8 DummyVecEnv, 256 steps, batch 256, 10 epochs, actor/critic [64,64], CPU; stop after full rollout 100,352 transitions.
- Preserve existing V2 checkpoints and reports. New output prefix `ppo_waypoint_v2_lowstd`.
- Exact fixed 100 local seeds 20261101–20261200 and 100 full seeds 20261001–20261100.
- Deterministic checkpoint evaluation; stochastic old-policy diagnosis repeats each full waypoint three times with reproducible per-episode sampling seeds.

## Tasks

### 1. Policy distribution and stochastic audit

Create `mujoco/rl/ppo_exploration_audit.py` and `mujoco/rl/test_ppo_exploration_audit.py`.

- [x] RED: tests require real checkpoint `DiagGaussianDistribution`, four trainable state-independent `log_std`, correct physical sigma conversion, fixed target identity and distinct sampled-vs-mean action accounting.
- [x] GREEN: inspect checkpoint, evaluate 100 deterministic and 300 stochastic full-task episodes through the unchanged environment, reuse existing braking/crossing summaries, save independent JSON report.
- [x] Verify raw Gaussian mean, clipped deterministic/executed action magnitudes and saturation, near-target actual/command/tangential metrics, crossing; never train.

### 2. Low-std factory and training

Modify `mujoco/rl/train_ppo_waypoint.py` only to accept optional `log_std_init=0.0` while preserving its default output. Create `mujoco/rl/train_ppo_v2_lowstd.py`, tests `mujoco/rl/test_train_ppo_v2_lowstd.py`.

- [x] RED: tests assert default policy kwargs unchanged; low-std differs only in `log_std_init`, initial sigma `exp(-2)`, same fixed PPO hyperparameters and complete-rollout milestone schedule; existing outputs reject overwrite.
- [x] GREEN: fresh PPO from Reward V2, checkpoint/updated-state callback at 20,480/51,200/100,352; record initial and each milestone's log-std/std, six PPO training metrics and episode means; save separate TensorBoard/monitor/history/report.
- [x] At each milestone evaluate exact fixed100 local/full deterministic tasks with existing braking/crossing definition; compare to original V2 100k using matching target sets.

### 3. Run and verify

- [x] Run full 100,352-step training only once; inspect all checkpoint std and metrics, complete report and fixed target equality.
- [x] Run one deterministic 100k full-task MuJoCo human Viewer episode; record observed behavior without tuning.
- [x] Run full RL regression, inspect report/model artifacts, and perform independent code review if available.
