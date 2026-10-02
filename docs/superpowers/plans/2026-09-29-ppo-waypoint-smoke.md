# MineUAV PPO waypoint smoke plan — 2026-09-29

## Boundaries

Use the existing 7D-observation, 4D-velocity-command `MineUAVEnv` unchanged.
Train only on CPU in `.venv`, headless, with SB3 PPO/MlpPolicy and eight
same-process `DummyVecEnv` instances. Do not change reward, controller,
timing, or dynamics. The project is not a Git repository, so no worktree or
commits are possible; changes stay in the user-specified project directory.

## Files and responsibilities

- `mujoco/rl/train_ppo_waypoint.py`: resource preflight; Monitor-wrapped
  training environments; exact PPO config; 20,000-step hard cap; periodic
  independent deterministic evaluation; checkpoint, TensorBoard, Monitor,
  and JSON metrics output.
- `mujoco/rl/eval_ppo_waypoint.py`: load checkpoint and run deterministic
  headless or human-viewer evaluation through the same Gymnasium env.
- `mujoco/rl/test_ppo_waypoint.py`: construction, logging, exact-budget,
  and evaluation contracts before implementation.
- `mujoco/rl/README.md`: commands and exact-rollout-budget caveat.

## Implementation sequence

1. Establish passing existing RL tests and install CPU-only PyTorch, SB3,
   TensorBoard in `.venv` (done before code editing).
2. Write failing tests for eight distinct headless Monitor envs, expected
   PPO network/rollout config, and terminal `distance_m`/reason logging.
   Implement minimal training setup, then rerun tests.
3. Write failing tests for deterministic evaluation statistics and the
   20,000-step cap. Implement shared evaluation and callback. Verify tests.
4. Add CLI training and evaluation with isolated paths. Run existing RL
   regressions; perform one 20,000-step smoke run. The cap interrupts the
   tenth incomplete rollout at 20,000 total samples; nine full 2,048-sample
   rollouts (18,432 transitions) receive PPO updates. Report both counts.
5. Load saved model for 20 random-waypoint deterministic episodes; run one
   actual desktop viewer episode. Verify checkpoint, logs, finite PPO
   metrics, and full RL regressions. No further training.
