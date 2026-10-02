# Reward V2 Alignment and Policy Representation Implementation Plan

**Goal:** Audit whether Reward V2 ranks successful scripted flights above PPO/zero/random, and test whether the unchanged 7D observation and 4D action can be learned by a fixed 64×64 MLP via supervised imitation.

**Architecture:** Both audits reuse `MineUAVEnv(reward_version="v2")` and fixed local/full waypoint sets from the curriculum experiment. Part A accumulates environment reward breakdowns without training. Part B collects separate scripted-success trajectories, splits by episode, trains only a PyTorch tanh MLP, and evaluates its own closed-loop actions with existing trajectory diagnostics.

**Tech Stack:** Python, NumPy, PyTorch, Stable-Baselines3 for loading the existing V2 checkpoint only, MuJoCo, Gymnasium, unittest.

**Spec:** Current user request, Parts A and B.

## Global Constraints

- No PPO training, Reward V5, Reward V2 edits, controller changes, curriculum changes, larger model, or other frameworks.
- Fixed evaluation: 100 local A seeds 20261101..20261200 and 100 original full seeds 20261001..20261100; exact target coordinates checked against `ppo_waypoint_curriculum.json`.
- Part A policies: unchanged `scripted_action`, existing deterministic V2 PPO 100k, zero action, reproducibly seeded random action. Every policy gets identical targets and initial conditions.
- Part A gamma=0.99; record undiscounted and discounted sums of progress/action/brake/success/failure/total, per episode and aggregate.
- Part B scripted dataset uses disjoint training, validation, and fixed evaluation episode seeds. No step-level random split. Only successful scripted episodes enter the supervised dataset; attempts/failures are reported.
- Part B network input7, tanh hidden64/64 matching SB3 default actor hidden activation, linear 4D mean head; MSE training on normalized scripted actions, clip only when used as environment action.
- Preserve old checkpoints/reports. New reports, dataset, and imitation model have independent names.

## Tasks

### 1. Part A return audit

**Files:** Create `mujoco/rl/reward_v2_alignment_audit.py`, `mujoco/rl/test_reward_v2_alignment_audit.py`.

**Interfaces:** `evaluate_episode_return(env, policy, target, seed, gamma)` gives per-episode reward components and terminal metrics; `audit_alignment()` loads the fixed target sets, evaluates four policies, writes `mujoco/reports/reward_v2_alignment_audit.json`.

- [x] Test exact discounted component arithmetic and target equality / reward decomposition on a small real environment episode; observe red.
- [x] Implement fixed-seed target reuse, per-step accumulation, paired policy comparisons; observe green.
- [x] Run full 100 local + 100 full x four policies, preserve report.

### 2. Part B dataset and network

**Files:** Create `mujoco/rl/scripted_imitation_sanity.py`, `mujoco/rl/test_scripted_imitation_sanity.py`.

**Interfaces:** `collect_success_episodes` produces observation/action pairs with episode IDs; `ActorMeanMLP` is 7→64 tanh→64 tanh→4 linear; train/validation NPZ are separate by whole episode; fixed supervised config logged.

- [x] Test architecture, episode isolation, action bounds, dataset retention of scripted behavior; observe red.
- [x] Implement training/validation collection with seed ranges outside fixed evaluation sets, save NPZ and metadata; observe green.
- [x] Train with MSE, save model, losses and offline validation metrics.

### 3. Closed loop and Viewer

**Files:** Extend `mujoco/rl/scripted_imitation_sanity.py`; create `mujoco/rl/watch_imitation_policy.py` and tests where suitable.

- [x] Test `predict` adapter feeds learned actions into the existing V2 diagnostics and viewer runner accepts a saved model; observe red.
- [x] Evaluate fixed 100 local and 100 full deterministic closed-loop trajectories without scripted actions; record success, final distance, length, near-target speed and crossing.
- [x] If stable enough to fly, run one real human-viewer full episode; report what is actually seen.
- [x] Run full RL regression suite; inspect reports/data/model artifacts and answer A–E only from evidence.
