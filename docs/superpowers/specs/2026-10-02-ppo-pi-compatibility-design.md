# PPO + Velocity PI Compatibility Experiment

The user wants to determine whether a PPO policy trained from scratch with the previously validated, fixed velocity PI loop regains precise nominal waypoint stopping and retains the PI loop's static-disturbance benefit. This is one-seed experimental evidence, not a new robust baseline or domain-randomization study.

## Invariants

- Reward V2, 7D observation, 4D normalized velocity/yaw-rate action, 500/100/25 Hz timing, actuator allocation and attitude loop remain unchanged.
- PI is exactly `Ki_xy=0.5 s^-2`, `Ki_z=0.8 s^-2`, XY/Z integral-acceleration caps `1.5 m/s²`, existing conditional anti-windup, and zero integral on every reset. No Ki search.
- PPO remains CPU, MlpPolicy, separate `[64,64]` actor/critic, `log_std_init=-2.0`, eight DummyVecEnv environments and all Baseline v1 hyperparameters unchanged. Train only one explicitly recorded seed, `0`, from random initialization for the last full rollout at 100,352 transitions. Seed 0 permits same-policy-seed A/B historical comparisons; model weights are newly initialized rather than resumed.
- Historical P-controller and direct-PI checkpoint reports, existing models, XML, reward, observation and action code are read-only. New outputs use `ppo_waypoint_pi_lowstd_*` paths.

## Flow and decisions

1. An opt-in PI environment wrapper configures the current `VelocityCommandController` after normal `MineUAVEnv` construction. Training envs remain headless, monitored and independently seeded. Evaluation wrappers also capture the world-frame integral state/contribution at policy boundaries without exposing it to PPO. Reset asserts zero state.
2. A new training callback logs full-rollout PPO metrics, saves updated 20k/50k/100k checkpoints (actual 20,480/51,200/100,352), and evaluates deterministic fixed100 benchmark and independent fixed100 holdout waypoints at each checkpoint. Integral traces are saved independently of summary metrics, with per-episode success/failure linkage.
3. The 100k nominal gate requires at least 90/100 success on **both** benchmark and holdout. If either fails, stop with the nominal report; do not evaluate disturbed dynamics or train further.
4. Only after the gate passes, evaluate the same 100k PI-trained checkpoint on the same holdout 100 for mass and thrust multipliers `0.95, 0.98, 1.0, 1.02, 1.05` and world-X force `-5, -2, 0, 2, 5 N`. Reuse the one nominal holdout result as all three zero points. No retraining.
5. Compare A: historical seed-0 PPO with P, B: the same historical seed-0 PPO with PI, C: new seed-0 PPO trained with PI, using paired holdout targets. Report success, distance, completion time, crossing, near-target actual/command speed, action saturation, integral-state bounds and failure association. A/B reports already exist, so no duplicate A/B simulation is necessary.

## Error handling and verification

Reject output-path collisions, wrong waypoint digests, wrong reward/controller settings, non-finite PPO metrics or integral state, wrong checkpoint update counts, and missing complete evaluations. Interrupted runs preserve completed checkpoint/report parts. Tests exercise PI reset, exact seeded environment setup, trace/evaluation alignment and the nominal gate. The experiment stops after conditional robustness evaluation and does not alter the default P environment or prior PPO policy.
