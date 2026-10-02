# Robustness attribution audit design

## Intent and scope

Determine whether the frozen Baseline v1 robustness cliff is primarily caused by the learned high-level policy or by the unchanged velocity/attitude/allocator stack. This is evaluation only: no PPO training, reward changes, controller tuning, or model-file edits.

## Comparable waypoint evaluation

Use the exact existing 100 full-task holdout seeds and targets for the verified `scripted_action` and all five low-std 100k PPO checkpoints. Use `RobustnessEnv` with Reward V2, headless rendering, deterministic PPO, and the same nominal low-level controller for both high-level policies. Evaluate one shared nominal condition, then mass and thrust factors 0.90, 0.95, 0.98, 0.99, 1.01, 1.02, 1.05, 1.10, and world-X forces -10, -5, -2, -1, +1, +2, +5, +10 N. Only the selected plant parameter changes; no simultaneous perturbations. Preserve the previous robustness report and parts. Each new condition gets an exclusive-create result file so interrupted work can be resumed without overwriting prior data.

For every episode record success, termination, length, final distance, successful completion time, crossing under the existing signed target-plane definition, and final-window mean world-frame position error (`target - position`), velocity, and mapped velocity command. The primary tail window is the last 5 simulated seconds available; also record the last 1 second to distinguish terminal stopping from the preceding approach. Record actual window span and sample count because successful episodes can end before 5 seconds. Aggregate each metric first within each policy over 100 waypoints, then report scripted and five-seed PPO means/SD/min/max; retain all per-policy results. A missing near-target or successful-completion statistic remains null, not zero.

## Low-level attribution probe

Use the same velocity-command controller and plant injection path but no waypoint policy. Start from level `(0,0,1)` with zero velocity and zero yaw rate; hold zero velocity command for a 2-second nominal settling interval. Then apply one of mass 0.95/1.05, thrust 0.95/1.05, force-X -5/+5 N, or nominal, and continue fixed zero command for 15 simulated seconds. The hold-only environment lifetime is 18 seconds so this 17-second probe is not cut by the unchanged 15-second waypoint-task limit. The environment target is deliberately set to a valid, distant waypoint solely to prevent the task's success termination; controller input remains zero throughout. Record position error relative to the original hover point, world velocity, command, terminal reason, and final 5-second/1-second means. Do not reset the MuJoCo/controller state at the perturbation instant.

## Interpretation and outputs

Produce one JSON report under `mujoco/reports/` and three success-rate curves with scripted alongside the five-seed PPO mean and min–max band; save per-condition compact JSON so every level is inspectable. The report compares nominal and each perturbation, terminal steady-state error and compensating command, and low-level hold behavior. Explain A/B/C using measured behavior, distinguishing a policy-driven stop from controller steady-state error. Avoid claiming true physical robustness or assigning blame from success alone. No Domain Randomization or controller modification follows automatically.

## Error handling and verification

Reject mixed target sets, wrong checkpoint order, overwritten report paths, malformed/nonfinite results, and early/empty trajectory windows. Check the new nominal scripted result against the existing holdout scripted reference if available, and new PPO nominal results against the previous robustness audit. Verify one physical runtime perturbation against the previous `RobustnessEnv` constructor path, the full grid/episode counts, output plot integrity, and relevant inference-only tests.

## Ruling on execution gate

The user supplied the experiment, limits, and expected conclusion, and previously granted full execution authority without repeated consent. Proceed with the documented implementation inline. Cost if wrong: the chosen 2-second nominal pre-hold and last-5/last-1-second summary convention may differ from an unstated preference; both are explicit in the report and require no controller or task change.
