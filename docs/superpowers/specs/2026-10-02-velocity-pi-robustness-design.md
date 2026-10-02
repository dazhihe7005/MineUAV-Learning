# Velocity PI Robustness Audit Design

## Intent and boundary

Test whether adding integral action to the existing world-frame velocity-command outer loop removes static-disturbance velocity error and improves frozen high-level waypoint performance. Do not train or alter PPO, Reward V2, observation/action mappings, allocator, attitude loop, dynamics XML, or the 100 fixed holdout targets. Preserve the historical P-only controller and all old reports as a reproducible reference.

## Controller design

The only flight-control algorithm change is in `mujoco/control/velocity_command_controller.py`. Its existing `kv=[1.5,1.5,2.0] s^-1` remains unchanged. Add optional `ki_xy` and `ki_z` gains, defaulting to zero for legacy P mode; PI experiments explicitly construct this same controller with conservative `ki_xy=0.5 s^-2` and `ki_z=0.8 s^-2`. At each 100 Hz control update, integrate world-frame velocity error using actual `control_dt` and compute `a_raw = kv*(v_cmd-v) + Ki*integral_error`. Keep the existing horizontal/vertical 3 m/s² acceleration limits and all downstream thrust/attitude/allocation logic.

Limit the XY integral *acceleration contribution* to a 1.5 m/s² Euclidean norm and Z contribution to ±1.5 m/s². Conditional anti-windup rejects an integration increment when the corresponding raw acceleration exceeds its existing limit and the increment pushes farther into saturation; increments that unwind saturation remain allowed. `reset()` clears integral state, last acceleration, and anti-windup counters. `MineUAVEnv.reset()` already invokes the controller reset, so no environment code or RL interface change is needed. Expose read-only integral contribution/last commanded acceleration for diagnostics.

## Evaluation design

An independent inference-only script under `mujoco/rl/` explicitly enables PI on each private `RobustnessEnv`; the default P setup and old attribution scripts remain unchanged. It first runs seven zero-velocity-command probes after 2 s nominal hover, with the same plant-only disturbances, target, and safety limits as the prior attribution audit. Record state, position/velocity error, integral state/contribution, desired acceleration, motor RPM, safety termination, and late-stage 1 s/5 s summaries. A zero-velocity command is not position hold: quantify residual displacement separately from final drift rate.

Only after nominal and disturbed low-level probes have finite, bounded states and no sustained oscillation, evaluate scripted and all five frozen PPO checkpoints on the same fixed 100 holdout targets at mass/kf −5,−2,−1,0,+1,+2,+5% and world-X force −5,−2,−1,0,+1,+2,+5 N. Use deterministic PPO and Reward V2. Reuse the prior attribution episode/tail/crossing definitions; evaluate each policy in its own private environment. Preserve old P results and write new PI parts/report/curves under distinct names. Report five-seed mean, sample SD, min, max; compare nominal, ±5 levels, late-stage command/actual velocity and position error, saturation/windup/oscillation, and whether frozen PPO robustness improves.

## Interpretation safeguards

PI can remove steady *velocity* bias without returning a zero-command aircraft to its starting *position*. Safety-terminated probes have shorter observation windows and are not steady-state evidence. One training seed or one waypoint never substitutes for the five-seed/100-waypoint grid. If the initial conservative PI gains destabilize a probe, make at most a documented minimal manual stability correction before the full waypoint grid; no search or training.
