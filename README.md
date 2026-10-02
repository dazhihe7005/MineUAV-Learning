# MineUAV-Learning

Robot Learning and Sim2Real stack for a ~7 kg mine UAV.

## Architecture

```text
MuJoCo dynamics
→ classical low-level control
→ Gymnasium
→ PPO high-level velocity policy
→ robustness / Sim2Real
→ ROS2 / PX4
→ LiDAR / Vision
→ World Model / WAM
```

The learned policy outputs:

```text
[vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
```

instead of directly commanding motors.

## RL Interface

Observation:

```text
[target error xyz, world velocity xyz, yaw error]
```

Action:

```text
[vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd]
```

Timing:

```text
physics    500 Hz
controller 100 Hz
policy      25 Hz
```

## Key Findings

### PPO exploration scale

SB3 PPO's default continuous-action exploration scale was too large for this precision waypoint task.

Changing only:

```text
log_std_init: 0 -> -2
```

produced a strong improvement. Five training seeds achieved about 95.6% mean success on both fixed benchmark and holdout waypoint sets.

### Robustness attribution

Mass, thrust coefficient and constant external force caused major failures under the proportional velocity loop. Scripted-policy and low-level hold tests showed this was not only an RL policy issue.

### PI controller and observability

Adding velocity PI with anti-windup restored strong constant-disturbance rejection, but changed the closed-loop dynamics and introduced hidden controller state.

A diagnostic observation augmented with PI integral acceleration restored seed-0 holdout performance to 100/100.

## Documentation

See:

```text
docs/PROJECT_JOURNEY_AND_INTERVIEW.md
```

for the complete project history, failed experiments, diagnostics, equations and interview preparation notes.

## Current Status

Next steps:

- multi-seed validation for PPO + PI observable state
- robustness evaluation
- Domain Randomization
- sensor noise / delay
- ROS2 / PX4
- history-based policy
- World Model / WAM

## Notes

Several physical parameters such as COM/inertia and yaw torque coefficient are engineering estimates and should be recalibrated before claiming real-world quantitative accuracy.
