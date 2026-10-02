# MineUAV Gymnasium interface baseline

This stage only establishes a waypoint environment; it does **not** train PPO
or send policy actions to motors. In the project `.venv`, install only
`gymnasium>=1.2,<1.4` in addition to the existing MuJoCo dependencies.

The policy output `[-1,1]^4` maps to world-frame `[vx,vy,vz]` limits
`[±1.5,±1.5,±1.0] m/s` and yaw-rate `±1.0 rad/s`. The velocity outer loop
computes `a_des = Kv(v_cmd − v)` with `Kv=[1.5,1.5,2.0] s⁻¹`, horizontal
acceleration limited to `3 m/s²` and vertical to `±3 m/s²`. It reuses the
existing `acceleration_to_thrust_vector`, `thrust_vector_to_attitude`,
quaternion attitude PD and bounded four-rotor allocator. The low-level yaw
target integrates the commanded yaw rate each 100 Hz controller update and
is reset to zero every episode. There is no motor lag or observation noise.

The MJCF physics timestep is `0.002 s` (500 Hz). The controller runs every
5 physics steps (`0.01 s`, 100 Hz), while the policy runs every 20 physics
steps (`0.04 s`, 25 Hz). Each normal `step(action)` holds the action for
4 controller updates / 20 physics steps; a failure can stop that step early.

Observation is a float32 Box with 7 components:
`[target_x−x, target_y−y, target_z−z, vx, vy, vz, wrap(target_yaw−yaw)]`.
Position and velocity are MuJoCo free-joint/body-origin values in world
coordinates; yaw target is zero in this baseline. The other attitude states
are internal to the low-level controller, not observations.

Reward per policy step is
`10*(previous_distance−current_distance) − 0.005*||action||²
 + 10*success − 10*failure`. `info["reward_breakdown"]` exposes all four
terms. Success requires distance `<0.10 m` and speed `<0.15 m/s` for
5 consecutive policy steps. Failure (`terminated=True`) covers non-finite
MuJoCo state, `z<0`, horizontal radius `>6 m` or altitude `>4 m`, and tilt
`>60°`. A 15 s limit sets `truncated=True` only if not terminated. Reset
puts the UAV at `(0,0,1)`, level and stationary, clears MuJoCo and yaw-target
state, and samples x/y uniformly from `[-2,2]`, z from `[0.7,1.5]`.

From project root:

```bash
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_*.py' -v
.venv/bin/python mujoco/rl/test_env_random_actions.py
.venv/bin/python mujoco/rl/test_env_scripted_policy.py
```

The random script calls `gymnasium.utils.env_checker.check_env` with the
render-only check disabled because this environment is intentionally headless.
Results are written to `mujoco/reports/rl_random_actions.json` and
`mujoco/reports/rl_scripted_policy.json`. The scripted policy uses
`v_cmd=(1/3)*position_error` (clipped to velocity bounds) and
`yaw_rate_cmd=wrap(yaw_error)` (clipped to ±1 rad/s). This is an interface
probe, not a learned policy or a tuned flight controller.

## Optional desktop viewer

`MineUAVEnv(render_mode=None)` remains fully headless: no viewer import,
window, synchronization or sleep. `MineUAVEnv(render_mode="human")` lazily
opens MuJoCo's official `mujoco.viewer.launch_passive(model, data)` on first
reset/render, using the **same** `MjModel` and `MjData` that `env.step()`
advances. Each policy step calls `viewer.sync()`, then sleeps only if the
simulation time is ahead of wall-clock time. The 500/100/25 Hz stepping
cadence is unchanged. Repeated resets reuse the window; `close()` requests
viewer shutdown and gives the passive X11/GLX thread a short human-only
cleanup interval. Closing the window manually makes the watch loop exit.

From the project root on a desktop X11 session:

```bash
DISPLAY=:0 .venv/bin/python mujoco/rl/watch_scripted_policy.py
```

The default repeats fixed-target `(0,0,1) → (2,2,1.5)` episodes until the
viewer closes or Ctrl+C. Use `--episodes 1` for one full flight. The terminal
prints state/command/distance at 5 Hz and holds the final frame for about
2 seconds. It imports the exact `scripted_action()` already used by the
headless validation; observation, action, reward, termination and controller
gains are unchanged. On systems where `DISPLAY` is already set, omit the
`DISPLAY=:0` prefix.

## PPO 20k smoke run

The project `.venv` uses CPU-only PyTorch plus `stable-baselines3` and
`tensorboard`. This stage does not change the environment's observation,
action, reward, controller, or termination logic. The training entry point is
headless; only the evaluation command with `--human` opens a Viewer:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python mujoco/rl/train_ppo_waypoint.py
DISPLAY=:0 .venv/bin/python mujoco/rl/eval_ppo_waypoint.py --human --episodes 1
```

Training uses eight independent `Monitor(MineUAVEnv(render_mode=None))`
instances in one `DummyVecEnv` process and a separate evaluation environment.
Every two complete rollouts it evaluates five waypoints deterministically;
after training it evaluates 20 randomly seeded waypoints. `Monitor` logs
episode return, length, final distance, and termination reason. The actor
and critic are separate 64×64 MLP branches after a parameter-free flattening
extractor. Each rollout collects `8 × 256 = 2,048` transitions, with PPO
batch size 256 and 10 epochs. A hard cap prevents SB3 from silently rounding
20,000 up to 20,480: exactly 20,000 transitions are sampled, while nine full
rollouts (18,432 transitions) participate in gradient updates. The final
1,568 samples are not used for an update because SB3 requires a complete
rollout. This is deliberate and reported explicitly, not a hyperparameter
change.

The checkpoint is `mujoco/rl/models/ppo_waypoint_smoke.zip`; TensorBoard
events are under `mujoco/rl/tensorboard/ppo_waypoint_smoke/`; per-environment
Monitor logs and periodic evaluation JSONL are under
`mujoco/rl/logs/ppo_waypoint_smoke/`; the final metrics report is
`mujoco/reports/ppo_waypoint_smoke.json`. The training command preserves an
existing checkpoint instead of overwriting it. To inspect logs:

```bash
.venv/bin/tensorboard --logdir mujoco/rl/tensorboard/ppo_waypoint_smoke
```

## Fixed 200k PPO baseline

`train_ppo_baseline.py` starts a **new** PPO model from scratch; it does not
load the smoke checkpoint. It reuses the exact same environment, eight
headless Monitor-wrapped DummyVecEnv instances, CPU PPO configuration, and
independent evaluation environment. Each of four milestones evaluates the
same 100 seeded random waypoints with `deterministic=True` and records mean
reward, episode length, final distance, success rate, and completion time
for successes. A separate final 100-waypoint evaluation uses a second fixed
seed set and records action quantiles and fractions with `|action| >= 0.95`.

For a new workspace without existing baseline checkpoint files:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python mujoco/rl/train_ppo_baseline.py
.venv/bin/python mujoco/rl/plot_ppo_baseline.py
DISPLAY=:0 .venv/bin/python mujoco/rl/eval_ppo_waypoint.py \
  --model mujoco/rl/models/ppo_waypoint_100k.zip --human --episodes 1
```

The four model files are `mujoco/rl/models/ppo_waypoint_{20k,50k,100k,200k}.zip`.
Actual trained-policy timesteps are 20,480, 51,200, 100,352, and 198,656.
With the fixed 2,048-transition rollout, 198,656 is the last complete PPO
update before the strict 200,000-step ceiling; the script does not sample or
train beyond it. The checkpoint hook runs only after the corresponding PPO
update. Training metrics are saved after each update in
`mujoco/rl/logs/ppo_waypoint_baseline/training_history.jsonl`, and the
TensorBoard run is under `mujoco/rl/tensorboard/ppo_waypoint_baseline/`.
The complete evaluation and action data are in
`mujoco/reports/ppo_waypoint_baseline.json`; four `ppo_*.png` training and
evaluation curves are generated in the same reports directory. The training
command refuses to overwrite existing baseline checkpoints or logs.

## Reward V2: near-target braking (100k)

`MineUAVEnv` defaults to the unchanged V1 reward. Select `reward_version="v2"`
to retain its progress, action, success and failure terms and add only
`-0.5 * clip((0.5 - distance) / 0.4, 0, 1) * (vx² + vy² + vz²)`. The V2
`info["reward_breakdown"]` exposes `progress`, `action`, `brake`, `success`,
`failure`, and `total`. No observation, action, controller, timing, success
criterion or PPO setting differs from the baseline.

The experiment uses a fresh model and the same fixed 100 evaluation seeds
as the baseline milestones (`20261001` through `20261100`). The three V2
checkpoints are saved after full updates at 20,480, 51,200 and 100,352
timesteps. The script re-evaluates the original 100k checkpoint under V1
with the exact same diagnostics and waypoint seeds. Conditional mean speed
at each distance threshold pools policy steps, not episodes. The standalone
`speed < 0.15 m/s` episode fraction includes early post-reset steps near
rest and must be interpreted alongside the simultaneous distance-and-speed
fraction. Raw V1/V2 returns are not performance-comparable.

```bash
.venv/bin/python mujoco/rl/train_ppo_reward_v2.py
DISPLAY=:0 .venv/bin/python mujoco/rl/eval_ppo_waypoint.py \
  --model mujoco/rl/models/ppo_waypoint_reward_v2_100k.zip \
  --reward-version v2 --human --episodes 1 --seed 20261001
```

The report is `mujoco/reports/ppo_waypoint_reward_v2.json`; TensorBoard
events are under `mujoco/rl/tensorboard/ppo_waypoint_reward_v2/`, and
Monitor, training-history and milestone logs are under
`mujoco/rl/logs/ppo_waypoint_reward_v2/`. Existing V2 artifacts are never
silently overwritten.

The completed run's TensorBoard near-0.1 m speed scalar originally encoded
undefined 20k/50k means as zero. The current event file omits those two
points while preserving all PPO scalars; the unmodified original event file
is archived under `mujoco/rl/logs/ppo_waypoint_reward_v2/original_tensorboard_events/`.
`repair_reward_v2_tensorboard.py` documents and reproduces this event-file
correction for an uncorrected run, without changing training checkpoints.

## Reward V3: continuous distance cost (100k)

Select `reward_version="v3"` to use every V2 reward term plus exactly
`r_distance = -0.02 * current_distance_m`. V3 keeps `lambda_brake=0.5`;
the observation/action interfaces, termination, controller, timing and PPO
configuration do not change. Its breakdown adds `distance` to V2's
`progress`, `action`, `brake`, `success`, `failure`, and `total` fields.

`train_ppo_reward_v3.py` starts fresh, saves three independent updated
checkpoints at 20,480, 51,200 and 100,352 timesteps, and evaluates each
on the same 100 seeded waypoints. The report checks all 100 target
coordinates against the existing baseline and V2 100k evaluations before
comparing performance. Episode rewards must not be compared across reward
versions because their definitions differ.

```bash
.venv/bin/python mujoco/rl/train_ppo_reward_v3.py
DISPLAY=:0 .venv/bin/python mujoco/rl/eval_ppo_waypoint.py \
  --model mujoco/rl/models/ppo_waypoint_reward_v3_100k.zip \
  --reward-version v3 --human --episodes 1 --seed 20261001
```

The V3 report is `mujoco/reports/ppo_waypoint_reward_v3.json`. Models,
TensorBoard events and Monitor/history logs use separate
`ppo_waypoint_reward_v3_*` names and directories. At 100k, the fixed
100-waypoint evaluation succeeded 0 times; Viewer and diagnostics still
show target-area crossings without stable stopping. No reward coefficient
was changed after this result.
