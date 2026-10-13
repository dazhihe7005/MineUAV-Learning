# PX4–MuJoCo Phase 3B: BC/PPO Policy Shadow Preflight

Date: 2026-10-13. Branch: `feat/px4-mujoco-policy-shadow-preflight`, based on `63e504eb1835a53a08d975476ba2dc10cc503d5d`.

## Scope and outcome

**PASS — offline inference/interface only.** Seven existing checkpoints loaded read-only, accepted identical recorded 7D observations, produced deterministic finite normalized actions, and converted them into *data representing* PX4 NED velocity/yaw-rate setpoints. No PX4 process, MuJoCo episode, arm request, Offboard message, model update or training was run. PX4 flight and policy transfer are **NOT TESTED**. The 25 recorded scripted PI-environment trajectories are nominal MuJoCo states, **not** FAST-LIO2 or SLAM estimates.

The replay selection was fixed before inspecting model outputs: `fresh_000_scripted` through `fresh_024_scripted` from the prior matched nominal evaluation, 3,620 policy-boundary samples. The raw NPZ/JSON traces remain local-only in `/home/keyousuke/MineUAV-Learning/mujoco/reports/uav_ppo_bc_nominal_comparison_parts`; every NPZ SHA256 was checked against its paired historical JSON, every JSON self-hash was recomputed, and the ordered trace-hash list was pinned to the **committed** matched-comparison report. That digest is `927cdd8705f58fb3fccf34cf5f9ed9fec1bc380cf5e8faef1082ec3f7cb3b2ac`. Reproduction can select a different local trace directory using `--trace-root`, but its selected 25 traces must match the pinned hashes. This fixed cohort is only an offline interface probe, not an estimate of generalization or closed-loop success.

## Source contract audit

The ground-truth contract is in `mujoco/rl/mine_uav_env.py::_get_obs` and `mujoco/control/velocity_command_controller.py::map_normalized_action`:

| Signal | Exact PI-training meaning | External preflight representation |
|---|---|---|
| Input 0–2 | `target_position - position`, MuJoCo world ENU, m | Same target-minus-current ENU error |
| Input 3–5 | MuJoCo world-frame translational velocity, m/s | Same world ENU velocity |
| Input 6 | `atan2(sin(target_yaw - yaw), cos(...))`, rad | Wrapped ENU yaw error; reset validity gate on target change |
| Output 0–2 | Normalized world ENU velocity command | Multiply by `[1.5,1.5,1.0]` m/s, then `enu_to_ned`: `[vy,vx,-vz]` |
| Output 3 | Normalized positive-ENU yaw rate | Multiply by `1.0` rad/s, then negate for NED yaw rate |

Both policies use 7D observation and normalized 4D action bounds `[-1,1]`. This common *semantic* contract does **not** imply common preprocessing. `mujoco/rl/uav_bc_policy.py::BCPolicy.raw_actions` first uses the checkpoint's Train-only observation mean/std, denormalizes the 4D network output using Train-only action mean/std, and `BCPolicy.predict` clips to `[-1,1]`. `mujoco/rl/uav_ppo_multiseed.py::config` specifies raw 7D float32 without VecNormalize; `uav_ppo_multiseed_eval.py::run` evaluates SB3 `PPO.predict(..., deterministic=True)` and retains the environment's normalized action space. The new `FrozenPolicy` preserves these paths rather than merging them. `px4_policy_shadow.py` uses the already-tested Bridge `enu_to_ned` transform, but imports no transport or Offboard sender.

The original BC and yaw-augmented BC actors are 7→128ReLU→128ReLU→4 (18,052 parameters each). The five PPO models use separate 7→64Tanh→64Tanh actor/critic branches with diagonal Gaussian action distribution; deterministic output is the clipped policy mean. Each PPO archive records its expected independent seed and 100,352 training steps. Historical SHA identities were checked against the committed matched-comparison/multiseed reports and again before/after inference; full hashes, parameter hashes and per-model normalization statistics are in the machine-readable report. **No checkpoint changed.**

## Validation and safety semantics

The external-observation constructor checks dimensions, finite values, world-frame error sign and yaw wrap. A per-episode gate rejects non-increasing microsecond timestamps, age above 120 ms, yaw errors outside ±π, and wrapped yaw changes exceeding `4 rad/s × sample interval + 0.02 rad`. These are fixed *offline interface validity checks*, not a validated PX4 flight envelope; the gate must reset on target changes. The action adapter rejects nonfinite/wrong-size values, out-of-range normalized commands and stale source samples; it does not silently clip substantial excursions. Its tiny `1e-6` boundary tolerance only handles numerical rounding. Elementwise bounds do not establish collision avoidance, estimator health or Offboard safety. No sender exists in the module.

The replay contained 3,620/3,620 accepted valid samples, 0 physical-invalid and 0 gate-rejected. The public `shadow_step` validity gate and action adapter were exercised for **every** accepted sample of **every** model, with a fresh gate per historical episode. Replay timestamps are reconstructed as `step_index × 40ms` and `now=sample_time`, so the replay verifies sequential ordering/finite/yaw continuity but **does not** validate real PX4 estimator age or clock mapping; stale rejection is supported by unit tests only. Model outputs were repeated exactly on the same input and parameter/checkpoint hashes remained unchanged. Positive ENU X maps to NED Y; positive ENU Y maps to NED X; positive ENU Z and positive ENU yaw-rate map to negative NED components. Unit tests cover signs, extremes, invalid/stale input and replay SHA/action indexing.

## Offline shadow observations

Single-process CPU inference, one Torch thread, 500 fixed sample timing probes per model after 50 warm-ups; timings are **model calls only**, with loading reported separately. They exclude state-estimation transport, ROS/MAVLink, scheduler overhead and simultaneous multi-model execution.

| Model | Load ms, approx. | Inference P95 ms, approx. | Normalized output saturation (`any |a|≥.95`) | Near-target mean command speed, m/s |
|---|---:|---:|---:|---:|
| Original BC | 2 | 0.05 | 0% | 0.040 |
| Yaw BC | 1 | 0.05 | 0% | 0.042 |
| PPO seed0 | 567 | 0.12 | 0% | 0.114 |
| PPO seed1 | 7 | 0.11 | 0% | 0.123 |
| PPO seed2 | 6 | 0.12 | 0% | 0.108 |
| PPO seed3 | 6 | 0.11 | 0% | 0.114 |
| PPO seed4 | 6 | 0.12 | 0% | 0.102 |

The table rounds one run; full exact load, P50/P95/P99/max timings are in the JSON and change slightly across CPU runs. The first PPO load includes library initialization and is not representative of the subsequent loads. All seven single-model call maxima were below the 40 ms/25 Hz policy period (**PASS for isolated CPU inference, not hard real-time**). Process peak RSS was 461.9 MiB; sampled minimum system available memory was around 10 GiB; one worker, no observed OOM. Per-model process RSS snapshots are in the report but allocator retention prevents treating their differences as clean model footprints.

Near-target is `distance <0.2m` on 961 recorded samples. The original/yaw BC near-target mean commands are close on this nominal cohort; PPO seeds command larger mean speeds. A command with negative dot product against current velocity is counted as a *descriptive braking tendency*, not actual deceleration under PX4. Action differences from the recorded Scripted output and within-episode action changes are in the report. These trajectories were generated by Scripted with the original PI stack; evaluating a different policy on their states is off-policy comparison only.

For input-response illustration, the yaw-error scalar of the first real recorded observation was artificially replaced by −30°, −10°, 0°, +10°, +30° while every other component stayed fixed. This is **synthetic counterfactual input, not a physical replay**. Both BC models' normalized yaw-rate output increases by about 1.03 across −30°→+30°; four of five PPO seeds have a negative slope and one a small positive slope. The PPO behavior is an output-sensitivity concern under large-yaw inputs outside their nominal training coverage, **not** proof of a coordinate conversion fault or physical-control failure. At +30°, original BC emits a markedly different XY command from yaw-augmented BC, consistent with prior yaw-coverage findings. See the two figures under `mujoco/reports/px4_mujoco_policy_shadow_figures/`.

## Reproduce

From the feature worktree, with the existing project virtual environment and local historical traces:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco:mujoco/rl \
  /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  mujoco/rl/px4_policy_shadow_analysis.py \
  --trace-root /home/keyousuke/MineUAV-Learning/mujoco/reports/uav_ppo_bc_nominal_comparison_parts
```

The script verifies recorded-trace/checkpoint hashes, loads and releases one model at a time, writes aggregate JSON atomically and never imports or calls a PX4 transport sender. Separate verification: **30 new offline policy-shadow tests passed**, and **356 inherited Bridge tests passed** (two existing nonfatal invalid-angle warnings); no full-repository/CAD suite was run. The independent read-only review initially found two Important issues (public gate bypass and unpinned local-only cohort), both fixed with regression tests; final re-review found **0 Critical / 0 Important** remaining and did not itself rerun tests. Timing values vary with CPU load and should not be used as hard real-time guarantees.

## Acceptance and next boundary

| Question | Status | Evidence/limit |
|---|---|---|
| Seven frozen models loaded with expected identity | PASS | SHA and parameter hashes, seed/budget checks |
| Original 7D and 4D semantics audited | PASS | Historical training/environment source plus converter tests |
| Model-specific BC/PPO preprocessing preserved | PASS | Direct prediction equivalence tests |
| ENU→NED velocity and yaw-rate signs | PASS | Literal positive/negative unit tests |
| CPU 25Hz isolated inference | PASS | All measured maxima <40ms on this machine |
| Offline no-transmission Shadow Mode | PASS | 3,620 recorded samples; pure return-value API |
| PX4 estimator-derived 7D observation | NOT TESTED | Phase3A live EV fusion not yet verified |
| PX4 Offboard transport, failsafe, closed-loop success | NOT TESTED | Deliberately excluded; no SITL run |

Remaining integration work, **not executed**: obtain validated PX4 estimator position/velocity/yaw with target provenance and freshness; verify frame and yaw semantics from estimator outputs; implement an independently safety-reviewed Offboard sender with mode/arming/failsafe and bounded command lifecycle; test disarmed reception before any separately authorized flight. A successful PI-environment BC/PPO baseline cannot be imported as a PX4 flight-success claim.
