# Frozen latent random-shooting MPC v1

Branch: `feat/latent-random-shooting-mpc`.
Base: `feat/joint-v1-v3-multiseed`, `e3ce35a0c736b3332c70dddd9c7d03cc20627b0d`.

Question: can the fixed seed0 autonomous-consistency v3 dynamics support useful
closed-loop waypoint action selection? This is not policy/reward learning,
Dreamer or a full model-based RL agent.

## Fixed protocol

- Frozen checkpoint `joint_latent_world_model_v3_autonomous_consistency.pt`,
  SHA256 `42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9`.
  E/T/D remain eval, requires_grad=False; no optimizer exists in this experiment.
- E is GRU11→64; T residual Tanh MLP68→128→128→64; D Tanh64→128→128→7.
  Use original Train-only observation/action normalization; no latent rescaling.
- Online causal E receives one real observation7 and previous executed normalized
  action4 at each decision; its hidden state resets at every episode. Incremental
  encoding is equivalent to encoding the complete real prefix, not zero-memory.
- 512 candidates, 10 steps (0.4s). Candidate0 is allzero, candidate1 repeats the
  previous command. For each of the other510, set a[-1]=previous executed command
  and a[k]=clip(a[k-1]+epsilon[k],-1,1), k0..9; epsilon iid Gaussian with per-axis
  std0.5×archived Train action std. First epsilon is applied before first action.
- Commands are **normalized**, not physical velocities: physical scale is
  `[1.5,1.5,1.0,1.0]` (m/s,m/s,m/s,rad/s). Sampling/smoothness use normalized units;
  decoded terminal position/velocity/yaw cost uses physical units.
- From common z_t, each candidate follows z'=z+T(z,normalized_action); D reads
  terminal z. No decoded observation feeds E; no simulator future is consulted.
- J=||error_xyz_H||²+0.5||velocity_xyz_H||²+0.1yaw_error_H²
  +0.05mean_k||a[k]-a[k-1]||². This is a fixed **hand-designed planning cost**,
  not RL reward or a dimensionally homogeneous physical energy objective.
- Select argmin (first index for ties), execute only its first action, then get
  next real observation and re-encode/replan. No CEM/MPPI/gradient planning/tuning.
- Existing MineUAVPIEnv RewardV2, nominal dynamics/PI/success/failure unchanged:
  distance<0.1m AND speed<0.15m/s for5 consecutive steps; 15s/375-step timeout.
  Benchmark100 and holdout100 archived coordinates/seeds, explicitly restored
  at reset and hashed against prior Oracle-PPO evaluation, for all3 controllers.
- Zero baseline emits zero4; scripted baseline reuses `scripted_action` unchanged
  (physical velocity error/3 and yaw_error proportional command). No PPO runs.
- Seed per episode: SeedSequence([0,task_index,target_index]).generate_state(1)[0].
  Fresh planner RNG and recurrent state each episode; model seed0 never selected
  by test/planning performance. There is only one configuration.

## Metrics and diagnostic boundaries

All targets, including failures/timeouts, enter success/finaldistance statistics.
Completion time is success-only (null when none), not timeout-censored average.
Near-target speed averages pre-action real states with distance<0.1m across steps;
absent visits are null, not zero. Command speed uses physical translation scale.
Crossings: first entry within0.2m establishes target-plane axis, then alternating
±0.02m hysteresis crossings counted, including terminal observed state. Saturation
is fraction of selected normalized action **elements** with abs>=0.95; it is not
rotor allocator saturation. Episode action variance and target-direction response
are descriptive checks against constant commands, not proof of model causality.

OOD uses only the exact Train rows from original supervised normalization:
standardized selected-action L2 norm and fraction outside each axis's central95%
range (2.5..97.5 quantiles). This is not joint95% coverage, nor proof of in/out-of-
distribution state-action sequences. No additional95% clipping occurs.

Prediction sanity uses first3 fixed episode identities per split, irrespective of
outcome, with legal10-step starts every25steps. At each saved decision latent,
run T under the **actually executed next10 actions**, compare steps1/10 to matching
real observations. Only actions enter prediction; future observations are targets.
This is not accuracy of the original selected10-step candidate, because receding-
horizon replanning usually replaces its future actions. Selected terminal costs
and realized-prefix forecasts must not be conflated.

CPU, one Torch thread, all512 candidate trajectories batched, inference_mode.
Planning timer includes sampling/transition/terminal decode/cost/argmin, excludes
physics/disk/plotting. Total decision timer additionally includes real-history E.
Report mean/median/p95/max and fraction<=40ms; no real-time sleeping enforced.

## Reproduction and local data

```bash
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_latent_mpc*.py'
.venv/bin/python mujoco/rl/run_latent_random_shooting_mpc.py
.venv/bin/python mujoco/rl/latent_mpc_verification.py
```

Original local `pi_hidden_state_seed0/train.npz` is required only to verify Train
action provenance and compute action quantiles (no dataset regeneration/training).
Original archived benchmark/holdout integral-trace JSONs are read only for saved
coordinates/seeds; the committed report also contains the complete target manifest.
Large arrays are excluded: `latent_random_shooting_mpc_seed0_parts/` holds six
condition fragments, six small fixed-episode traces and two timing/action arrays,
with hashes in the committed report. Resuming trusts only complete conditions with
identical checkpoint/source/target/config identity; partial results are not used.
Post-review integrity guards bind condition caches to environment/PI/controller,
rotor XML/config/inertia, dataset manifest, and original hardware/runtime. They
reject incompatible reuse rather than relabel previous outcomes/latency. The
report verifier requires exactly3controllers×2splits×100episodes and derives
the count. Same-protocol evaluation was repeated after these metadata/verification
fixes; core sampling/cost/encoding/control code and all model parameters unchanged.

## Results (2026-10-08)

| Controller | Benchmark successes | Holdout successes | Final distance benchmark/holdout (m) |
|---|---:|---:|---:|
| Zero | 0/100 | 0/100 | 1.5681 / 1.5299 |
| Scripted | 100/100 | 100/100 | 0.07144 / 0.06874 |
| Frozen v3 shooting MPC | 0/100 | 1/100 | 4.2033 / 4.3860 |

MPC has153 physical failures (81outside-flight-area,72excessive-tilt),46timeouts
and1success, across200targets. Completion time1.24s represents **only one** success,
not a speed improvement over scripted5.54/5.81s across100successes per split.
Near-target actual/command speeds benchmark0.818/1.760, holdout0.408/1.654m/s;
only29/81near-target steps. Crossing11%/9% and command-element saturation39.15%/
40.24%; low crossings here mostly reflect failure to approach, not safe braking.

CPU AMD Ryzen9 7940HX, one Torch thread:44229MPCdecisions. Planning mean/median/p95/
max6.535/6.494/6.765/13.422ms; including E6.704/6.654/6.991/14.445ms. All observed
decisions are below40ms (offline run, not an OS real-time guarantee).

Selected actions exceed at least one coordinate's central Train95% range85.96%
of the time;45.35%of elements outside, mean standardized norm3.445. Commands span
the entire[-1,1]range and have high temporal variance. First-action correlation
with target x/y/z0.182/0.051/0.284: weak response, not a constant action or scripted
proportional equivalent, and certainly not evidence of good model decisions.

Fixed first3episodes/split provide37realized-prefix windows. Benchmark normalized
RMSE H1/H10=0.2224/1.2212, holdout0.2508/0.4107. One-step and10step discrepancies
are substantial under the planner's changed visitation; this subset does not
estimate all200episode errors. Keep the historical recorded-action Test metrics
separate. Yaw error uses the unchanged environment wrap and ordinary per-axis
error comparison; no circular-representation modification or prediction clipping.
Yaw branch-cut crossings can inflate ordinary RMSE (including the benchmark
H10 sanity subset); do not interpret the entire normalized discrepancy as a
geodesic angular error or a uniform all-episode prediction estimate.

Conclusion: **negative decision-usefulness result** for this one fixed planner.
It is fast enough on this CPU but worse than scripted and even zero on final
distance/safety. High action-support departure and forecast mismatch are
compatible with model exploitation/OOD risk, not proof of a unique cause; cost,
short horizon, candidate coverage, state OOD and partial dynamics also remain.
Do not upgrade to CEM/MPPI or present this as a usable controller yet.

One recommended next controlled experiment only: keep model/N/H/cost unchanged
and compare environment-bound shooting to a Train-supported action-constrained
shooting condition, testing whether action-support departure contributes to the
failure. This was **not performed**, and is not a proposal to alter PI/reward.
