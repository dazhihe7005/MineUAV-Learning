# UAV Behavior Cloning Policy Baseline — seed 0

## Outcome and scope

The independently executed neural policy succeeds on **100/100 benchmark and
100/100 holdout** episodes, matching the existing Scripted controller. There are
no physical failures, timeouts, action saturation, or deployment clipping. This
establishes a supervised policy baseline for the nominal PI waypoint task, not a
robust flight controller, visual policy, PPO experiment, or world-model result.
Canonical world-model v3 remains unchanged and is not used by BC.

Branch: `feat/uav-behavior-cloning-baseline`
Base: `feat/world-model-onpolicy-adaptation` / `4858ce65a29806f37a813f0225b6aa6adfa6cd15`

## Expert and environment

Expert is the existing `test_env_scripted_policy.scripted_action`. It reads only
current position error xyz and yaw error from the same 7D observation as BC. It
does **not** read velocity, PI integrals, attitude, motors, previous commands,
future states, or other privileged information. Its deterministic rule is:

```text
velocity_command = clip(position_error / 3, [-1.5,-1.5,-1], [1.5,1.5,1])
environment_action_xyz = velocity_command / [1.5,1.5,1]
environment_action_yaw = clip(yaw_error, -1, 1)
```

Thus no expert-information deficit is demonstrated here: the chosen expert is
fully representable from current 7D input. This does not establish that 7D is a
sufficient physical Markov state or that every possible controller can use it.
Expert labels are actual high-level commands, not actuator PWM or predicted
world-model trajectories.

Unchanged `MineUAVPIEnv(reward_version='v2')`: physics 500 Hz, control 100 Hz,
policy 25 Hz. Observation is target-minus-position xyz, world-frame velocity xyz,
and wrapped yaw error. Four legal normalized commands in [-1,1] map to
`[vx_cmd,vy_cmd,vz_cmd,yaw_rate_cmd]` with scale `[1.5,1.5,1,1]` in m/s and rad/s.
Success remains distance <0.10 m and speed <0.15 m/s for five consecutive policy
steps. PI, dynamics, reward, failure conditions, and timeout are unchanged.

## Dataset and isolation

Only target coordinates/IDs and grouped split from `pi_hidden_state_seed0` are
reused. Fresh complete expert episodes use seed `2026100900 + target_id`:

| Split | Targets/episodes | Action-labelled samples | Successes |
|---|---:|---:|---:|
| Train | 84 | 8,974 | 84 |
| Validation | 18 | 1,888 | 18 |
| Test | 18 | 2,199 | 18 |
| Total | 120 | 13,061 | 120 |

Grouping is by target, with IDs and coordinates checked for overlap. Archived
100 benchmark and 100 holdout target/reset-seed manifests are unchanged and have
zero coordinate overlap with all 120 dataset targets; no replacement evaluation
cohort was needed. Train-only normalization and region thresholds are saved.
Each episode has `o[0:n+1]`, `action[0:n]`, previous action (zero initially),
step index, target ID, episode ID, and final success/failure labels. All outcomes
are retained, not success-filtered. No Test samples are loaded by the trainer.

Target split, dataset and closed-loop manifests record source/runtime/config
identity, seeds, counts and per-file SHA256. Raw NPZ files stay ignored/local-only;
their absolute local paths are references, not vendored data. Resume rejects
stale identities, altered hashes, mislabeled targets, duplicate/omitted episodes.

## Model, scaling, and fixed training

MLP `7 -> 128 ReLU -> 128 ReLU -> 4 linear`, **18,052 parameters**. Train
observation mean/population std normalize input. The network predicts
Train-standardized actions; inverse scaling gives normalized environment commands.
Deployment clips only to original legal bounds. All stds use a 1e-6 floor.

```text
L = mean over samples and 4 dimensions of
    ((raw_predicted_command - expert_command) / Train_action_std)^2
```

CPU PyTorch, deterministic seed 0, Adam lr=0.001, batch 512, exactly 100 epochs.
No sweep, early stopping, gradient clipping, teacher fallback, recurrent policy,
or extra loss. Lowest Validation loss alone selects the checkpoint.

Best epoch **100**. Best Train loss **0.0000812689**, Validation loss
**0.0002641008**. Epoch-100 gradient norm mean/max **0.0110725 / 0.0171335**;
maximum gradient norm over training **0.837776**, all finite. Epoch-100 Validation
raw action RMSE/MAE **0.00155053 / 0.000759710**.

Best parameter SHA256: `771382758078def13665abef25ec5a9b3702b7938f265f55294605bcc5b60cfa`.
Checkpoint SHA256 and initial parameter hash are recorded in the JSON report.

## Offline Test imitation

2,199 independent Test samples: action MSE **0.000003510915**, RMSE **0.001873744**,
MAE **0.001037225**; Train-action-std normalized MSE **0.0004307048**, normalized
RMSE **0.02075343**. Raw and deployment-clipped errors coincide; no clipping.

| Command | Normalized-command RMSE | MAE | Correlation |
|---|---:|---:|---:|
| vx | 0.00239668 | 0.00170246 | 0.999828 |
| vy | 0.00273742 | 0.00182284 | 0.999735 |
| vz | 0.000897631 | 0.000611098 | 0.999790 |
| yaw rate | 0.0000187293 | 0.0000125061 | 0.999614 |

Horizontal physical-command RMSEs are 0.00359502 / 0.00410613 m/s; z and yaw use
unit scale. Distance-region normalized action RMSE (<.1/.1–.2/.2–.5/>=.5 m):
**.009474 / .011868 / .014835 / .030572**. Velocity/action-magnitude terciles use
Train thresholds only; their counts/errors are in the report.

## Independent closed-loop execution

BC receives current observation only. It never calls the expert, mixes teacher
actions, or accesses a world model during control. Expert labels on BC states are
computed **after** each finished episode solely for offline error diagnosis.
Scripted is separately executed for the matched comparison. Every fixed episode
is counted and uses the same target and reset seed across controllers.

| Metric | Scripted benchmark | BC benchmark | Scripted holdout | BC holdout |
|---|---:|---:|---:|---:|
| Success | 100/100 | 100/100 | 100/100 | 100/100 |
| Mean final distance (m) | .071441 | .066642 | .068744 | .063369 |
| Mean completion time (s) | 5.5376 | 5.7244 | 5.8132 | 6.0464 |
| Physical failures / timeouts | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Near-target actual speed (m/s) | .192370 | .192324 | .194906 | .193490 |
| Near-target command speed (m/s) | .019554 | .018658 | .018700 | .017867 |
| Crossing episode fraction | .38 | .43 | .43 | .43 |
| Command-element saturation | 0% | 0% | 0% | 0% |

Near-target speed averages **all** distance<.1 m pre-action steps, including
approach/transit before the success-speed hold. It is not terminal success speed;
there is no change to the success criterion. Saturation means |normalized a|>=.95.
BC completes slightly slower (+.1868 / +.2332 s) and preserves the expert's
nominal behavior, rather than proving overall superiority. All initial-distance
bins have 100% success; counts are saved, including the few shortest targets.

## Distribution shift and limitations

BC visited-state posthoc expert action RMSE **.003925 / .003213** exceeds offline
Test .001874. Train-standardized observation RMS means: Train .8719, BC
benchmark/holdout .9989/.9631, Scripted 1.0344/.9905. BC marginal central95% box
departure is 32.81%/29.78%, compared with Scripted 35.89%/31.86%. These are coarse
distribution diagnostics, not strict OOD labels or a causal shift test.

Some shift/action-error growth exists, but no failure accumulation or catastrophic
covariate shift was observed on these nominal targets. BC final distances are
slightly smaller, not grounds to tune the policy. The deterministic expert is
simple; target-yaw is nominal and Train yaw-error/action std is only .00090545.
Successful imitation does not demonstrate large-yaw, saturation, recovery, noisy
sensor, perturbed reset, unseen physics, or real-drone robustness. One training
seed and a ceiling-success benchmark also limit comparisons.

There is **no failed BC episode** in the fixed 200. The required failure figure
is an explicit no-failure notice, not a fabricated trajectory or a new experiment.

## Resource safety and verification

Exactly **one** MuJoCo worker. Simulation and training phases are serialized by a
lock. A stdlib-only supervisor measures full child-process-tree RSS/HWM, available
physical memory and kernel OOM counter every .1 s; start requires >=5.5 GiB
available, runtime aborts below 2 GiB (1.5 GiB reserve + .5 GiB margin), above
4 GiB phase RSS, or on a new OOM. Atomic fsync/replace publication and completed
episode records support recovery. No swap change, process killing outside the
owned phase, or intentional resource reservation for other apps was performed.
Polling telemetry is not a hard cgroup memory guarantee. No new OOM occurred.
Detailed phase peaks/minimum headroom are in the report; experiment-only peak
process-tree HWM before full-suite testing is **1.069 GiB**.

Verification independently checks all 120 expert labels/indices/hash identities,
400 closed-loop episode records and outcomes, Train-only scaling, 100-epoch
budget, Validation-only selection, offline metrics, and BC output independence.
Two real full episodes re-executed with bitwise-identical arrays. Frozen BC and
canonical v3 hashes are unchanged after evaluation. Regression tests cover the
actual supervised child shutdown and concurrent-phase rejection as well as split,
command/indexing, cache identity, checkpoint reload, no teacher access, seed
reproducibility, regions and complete failure accounting.

All **20 new tests pass**, including the three final-review regressions. The
pre-review full filesystem discovery with the same read-only historical Git
fixtures as the base experiment: **544 passed, 4 known skipped, 548 run**, no
failures. **Post-fix fresh full suite: 547 passed, 4 known skipped, 551 run**;
no failures. This loads missing legacy modules
`reward_v2_alignment_audit` / `ppo_reward_success_alignment` and their missing
diagnosis report directly from Git commit
`8879754a43302e9c55b27de4ce3c629c7e4e1a53` into memory; it does not restore or
modify those historical files. Pre-review bare discovery ran **486 tests: 455 passed,
30 pre-existing import errors, 1 known skipped**. All 30 errors are the same
missing `ppo_reward_success_alignment` dependency in historical PI/PPO tests
already disclosed by the base experiment; error test names and log hashes are
included in the report. No BC tests depend on this fixture and no new algorithm
test fails. Bare discovery is **not** claimed globally green.

Pre-review full-suite sampled process-tree peak RSS **2.654 GiB**, summed process HWM
**3.733 GiB**, minimum MemAvailable **8.542 GiB**, zero new OOM. The final post-fix
suite peak sampled RSS is **2.851 GiB**, summed HWM **3.908 GiB**, and minimum
MemAvailable **8.258 GiB**, also zero new OOM. Summed HWM can
combine non-simultaneous per-process peaks and is conservative, not a measured
simultaneous peak. Historical unrelated artifacts are preserved rather than
restored or included in this experiment.
The separate bare-suite run also recorded zero new OOM, minimum available
**8.508 GiB**; its nonzero exit is the historical import errors, not resource
failure. Full experiment/test-phase telemetry is retained in the aggregate report.

The sole independent, read-only code review found two Important lifecycle/report
edge cases. Both were fixed in one RED-to-GREEN pass: SIGTERM/abnormal supervisor
exit now cleans up its owned child before releasing the phase lock and records
interruption; publication/trajectory notices derive success/failure claims from
actual outcomes, with synthetic all-success, zero-success and mixed tests. The
real 200/200 BC result, checkpoint, labels, metrics and physics are unchanged.
No deferred Minor findings; no second review or extra experiment. Abrupt SIGKILL
or an OS crash is not a guaranteed recoverable cleanup path; atomic completed
records and identity checks remain the recovery basis.

## Reproduce

From repository root with the existing `.venv` and original local target archives:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python mujoco/rl/run_uav_behavior_cloning.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s mujoco/rl -p 'test*uav_bc*.py'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s mujoco/rl -p 'test_verify_uav_behavior_cloning.py'
```

The complete runner executes guarded sequential phases, validates existing
completed episode caches, reuses a complete matching trained model, and verifies
outputs before publication. Interrupted incomplete training restarts seed0's
entire fixed budget; it never silently continues with a shorter budget. Changing
source/runtime/data identities fails closed instead of relabeling old artifacts.

Artifacts: best `uav_bc_mlp_seed0.pt` (~75 KiB), aggregate JSON, split/dataset/eval
manifests, source/tests/verifier, seven figures, this document and experiment log.
Raw episodes, runtime/training logs, caches and scratch remain local-only.

## Conclusion and only recommended next experiment

Accept BC as the learned nominal policy baseline alongside Scripted/PPO; canonical
deterministic v3 is untouched. Next only recommend a fixed-manifest
**perturbed-initial-state BC robustness evaluation**, with no training/DAgger.
It is not executed here. No subsequent experiment starts automatically.
