# Phase 3E-L9S — Repaired PPO End-to-End Integration Regression

Base: `77e921f15a3e275df448641672f348991d27a3ce`.
Branch: `test/lidar-ppo-repaired-integration`.

## Scope and frozen identity

This is an independent, single-attempt integration Pilot, not a formal retraining
experiment or evidence of learned obstacle avoidance. Experiment ID:
`L9S_REPAIRED_PILOT_ONLY_SEED0`. The only permitted training budget is seed0,
one CPU worker, 4096 actual environment steps from a newly constructed random
PPO. One subsequent deterministic evaluation covers the six frozen Validation
scenes. Test episodes remain zero. No checkpoint initializes training.

L9R remains unchanged, including its nine runtime source files, repair protocol
`e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f`, and
`formal_retrain_authorized=false`. The original L9 remains **FAIL**. Its eleven
raw files, attempt, checkpoint, report and documentation are read-only.

The independent L9S protocol is published in
`mujoco/reports/mujoco_lidar_ppo_repaired_integration_protocol.json`:

`09837c2104b12a1780cf81b6eb06c9b6c7f6e167d3378381def706e6af9f5cb3`

The reviewed Pilot runtime source SHA256 is
`7ec0c79c5eb90e814f0a904e289fba8a7e81e98d96feadc9167cb81c252e1ff1`.
Python JSON float/int types are retained when checking the protocol digest.

## Integration and authorization

`ppo_repaired_pilot.py` composes the existing scheduler, `make_ppo`,
`execute_segments`, Health/TrainingMetrics, CheckpointManager and Validation
evaluator. It never invokes the old blocked Driver or changes its authorization.
The new CLI requires both `--l9s-pilot` and `--authorize-l9s-pilot`, and the
expected new protocol digest. The canonical output directory is
`mujoco/reports/mujoco_lidar_ppo_repaired_integration_parts/run_01`.
An exclusive directory reservation and atomic `pilot.attempt` prohibit retry
or overwrite, including after failure. There is no initializer or formal mode.

The unchanged frequencies are 500Hz physics, 100Hz PI, 25Hz policy and 10Hz
LiDAR, with 24000 rays/frame. Observation remains LiDAR `[3,12,24]`, state7,
heading2 and age1. Actions remain normalized4D with the original environmental
velocity scaling. CNN, Reward `NAV_V2_COLLISION_PREFLIGHT_V1`, PPO parameters,
12/6/6 manifest, original7kg model and seven historical checkpoints are frozen.

FormalHealth already inherits TrainingMetrics, so the environment is wrapped
once. The shared classifier preserves `terminal_class` and
`raw_termination_reason`, requiring mutually exclusive flags and consistent
terminated/truncated semantics. Unknown or contradictory transitions fail closed.
Actual physics ticks are summed from transitions; early terminal transitions
are not assumed to execute20 ticks.

The final checkpoint is saved after32 complete rollouts and their optimization.
SB3's64 epoch-update count is distinct from128 Adam step calls. Temporary Adam
hooks validate finite gradients/state and count actual calls; hooks are removed
before serialization. Reload compares model/optimizer contents and deterministic
prediction without additional environment steps. Python/NumPy/Torch RNG states
are preserved around reload and Validation. The Train scene is closed before
sequential Validation scenes, and its scheduler state is checked unchanged.
Resume remains `RESTART_REQUIRED`; reload verification is not continued training.

## Evidence and failure guards

Both Train and Validation check the1.5GiB MemAvailable and2GiB disk thresholds.
Validation uses a scoped passive environment factory around unchanged NavigationEnv
methods; the binding is restored in `finally`. It checks finite observations,
qpos/qvel/ctrl and causal LiDAR timing, and retains known scene identity and prior
reward in a post-transition exception. No dynamics, reward or termination rule is
modified. Global guard observations contribute to the reported resource minima.

Each completed training episode retains canonical/raw reasons, scene identity,
return, action norm, distance and trajectory length. Every reset/step also records
actual simulation time, original LiDAR sample time/age and physics ticks. The
read-only artifact analyzer independently recomputes these statistics and checks
the completion manifest, protocol, checkpoint and six Validation identities.

Failure files use the existing bounded recorder with unique IDs and exclusive
atomic publication. Returned transitions are distinguished from last-known context;
unavailable fields stay `NOT_AVAILABLE` and nonfinite values use explicit strings.
Failures latch the wrapper. A publication failure rolls back a newly linked
completion artifact. Required environment cleanup precedes completion publication.
An interrupted process, SIGKILL or an unwritable disk cannot be guaranteed to
produce a complete diagnostic; the consumed attempt never authorizes a retry.

## Offline regression and independent review

The restored fixed upstream repository is the same MuJoCo-LiDAR commit
`e6b879df4b96daaed30ef10fe02650d29048fa9b`; pattern SHA256
`9fa0165576f0060488254f04bd647a891004d4f4faa6e9b03b69d830c117c9a3`.
The old `/tmp` copy was absent at initial collection, which exited2 without running
tests. Restoring the pinned source enabled the unchanged baseline:777 PASS,
22 deselected. No dependency was installed or upgraded.

New forced diagnostics exercise all five classes and four physical raw causes
through NavigationEnv → TrainingMetrics → FormalHealth, compare original rewards
and observations exactly, associate statistics with an untrained checkpoint,
and exercise Validation classification. These are `FORCED_TEST_STATE`, zero-update
tests and are never pooled with natural Pilot failures or navigation performance.

RED evidence includes the missing independent entry, missing read-only artifact
analyzer, missing Validation guard and a caught directory-fsync failure leaving
a completion marker. Final GREEN and Pilot measurements are recorded below.

Independent pre-review required fixes to Validation guards, completion publication,
and first-record provenance. The final runtime SHA above received code-review GO
with no unresolved Critical/Important findings, conditional on completed tests.

Of the old22 exclusions, seven safe early-rejection tests are restored unchanged
(five parameterized authorization/seed/budget/hash checks, existing output, low
resources). Eight obsolete L8/L9 identity or defect-expectation tests remain
excluded. Seven tests that independently execute optimizer updates also remain
excluded; their capabilities are checked using this single Pilot's artifacts.
The JSON report identifies every excluded original node and replacement evidence.
These exclusions are not reported as original-test PASS.

## Runtime results

The sole authorized Pilot completed: **PASS**. Its runtime source remained
unchanged after execution. There were no retries or additional training runs.

| Measurement | Actual result |
|---|---:|
| Train environment steps | 4096 |
| Physics ticks | 81729 |
| Complete rollouts / PPO epoch updates / Adam calls | 32 / 64 / 128 |
| Completed Train episodes | 21 |
| Success / collision / timeout | 0 / 17 / 4 |
| Natural physical / sensor failures | 0 / 0 |
| Train reset visits / distinct Train identities | 22 / 9 of 12 |
| Validation episodes / successes / collisions | 6 / 0 / 6 |
| Test episodes | 0 |
| Maximum observed LiDAR age | 0.098 s |

The three unvisited Train identities remain explicitly zero in the report; the
budget was not increased to force coverage. Deterministic all-scene initialization
is inherited test evidence, not natural Pilot visitation. Natural physical failure
is **NOT OBSERVED**. The four physical causes are covered separately by
`FORCED_TEST_STATE` regressions, not presented as training outcomes.
The last completed episode ended at step3896; the final200 steps belong to an
unfinished episode and are excluded from completed-terminal counts.

Actor and Critic parameter hashes both changed. All recorded gradients, model
parameters and Adam states were finite; Adam state counters reached128. Saving
and reloading preserved policy parameters, optimizer state and deterministic
prediction. The local-only final checkpoint SHA256 is:

`5fe8c3f2a89a922dda8bcd9ad4025944040adc2fff863d6a34d3892b626bff35`

The six-scene Validation used deterministic prediction, changed neither policy
nor optimizer, and preserved Train scheduler/RNG state. All six episodes collided;
mean return was -23.96365. Train completed-episode mean return was -24.56364,
mean path3.15889m and mean final goal distance1.64289m. These results **do not show
learned avoidance**. The repaired integration can run; navigation competence and
million-step stability remain unproven.

The read-only analyzer checks all ten sealed artifact hashes, archive integrity,
4096 step rows plus22 reset rows, all terminal/raw counts and six Validation
identities. It pins the published protocol and artifact payload to the exact
authorized digest and rejects duplicate JSON keys and nonfinite float overflow.
It does not deserialize tensors or rerun optimization: gradient finiteness and
reload equality are explicitly hash-bound evidence from the frozen producer.
The original L9's eleven raw files and FAIL conclusion remain unchanged.

## Measured performance and resources

| Operation (ms) | Samples | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| Environment step | 4096 | 1.170 | 11.952 | 12.298 | 14.111 |
| LiDAR raycast | 1625 | 6.274 | 6.705 | 6.904 | 7.916 |
| Encoding | 1625 | 4.072 | 4.307 | 4.489 | 5.605 |
| PPO rollout update | 32 | 25.281 | 26.334 | 33.552 | 36.710 |

There were1647 full-ray scans including22 reset scans. Each step timing includes
whatever physics/scanning work was actually due; a non-scan step is not a24k-ray
scan benchmark. Training-phase wall time35.827s includes initialization and
checkpoint/reload, but excludes final Validation; throughput114.327 steps/s.
Total wall time42.406s and CPU40.150s correspond to about94.68% of one CPU core.
The summed physical simulation time is163.458s, across episode resets.

Pilot peak RSS1607.422MiB; minimum guarded MemAvailable9256.953MiB; minimum
guarded free disk71999062016 bytes. No OOM was observed. Reset RSS ranged from
1110.238 to1403.273MiB and ended at1230.824MiB; this short run cannot establish
long-run memory bounds. Regression-suite resource peaks are reported separately.

## Final verification status

The complete final selected suite passed **885 tests**, with0 failures,0 skips,
15 explicit deselections and2 inherited warnings, in132.507s. This comprises777
previous L9R cases, seven restored unchanged historical cases and101 new L9S
cases. The regression entry blocked learning/optimizer calls; no forbidden call
occurred. It is not a full-repository/all-history PASS. Final regression peak
RSS1850.988MiB and sampled minimum MemAvailable8857.660MiB are separate from
the Pilot measurements; an earlier completed regression peaked at1955.391MiB.

Independent runtime pre-review and post-run analyzer/evidence review both returned
GO with no remaining Critical/Important findings. Seven added analyzer regressions
first reproduced protocol-substitution/JSON-overflow weaknesses (RED), then passed
after offline fixes; all57 analyzer cases are included in the final885-test run.
Final independent document/result review also returned **GO**, with zero
Critical/Important findings, after checking raw counts, exclusion classifications,
the partial episode, all21 protected historical files and nine L9R source hashes.
`REPAIRED_TRAINING_CHAIN_READY=YES` applies only to this bounded integration
regression. The experiment's runtime PASS does not authorize formal retraining.
`FORMAL_RETRAIN_AUTHORIZED=NO` remains unconditional.

## Reproduction and limitations

Use system Python3.10 with the existing project virtual-environment packages:

```bash
export PYTHONPATH=/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco:/home/keyousuke/MineUAV-Learning-px4-sitl/mujoco/rl:/tmp/mineuav-mujoco-lidar-audit/src:/home/keyousuke/MineUAV-Learning/.venv/lib/python3.10/site-packages
export MINEUAV_LIDAR_UPSTREAM=/tmp/mineuav-mujoco-lidar-audit
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python3 -m lidar_integration.verify_repaired_integration --evidence NEW_TEST_EVIDENCE.json
python3 -m lidar_integration.ppo_repaired_pilot --l9s-pilot --authorize-l9s-pilot \
  --expected-protocol-sha256 09837c2104b12a1780cf81b6eb06c9b6c7f6e167d3378381def706e6af9f5cb3 \
  --upstream /tmp/mineuav-mujoco-lidar-audit
```

The second command is the one-shot entry; an existing attempt must be retained
and a second invocation is rejected. It is not a recommendation to rerun it.
MuJoCo3.14.0 is the real package with MjModel API, SB3 is2.9.0, Torch2.14.0+cpu,
Gymnasium1.3.0 and NumPy2.2.6. SB3's optional Atari import encounters the existing
OpenCV/NumPy ABI error and catches ImportError with `cv2=None`; this navigation
chain does not use Atari/OpenCV. Two inherited warnings concern Matplotlib3D
package mixing and an unregistered Gymnasium alternate render mode.

This bounded check cannot establish million-step memory stability, learned
avoidance, generalization, real MID360S fidelity or PX4 control compatibility.
The collision envelope remains the previously declared placeholder. Formal
retraining authorization stays **NO**. PX4, FAST-LIO2, real hardware, Test episodes
and additional training seeds remain outside this task.
