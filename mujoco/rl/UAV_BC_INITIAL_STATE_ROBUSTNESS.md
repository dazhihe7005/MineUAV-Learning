# BC Initial-State Perturbation Robustness

Evaluation only, seed0 frozen BC against the existing Scripted Controller. No training, fine-tuning, PI tuning, observation/action changes, or follow-on experiment.

Branch: `feat/uav-bc-initial-state-robustness`
Base: `feat/uav-behavior-cloning-baseline` / `11d3e6bed47d87e86956349db207e89727d08bef`

## Protocol fixed before evaluation

The checkpoint is `models/uav_bc_mlp_seed0.pt`, SHA256 `7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c`. The standalone 7→128 ReLU→128 ReLU→4 actor has 18,052 parameters, runs in eval mode, and has all parameters frozen. Input and output scaling are the unchanged checkpoint's Train-only statistics. The Scripted controller uses position error and yaw error from the same 7D observation; it does not use privileged state. Its actions are never consulted by BC execution.

Nine predefined conditions retain the requested magnitudes:

| Condition | Position norm (m) | World linear velocity norm (m/s) | Actual initial yaw |
|---|---:|---:|---:|
| Nominal | 0 | 0 | 0° |
| Position small / large | 0.10 / 0.30 | 0 | 0° |
| Velocity small / large | 0 | 0.15 / 0.40 | 0° |
| Yaw small / large | 0 | 0 | ±10° / ±30° |
| Combined small / large | 0.10 / 0.30 | 0.15 / 0.40 | ±10° / ±30° |

Gaussian unit directions are deterministic from `SeedSequence([2026101001, split_id, episode_index])`. Position and velocity directions are separate draws, reused across magnitudes and component/combined conditions. Yaw signs are a fixed permutation of 50 positive/50 negative signs per split. No rejection sampling based on controller performance is used. All 1,800 initial states were checked and the complete manifest frozen before smoke scores or formal evaluation. These are engineering test amplitudes, not certified safety or recovery limits.

The original benchmark100 and holdout100 targets/reset seeds are reused verbatim from `uav_bc_target_splits_seed0.json`; they were previously checked disjoint from the BC Train/Validation/Test target groups. Each condition has 200 unique base episodes, each evaluated once by each controller: 3,600 controller episodes. Smoke checks reuse a subset of these exact tasks, rather than adding or duplicating formal samples.

## Physical correctness and paired reset

Perturbations change real free-joint `qpos[:3]`, world translation `qvel[:3]`, and a normalized wxyz yaw quaternion `[cos(yaw/2),0,0,sin(yaw/2)]`. `mj_forward` updates derived geometry/observation. Initial position is `[0,0,1]+offset`; angular velocity stays zero. The unchanged task yaw target and low-level controller yaw target remain zero, so yaw perturbation is an actual orientation error rather than a task-target change. PI integral, actuator controls, previous action, episode counter and success streak remain nominal zero. `previous_distance` is updated to the physically perturbed distance, avoiding an artificial first-step reward jump.

The v2 model uses a coarse collision box (half-height0.05m), estimated inertia/COM and instantaneous rotor response. Pure yaw keeps the box vertical extent unchanged. Every initial state has positive floor clearance, no penetrating contact, legal observation and no initial failure. The full range is inside the existing 6m horizontal/4m altitude/60° tilt failure limits. No geometry or controller parameters were changed.

`decision_fidelity_snapshot` copies **all MjData**, including integration/act/ctrl/time/warmstart/derived fields, and deep-copies all nonresource environment fields/RNG, nested PI/yaw/controller fields, allocator state and previous action. Each pair restores the same exact snapshot into the same compiled model; there is no reset after restore. The integration/controller/RNG fingerprint must match the immutable manifest and both controller records. Repeated rollout tests cover nonzero controller memory as well as perturbed initial states.

## Metrics and statistical boundaries

Success is unchanged: distance<0.10m and speed<0.15m/s for five consecutive policy steps. Yaw recovery is **not** a success condition. Physics/control/policy frequencies remain 500/100/25Hz and timeout15s. All successes, physical failures and timeouts count in denominators. Failed completion times are null, not timeout durations.

The report gives per-split and pooled success counts/Wilson95% intervals, all-episode final distance, successful-only completion times, physical failures/timeouts, near-target actual and commanded speed, crossings, command-element saturation at `|a|>=.95`, max actual speed, max position displacement from the **perturbed initial position**, minimum distance/target-region visits and distance reduction. Near-target speed includes approach/transit samples and is not identical to the success-terminal speed.

Paired statistics use matched `(condition,split,target_id)` keys: BC-minus-Scripted success/distance, the four success concordance/discordance cells, termination-reason pairs, and completion-time differences only when **both** succeed. The fixed-seed4,096-resample paired bootstrap is descriptive/exploratory. Zero observed failures and degenerate bootstrap intervals do not imply zero population risk. Samples share the same targets across conditions; direction bins are exploratory and do not isolate causal direction effects.

Velocity/yaw settling means the first five successive sampled states below0.15m/s or2° respectively. A initially stationary vehicle may satisfy the velocity criterion before later accelerating to travel to its target; max-speed and full curves are also reported. Recovery curves never carry terminated states forward and show the number of remaining trajectories at3s.

## Reproduction and recovery

Use one CPU MuJoCo worker and one numerical thread. Run phases sequentially under the existing memory guard:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python mujoco/rl/uav_bc_safety.py \
  --directory mujoco/reports/uav_bc_robustness_seed0_parts --phase smoke \
  -- .venv/bin/python mujoco/rl/uav_bc_robustness.py --smoke
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python mujoco/rl/uav_bc_safety.py \
  --directory mujoco/reports/uav_bc_robustness_seed0_parts --phase evaluate \
  -- .venv/bin/python mujoco/rl/uav_bc_robustness.py
```

Rerun the evaluator under a `resume` phase to validate/reuse completed records. Atomic fsync/replace writes publish NPZ first, then SHA-bound metadata; interrupted orphan NPZ is regenerable, while existing corrupted/mislabeled/source-mismatched completed records cause a hard error. Phase lock prevents overlapping jobs in this experiment directory. A complete result is published only after exact key-set/count validation and verification; no duplicate or missing tasks are silently accepted.

Memory protection reserves1.5GiB for system services, refuses starts with<5.5GiB available, and aborts when available<2GiB, sampled process-tree RSS>4GiB, or a new system OOM kill appears. Poll interval0.1s. No swap changes or throughput-driven extra workers. The telemetry is sampled RSS/summed per-process HWM, not a hard OS guarantee. Raw trajectories, cached task records, runtime logs and temporary review/plan material remain in ignored `uav_bc_robustness_seed0_parts`; Git contains report/manifest/hash summaries only.

## Results and conclusion

Each B/H cell below is successes/100. All conditions have 200 paired episodes; no initial states were excluded.

| Condition | BC B / H | Scripted B / H | Pooled BC−Scripted (pp) | Expert-only successes | Both failed |
|---|---:|---:|---:|---:|---:|
| Nominal | 100 / 100 | 100 / 100 | 0 | 0 | 0 |
| Position0.10m | 100 / 100 | 100 / 100 | 0 | 0 | 0 |
| Position0.30m | 100 / 100 | 100 / 100 | 0 | 0 | 0 |
| Velocity0.15m/s | 100 / 100 | 100 / 100 | 0 | 0 | 0 |
| Velocity0.40m/s | 98 / 99 | 95 / 96 | +3.0 | 0 | 3 |
| Yaw10° | 73 / 76 | 100 / 100 | −25.5 | 51 | 0 |
| Yaw30° | 1 / 1 | 100 / 100 | −99.0 | 198 | 0 |
| Combined small | 75 / 72 | 100 / 100 | −26.5 | 53 | 0 |
| Combined large | 1 / 3 | 94 / 90 | −90.0 | 180 | 16 |

Velocity0.40 has six BC-only successes, zero Scripted-only successes and three shared timeouts. Combined-large has16 shared timeouts: the whole condition's failure cannot be attributed solely to BC. Across1,800 episodes/controller: BC1,299 successes/501 timeouts; Scripted1,775 successes/25 timeouts. Both have **zero physical failures**, including no nonfinite-state, excessive-tilt, altitude/area failure. Severe task failure under yaw is real but is not an observed physical crash/catastrophic dynamics failure.

| Condition | Mean final distance BC / Scripted (m), all200 | BC−Scripted completion (s), both-success pairs | Mean max actual speed BC / Scripted (m/s) |
|---|---:|---:|---:|
| Nominal | .065006 / .070092 | +.2100 (n200) | .758703 / .766621 |
| Position small | .065322 / .071305 | +.2590 (n200) | .758724 / .766455 |
| Position large | .066912 / .071857 | +.2400 (n200) | .764065 / .771242 |
| Velocity small | .064436 / .066016 | −.0140 (n200) | .764918 / .773579 |
| Velocity large | .073652 / .074315 | +.0329 (n191) | .826913 / .837736 |
| Yaw small | .084500 / .069444 | +6.7611 (n149) | 1.112696 / .769168 |
| Yaw large | .258092 / .066152 | +7.0000 (n2) | 2.189000 / .789532 |
| Combined small | .086447 / .065879 | +5.9238 (n147) | 1.116777 / .775610 |
| Combined large | .256575 / .078205 | +2.5800 (n4) | 2.183338 / .866008 |

Timing under yaw-large/combined-large concerns only2/4 paired successes and is not a population recovery-time claim. Timeout durations are never counted as successful completion. All-episode final-distance deltas and full duration/near-target/excursion distributions are retained in the report.

Velocity-only BC does not show greater mean peak-speed overshoot than Scripted. In contrast, yaw30° raises BC mean peak speed to2.189m/s (Scripted.790), and combined-large to2.183 (Scripted.866). Near-target actual speed under yaw-large is .469558m/s for BC versus .206139 for Scripted, consistent with difficulty satisfying the unchanged low-speed hold. Initial yaw is nevertheless largely corrected by BC: mean terminal absolute yaw is .314°/.422° for yaw-small/large, compared with Scripted3.077°/7.228°. **Poor waypoint success is not simply failure to correct yaw itself**; task success does not require yaw, and the neural policy's response to off-nominal yaw is associated with strong translational transients and delayed arrival/braking.

Scripted has zero command saturation in every condition. BC aggregate saturation is0 for nominal/position/velocity, .0135% yaw-small,1.0638% yaw-large, .0111% combined-small,1.0858% combined-large (fraction of all command elements). These low aggregate percentages do not exclude important short initial saturation bursts. Direction/sign strata and maximum position excursion are reported separately, without post-hoc direction filtering or causal claims.

Exploratory yaw-sign asymmetry: at10°, BC positive/negative yaw successes are67/100 vs82/100 (combined-small68/100 vs79/100). At30°, the gap is broad across both signs:2/100 vs0/100 (combined-large4/100 vs0/100). The fixed targets in the two sign groups differ, so these are descriptive strata, not an isolated sign-effect experiment. BC mean/max displacement from the perturbed initial position under yaw-large is3.479/5.759m, combined-large3.467/5.712m; these are excursions, not target errors or instantaneous flight-radius violations. Full position/velocity dominant-axis strata are in the report.

The checkpoint's nominal Train yaw std is about.00090545rad;10°/30° are about193/578 nominal standard deviations. This is a substantial coverage/extrapolation warning, **not proof that normalization alone causes failure**. Scripted uses the same observation and succeeds in all yaw-only episodes, so no privileged-information deficit was found. The seven scalars still do not represent the complete simulator/controller state; this experiment does not establish general Markov sufficiency.

Zero failures in100 trials have Wilson95% success interval approximately[96.30%,100%]; pooled200/200 gives[98.12%,100%]. These are finite fixed-target results, not guaranteed safety. See `../reports/uav_bc_initial_state_robustness_seed0.json` for all per-split intervals, paired bootstrap intervals, actual sample counts and exploratory direction groups.

**Conclusion:** Keep BC as a nominal and tested position/small-velocity perturbation policy baseline. Do not promote it as a yaw/combined-perturbation-robust baseline. There is a large BC-specific robustness gap relative to its nonprivileged expert, especially30° yaw. Large combined perturbations also reveal some expert limitations. No policy retraining or controller tuning was done.

**Only next recommendation:** A separate controlled yaw-coverage expert-data BC experiment with unchanged architecture and strict target isolation. Not executed; no DAgger/PPO/SAC/world-model/MPC changes are started.

## Verification and resources

The independent publication verifier validated all3,600 metadata/trace hashes, complete key-set uniqueness,1,800 equal paired initial fingerprints, action/observation/previous-action indexing and physical-position consistency. All400 nominal controller trajectories exactly reproduced the previous baseline's action and observation arrays. The checkpoint SHA256 and parameter hash `771382758078def13665abef25ec5a9b3702b7938f265f55294605bcc5b60cfa` are identical before/after. Canonical world-model SHA256 `42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9` also remains unchanged and is unused.

Fifteen new robustness tests plus18 existing BC tests passed (33 total), including the injected nonfinite-physics failure regression. Final bare whole-project discovery ran504 tests:473 passed,30 pre-existing legacy dependency import errors,1 skipped, zero assertion failures. The30 error names are retained in the JSON `tests.bare_full_suite.errors`; the bare suite is **not** claimed green. Using read-only in-memory historical modules `reward_v2_alignment_audit` and `ppo_reward_success_alignment`, plus the missing archived diagnosis JSON, from commit `8879754a43302e9c55b27de4ce3c629c7e4e1a53`, final full discovery ran566 tests:562 passed/4 known skips/0 failures or errors. No old file restoration or unrelated dependency fix was committed. Existing trainers are exercised only by tiny unit fixtures in temporary directories, never against experimental checkpoints or this frozen evaluation.

All phases used one MuJoCo worker and no concurrent large simulation/training. The final full evaluation took720.20s; sampled process-tree peak RSS0.998GiB, HWM1.146GiB. Across both the archived first run and the final smoke/evaluation/resume/comparison/tests, peak sampled RSS2.802GiB and summed per-process HWM3.728GiB (the maximum occurs in whole-repository tests); minimum available physical memory7.248GiB. No new OOM kill, memory-guard abort or swap configuration change. Resume revalidated all3,600 completed records; it did not repeat the formal controller episodes. Scientific plots and reports are atomically published.

## Independent-review failure-retention fix

The fresh read-only review found one Important numerical-failure edge case: the environment's terminal sanitized zero observation was incorrectly treated as a physical measurement, and NaN physical position could prevent strict JSON publication. A real-environment injected `nonfinite_state` regression reproduced the failure before the fix. The evaluator now saves a physical-validity mask, keeps the failure in the denominator, marks unavailable terminal measurements null, and excludes placeholder zeros from target arrival, minimum distance, yaw/velocity recovery and plotting. Aggregate/paired metrics expose available-measurement counts. This changes no controller, physics, perturbation or finite-episode formulas.

The first run had no nonfinite or physical failures, so its scores were unaffected. Its cache/manifest were recoverably archived locally; the complete identical3,600-task cohort was rerun with a fresh source-bound identity, rather than accepting stale hashes. Every original action/observation/physical trajectory array is bitwise identical and all original scalar outcomes are identical to the final run, in addition to the400 original nominal-baseline comparisons. The two runs remain one3,600-task scientific cohort, not7,200 independent samples. Final test/resource numbers and the explicit post-fix reproduction result are recorded in the report. Both ignored raw/cache directories total about198MiB and remain local-only.

Deferred low-priority review items: generation identity does not enumerate every imported metrics implementation (publication separately recomputes trace metrics); the frozen manifest's absolute checkout path restricts direct cache resume after moving the repository; completion-figure count annotations touch the top border and one overlaps the legend. They do not alter the current results; no cosmetic/dependency-isolation expansion was included in the failure-retention fix.

Review boundaries: this remains simulation-only, fixed-target descriptive evidence, not real-aircraft safety, uniquely identified yaw causality, population guarantees, general observation sufficiency or adversarial artifact authenticity. Hashes validate integrity/reproduction, not historical chronology. The reviewer checked code/artifacts; fresh full experiments/tests/nominal comparisons, exact staging and remote handoff are executor responsibilities. Unrelated legacy errors and untracked historical work are preserved. No next research stage is authorized or executed.
