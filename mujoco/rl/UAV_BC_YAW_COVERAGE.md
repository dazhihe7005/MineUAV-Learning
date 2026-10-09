# Controlled Yaw-Coverage Behavior Cloning

Branch `feat/uav-bc-yaw-coverage`, base `b05cf751cb776f2a17d21b19f4c4e629de9a8775`. This is an expert-data intervention, not a controller/interface fix, BC fine-tune, DAgger, PPO/SAC or world-model experiment.

## Read-only preflight

Observation is `[target_xyz−position_xyz, world_vx, world_vy, world_vz, wrap(target_yaw−actual_yaw)]`. Error and free-joint translational velocities are world-frame quantities. Quaternion is MuJoCo wxyz; angle wrap is `atan2(sin(angle),cos(angle))`. The expert uses `clip(error/3)` world velocity, normalized by `[1.5,1.5,1]`, plus clipped yaw error as yaw-rate. The environment maps normalized commands back to world velocity and rad/s; no body-frame reinterpretation or motor labels occur.

At equal physical xyz/velocity, changing initial yaw0/±10/±30° leaves the expert xyz commands unchanged and changes only yaw-rate. Actual physical motion can still differ: world-velocity PI produces desired world acceleration/thrust; yaw-dependent desired attitude, attitude feedback and rotor allocation change transients. Controller/PI/allocator/task-yaw target are not modified. The expert does not need hidden information beyond the same7D observation. This does not prove the observation fully represents simulator/controller Markov state.

Original Train has8,974 samples, yaw std.00090545055rad, absolute maximum.28864449°. Initial±10/±30° are about193/578 original yaw standard deviations. Original labels, physical state/frame mapping, sign and wrap were checked; no interface or label error was found. This is a coverage/extrapolation warning, not evidence that yaw cannot be corrected or that normalization alone causes failure.

## Fixed controlled design

A is the frozen Original BC. B and C are fresh seed0 7→128ReLU→128ReLU→4linear actors,18,052parameters. B=Nominal-Expanded, C=Yaw-Augmented. No original weights are used to initialize them. Both use the exact Original Train-only observation **and** action mean/std; no re-fitting, yaw-specific weights, clipping changes, hyperparameter search or new architecture.

Original84Train and18Validation target groups are reused; original18Test are not training/selection inputs. Each target supplies10 unique base states. Fixed seed2026101101 generates separate Gaussian unit position/velocity directions with independent uniform radii0–.10m and0–.15m/s. Each B/C pair shares target, reset seed, position, velocity, nominal controller memory and all settings. B actual initial yaw is0; C has4nominal/3yaw10/3yaw30 episodes per target, with globally balanced signs. Subsequent trajectories are allowed to differ.

840 expert episodes per training source and180 common yaw-mixed Validation episodes were physically simulated. All succeeded; no success filtering was used. Raw failures would still be retained, with invalid terminal zero observations excluded from imitation inputs rather than treated as physical measurements. Labels are the expert's actually executed4D commands at the same pre-action observation, never BC/world-model outputs. Full raw trajectories remain local-only.

Each Train dataset has exactly20,000 unique transitions; common Validation4,000. Disjoint phase rules, in priority order:

1. Early: policy index<10 (first.4s).
2. Near braking: non-early distance<.2m.
3. High speed: remaining speed≥.5m/s.
4. Approach-other: remaining samples, including low-speed far-target states.

Train quotas respectively2,000/6,000/5,000/7,000; Validation400/1,200/1,000/1,400. Within every phase the initial-yaw mixture is40/30/30%, with nonzero signs50/50. Thus C Train has8,000nominal +3,000each signed yaw10/30; Val1,600nominal +600each signed yaw10/30. B uses the same **shadow** yaw-group quotas to match selection, but its actual initial yaw is0 for all20,000 samples. Shadow groups are not real yaw coverage. Sampling is fixed-seed without replacement; no duplicated physical observation/action rows, episode-step IDs, omitted targets or quota padding. Shortage would stop the experiment, not change the budget.

This stratification guarantees transient/high-speed/braking coverage but does not reproduce natural visitation frequencies. The common small xyz/velocity variations are shared by B/C, so the causal contrast is their additional yaw coverage, not Original-versus-new-data alone.

## Matched training and recovery

Loss is MSE of `(predicted_action−expert_action)/Original Train action std`, with the Original normalized observation input. Adam lr.001,batch512,100epochs,seed0,no new gradient clipping.20,000 samples require40updates/epoch, hence4,000updates/model. Both receive the same index permutations and initial parameter hash. Original's smaller8,974-sample historical experiment had about1,800updates at the same100epoch cap; it is a reference, not a budget-matched causal control.

Only the **shared Validation normalized action MSE** selects checkpoints. Nominal/yaw10/yaw30 Validation metrics are separately recorded; Final closed-loop success is never selection input. Train/Val loss, physical action errors, gradient norms, batch-order hashes and complete100epoch histories are saved. Large yaw scales and their per-dimension weighting are deliberately retained, not silently normalized away.

Recovery atomically saves completed epoch actor/Adam/RNG/history/best-state together. An interrupted partial epoch rolls back to the durable epoch and cannot duplicate committed updates. Final best model is exported only after the fixed budget completes. Tiny real-training interruption/reload tests verify identical uninterrupted/resumed histories and best parameters. Intermediate recovery files are local-only.

## Strict new final evaluation

100 new target groups are drawn once from unchanged task bounds x/y±2m,z.7–1.5m using fixed seed2026101101, with new target IDs/reset seeds. Coordinates and identities are disjoint from original BC Train/Val/Test, old benchmark/holdout and all adaptation Train/Val/Final targets. Targets and design are frozen before training. These are a new single100-target cohort, not the old two100-target benchmark/holdout sets.

Nine unchanged robustness conditions: nominal; position norm.10/.30m; world velocity norm.15/.40m/s; actual yaw±10/±30°; small/large combinations. Original fixed direction sampler/sign balance is reused. All900 initial states are physically validated, then each is restored from the same complete MjData/environment/PI/yaw/allocator/RNG snapshot for Scripted/A/B/C. Real qpos/qvel/unit yaw quaternion plus `mj_forward`, not observation-only perturbations.3600unique controller episodes, all failures retained.

Unchanged500/100/25Hz, world-action limits, PI RewardV2,15s timeout, success distance<.10m andspeed<.15m/s for5consecutive policy steps. Yaw is not a success condition. BC inference only sees7D observation; no expert, reference action, history correction or fallback is consulted.

All-episode distance, speed/excursion, physical failures/timeouts, near-target actual/commanded speed(d<.1), saturation, yaw and recovery metrics are reported. Failed completion time is null. Pairwise time differences only use both-success episodes. Distance rebound overshoot is `max(d−cumulative_min(d))`, not instantaneous absolute error; post-peak speed recovery is the first five samples below.15m/s after episode speed peak. Invalid terminal placeholders never count as recovery or arrival. Wilson95% success intervals and descriptive paired bootstrap are not population/safety guarantees.

Offline Test action errors use all valid pre-action observations/actions from these unseen targets' Scripted nominal/yaw10/yaw30 trajectories. Both raw and deployed-clipped4D action errors, per dimension and early-large-yaw/near-braking regions are reported. Actions are normalized policy commands; multiply xyz by[1.5,1.5,1] to convert error to commanded m/s, yaw action to rad/s.

Predeclared engineering references: yaw30≥90/100,nominal≥98/100. A>5percentage-point B/C position/velocity loss is flagged as regression, not used to tune/select models. Failure to reach references is a negative result; no automatic tuning or DAgger.

## Reproduction and resource safety

Run collection, training, smoke, full evaluation and resume **sequentially**, one CPU MuJoCo worker/one numerical thread, each under `uav_bc_safety.py`. Example:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python mujoco/rl/uav_bc_safety.py \
  --directory mujoco/reports/uav_bc_yaw_coverage_seed0_parts --phase collection \
  -- .venv/bin/python mujoco/rl/uav_bc_yaw_data.py
```

Subsequent child commands: `uav_bc_yaw_training.py`, `uav_bc_yaw_evaluation.py --smoke`, `uav_bc_yaw_evaluation.py`, then a hash-checked evaluation resume. Phase lock and fsync/replace prevent overlapping jobs/incomplete publication. Guard reserves1.5GiB, refuses starts below5.5GiB available, aborts below2GiB available/treeRSS>4GiB/newOOMkill. No swap changes. Sampled RSS/HWM are monitoring evidence, not a hard OS guarantee.

Raw1860expert trajectories, selected datasets,3600final traces, optimizer recovery files and logs remain in ignored `mujoco/reports/uav_bc_yaw_coverage_seed0_parts`. Git contains only bestB/C models, code/tests, reports/manifests/hashes, eight figures, this document and experiment log. Canonical world-model and Original BC are not modified. Final publication is gated by complete task/label/hash/selection/pair verification and tests.

All eight figure filenames are preserved inside `mujoco/reports/uav_bc_yaw_coverage_figures/`, avoiding a collision with the historical adaptation experiment's `nominal_regression_comparison.png`. That historical tracked figure is byte-identical to the base commit; no prior experiment picture is replaced.

## Results: fixed unseen 100-target cohort

All 3,600 controller episodes completed, with no missing/duplicate tasks. Each cell below has denominator 100. All failures were timeouts; no physical, nonfinite, tilt or flight-area failures occurred.

| Initial condition | Scripted | Original A | Nominal-expanded B | Yaw-augmented C | Paired C−B, pp |
|---|---:|---:|---:|---:|---:|
| Nominal | 100 | 100 | 100 | 100 | 0 |
| Position .10 m | 100 | 100 | 100 | 100 | 0 |
| Position .30 m | 100 | 100 | 100 | 100 | 0 |
| Velocity .15 m/s | 100 | 100 | 100 | 100 | 0 |
| Velocity .40 m/s | 97 | 98 | 98 | 94 | −4 |
| Yaw ±10° | 100 | 74 | 92 | 100 | +8 |
| Yaw ±30° | 100 | 0 | 6 | 100 | +94 |
| Combined small | 100 | 74 | 93 | 100 | +7 |
| Combined large | 92 | 2 | 6 | 98 | +92 |

Predeclared yaw30≥90 and nominal≥98 engineering references were met without changing configuration. C has 892 successes/8 timeouts overall. Yaw30 C/B discordance: 94 C-only successes, 0 B-only, 6 both success. Combined-large: 92 C-only, 0 B-only, 6 both success, 2 both timeout. Velocity-large: 2 C-only, 6 B-only, 92 both success. Relative to Scripted, C has 6 expert-only velocity-large successes and 3 C-only; both combined-large failures are shared with Scripted. Thus a real remaining BC-specific large-velocity gap exists, despite yaw repair.

100/100 Wilson95% interval is approximately [96.30%,100%], not a zero-risk guarantee. Descriptive paired C−B intervals are yaw10 [3,13]pp, yaw30 [89,98]pp, velocity-large [−10,1]pp. New targets differ from the previous benchmark/holdout cohort; previous 2/200 yaw30 success is not a paired comparator to the present 0/100.

### Training and offline imitation

| Model | Best epoch | Train normalized MSE | Shared Val normalized MSE | Val yaw0 / yaw10 / yaw30 |
|---|---:|---:|---:|---|
| B | 35 | .000245292 | 1.534668607 | .000364381 / .541202199 / 4.573874143 |
| C | 84 | .003143932 | .003336325 | .002186537 / .002371953 / .005833748 |

Both completed all 100 epochs/4,000 updates. Initial hash for both: `668f407652d04f9a5a3af3eccfe47bf8751ce0358fc9a30dd77e230243f7d8d1`; every epoch's batch-order SHA matches. Best-epoch gradient norm mean/max B=.023232/.039400, C=29.4507/53.2886; all gradients/losses finite, no new clipping. Large C gradients coexist with the deliberately retained tiny Original yaw normalization scale; magnitude alone is not numerical explosion.

Offline Test has 45,337 unique valid expert transitions: yaw0 14,318; yaw10 14,711; yaw30 16,308. Early-large-yaw 1,000 and near-braking 11,585 are diagnostic subsets, not extra independent samples. Numbers are deployed clipped normalized-policy-command errors (not normalized loss, motor commands or physical velocity error).

| Model | Overall action RMSE / MAE | Yaw0 RMSE | Yaw10 RMSE | Yaw30 RMSE | Early yaw30 RMSE | Near braking RMSE |
|---|---|---:|---:|---:|---:|---:|
| A | .181207 / .072680 | .002721 | .108271 | .284085 | .500307 | .182137 |
| B | .113588 / .048511 | .002745 | .062366 | .179871 | .379369 | .105390 |
| C | .005668 / .003207 | .004492 | .004316 | .007402 | .015470 | .004408 |

C per-dimension overall RMSE vx/vy/vz/yaw-rate = [.006450,.008962,.002565,.00004213]; MAE = [.004730,.006245,.001829,.00002314]. Correlations = [.999145,.998297,.998452,.999999960]. Multiply vx/vy by1.5 and vz by1 to express commanded velocity errors in m/s; yaw action is rad/s. Full raw/clipped MSE/MAE/RMSE/correlations for every yaw and phase subset are in JSON.

Nominal offline RMSE is higher for C than A/B (.004492 vs .002721/.002745); this is an imitation-precision trade-off, not hidden by the 100/100 nominal closed-loop result. No nominal success regression, but all-episode mean final distance rises B .061963→C .065769m. Do not describe C as uniformly superior in every metric.

### Translational recovery and braking

All-episode means at yaw30 (Scripted / A / B / C):

| Metric | Scripted | A | B | C |
|---|---:|---:|---:|---:|
| Peak translational speed, m/s | .780119 | 2.173531 | 1.563390 | .777216 |
| Final distance, m | .063311 | .262867 | .127629 | .059448 |
| Distance rebound overshoot, m | .258167 | 2.176317 | .948151 | .256673 |
| Near-target actual speed, m/s | .204166 | N/A | .263256 | .202678 |
| Near-target command speed, m/s | .018867 | N/A | .028330 | .021008 |
| Command-element saturation, % | 0 | 1.0033 | .1802 | 0 |

Near-target means include moving/braking samples with distance<.10m, not just the five final low-speed success samples. A has zero such samples at yaw30; N/A is not zero velocity. C mean peak drops 50.29% from B and 64.24% from A, while mean rebound drops 72.93% from B. C near-target motion and rebound closely match expert behavior; no saturation occurs for C in any condition.

Yaw30 mean final absolute yaw is Scripted .128281rad (7.35°), A .007338rad (.42°), B .007475rad (.43°), C .121236rad (6.95°). C/expert finish sooner, while A/B continue until near timeout. Yaw is not part of the success definition; larger yaw at earlier successful termination is not evidence of failed yaw control. This supports repair of yaw-conditioned translational extrapolation/braking rather than simply faster yaw correction.

C mean successful completion times nominal/yaw10/yaw30/combined-large are 5.670/6.285/6.616/9.869s. Paired C−B completion changes (only both-success episodes): nominal−.252s (n100), yaw10−5.222s (n92), yaw30−6.913s (n6), combined-small−5.343s (n93), combined-large−3.207s (n6). Small surviving subsets must not be generalized to all episodes. Timeout completion times remain null.

Combined-large peak speed A/B/C=2.152923/1.592247/.862328m/s; C/expert mean final distance .074862/.077409m. Velocity-large C success94 vs B98 and expert97: all six C failures reached near the target but did not sustain both thresholds for five steps before15s. Five even satisfy instantaneous terminal distance/speed thresholds, which is not sufficient for the unchanged hold rule. Two combined-large C failures similarly visited the target region, final distances .10952/.11582m, and share expert timeouts. No catastrophe or concealed failure filtering.

## Interpretation and next step

The controlled C-versus-B contrast supports expert yaw coverage as an important, intervenable source of the previous yaw-conditioned translational failure: +94pp yaw30 success, +8pp yaw10, +92pp combined-large, much lower transient/braking action errors and expert-like speed/rebound. B-versus-A improvement at yaw10 also demonstrates common extra-data/coverage effects; Original is not the causal budget-matched control. This is one training seed, fixed perturbations and100new targets, not proof that yaw coverage is the unique mechanism or of real-world flight safety.

C is a useful yaw-covered policy baseline for this evaluated simulation envelope. Original remains preserved for historical nominal reference. Small nominal offline precision regression and large-velocity timeout gap prevent an unqualified all-metric dominance claim. No demonstrated privileged-information gap is required by the Scripted expert, although7D input is not a complete physical/controller state.

DAgger may eventually address off-expert-state failures, but these results do not require immediately starting it; yaw30 was repaired by ordinary supervised coverage. **The only next recommendation is one fixed-policy persistent external-disturbance recovery evaluation of C**, with training/controllers unchanged. Not executed. No DAgger, PPO/SAC, world-model training, MPC modification, size/loss sweeps or next experiment was started.

## Verification and measured resources

Independent publication checks reconstruct all selected expert labels from hashed original episode/step arrays; verify all1,860episode manifests, target isolation, phase quotas, initialization/budgets/batch schedules/Val-only selection/normalization; and recompute closed-loop geometry, scalar outcomes, near-target/saturation/peak statistics and all paired summaries from real traces.900states/3,600unique tasks, all hashes checked, no missing/duplicate samples. Complete evaluation resume validates/reuses these same tasks rather than adding independent samples. Original checkpoint SHA `7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c` and parameter hash remain identical before/after. Canonical world-model SHA `42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9` also unchanged and unused.

Best B checkpoint SHA `d426169df719167bbe03b08572b0b35ededacfa919df7912932d03b0615ff8dd`; C `d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce`. Both checkpoint and parameter hashes remain unchanged throughout formal inference. Architecture and action output interface are identical to Original.

Fresh BC-related tests:56/56 passed, including23new yaw-data/training/evaluation/publication tests. Read-only historical-fixture full suite:589run,585passed,4knownskips,0errors/0failures. Bare full discovery:527run,496passed,1skip,30pre-existing historical missing-dependency errors,0assertion failures. **Bare full suite is not green.** Historical `reward_v2_alignment_audit`, `ppo_reward_success_alignment` and archived diagnosis report are injected read-only into test memory from commit `8879754a43302e9c55b27de4ce3c629c7e4e1a53`; no historical files are restored or changed. All30error test names/commands are preserved in report `tests.bare_full_suite.errors`, and full logs remain local-only. The related suite can be run directly:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m unittest discover -s mujoco/rl -p 'test_uav_bc_*.py' -q
```

All heavy phases were serialized, one MuJoCo worker and one numerical thread. A premature test launch was refused by the phase lock before a child ran; the refusal log is retained, then tests were launched after verification finished. No uncontrolled concurrent workload or corrupted cache resulted.

No OOM, no NaN/Inf training failure, no swap change. Formal evaluation756.49s; sampled peakRSS .934GiB (HWM1.035GiB). Published completed phases including retained earlier test-guard telemetry: sampled process-tree peakRSS2.986GiB, summed per-process HWM5.132GiB, minimum available physical memory8.362GiB. Summed HWM combines each process's historical high-water mark and is NOT a simultaneous actual RSS peak or a breach of the4GiB sampled-RSS limit. Tests dominate phase memory; collection peakRSS1.117GiB, training .356GiB, smoke .979GiB. These are measured monitoring values, not a hard system or real-time guarantee. Safety guard still reserves1.5GiB and enforces4GiB sampled treeRSS/2GiB available abort limits. The first figure publication failed on Wilson0/100 endpoint roundoff; real regression tests failed first then passed after rendering-only correction. No scientific data/statistics/checkpoint changed, and fresh full verification was repeated.

Fresh read-only review identified a physical-failure timing gate: failures can stop midway through a40ms action. A real outside-flight-area injection after the second action's first500Hz tick produced .042s (not .080s), failed first, then passed after outcome-aware duration validation. Success/timeout timings stay exact; physical failures must lie within the final interval on the500Hz grid. Formal cohort has no such failures, so no outcome/data/checkpoint changed. No second review or additional experiment.

Deferred minor review disclosures: deterministic training recovery was tested after a durable epoch, not an injected mid-epoch optimizer-update interruption. Also, the broad phase-metric wording above must be interpreted using the actual JSON schema: overall/per-yaw metrics are raw+clipped; early-large-yaw and near-braking diagnostics are **clipped-only**. No raw phase diagnostics are claimed as computed evidence.
