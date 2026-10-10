# Frozen BC External Disturbance Robustness

Branch: `feat/uav-bc-external-disturbance`
Base: `feat/uav-bc-yaw-coverage` / `e30984ad879c2263081c0f9244b3ad1dc1bea8d0`

## Research question and fixed scope

Can unchanged Scripted, Original BC and Yaw-Augmented BC complete the original waypoint task under sustained lateral physical force or a transient gust? This is evaluation only: no fitting, optimizer, new observations, controller tuning, reward changes, world-model use, MPC changes or next-stage experiment.

Both neural policies remain standalone7→128ReLU→128ReLU→4linear,18,052parameters, original saved Train-only observation/action normalization and bounds[-1,1]. Actions command world-frame vx/vy/vz scaled by[1.5,1.5,1]m/s and yaw-rate by1rad/s. They cannot receive force magnitude or access Scripted outputs. The expert remains `scripted_action`: world position-error/yaw feedback using the same7D observation, no privileged disturbance input; velocity is present in the observation but not used by the expert formula.

## Frozen artifacts

| Policy | Checkpoint SHA256 | Parameter SHA256 |
|---|---|---|
| Original | `7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c` | `771382758078def13665abef25ec5a9b3702b7938f265f55294605bcc5b60cfa` |
| Yaw-Augmented | `d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce` | `b1cdac0dd564baaee225b2c13e8e19a0540a842b5764da47212ebba8efffa5f5` |

Files are `models/uav_bc_mlp_seed0.pt` and `models/uav_bc_yaw_augmented_seed0.pt`. Loading sets eval/requires_grad=False. Before/after inference hashes are checked independently; neither checkpoint is rewritten. Canonical v3 World Model is unused and unchanged.

## Physical preflight and injection

The compiled model is a single7kg `mine_uav` rigid body (body1), world+Zup, gravity9.81m/s². This is the existing engineering-estimated mass/inertia model, not a measured aircraft. Estimated COM in body coordinates is[.015134357576683754,.00008124967855399744,.03602788555838244]m.

The private `DisturbanceEnv` inherits the unchanged PI environment. Immediately before every500Hz `mj_step`, its existing audit hook clears the Cartesian force buffer and assigns the desired worldXY force to `xfrc_applied[1,:3]`. Torque entries remain zero. This API acts at the moving body COM (`xipos`), not the body-frame origin. An independent `mj_applyFT` at world COM gives the same generalized force even with rotated yaw. Force acts through integration; no observation/position/velocity edits simulate it. Normalized policy actions,100Hz PI/attitude/allocator updates and25Hz policy timing are unchanged. Nominal subclass trajectory is tested bitwise against the original PI environment.

| Condition | Force | Schedule |
|---|---:|---|
| Nominal |0N|Always zero|
| Constant-Low |3.4335N=.05mg|Task start to termination|
| Constant-Medium |6.867N=.10mg|Task start to termination|
| Constant-High |13.734N=.20mg|Task start to termination|
| Gust-Medium |6.867N|Physics ticks1000–1999,2≤t<4s|
| Gust-High |13.734N|Physics ticks1000–1999,2≤t<4s|

Schedules use rounded integer physics ticks to avoid floating-time drift at2/4s; every actual pre-integration force vector/time is retained locally and independently checked. Assignment rather than addition prevents double force. Reset clears all external force and task/controller history. Full snapshots include MjData (integration/solver warmstart/act/ctrl/external-force buffers), PI integral/yaw target/allocator, task/RNG state and previous action. Restore/replay is deterministic.

High statically needs70.029934N and11.31° tilt, below158.3910N available thrust,60° failure limit and3m/s² horizontal acceleration command limit. However,1.962m/s² disturbance acceleration exceeds the unchanged1.5m/s² PI integral-compensation cap. Static force feasibility does not guarantee successful task recovery or zero steady bias. No force strength is changed after smoke outcomes. These are fixed simulation stress loads, not an aerodynamic model or a conversion to wind speed.

## Cohort, pairing and outcomes

100 fresh targets use fixed seed2026101201, original full-task uniformxy±2m/z.7–1.5m, reset seeds950140000+i. Coordinate/ID/reset-seed checks exclude all original84Train/18Val/18Test, benchmark/holdout200, adaptation180 and yaw-final100 target groups (600 exclusions). Force directions use100 permuted angular bins with seeded within-bin jitter, cover the full worldXY circle, and are shared across conditions/controllers.

All600 target-condition full snapshots are valid and reused by all three controllers:1800 episode tasks, no failure filtering. Original success is unchanged: distance<.10m AND speed<.15m/s for5consecutive25Hz steps; physical failures and15s timeout retain original semantics. Failed completion times are null; paired completion differences only use pairs where both succeeded. New targets are not a re-evaluation of the old benchmark/holdout cohort.

## Metric definitions fixed before scores

- Peak speed is the maximum of valid25Hz boundary world-speed samples, not a continuous-time maximum. Near-target actual/command speed uses pre-action d<.10m samples, including braking transients; no such samples means unavailable, not zero.
- Distance overshoot is maximum rebound above running minimum target distance. Maximum target-distance excursion and displacement from initial position are separate fields.
- Command saturation is fraction of action elements with|a|≥.95. Allocator saturation is separately reported; neither modifies bounds or control.
- Gust recovery requires the full2s force exposure and5consecutive valid, completed25Hz boundaries at t≥4s satisfying both original distance/speed thresholds. A partial terminal step or physical-failure terminal observation cannot complete recovery. Onset/confirmation lag is measured from4s. Termination before force start or end is explicitly counted/censored; an unexposed successful episode is not recovered. No single final-frame recovery inference.
- Constant “steady-state” output is explicitly a finite terminal-window proxy: last1s mean/std distance and speed, distance slope and target-hold fraction, only for episodes lasting≥5s with a valid terminal state. Eligibility/sample count is reported. The task still stops at original success; this cannot prove indefinite/asymptotic target holding.
- Wilson95% individual success intervals and descriptive fixed-cohort paired bootstrap intervals are supplied.100/100 observed successes do not imply zero true failure probability. Paired comparisons include all failures and separately count expert-only, BC-only, shared failures and timeout discordance.

## Results and interpretation

All1800 formal tasks and exact same-cohort resume completed. Independent verification recomputed physics force timing, source/model/trace hashes, action independence, outcome/scalar metrics and all600 triple-paired states. No omitted/duplicate tasks. All formal observations remained finite; no placeholder observation was counted as a measurement.

Final read-only review found two Important processing issues, corrected once with RED→GREEN regression tests: incomplete failed steps could falsely confirm gust recovery, and publication omitted the evaluator identity after removing its repeated per-row copies. The immutable acquisition cache and traces were not rewritten or reacquired. The public report now retains the original acquisition identity and cache hash once, plus separate current metric/verifier/publisher/plotting source hashes and the explicit reviewed revision. Original acquisition evaluator SHA256 `42115dc66b7217a5c58b18ac4335a8160c4621b2878eb438c59c7f02d41911d7` is retained and verified from its local archive. All1800 cached scalar rows and aggregate results recompute unchanged under the corrected processor; the actual gust cohort has only full-duration timeouts.

One Minor generic-field limitation is deferred: `interpretation.shared_failed_conditions` means that every controller has some failure in a condition, not necessarily on the same targets. Same-target shared failures must instead use `three_way_paired_outcomes.all_failed`. Current High/gust cohorts have100/100 triple failures, so this naming issue changes no current conclusion. No second review or additional research run.

### Success and failure accounting

Each cell is success/100. All failures are retained.

| Condition | Scripted | Original BC | Yaw-Augmented BC | Yaw-Augmented−Original | Yaw-Augmented−Scripted |
|---|---:|---:|---:|---:|---:|
|Nominal|100|100|100|0pp|0pp|
|Constant-Low|100|100|100|0pp|0pp|
|Constant-Medium|100|97|94|−3pp|−6pp|
|Constant-High|0|0|0|0pp|0pp|
|Gust-Medium|0|0|0|0pp|0pp|
|Gust-High|0|0|0|0pp|0pp|

Constant-Medium paired C−A:91both successes,3C-only,6Original-only,0both failures; C−Scripted:94both successes,6Scripted-only. Timeout discordance mirrors these counts. High and both Gust conditions have100/100 **same-episode** shared failures, not just equal aggregate success rates. High physical failures (outside_flight_area) Scripted/A/C=2/1/2, timeouts98/99/98. All other failures are timeouts: Medium0/3/6; each Gust100/100/100. Across600episodes/controller: Scripted300successes/2physical/298timeouts; Original297/1/302; C294/2/304. No nonfinite state, excessive-tilt or below-ground failures. Physical flight-area failures are unsafe outcomes, even without numerical blow-up.

100/100 Wilson95% interval is approximately[96.30%,100%];0/100 approximately[0%,3.70%]. These fixed-cohort observations are not population guarantees. Medium97/100 and94/100 and every paired bootstrap interval are retained in JSON. The original old-target nominal finding is reproduced on this new cohort, not by reusing old episodes.

### Task progress, speed and braking

Mean final distance includes every episode, including finite physical failures. Completion time includes successful episodes only.

|Condition|Final distance m S/A/C|Successful completion s S/A/C|Mean peak speed m/s S/A/C|Near-target actual speed m/s S/A/C|
|---|---|---|---|---|
|Nominal|.073668/.067980/.066201|5.450/5.688/5.753|.761744/.754550/.754180|.187839/.189485/.188111|
|Constant-Low|.078710/.077321/.078369|10.287/10.496/10.765|.859575/.850628/.846667|.212366/.207359/.212391|
|Constant-Medium|.055920/.061910/.067928|12.066/11.871/11.977|1.103025/1.096313/1.088197|.221817/.206946/.214630|
|Constant-High|1.007850/.988912/1.040618|N/A/N/A/N/A|1.793786/1.795975/1.790496|2.497170/2.497671/2.496546|
|Gust-Medium|.235642/.226688/.233598|N/A/N/A/N/A|1.506488/1.496542/1.478409|.515582/.501754/.488080|
|Gust-High|.114990/.097200/.113910|N/A/N/A/N/A|2.100862/2.089564/2.065649|.676105/.628485/.623060|

S/A/C means Scripted/Original/Yaw-Augmented. Each High near-target statistic has only1episode/2samples; crossing the target at high speed is not successful braking, and these sparse values are not a typical-controller estimate. Near-target averages can exceed the.15m/s success threshold because they include approach/braking/crossing observations. Original Gust-High mean final distance below.10m still gives0successes: distance alone is insufficient.

Mean distance-rebound overshoot m S/A/C: Nominal .181328/.189467/.187725; Low .234434/.223269/.226200; Medium .458234/.464777/.443819; High2.239860/2.253272/2.234408; Gust-Medium .918829/.916811/.905875; Gust-High2.332816/2.331551/2.312690. Mean maximum target distance High3.431050/3.444417/3.424935m; Gust-High2.792412/2.798513/2.780202m. Complete distributions and initial-position excursion are in JSON.

All command-element and allocator saturation fractions are0 in every condition. Thus saturation is not observed as the proximate explanation here. Final yaw means remain small: High .0895°/.0779°/.1175° and Gust-High .0819°/.0869°/.0758°. Failure is not simply inability to correct yaw.

Paired C−A/C−Scripted mean final-distance changes m: Nominal−.001779/−.007466; Low+.001049/−.000341; Medium+.006018/+.012008; High+.051706/+.032768; Gust-Medium+.006910/−.002043; Gust-High+.016710/−.001080. Completion deltas, restricted to both-success pairs: Nominal+.0644/+.3028s(n100/100), Low+.2688/+.4784s(n100/100), Medium+.1560/−.0047s(n91/94). No completion comparison for all-failed conditions. Mean peak/near-speed paired differences and timeout discordance are retained in JSON.

### Constant finite terminal-window proxy

|Condition|Eligible S/A/C|Mean distance m S/A/C|Mean velocity m/s S/A/C|
|---|---|---|---|
|Low|95/95/98|.121109/.119038/.116630|.155299/.144514/.147224|
|Medium|100/100/100|.102663/.112273/.115447|.196103/.182585/.177507|
|High|98/99/98|.956015/.964914/.985876|.017222/.025372/.043107|

High terminal-window target-hold fraction is0 for all eligible episodes, despite small final speed: settling at an offset is not recovering the target. Low/Medium proxy hold fractions mostly around.19 reflect the original five-sample success stop within the last1s, not proof of prolonged holding. Unequal eligible counts are disclosed and not treated as identical controller cohorts; these proxies are descriptive, whereas explicit paired results use common episodes. Static High compensation analysis is consistent with a shared PI-cap limitation (simple quasistatic Scripted approximation about.924m offset), but neither this approximation nor this evaluation establishes a unique cause.

### Gust recovery

All100targets/controller in both gusts actually received the full2s force;0terminated before gust onset;100survived force end. At every post4s physics tick the force is exactly zero. None achieved the prescribed post4s five-consecutive-sample position+velocity recovery within the unchanged15s budget. Recovery time is therefore null/right-censored, **not15s**, and not a claim of permanent inability to recover.

Fixed illustrative target000 Scripted Gust-Medium: distance/speed at4s1.265m/.609m/s; at8s.094m/.895m/s; at15s.201m/.208m/s. It passes near the goal rapidly and later oscillates, rather than stably braking. Gust-High final distance can be small while velocity remains too high. Scripted and both learned policies exhibit this shared transient-recovery limitation; no integral/root-cause intervention was performed.

### Conclusion

Frozen Yaw-Augmented BC has initial simulated **Low sustained-force** robustness (100/100), but not broad external-disturbance robustness: Medium94/100 is below Original97 and expert100; High and both gusts fail for all controllers. Yaw-covered data did not unexpectedly produce a clear sustained/gust advantage. There is a BC-specific Medium gap and larger shared High/gust limitations. Low/Medium task success does not establish indefinite station keeping or real-world mine-wind resistance. The only next recommendation is the fixed-controller post-gust PI/braking audit; not executed.

## Resource safety and reproduction

Use repository `.venv`, one MuJoCo worker, CPU inference with torch/BLAS threads1. Existing phase supervisor holds a nonblocking experiment lock, monitors process-tree RSS and global OOM count every.1s, limits sampled tree RSS to4GiB, requires5.5GiB available before startup, and aborts below2GiB available (1.5GiB OS reserve). It atomically writes phase telemetry and terminates owned process groups on failure. No swap modifications. Heavy simulation/test phases run sequentially. Atomic fsync/replace records contain unique task IDs, model/source/manifest/trace hashes; resume validates contents and refuses corrupt, changed-source, duplicate or missing tasks.

Formal simulation585.36s, sampled treeRSS1.101GiB; all completed phases peak sampledRSS3.122GiB (whole-suite tests dominate), minimum available7.545GiB, OOM counter delta0. Maximum summed historical processHWM3.934GiB is not a simultaneous memory peak. Measured polling telemetry is not a hard OS stability guarantee. No swap or worker increase; task/trace completeness and exact resume were verified.

Fresh tests:84BC-related passed (28new); read-only archived-dependency full suite617run/613passed/4knownskips/0errors/failures. Bare full suite555run/524passed/30pre-existing missing-dependency import errors/1skip/0assertion failures; **bare suite is not green**. The same historical error-name set as the previous stage was required, not silently ignored. Read-only fixture obtains old `reward_v2_alignment_audit`, `ppo_reward_success_alignment` and a diagnosis JSON through `git show 8879754a…` into memory, never restoring historical working-tree files. Full commands, error names and summaries are in the report's `tests` field. Tiny synthetic training unit tests exercise older code only inside temporary fixtures; no experimental policy/world-model checkpoint is updated.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python mujoco/rl/uav_bc_safety.py \
  --directory mujoco/reports/uav_bc_external_disturbance_seed0_parts --phase evaluation \
  -- .venv/bin/python -u mujoco/rl/uav_bc_external_evaluation.py
.venv/bin/python mujoco/rl/uav_bc_external_evaluation.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_uav_bc_external_*.py' -q
```

Use the same guarded command for smoke (`--phase smoke`/child `--smoke`), resume and verification. Publication requires verified complete simulation/resume/test/resource evidence. Reports/manifests/eight figures use an experiment-specific directory and do not overwrite older plots. Raw policy/physics traces, cache, phase logs and review scratch remain local in ignored `uav_bc_external_disturbance_seed0_parts`.

## Limits and one next recommendation

Single fixed100-target cohort, estimated nominal MuJoCo geometry/inertia, force resultants rather than wind aerodynamics, no disturbance observation, original termination censors long-term holding and some gust exposure, fixed15s budget. No claim of force estimation, physical deployment safety or mine-wind capability. The sole next recommendation is a read-only fixed-controller PI-integral/braking-response audit after gust removal, with paired Scripted/BC traces and no parameter tuning; it is not executed here.
