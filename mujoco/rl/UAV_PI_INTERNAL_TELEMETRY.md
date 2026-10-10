# Minimal PI gust-recovery telemetry

## Outcome

The preselected Scripted Gust-Medium episode reproduces the historical timeout exactly. The directly observed failure is oscillatory translation whose position and speed gates remain out of phase, despite correctly updated braking requests. The PI integral retains force-compensation memory and sometimes opposes the P braking term, but **neither integral clamping nor saturation-driven anti-windup is triggered**. Actual attitude/thrust direction reverses after the requested direction. This supports a shared cascade transient-response limitation; it does not isolate one component as the unique cause.

No controller, PI gain, policy, model, physics, external-force schedule, reward, success rule, or timeout was changed. No training or next-stage experiment was started.

## Fixed design and exact reproduction

- Branch: `feat/uav-pi-internal-telemetry`.
- Base: `a3e6ee4da8eb437f04d2e2c030f5be7b7b5d4737`.
- Before any new run, select old external-manifest index000, `external-final-2026101201-000`, seed950140000.
- Target: `[-0.7051126477516996, -1.5917973670742391, 1.14849756028413]`m.
- Direction: `[0.9845354832399106, 0.1751852797513414, 0]`, worldXY.
- Initial position `[0,0,1]`, velocity0, identitywxyz quaternion, controllerI/yaw/counters/previous action0.
- Constant-Medium:6.867N throughout. Gust-Medium:6.867N at physics ticks1000–1999, exactly `[2,4)`s.
- Physics500Hz/.002s, control100Hz/.01s, policy25Hz/.04s. Original maximum375policy steps/15s.
- Original success: distance<.10m AND speed<.15m/s for5consecutive completed25Hz steps, no physical failure.

The existing full snapshot implementation copies allMjData, including act/ctrl/integration history and solver warmstart, plus environment/RNG, controller/I/yaw/attitude parameters, allocator and previous action. Each condition reconstructs its old manifest snapshot hash. Full physical/controller pairing is verified separately by removing **only** the deliberately different `force_condition` name from the environment fingerprint; direction and every physical/controller field remain included. The common fingerprint is in the report.

Both new retained traces are **bitwise identical** to all arrays in the old nontelemetry raw files: observation, position, actions/previous actions, indices, validity/streak, policy timestamps,500Hz applied force/time and allocation/update counts. Step count, final state and termination also agree. A predeclared fallback absolute tolerance1e-12/relative0 exists, but was not needed. Verification/resume performs0new flight episodes.

Actual execution accounting:2unique experimental episodes, **3flight executions**. The first Constant flight finished, but an analysis-layer validator mistakenly treated the legacy25Hz `control_update_counts` array as100Hz telemetry and failed before publishing its raw data. Its exact regression test was observed failing, then passed after the validator fix. Only that same fixed Constant episode was repeated, followed by the one Gust episode. Initial selection/attempt/resource/source-revision evidence is retained locally. This was not a physics failure, new sample, configuration change, orOOM. An incomplete attempt otherwise fails closed rather than silently reacquiring.

## Actual code audit

The task uses [Scripted policy](test_env_scripted_policy.py), [PI configuration](audit_velocity_pi.py), [velocity controller](../control/velocity_command_controller.py), [attitude/thrust helpers](../control/position_controller.py), [quaternion attitudePD](../control/hover_controller.py), [allocator](../control/control_allocator.py), [environment](mine_uav_env.py), and the unchanged [external-force hook](uav_bc_external_disturbance.py).

1. Scripted requests world velocity `clip(position_error/3)` at25Hz and proportional wrapped-yaw rate. It does not use actual velocity to calculate its action.
2. VelocityPI measures world velocity at100Hz: `e_v = v_cmd - v`; `P = [1.5,1.5,2]*e_v`.
3. Candidate integral: oldI+`e_v*.01`. `_integral_error` stores metres. Ki `[.5,.5,.8]` converts it to acceleration.
4. CandidateXY I-contribution norm is capped1.5m/s²; Z is clipped±1.5m/s². Equivalent state limits are XYnorm3m/Zabs1.875m.
5. Conditionalanti-windup: if trialP+I exceeds output acceleration limit3m/s² AND candidateI increment increases that demand, freeze that part ofI to its old value; separateXY/Z counters increment. No leak, back-calculation state or gust-triggered reset exists.
6. Committed P+I is then limited to XYnorm3/Zabs3m/s².
7. `mass*(acceleration-gravity)` becomes desired world thrust and body-to-world attitude/yaw. QuaternionPD (`kp=[6,6,2]`, `kd=[4,4,2.8]`) requests inertia/gyro-compensated body torque. Thrust magnitude is bounded and the rotor allocator produces bounded omega² commands.
8. Existing reset clearsI, saved acceleration, freeze counters and yaw. Full snapshots restore all these states. No controller field or parameter was added.

Policy-action saturation, PI integral-clamp events, acceleration limiting/anti-windup, scalar thrust limiting, and allocator saturation are different quantities. Do not substitute one for another.

## Passive measurement and signal meanings

[Observer](uav_pi_telemetry.py) uses external CPython `sys.settrace` with exact object/function/frame identity and source-AST line probes. It copies existing locals and returned command/allocation values, never writes frame locals or production state. It refuses an existing debugger/trace, and always removes the observer on exit/exception. Original controller and simulation source hashes remain unchanged.

100Hz records: simulationtime, actualXYZ/quaternion/body angular velocity, target, world velocity/command/error, old and committed integral, actual clamp branch events, freeze counters/events, trial acceleration, committed pre/post-limit acceleration, desired quaternion/rotation/thrust/wrench, actual returned allocator wrench/rotor command/saturation, applied `data.ctrl`, and saved successstreak at update entry.

- P contribution is an explicitly **derived diagnostic** from measured error and actual gains; no separately stored P variable exists. I is the controller's actual returned I-contribution property.
- `anti_windup_trial_m_s2` is the actual `raw` local before candidateI freezing. `output_before_limiting_m_s2` is the actual final helper local **after**I commit but **before** clipping. They are deliberately distinct.
- Force is the actual500Hz applied force, indexed to the matching100Hz timestamp **after** the real before-physics hook. The buffer read at compute entry would be stale, particularly at2/4s; it is not used as current force.
- Desired thrust norm, bounded requested wrench and allocator commanded thrust/actualctrl are recorded. There is no independent measured aerodynamic/rotor thrust sensor: **not observable** here. Commanded thrust is not labeled measured thrust.
- Actual attitude quaternions are measured; angle/body-Z projections are derived. `MjData.qacc` is copied from the prior physics-step cache, not claimed to be the instantaneous derivative of the newly issued command. At the4s boundary it can still contain the final forced tick's acceleration.
- PositionPI integral, anti-windup back-calculation state, and model-learning signals: **not applicable**; the actual code has none.
- Success/timeout are evaluated only at original25Hz policy boundaries.100Hz records carry the pre-update streak; final flags remain in the episode record. No100Hz success rule was invented.

## Measured results

| Signal | Constant-Medium | Gust-Medium |
|---|---:|---:|
| Termination | success12.64s | timeout15s |
| Policy steps |316|375|
|100Hz updates|1264|1500|
|500Hz force samples|6320|7500|
| Final distance, m |.054348|.200980|
| Final speed, m/s |.129326|.208214|
| MaxXY I-contribution norm, m/s² |1.010090|.809576|
|XY/Z clamp events|0/0|0/0|
|XY/Zanti-windup freeze events|0/0|0/0|
|XY/Z acceleration-limit events|0/0|0/0|
|Allocator saturation events|0|0|
|After4s joint gates at100Hz|19|0|
|After4s joint gates at25Hz|5|0|

### Why does speed peak after force removal?

Gust distance increases from1.265298m at4s to1.368262m at4.34s (+.102964m). Motion then reverses and approaches/crosses the target while carrying substantial speed. Translational speed peaks1.478594m/s at5.51s,1.51s after force withdrawal; this is not residual applied force.

Along the fixed force direction, observed reversal brackets are:

|Signal|First post4s sign reversal|
|---|---:|
|Actual velocity|4.32–4.33s|
|P contribution|4.55–4.56s|
|CommittedPI output and desired bodyZ|4.81–4.82s|
|Actual bodyZ / cachedphysical acceleration|5.50–5.51s|
|Position error|5.67–5.68s|
|I contribution|6.10–6.11s|

At4s, I along the force axis is about−.665m/s². At5.51s, P=+2.075234, I=−.411827, and committedPI=+1.663408m/s² along that axis. Actual bodyZ projection is only+.001501, just reversing its lateral direction. Thus P/PI have already requested reversal while the physical thrust direction is still catching up. In this segment, P→PI sign reversal is delayed about.25–.27s by the opposingI contribution; requested→actual bodyZ reversal separation is about.68–.70s. These are descriptive crossing-time separations, **not** identified pure delays/Jacobian/transfer functions.

I can legitimately provide persistent compensation under Constant force. Indeed the Constant integral stays directed against the force and the task succeeds. Gust removes that need, but existingI releases only through measured velocityerror integration. LargerI alone would not implywindup; here actualI remains well below its cap and no limiter activates. Retention plus finite attitude/plant response is a supported contributor, not a uniquely proven causal mechanism.

### Five fixed time windows

|Window|Constant updates|Gust updates|Constant mean speed|Gust mean speed|Gust P/I-opposition fraction|
|---|---:|---:|---:|---:|---:|
|0–2s|200|200|.5540|.3833|.3750|
|2–4s|200|200|.7547|.7032|.2100|
|4–6s|200|200|.2853|.9826|.7250|
|6–10s|400|400|.2678|.6015|.5025|
|10–15s|264|500|.1602|.3320|.5300|

Speeds are m/s. Windows are half-open with explicit floating-clock boundary tolerance. Constant is censored at12.64s; its last row is not a fabricated full5s window. P/Iopposition is a signedXYdot-product diagnostic, not necessarily incorrect control. Report includes all phase distances, meanI, events and actual/desired attitude error.

### Why are position and speed not jointly acceptable?

Gust has5post4s signed-error crossings and5signed-velocity crossings along the force axis. The response is oscillatory and decays over the finite window, not monotonic divergence. Near target, inertia carries the vehicle through; near turning points speed is small but position remains away from target.

- After4s, at100Hz:105distance-only samples,60speed-only samples,0joint samples.
- During distance<.10m, minimum observed speed is.327396m/s, more than twice the speed threshold.
- During speed<.15m/s, minimum observed distance is.211435m, more than twice the distance threshold.
- At25Hz:26distance-only,15speed-only,0joint. Maximumsuccess streak0.

Therefore this episode does **not** fail solely because of the5-step dwell or25Hz sampling. It cannot even meet the instantaneous joint gates in the100Hz records. Eventual settling beyond15s remains unknown. No new timeout/success rule is justified by this result.

Constant and Gust already differ from0–2s because Constant starts at0s and Gust at2s. Their controller states at4s are different. This is the requested paired condition comparison, not an identical-at-removal intervention isolating onlyI carryover.

## Evidence classification and next step

- **Confirmed:** actualboundedI retention; no actual clamp/freeze/output/allocator saturation;100Hz command updates; delayed actual versus desired thrust direction; repeated phase-separated target/velocity crossings; unchangedforce cessation and task accounting.
- **Supported hypothesis:** integral/P phase opposition plus finite attitude/cascade response contributes to insufficient transient braking/settling within15s. The shared lower cascade is a specific candidate link supported by newinternal measurements.
- **Unresolved:** pathologicalwindup, uniquePI-versusattitude-versusouterposition cause, exact damping ratio/asymptotic stability, causal effect ofI removal, and generalization to High/BC/all600old failures.

No automatic low-level controller change is warranted. One next recommendation only: a separate same-state local attitude-cascade response validation, comparing requested braking with actual thrust-direction response while preserving current gains/task criteria. Not executed. Do not formally enterPPO/SAC on the assumption this common recovery limitation has been resolved; nominal learningbaselines remain valid, but robust-control readiness is not established.

## Verification, resources and reproduction

45/45relevant tests passed:19newtelemetry/analysis tests,18saved-trajectory audit tests,8actualcontroller tests. No skipped/error/failedfinaltests. Test physics setup checks exactreset without integrating a new flight episode. Bare project-wide tests were deliberately not run because unrelated fixtures perform simulation/training, inconsistent with the requested minimal budget.

OneMuJoCoworker; guarded serialized phases, atomicfsync/replace writes, unique IDs/rawSHA/completedrecord hashes, fail-closedresume. NoOOM/swapmodification. Sampled process-treeRSSpeak≈1.178GiB; maximumsumhistoricalVmHWM≈1.184GiB (not a simultaneous treepeak); exactbytes and per-phase evidence are in the report. The highest memory use is imports/controller-test setup, not a large dataset.

OriginalBC SHA256:
`7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c`

Yaw-AugmentedBC SHA256:
`d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce`

Canonicalv3 SHA256:
`42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9`

All are unchanged before/after. They are not loaded, executed or modified in these Scripted runs. Controller parameters and production source bytes are unchanged; no optimizer exists in the experiment.

Artifacts: report, this document, telemetry/acquisition/analysis/plot source, tests,8namespaced figures andEXPERIMENT_LOG. Raw100Hz/500Hz/25Hz arrays, attempts/selection/recovery/evidence/cache remain local-only in `mujoco/reports/uav_pi_internal_telemetry_parts/`. Historicalraw data/untracked files are preserved.

```bash
# Read-only verification of existing completed tasks;0new flights.
.venv/bin/python mujoco/rl/run_uav_pi_internal_telemetry.py --verify-only
PYTHONPATH=mujoco/rl .venv/bin/python -m unittest test_uav_pi_internal_telemetry test_uav_gust_recovery_audit test_velocity_command_controller -q
# Analysis reads existing raw artifacts; does not simulate.
.venv/bin/python mujoco/rl/uav_pi_telemetry_analysis.py --finalize
```

The acquisition script refuses stale/unfinished tasks. The special logged verifier-error recovery applies only to the exact initially recorded validation-source revision and one same-Constant repeat; it is not a general permission to silently rerun experiments.
