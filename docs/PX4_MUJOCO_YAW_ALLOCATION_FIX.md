# PX4–MuJoCo Phase 2C — Yaw allocation compatibility

Branch: `feat/px4-mujoco-yaw-allocation-fix`.
Research base: `fbd273be0f7169f738f6c8024e7ec7be37e14941`.
Original PX4: `8f9b91b2bdd3a65952601a084c3aa35e96fee59d`.
This is an isolated SITL-only compatibility experiment, not hardware approval.

## Root cause and alternatives

Pinned `src/modules/control_allocator/ControlAllocator.cpp` removes a whole
control row when every entry has absolute magnitude ≤0.05. With CT=1 and
KM=±0.017207708, it removes yaw, while roll/pitch/vertical thrust survive.
Prior immutable Phase2B ULog corroborates desired yaw100°, positive body-z
rate/torque requests, unallocated yaw and almost unchanged true yaw90°.

Scheme A (matrix/actuator scaling) is **rejected for flight**: B'=sB with
physical c=s·v requires v∈[0,1/s], not stock PX4 [0,1]. Merely changing CT/KM
and using an inverse adapter would change maximum force or saturate/reject
valid PX4 outputs. Internal axis normalization further prevents an assumed
end-to-end equivalence. No full allocator/range/hover/saturation proof exists.

Scheme B is a narrow source exception in an independent PX4 worktree.
It retains the **unchanged** weak yaw row only for a single multirotor4-motor
matrix, with positively observed fault-free motor health and no stopped motor,
non-tilting CT=1, two yaw signs,
roll/pitch surviving the original filter, well-separated normalized rows and
a zero-moment unit-collective trim strictly inside [.1,.4] per motor.
Zero/dependent/near-degenerate/nonfinite axes remain rejected. This certifies
local authority around interior trim, **not** feasibility of arbitrary torque.
The unchanged allocator/desaturation and min/max constraints still apply.
Raw motor failure status is checked even when upstream failure mode is Ignore;
health eligibility transitions explicitly refresh the matrix. Unknown health
fails closed. This does not certify detection of unobserved hardware faults.

The build explicitly refuses non-SITL boards. Its configuration merges the
default SITL configuration with only three Gazebo module exclusions; no
Gazebo engine is built or run. No controller PID/EKF parameters are tuned.

## Physical mapping and provenance

MuJoCo3.14.0, mass7kg, 500Hz physics, 250Hz lockstep sensor/actuator exchange.
Rotor order, COM/inertia/sites, kf=4.9977603696661616e-5 and km=8.6e-7 unchanged.
Adapter c∈[0,1] → omega²=c·(8500·2π/60)²; ideal instantaneous actuator,
THR_MDL_FAC=0. Rotor handedness and km are engineering assumptions, not
independently measured deployment parameters.

In FRD at COM, W=B·c·kf·omega_max² (rows Tx/Ty/Tz/Fx/Fy/Fz). Keeping every B
entry and the entire motor box unchanged preserves the full physical wrench
map. Actual MuJoCo generalized actuator force, corrected from origin torque
to COM, is checked at all basis/extreme inputs and a mixed input. Pure ±yaw
around hover leaves Fz/RP unchanged. An unreachable100Nm request is not
claimed achieved after clipping. Tests compile pinned PX4's actual selected
SequentialDesaturation allocator, real parameter store and default airmode;
zero-yaw hold/RP outputs match the original filter. Two no-op command registry
hooks satisfy standalone linking only; allocation and parameter code are real.

`mujoco/px4_bridge/compat/px4_yaw_compat.patch` archives the exact custom
source/config diff. `px4_yaw_build_identity.json` explicitly pins commit,
binary, version header, startup tree, board configuration and patch hashes.
The runner rejects mismatches and retains the original unpatched entrypoint.
Custom source/binary are at `/home/keyousuke/PX4-Autopilot-mineuav-yaw`;
the original `/home/keyousuke/PX4-Autopilot` is not overwritten.

## Protocol and reproduction

All pre-existing telemetry freshness checks/watchdogs/physical envelopes and
magnetic reference remain unchanged. An actual allocator yaw-row diagnostic
is additionally required in matching disarmed proof and before active flight.
No forced arm/precheck bypass. One disarmed16s check then, only if passed,
one45s bounded yaw test: relative altitude2m, NED yaw90→100°, step≥32s.
Settling: error<2°, |FRD body-z rate|<.05rad/s for≥2s, samples gaps≤.15s.
No automatic flight retries or criteria changes.

From the research worktree:

```bash
export PYTHONPATH=mujoco
PY=/home/keyousuke/MineUAV-Learning/.venv/bin/python
PX=/home/keyousuke/PX4-Autopilot-mineuav-yaw
ID=mujoco/px4_bridge/compat/px4_yaw_build_identity.json
PARTS=mujoco/reports/px4_yaw_allocation_parts
$PY -m px4_bridge.allocation_runner --px4-root "$PX" --compatibility-manifest "$ID" --runtime "$PARTS/disarmed_01" --mode disarmed --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
# Only after independent offline safety review and successful disarmed proof:
$PY -m px4_bridge.allocation_runner --px4-root "$PX" --compatibility-manifest "$ID" --runtime "$PARTS/yaw_01" --mode yaw --smoke-proof "$PARTS/disarmed_01/smoke_summary.json" --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
$PY -m px4_bridge.analyze_yaw_allocation --px4-root "$PX" --compatibility-manifest "$ID" --dependency-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

Runtime directories must be **new and empty**. Commands above describe the
authorized experiment, not permission to repeat flight after a failure.
Binary/build/ULog/raw summaries/resource logs remain local-only.

## Results

**Allocation compatibility PASS; complete yaw validation BLOCKED.** No further
flight was launched after the first active run stopped.

| Acceptance | Status | Evidence |
|---|---|---|
| Real physical motor mapping unchanged | PASS | Basis/extreme/mixed MuJoCo force tests; unchanged full motor box |
| Original weak-yaw root cause | PASS | Pinned source, old ULog unallocated torque, actual geometry |
| Isolated build identity | PASS | Clean source, archived exact patch, pinned binary/config/header/rootfs |
| Disarmed bidirectional communication | PASS | One16s run;4000 IMU,160 GPS,3985 actuator messages |
| Actual PX4 yaw row retained | PASS | Live allocator diagnostic in both runs:±.017207708 |
| Complete45s bounded flight | BLOCKED | Query-service deadline at simulation15.484s |
| +10deg yaw direction/tracking/settling | NOT TESTED | Step was never sent; all response metrics null |
| Active-window abnormal actuator saturation | PASS | Zero upper-bound physical samples and zero allocator limit flags after arm request |
| Final in-flight heading alignment in custom run | BLOCKED | Stopped during ascent before final alignment; not evidence of alignment regression |
| Safe stop and owned process cleanup | PASS | Four controls zero; PX4exit0; no retry |
| Phase3 entry | BLOCKED | Required maneuver and sustained-health evidence absent |

### Exact mathematical evidence

Matrix rows are `[Tx,Ty,Tz,Fx,Fy,Fz]`, FRD at the actual COM. Original
filtered yaw is `[0,0,0,0]`; retained yaw is
`[-.0172077077809,+.0172077077809,-.0172077077809,+.0172077077809]`.
Roll, pitch and vertical thrust rows are bitwise unchanged in the source
algorithm. Fx/Fy remain zero; they are not falsely declared controllable.

| Row | Motor1 | Motor2 | Motor3 | Motor4 |
|---|---:|---:|---:|---:|
| Tx | .2198027503 | -.2191992497 | -.2191992497 | .2198027503 |
| Ty | -.2340983576 | -.2340983576 | .2049046424 | .2049046424 |
| Tz | -.0172077078 | .0172077078 | -.0172077078 | .0172077078 |
| Fz | -1 | -1 | -1 | -1 |

Filtered rank3→4; row-normalized condition number1.068724. The dimensionful
physical matrix condition number58.12 is **not** coordinate-invariant.
Physical Fz per full motor−39.597751N; Tz per motor±.681387Nm. Hover controls
`[.40412044,.40531244,.46297427,.46178227]` yield Fz−68.67N and essentially
zero moments. Zero-yaw nominal mixer-column difference≤1.12e−15. Actual
SequentialDesaturation tests preserve nominal hold/RP outputs and bound
unreachable requests rather than claiming arbitrary yaw torque achieved.

### Build provenance

Custom PX4 commit: `142c24cffb0cdfd57ab0a9f2506d3203610ce737`.
Binary SHA256: `2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004`.
Original binary SHA256 remains
`4f491ce85d6c79fbd5713894112c8911903fcefd1bf45b7da3579cd1812beac4`.
GCC11.4, RelWithDebInfo, Ninja, configuration`px4_sitl_mineuav_yaw`.
Builds used one outer and one nested job. The exact5-file diff and build
identity are tracked; build products are not. Dependencies were reused from
existing local submodules; an interrupted MicroCDR download attempt was
replaced with an existing local Git repository. No system installation.

Rebuild in the isolated worktree with the same existing dependencies:

```bash
env -u CI -u VSCODE_PID -u CLION_IDE GIT_SUBMODULES_ARE_EVIL=1 CMAKE_BUILD_PARALLEL_LEVEL=1 make px4_sitl_mineuav_yaw -j1
```

This is a provenance/recipe record, **not** an independent reproducible-build
certificate. Any different binary must undergo a new explicit review and
identity update; it is not automatically admitted by the runner.

### Stopping evidence and interpretation

The disarmed CLI first reported exit1 because ANSI color reset bytes followed
the numeric allocator row. A real-line RED→GREEN test corrected parsing of
SGR bytes only; missing/NaN/later-zero rows remain rejected. The original
completed raw proof was subsequently validated offline without altering it
or rerunning that episode. This parser correction preceded the only flight.

The active run armed normally at11s and stopped at15.484s of simulation
(16.088s elapsed wall time), before32s yaw authorization. Final physical
height1.461455m; maximum sampled XY excursion.011646m; peak sampled speed
1.264570m/s; maximum motor fraction.561637. No physical envelope violation
was reported. These ascent observations are **not** validation of2m hold or
post-step coupling. No final heading alignment, yaw overshoot, steady-state
error, or settling time is available for this run.

**Confirmed:** last safety query wall time15.435843s plus its unchanged.65s
service deadline gives16.085843s, earlier than the actuator deadline16.333173s.
The final monitor fault is `safety query service stopped`. Last recorded
actuator age is only.002078s. The generic receive exception text says
`PX4 actuator absolute watchdog expired` for the minimum of several safety
deadlines; it is **not evidence that actuator messages stopped**.

Wall intervals for the last five queries increased
`.53678,.55974,.58786,.60974,.64164s` for nominal.5s simulation intervals.
The new high-frequency yaw-row check scans accumulated console text; final
text is265803 characters. An offline100-call benchmark costs approximately
1.7ms/check on this machine. **Supported hypothesis:** this diagnostic
overhead contributes to wall-time schedule overrun. There is no in-flight
CPU attribution or controlled counterfactual, so host load and other
diagnostic work cannot be ruled out. No watchdog threshold was relaxed,
no runtime scheduling fix was made after flight, and there was no retry.

Actual ULog saturation flags show20 pre-arm motor/sample entries at the
minimum bound (five timestamps.3–.9s, four motors each); these normal zero
outputs must not be confused with active-flight upper saturation. After
the non-force arm request, all observed allocator flags are zero. No
post-step saturation result exists because the step did not occur.

388 controller/EKF/allocator ULog parameter values were compared with the
old Phase2B flight:387 match. The only difference is volatile
`EKF2_MAG_DECL` (old3.405605°, new0°); logs start at different times1.344s
and.304s. Both source builds configure default0/DECL_TYPE3, and
`EKF2::UpdateMagCalibration` automatically saves inferred declination
without notification. Explicit environment overrides, initial qpos,
synthetic magnetic field and noise are identical. Exact unlogged update
time is not observable; do not describe all recorded parameter values as
identical or treat this mutable estimate as deliberate EKF tuning.

## Independent review and verification

One fresh-context read-only reviewer independently ran134 preflight tests,
finding0Critical/4Important/0Minor. All four findings were addressed offline
before flight: raw motor-health revocation/refresh, full disarmed evidence
binding, honest analyzer admission/provenance, and the actual selected
SequentialDesaturation native test. Author verification:147 tests before
flight;150 tests after analysis-only regressions,0skip, original109 retained.
The reviewer did **not** independently recheck the final fixes or rebuild
the binary; no second review or flight is claimed. An earlier reviewer
invocation was unavailable due quota before doing work, not a second review.

The two post-flight analysis failures (null pre-query gate plotting and
incomplete-step status classification) were handled by offline regressions.
No flown runtime/safety file or custom PX4 source/binary changed afterward.
Final analysis validates the frozen runtime hashes and14 protected historical
files against the research base. Original PX4 source/binary and all87 original
untracked files remain preserved. All raw summaries and ULogs are retained.

Four plots show only the actual truncated recording, not synthetic step
response. Discrete targets/motors/health use steps; the red line marks the
recorded stop. The last sampled gate flag precedes final fault latching.
Research source and isolated PX4 source diff whitespace checks pass. Checking
the exact archived unified diff as newly added text flags its required
context-prefix spaces/tabs; those approved provenance bytes are preserved,
not normalized or incorrectly reported as a clean all-file check.

## Resources and limitations

One MuJoCo/one PX4; exactly one disarmed and one active invocation; zero
flight retries. Bridge peak772112384bytes(.719GiB), sampled PX4 peak
10018816bytes(9.56MiB), minimum available8817274880bytes(8.21GiB).
Final incremental build child peak406724608bytes; process-group samples
may miss short-lived peaks. No OOM observed in owned runs, no kernel-wide
OOM telemetry or hard-real-time guarantee. No swap modifications.

**Unresolved:** full positive yaw response, final in-flight alignment and
hover coupling on the custom build; precise attribution of wall-time
query overrun; hardware motor direction/identified drag/ESC lag. No real
hardware, ROS/Odometry, negative step or robustness qualification.

## Rulings and next step

- User's autonomous scope replaces redundant approval/merge menus; cost:
  design revisions remain possible, no extra experiment budget.
- Choose guarded source fixB over unproved scalingA; cost: build or flight
  may remain blocked, never silently alter CT/KM or motor limits.
- Review before flight and repair Important findings offline; cost: final
  fixes have author tests, not a second independent review certificate.
- Exclude only three Gazebo build modules in the isolated board overlay;
  cost: custom build/configuration must be explicitly pinned.
- Preserve historical files, local worktrees and ignored planning artifacts;
  cost: retained local disk usage instead of destructive cleanup.
- Available reviewer substituted after quota failure; cost: no claim of
  frontier-model or multiple independent reviews.
- Accept immutable disarmed proof after ANSI-only parser repair; cost:
  no rerun evidence, while the final parser is the one used in flight.
- Live readiness/response judgments deferred to actual runs; cost: this
  stopped run means BLOCKED/NOT TESTED, no guessed flight success.
- Hardware controllability and independent binary reconstruction remain out
  of scope; cost: no deployment approval or reproducible-build claim.
- Unrelated CAD ABI suite remains blocked without installation; cost:
  report the scoped Bridge result, not whole-repository PASS.
- Preserve the exact approved patch archive despite outer text-whitespace
  warnings; cost: disclose this narrow check exception, never silently
  change the pinned source-diff identity.
- Stop after the actual deadline failure with no additional flight; cost:
  the original yaw-validation goal remains incomplete rather than fabricated.
- Repair only offline analysis after flight; cost: final analysis has author
  regressions, while the flown runtime stays byte-identical to its freeze.

Deferred Minor findings: none. The operational query-scheduling blocker is
**not** dismissed as a minor; it prevents flight completion and Phase3.

**One next recommendation only:** a separately authorized offline audit of
bounded/incremental allocator diagnostics and wall-versus-simulation query
scheduling, preserving fail-closed deadlines. Do not rerun flight, adjust
gains or enterPhase3 in this stage.

## Verification limits

Full repository pytest collection is blocked by pre-existing CAD SciPy versus
NumPy2.2.6 ABI incompatibility (`cad_check/test_inspect_stl_assembly.py`). No
installation/global dependency changes are made. Bridge regressions are run
separately, including the original109 tests and new mathematical/identity
checks. Resource guard reserves1.5GiB; one outer/nested build job, one SITL
and one MuJoCo only. No history/checkpoint/reward/environment modifications.
