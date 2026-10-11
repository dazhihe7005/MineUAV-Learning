# PX4–MuJoCo Phase 2: heading and yaw validation

Date: 2026-10-11. Branch: `feat/px4-mujoco-yaw-validation`.
Base: `feat/px4-mujoco-sitl-integration`,
`3b36e9d77f2f6b89699b1ebd065e1e38310999be`.

## Outcome

The approximately3.4° Phase1 heading bias is confirmed to originate from an
inconsistent simulated geomagnetic reference, not a3.4° physical yaw error.
Correcting that sensor reference removes the bias without changing EKF
parameters, gains, frames, actuator mapping, physics, preflight checks or
health flags. A2m ascent completes actual in-flight magnetic alignment.

The independent+10° yaw response is **not fully validated**. The request was
authorized by actual healthy/final-aligned states, but the live diagnostics
freshness gate stopped the final attempt0.192s later. No further retry was
performed. Settling time and full yaw actuator dynamics remain untested;
Phase3 external Odometry entry is blocked pending completion of this gate.

| Acceptance item | Status | Evidence / boundary |
|---|---|---|
| Source of3.4° difference | PASS | Original field has0° declination; actual PX4 grid/ULog uses3.405605° |
| ENU/NED, FLU/FRD, yaw/unit math | PASS | Proper rotations and wire/sensor interface tests; defaults preserved |
| Physically based sensor correction | PASS | Field reconstructed from pinned PX4 WMM tables at the synthetic GNSS location |
| Estimate–truth bias correction | PASS | Flight prestep5s mean+0.002979°, max absolute0.010692°; original+3.413907° |
| Final in-flight EKF heading alignment | PASS | Actual flags at15.848s, heading-good at16.856s; true through final recorded flight |
| Step authorized with real health | PASS | Logged original topic timestamps and true gate fields at32.004s |
| Complete independent yaw response | BLOCKED | Diagnostics freshness abort at32.196s |
| Settling time / full yaw motor response | NOT TESTED | Censored observation, no fabricated steady-state result |
| Offline motor mapping | PASS | Existing Phase1 interface suite and unchanged geometry |
| General absence of actuator faults | NOT TESTED | No saturation in this bounded ascent; this does not validate full yaw dynamics |
| External Odometry phase entry | BLOCKED | Complete yaw response and robust live health telemetry still needed |
| ROS/MID360S/FAST-LIO2/Odometry integration | NOT TESTED | Not implemented in this phase |

No acceptance claims real-aircraft safety or general flight robustness.

## Pinned implementation and coordinate audit

PX4: `v1.18.0-beta1-575-g8f9b91b2bd`, commit
`8f9b91b2bdd3a65952601a084c3aa35e96fee59d`. Existing binary SHA256:
`4f491ce85d6c79fbd5713894112c8911903fcefd1bf45b7da3579cd1812beac4`.
MuJoCo3.14.0, Python3.10.12, pymavlink2.4.49; existing pyulog only.
No install, compile, Gazebo, hardware connection or neural-network training.

MuJoCo ENU→PX4 NED: `(x,y,z)→(y,x,-z)`.
MuJoCo FLU→PX4 FRD: `(x,y,z)→(x,-y,-z)`.
Both matrices have determinant+1. Attitude transformation is
`R_NED_FRD = R_ENU_NED @ R_ENU_FLU @ R_FLU_FRD`; wxyz quaternions.
Identity MuJoCo attitude corresponds to90° NED yaw, not0°. Positive ENU
yaw decreases NED yaw. FRD body-z angular rate is negative FLU body-z rate
(a body angular rate, not an exact Euler yaw derivative during tilt).
Existing sensor adapter already converts body specific force, angular
rate, GNSS position and magnetic field consistently. No frame change
was required. Existing normalized-thrust→omega² mapping is unchanged,
including rotor1–4 ordering/FRD allocator geometry and yaw torque signs.
Ideal instantaneous motor response remains a simplified assumption.

## Confirmed magnetic reference mismatch

Original `SensorAdapter`: NED magnetic field `[0.215,0,0.427]`gauss,
therefore horizontal declination0°. GNSS reference: latitude47.397742°,
longitude8.545594°, altitude488m. Actual ULog parameters:
`EKF2_DECL_TYPE=3`, `EKF2_MAG_TYPE=0`, `EKF2_MAG_DECL=3.405605316°`.

Source audit:

- `PX4/src/lib/world_magnetic_model/geo_mag_declination.cpp` and
  `geo_magnetic_tables.hpp`:10° grid,
  bilinear interpolation, table scaling and longitude handling.
- `PX4/src/modules/ekf2/EKF/aid_sources/magnetometer/mag_control.cpp`,
  `updateWorldMagneticModel`: inclination/declination rotate the field.
- `mag_control.cpp::checkHaglYawResetReq`: final magnetic reset requires
  in-air/yaw-aligned state and estimated HAGL>1.5m.
- `EKF/ekf.h::isYawFinalAlignComplete`, `EKF2.cpp`: magnetic alignment
  and more than1s after reset determine `heading_good_for_control`.

Source-matched interpolation returns declination3.405605306°,
inclination63.157199846°, strength0.482387063gauss. For positive east
declinationD and positive downward inclinationI:

`B_NED = S * [cos(I)*cos(D), cos(I)*sin(D), sin(I)]`.

Phase2 field:
`[0.21743440698304717, 0.012939331084370323, 0.4304092603315878]`gauss.
Grid SHA256 `3e985cba6a1fd9b246d628bf1d24b843b306a5a6d60ec69900188ec535a3b8cb`;
field configuration hash `2a7b675a6a3722654294ed96db075a3008b6325e5ede816bc75441270e57141b`.

This is the PX4-pinned WMM2020 grid dated2024.41257, consistent with its
own EKF reference; it is not a measured2026 geomagnetic field or calibrated
sensor model. Added sensor configuration is opt-in: historical default
field, noise and Phase1 runner behavior remain unchanged.

Old hover physical yaw was approximately90.006°, while estimated yaw
was93.427° and desired yaw approximately90°. Thus estimation error and
physical tracking error were distinct. Original final5s estimate–truth
mean+3.413907°; corrected disarmed final5s−0.004765°, corrected flight
prestep5s+0.002979°. The intervention plus code/parameter evidence
confirms the simulated magnetic-reference mismatch as the bias source.

## Real alignment and finite test protocol

Phase1 hovered near0.6m, below the actual source>1.5m HAGL reset condition.
Ground startup may briefly show heading-good before magnetic fusion
activates; that transient is **not** counted as final alignment.
Offline alignment requires actual `cs_in_air` together with
`cs_mag_aligned_in_flight`, then `heading_good_for_control`.

Model is7kg with no overhead obstacle at2m and adequate static thrust
margin. Only this bounded nominal ascent was authorized. One disarmed
16s smoke followed by fixed45s maximum flights; start/hold90° NED yaw,
height setpoint−2m NED, planned+10° step at32s. Envelope: height<2.6m,
existing tilt/position/velocity checks. Actual step requires all three
topic states fresh, tilt/yaw aligned, in-air magnetic alignment, healthy
mag/baro, GNSS position, valid position/velocity, preflight pass,
armed/offboard, true height2.051±0.15m, XY excursion<0.2m,
speed<0.15m/s and estimate–truth yaw<1°. No precheck disable/health write.

Actual final run: magnetic in-flight alignment15.848s, final heading-good
16.856s. At32.004s gate used LP timestamp33000000, flag timestamp32192000,
status timestamp32668000μs, decision clock33004000μs. `heading_good`
and `cs_mag_aligned_in_flight` were truly set. The max physical height
was2.052484m, XY excursion0.023690m; peak ascent speed1.262715m/s.
Motor maximum normalized thrust0.561665;0 saturation samples.

## All attempts and stopping behavior

All four Phase2 SITL invocations and raw logs are retained locally;
no failed attempt was dropped or labeled complete.

| Run | End (sim s) | Outcome |
|---|---:|---|
| disarmed_01 |16.000| Complete corrected-field communication smoke |
| yaw_01 |32.004| No step; listener title `3 instances` not parsed correctly; safe stop |
| yaw_02 |23.556| Interrupted after a test/launch ordering mistake; no step; safe stop |
| yaw_03 |32.196| Step authorized32.004; freshness gate aborted0.192s later |

`yaw_01` was a passive-parser defect; corrected against captured console
and tests. Only EKF primary instance0 is active; registered but unpublished
instances1/2 must neither be merged into nor invalidate its fields.
`yaw_02` was launched in the same tool batch before the failed regression
test result was read. It was immediately stopped by SIGINT, exit130;
an atomic incomplete summary, zero commands and owned PX4 exit0 were
verified. This was a workflow violation, explicitly disclosed. Parser
fixes and51/51 fresh tests were read in a separate call before `yaw_03`.

Original single-flight implementation budget was explicitly revised
for these two instrumentation attempts, then hard-stopped after the
final run; no tuning or new scenario. All shutdowns zeroed four MuJoCo
controls and terminated only owned SITL processes. No physical abnormality
was observed, but no complete45s yaw test exists.

## Final blocking mechanism and censored response

`EKF2.cpp::PublishStatusFlags` publishes approximately1Hz, or immediately
when control/fault flags change. Console listeners were queried2Hz;
the fail-closed gate required each publication timestamp age≤1s at every
250Hz sensor tick. At simulation32.196s the clock33196000μs minus last
queried flags32192000μs was1.004s. All required boolean states still
showed healthy/final-aligned; ULog likewise did not show loss of alignment.
This is a **confirmed telemetry publication/query freshness mismatch**,
not evidence of EKF instability, frame failure or motor fault.

No gate threshold was relaxed after this outcome and no further flight
was run. Future validation must make receipt/publication/sample time
semantics source-consistent without replacing stale state with fake health.

Only9 physical samples with valid aligned desired/estimated angles exist
after step, and only0.192s observation total. Max observed desired–truth
error10.002422° and estimate–truth error0.008864° are early transient
values, **not steady-state errors**. Requested-target–truth max10.002408°.
Physical yaw change over the9 samples is−0.000518°; this tiny censored
change is not sufficient to identify torque sign, lag, normal tracking
or a fault. Settling time is null/NOT TESTED. The criterion was error<2°
and |FRD body-z rate|<0.05rad/s continuously through the remaining record,
with at least2s dwell and no sample gap>0.15s; incomplete runs cannot pass.

## Reproduction and analysis

Existing local environment only; no dependency installation. From this
isolated worktree, use `.venv` of the original repository. All test/flight
commands were sequential, never tests with active SITL.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/keyousuke/MineUAV-Learning/.venv/bin/python -c 'import sys;sys.path.append("/usr/lib/python3/dist-packages");import pytest;raise SystemExit(pytest.main(["mujoco/px4_bridge","-q","--tb=short"]))'

# Offline only; requires preserved local raw Phase1/Phase2 logs.
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.analyze_yaw --px4-root /home/keyousuke/PX4-Autopilot --dependency-search-path /home/keyousuke/.local/lib/python3.10/site-packages

# Historical exact Phase2 invocations; do not reuse nonempty runtime dirs.
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.yaw_runner --px4-root /home/keyousuke/PX4-Autopilot --runtime mujoco/reports/px4_yaw_phase2_parts/disarmed_01 --mode disarmed --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.yaw_runner --px4-root /home/keyousuke/PX4-Autopilot --runtime mujoco/reports/px4_yaw_phase2_parts/yaw_03 --mode yaw --smoke-proof mujoco/reports/px4_yaw_phase2_parts/disarmed_01/smoke_summary.json --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

Existing Phase1 starts PX4 as a child of the bridge; no separate simulator
or duplicate PX4 should be started. Loopback TCP4560 simulator transport,
500Hz physics,250Hz sensors/actuators,10Hz GPS,20Hz offboard setpoint,
2Hz safety listener requests,50Hz Phase2 physical logs. Physical clock
is `t*1e6+1e6`. Attitude sample uses ULog `timestamp_sample`; desired
attitude uses publication timestamp and causal zero-order hold (no future
command ramp); estimated angles unwrap before interpolation,
with no endpoint extrapolation. Hashes identify each summary and ULog.
Raw ULogs, parameters, consoles and physical arrays stay local-only.

## Resources, integrity, interpretation

One MuJoCo and one PX4 at a time; no OOM observed, no swap changes.
Maximum measured bridge process HWM581996544bytes (555.0MiB), PX4
sampled RSS20418560bytes (19.47MiB). Minimum system available memory
11299450880bytes (10.52GiB). Startup reserve2.5GiB/runtime reserve1.5GiB.
Offline analysis process peak is recorded separately in JSON (includes
plot/library loading). A single runtime CPU probe observed bridge31.6%
and PX47.3% of one CPU core; this is not a complete CPU peak measurement.

Historical physics/config/actuators, Original/Yaw-Augmented BC,
canonical v3 and Phase1 report all match base-commit hashes. PPO/RL/control
source is unchanged. Original worktree87 untracked files are preserved;
main is untouched. Scoped complete bridge suite, no historical training
or heavyweight RL tests. Independent review and fresh verification are
recorded in the final handoff/experiment log.

Confirmed: magnetic reference mismatch and corrected estimate bias;
actual final EKF alignment; diagnostics time-age abort.
Supported hypothesis: physical2m ascent provides the source-required
alignment opportunity, consistent with HAGL logs.
Unresolved: full yaw tracking/settling, motor dynamics, external Odometry
end-to-end readiness. No physical control-gain adjustment is justified by
the censored response. No Phase3 work is started.

Only next recommendation: correct live health-telemetry timing semantics
with offline tests, then repeat this same bounded yaw protocol before
external Odometry integration. This recommendation is not executed.

## Independent review and offline safety fixes

Fresh read-only review independently passed59/59 tests (0skip), reconstructed
all reported hashes/metrics and the exact stored step-gate console prefix.
It found0Critical and2Important gate defects: truncated new headers could
retain older healthy records; GNSS-fusion-enabled did not require explicit
absence of GNSS faults. Actual recorded step-state faults were false, so
these were safety edge cases, not detected faults in the recorded flight.

One offline TDD fix pass added10 failing regressions, then passed the complete
69/69 suite (0skip). Every header is now a boundary, truncated prefixes
invalidate health, and unfinished record bodies cannot authorize. Complete
record termination accepts actual pinned-PX4 blank-line or shell-prompt forms.
Primary instance0 remains isolated from other slots. GNSS position/yaw/height
fault flags and vehicle failsafe must be explicitly false; absent fields
reject. The original SHA256-matched step prefix still passes these tightened
checks offline; partial prefixes and asserted faults do not. No additional
SITL run or second review was performed. Raw evidence is immutable.

Known1s publication/query mismatch remains an operational blocker, explicitly
retained rather than weakening health checks. One deferredMinor: ordinary
line plots draw slopes across discrete flags/held setpoints; numerical
setpoint interpolation is causal, but native-timestamp step plotting would
improve presentation. Current figures must not be used as pure-delay evidence.

Review boundaries: full yaw dynamics; general flight/other seeds/locations/
hardware;2026 measured magnetics; ROS/localization; full historical RL behavior;
every forced-kill/secondary-cleanup exception; continuous CPU/concurrency/OOM/
swap history; and independent authorization-history reconstruction were not
established by this review. Coordinator preserved the isolated workspace and
87 original untracked files, checked all stop artifacts and observed memory
records. Absence of every unsampled resource event is not claimed. Detailed
decisions and their costs are reported at handoff; no next phase is authorized.
