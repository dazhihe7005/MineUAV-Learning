# PX4–MuJoCo Phase2B: telemetry freshness and complete yaw observation

Date:2026-10-11. Branch:`feat/px4-mujoco-yaw-telemetry-fix`.
Base:`feat/px4-mujoco-yaw-validation`, `601f517bc9bc6034197808646931f8214e9596a4`.

## Outcome

The telemetry defect is fixed and the **one authorized45s flight completes**
without a telemetry safety abort. This is **not a successful yaw response**:
the target changes90→100° at32.004s, but physical heading remains approximately90°.
Maximum poststep target error10.001821°, final2s mean target error9.996366°,
maximum physical FRD-z rate0.000153485rad/s, settling **FAIL / not achieved**.
The complete recording makes this a negative result, not a censored
`NOT TESTED` response. No additional flight or parameter fix was attempted.

| Item | Status | Evidence |
|---|---|---|
| Original freshness defect | PASS |1Hz publication +2Hz read can normally exceed1s age; original flags age1.004s |
| Distinct source/query/receipt/liveness times | PASS | All preserved separately; duplicate reads do not renew publication |
| Offline fail-closed checks | PASS | Original69 + new regressions, including original ULog replay |
| Disarmed sensor/actuator exchange | PASS |4000HIL_SENSOR,160HIL_GPS,3941HIL_ACTUATOR; no arm request |
| Full45s bounded recording | PASS |11250HIL_SENSOR,450HIL_GPS,11187HIL_ACTUATOR |
| Final EKF heading alignment | PASS |Mag aligned15.864s; heading-good16.872s, maintained |
| Absence of spurious freshness abort | PASS |MonitorfaultNone;45s completed |
| +10° yaw tracking and settling | FAIL |Setpoint reaches100°, truth does not turn; no qualifying2s window |
| Motor saturation | PASS |No physical motor upper-bound saturation; active allocator channels0–3 not saturated |
| Allocator yaw authority | FAIL |Pinned PX4 weak-row filter eliminates the existing yaw-effectiveness row |
| Height/XY envelope | PASS |Maximum height2.056629m; XY0.030747m; unchanged envelope never violated |
| Phase3 entry | BLOCKED |Yaw authority/response must be resolved first |
| External Odometry integration | NOT TESTED |No ROS/MID360S/FAST-LIO2 or external pose estimator integrated |

## Root cause and matching-source audit

PX4:`v1.18.0-beta1-575-g8f9b91b2bd`, commit
`8f9b91b2bdd3a65952601a084c3aa35e96fee59d`; existing binary SHA256
`4f491ce85d6c79fbd5713894112c8911903fcefd1bf45b7da3579cd1812beac4`.
MuJoCo3.14.0. No install/build or global environment change.

Actual pinned source, not an assumed generic telemetry cadence:

- `src/modules/ekf2/EKF2.cpp::PublishStatusFlags`:~1Hz or immediately when control/fault flags change. `timestamp_sample` is delayed EKF time; `timestamp` is publication time.
- `EKF2::PublishLocalPosition`: published on EKF updates; `heading_good_for_control` comes from actual final-yaw alignment, not our monitor.
- `src/modules/commander/Commander.cpp`: vehicle_status at2Hz or immediately when state/failsafe/arming changes.
- `src/systemcmds/topic_listener/listener_main.cpp::listener` and `topic_listener.hpp::listener_print_topic`: one-shot `orb_copy` prints the retained snapshot, not a new publication. Multi-instance topics include registered, unpublished slots; only instance0 is monitored here.
- `src/modules/mavlink/streams/ESTIMATOR_STATUS.hpp`: maps estimator solution flags and innovation ratios, not the precise `cs_mag_aligned_in_flight`/`heading_good_for_control` pair. `SYS_STATUS.hpp` has aggregate health but not an equivalent final magnetic alignment assertion. Increasing these stream rates would not manufacture the missing exact state.

Thus a250Hz gate cannot equate an unchanged1Hz status snapshot with a250Hz
heartbeat. The [MAVLink definitions](https://mavlink.io/en/messages/common.html#ESTIMATOR_STATUS)
also distinguish estimator status from transport presence. The pinned local
source remains the authority for this version.

## New monitoring semantics

Independent checks, all fail-closed, in `telemetry_freshness.py`:

| Time / state | Meaning | Handling |
|---|---|---|
| PX4 `timestamp_sample` | Sampling/delayed-filter epoch, where available |Preserved; finite, not future relative to publication, nondecreasing |
| PX4 `timestamp` | Actual uORB publication |Never replaced by query/receipt time; backwards/future or inconsistent duplicate payload latches failure |
| Query time |Bridge request, lockstep clock and host monotonic seconds |2Hz; finite response deadline150ms; overlap/absence/truncation rejected |
| Receive time |Actual receipt/parse observation |Recorded independently, including duplicate publications |
| Actuator time |Accepted exact lockstep HIL actuator timestamp + receive wall time |250Hz normal;250ms wall watchdog, strict protocol/order/peer/CRC/process checks |
| State values |Exact parsed flags, primary0 |Real loss of health/alignment/GNSS/localization/failsafe/arming/nav-state denies flight |
| MuJoCo time |Physics-driven clock with1s initial offset |Monotonic, no unexplained jumps; not a wall clock |

Flags/status source leases use their1s/.5s cadence +500ms poll interval
+150ms bounded response allowance: flags1650ms, vehicle_status1150ms.
Local position actually publishes at125Hz; its fixed650ms operational lease
is conservative relative to8+500+150=658ms. Extreme jitter can therefore cause
a conservative stop even with an otherwise timely response; the lease is
never extended in flight. This is **not the sole fix**: duplicate source-time
tracking, independent real transport liveness, query-response progress,
wall watchdogs, timestamp validation and fault-value evaluation are all
required. A healthy cached flag alone can never indefinitely authorize flight.

A query in progress immediately forbids **starting a new maneuver**. An
existing hold/maneuver can use its still-valid published snapshot only within
the finite response deadline and all independent leases. A complete unsafe
topic denies flight as soon as observed, even before the other topics finish.
An unfinished/missing response times out; no partial record is relabeled healthy.
Normal startup/ascent legitimately precedes final in-flight alignment: the
common operational health gate runs during flight from12s, while the original
final alignment gate additionally applies from32s and throughout the yaw test.

For250Hz checks and functioning polling/transport, sustained changed health
is observed within≤500+150+8=658ms from publication, including console service
every≤8ms while blocking for an actuator. Source stoppage is detected
within≤654/1154/1654ms from the last local/status/flags publication; synthetic
flags-only freeze detects at1652ms on the4ms grid. A truncated query detects
at152ms on that grid. Blocking reception also uses the **earliest** pending
query/actuator/query-service/receipt absolute wall deadline, not a renewed
250ms allowance on each receive. Late responses/actuators cannot erase a
previously expired lease. Transport loss is bounded by the250ms receive budget;
closed socket/protocol errors stop immediately when observed. These are
software deadlines, **not hard real-time guarantees**, and transient faults
that start and recover wholly between polls may not be captured.

## Offline regression and execution budget

Original69 tests retained. New cases cover normal1Hz status,2Hz asynchronous
query,250Hz checks,40ms source jitter, duplicates, source-only freeze with
live communication, old healthy cache during disconnect, ordering/rollback/
future timestamps, sample clock validity, partial/missing fields, fault flags,
final-alignment loss, GNSS faults, failsafe, localization invalidity, pending
query exclusion, and frozen simulation time with continuing wall time.
The original Phase2 ULog is replayed over its last5.196s: no fake health
fields, new monitor passes where the old1s gate falsely denies.

RED:23 monitor tests failed before implementation; opt-in runner and analysis
also had independently observed missing-feature failures. Initial GREEN:
bridge103passed0skipped, independently reproduced by the read-only reviewer.
Review found0Critical,2Important,1Minor; no areas were declined for judgment.
The Important findings were complete unsafe records hidden by a subsequent
fragmented title, and expired deadlines erased by late arrivals or exceeded
by a blocking receive. Six new regressions failed before the single offline
fix pass; final **109passed0skipped**, including the original69.

The duplicate-publication regression also fed the same actuator timestamp
twice, accidentally testing communication ordering instead of publication
expiry. This test-isolation finding was regraded Important for safety evidence
and corrected in the same pass; it now asserts source expiry at652ms. The
independent flags-only test still detects expiry at1652ms. Deferred minors:none.
No second review or extra flight was performed.

**Execution provenance:** the two authorized SITL runs used the monitor before
this review hardening. Their raw summaries/ULogs remain immutable. The final
parser/deadline changes passed offline tests but were **not re-flown**; current
source hashes identify final code, not a byte-exact attestation of the earlier
executed version. Normal captured-flight success means completed observation,
not proof of live testing every final safety branch.

Repository-wide pytest was attempted but blocked in **collection**, before
tests ran: `cad_check/test_inspect_stl_assembly.py` imports system SciPy built
for NumPy1.x against this venv's NumPy2.2.6 (`numpy.dtype size changed` /
`_ARRAY_API not found`). No dependency change/install was attempted. This is
not a claim that the historical CAD/RL suites pass; the full affected Bridge
suite is the verification scope.

Only two SITL invocations:one16s disarmed check, then one45s bounded flight.
No retry. Both exit0 and zero controls on shutdown. Atomic summaries and
ULog/raw identity hashes are retained; no raw data was overwritten.

## Negative yaw result: allocator configuration incompatibility

Confirmed source/config with corroborating ULog evidence:

1. `ActuatorAdapter.px4_geometry` leaves the existing physical ratio intact:
   `CT=1`, `|KM|=0.017207708`. Correct rotor/frame signs are unchanged.
2. `ActuatorEffectivenessRotors::computeEffectivenessMatrix` gives
   `moment = CT*p×axis - CT*KM*axis`, axisFRD−z. The raw yaw row therefore
   has magnitudes0.017207708.
3. Pinned `ControlAllocator.cpp` lines581–602 removes an axis if **every row
   entry has absolute value≤0.05**, before assigning the effectiveness matrix.
   The yaw row is consequently zeroed. This matrix is analytical source/config
   evidence, not a fictitious internal matrix sensor.
4. Actual `vehicle_rates_setpoint.yaw` is about0.488rad/s, requested yaw torque
   about0.097 in PX4 controller units. `control_allocator_status` reports this
   torque unallocated and `torque_setpoint_achieved=False`; motor outputs remain
   balanced for negligible physical yaw. Unused channels4–15 marked LOW are
   **not four active motor saturation**.

Hence observing a good EKF and delivery of a yaw command does not validate
the complete control chain. This is not fixed by relaxing health flags or
tuning yaw gains. Changing effectiveness scaling or PX4 source would exceed
this phase's fixed parameters, so the issue is explicitly **BLOCKED**, not
silently repaired and re-flown.

## Results, timestamps and figures

All angles NED; true attitude ENU/FLU transformed before yaw extraction.
True FRD-z rate is negative MuJoCo FLU-z rate. PX4 attitude uses sample-time
circular interpolation, setpoints causal publication-time hold, no endpoint
extrapolation. Physics500Hz, HIL250Hz, physical logging50Hz. Discrete state
and command plots use `step(...,where="post")`.

Poststep observation12.996s; maximum sample gap20ms. Fixed settling rule:
|target−true yaw|<2°, |FRD body-z rate|<.05rad/s continuously≥2s with gaps≤.15s.
No such window occurs. Overshoot0° is not evidence of successful tracking;
the aircraft never reaches the target. EKF prestep mean estimate-minus-truth
0.014647°, poststep mean−0.007304°, maximum absolute poststep0.029806°.

Numerical distributions, original source/receipt clocks, delays, actuator
and allocator statistics are in
`mujoco/reports/px4_mujoco_yaw_telemetry_fix.json`. The height and XY plot
clearly distinguishes physical body origin from PX4 relative position;
they are not identically referenced absolute position sensors.

## Reproduction (SITL only; do not rerun without a new authorized budget)

Use existing environments; append user dependencies per-process, no install:

```bash
cd /home/keyousuke/MineUAV-Learning-px4-sitl
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.yaw_telemetry_runner \
  --px4-root /home/keyousuke/PX4-Autopilot \
  --runtime mujoco/reports/px4_yaw_telemetry_fix_parts/disarmed_01 \
  --mode disarmed \
  --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.yaw_telemetry_runner \
  --px4-root /home/keyousuke/PX4-Autopilot \
  --runtime mujoco/reports/px4_yaw_telemetry_fix_parts/yaw_01 \
  --mode yaw \
  --smoke-proof mujoco/reports/px4_yaw_telemetry_fix_parts/disarmed_01/smoke_summary.json \
  --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

The existing directories are nonempty and intentionally refuse reuse.
The runner owns one pinned PX4 process; no separately launched flight
controller is needed. Original `yaw_runner` remains the archived Phase2
protocol; Phase2B uses the explicit new entrypoint. No serial connection,
force arm, precheck bypass, EKF flag rewriting, or secondary simulator.

Offline reproduction is safe and starts no PX4/MuJoCo instance:

```bash
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.analyze_yaw_telemetry --px4-root /home/keyousuke/PX4-Autopilot \
  --dependency-search-path /home/keyousuke/.local/lib/python3.10/site-packages
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -c 'import sys;sys.path.append("/usr/lib/python3/dist-packages");import pytest;raise SystemExit(pytest.main(["mujoco/px4_bridge","-q","--tb=short"]))'
```

## Resources and next recommendation

One worker/one PX4; no observed OOM. Bridge peak584245248bytes(557.18MiB),
PX4 sampled peak20361216bytes(19.42MiB); minimum available10733146112bytes
(9.996GiB). Runtime guard1.5GiB, startup2.5GiB; swap unchanged.
Historical models, physics, rewards and original87untracked files preserved.
Raw summaries, console streams, parameter files and ULog remain local-only.

Only recommendation:authorize a separate PX4 actuator-effectiveness scaling
compatibility fix, prove the physical torque mapping unchanged offline, then
authorize a new bounded yaw test. **Not executed; no Phase3 entry.**
