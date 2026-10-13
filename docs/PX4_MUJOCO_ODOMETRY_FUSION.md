# PX4–MuJoCo Phase 3C: GNSS-retained external odometry fusion

Base: `f202d9d2985312e80cf180a2f239077e3734aa81` on `feat/px4-mujoco-policy-shadow-preflight`.
Branch: `feat/px4-mujoco-odom-fusion-verification`.

## Scope and identity

This experiment permits one unarmed, at-most-20-second SITL communication/fusion test. Its source is **SIMULATED GROUND TRUTH ODOMETRY**. FAST-LIO2, SLAM, GNSS-denied localization and learned-policy flight are outside the validation.

The isolated PX4 source is `142c24cffb0cdfd57ab0a9f2506d3203610ce737`, based on `8f9b91b2bdd3a65952601a084c3aa35e96fee59d`. The approved binary SHA256 is `2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004`. MuJoCo is 3.14.0; the unchanged 7 kg model SHA256 is `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`.

## Actual PX4 path and parameters

Pinned source evidence, relative to `/home/keyousuke/PX4-Autopilot-mineuav-yaw`:

- `src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp:862`: `handle_message_odometry` accepts the simulator TCP MAVLink ODOMETRY packet. UNKNOWN estimator type routes to `vehicle_visual_odometry`.
- `src/modules/ekf2/EKF2.hpp:341`: EKF2 subscribes to `vehicle_visual_odometry`.
- `src/modules/ekf2/EKF2.cpp:2399`: `UpdateExtVisionSample` accepts NED position and BODY_FRD velocity, checks finite values, and applies configured noise lower bounds.
- `src/modules/ekf2/EKF2.cpp:298` and `:1139`: configured EV sources are advertised and their actual aid-source statuses are published.
- `src/modules/ekf2/params_external_vision.yaml:5`: EV control bits 0/1/2/3 are horizontal position/vertical position/3D velocity/yaw.
- `src/modules/ekf2/params_gnss.yaml:5`: GPS bits 0/1/2 are horizontal position/height/3D velocity.
- `src/modules/ekf2/module.yaml:87`: height reference 1 is GPS.
- `build/px4_sitl_mineuav_yaw/etc/init.d-posix/rcS:247`: `PX4_PARAM_*` environment values are applied with `param set` before module startup.
- `src/modules/logger/logged_topics.cpp:179`: estimator topics, including aid sources, are logged; the estimator-replay/vision profiles include `vehicle_visual_odometry`.

The isolated runtime requests `EKF2_EV_CTRL=5`, `EKF2_GPS_CTRL=7`, `EKF2_HGT_REF=1`, `SDLOG_PROFILE=131` (default + estimator replay + vision logging; the historical Phase2F value was already 131). ULog parameter values must agree before a fusion PASS can be reported. EV yaw is not enabled.

## Production, frames and clocks

`odometry_producer.py` consumes the existing `SensorAdapter`'s real 250 Hz physics samples. `SensorAdapter.sample` executes `mj_forward`, reads body `xipos` (COM), and derives COM velocity with `mj_jacBodyCom @ qvel`. This accounts for the model's nonzero inertial offset; freejoint position is not substituted for COM position.

Every fifth sensor tick emits one ODOMETRY packet: 50 Hz at exact 20,000 µs source intervals. Phase3A's `OdometryAdapter` and frame transforms are reused. Pose is LOCAL_NED (1), twist is BODY_FRD (12), quaternion is wxyz body-FRD to world-NED, estimator type is UNKNOWN (0), reset counter is 0, quality is the declared synthetic value 100. All six pose and twist covariance diagonal entries are 0.01 in the respective squared SI units. These are synthetic assumptions, not sensor characterization. The tiny-covariance symmetry check now uses relative tolerance and rejects large relative asymmetry at small magnitudes.

MuJoCo source capture time is `1,000,000 + physical_time_s * 1e6`. MAVLink `time_usec` carries that capture time. The source sidecar records successful Bridge send times from the host monotonic clock. **SimulatorMavlink overwrites `timestamp_sample` with PX4 receive time (`hrt_absolute_time`)** and then sets uORB publication time. Source-time synchronization and FAST-LIO2 delay compensation remain NOT TESTED. Nearest-prior source/receipt comparison is only a low-latency diagnostic; it never uses a future truth sample.

## Runtime isolation and safety

The historical `simulation_runner.py` remains byte-for-byte unchanged so historical proof replay retains its provenance. A single-threaded scoped adapter adds ODOMETRY to the original HIL_SENSOR/HIL_GPS list, applies the explicit runtime parameters, and observes successful ODOMETRY sends. The original bounded send path, incremental console parser, allocator-row checks, 150 ms query response, 250 ms actuator watchdog, 650 ms query service deadline and source leases remain active. Scoped bindings restore on every exit.

At the first atomic write, this run's raw summary is assigned `kind=disarmed_odometry_fusion_verification` and `not_flight_proof=True`. Old flight proof gates reject that kind. The regression test supplies every other old-proof prerequisite and checks rejection before `Popen`; this result cannot serve as an arming proof.

Runtime files and eight historical checkpoints are hashed before/after the run. PX4 source/build identity is verified before startup and after cleanup. Raw summary, source manifest, console and ULog remain local-only and are identified by SHA256 in the report.

## Evidence rules

Transport, uORB receipt and EKF fusion are separate criteria. Receipt requires finite, monotonic visual odometry with no gap over 250 ms and coverage to within 250 ms of the run end. EV position/velocity require the final three aid-source samples to be fused, unrejected and have strictly advancing `time_last_fuse`, with the last fuse/publication no more than one second old. Fusion flags and valid local position/velocity must also be current. Missing aid evidence is BLOCKED; recorded failure to fuse is FAIL. ULog publications after the tested physical endpoint are excluded from acceptance.

## Reproduction command

Execute only under a fresh authorization for a new single run; this command does not retry:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco:/usr/lib/python3/dist-packages \
 /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.odometry_fusion_runner \
 --px4-root /home/keyousuke/PX4-Autopilot-mineuav-yaw \
 --runtime mujoco/reports/px4_odometry_fusion_parts/disarmed_01 \
 --compatibility-manifest mujoco/px4_bridge/compat/px4_yaw_build_identity.json \
 --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

Analysis is offline and repeatable:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco:/usr/lib/python3/dist-packages \
 MINEUAV_PYULOG_SEARCH_PATH=/home/keyousuke/.local/lib/python3.10/site-packages \
 /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.analyze_odometry_fusion \
 --runtime mujoco/reports/px4_odometry_fusion_parts/disarmed_01 \
 --report mujoco/reports/px4_mujoco_odometry_fusion.json \
 --figures mujoco/reports/figures/px4_odometry_fusion
```

## Results

**EV/GNSS aid-source fusion evidence: PASS. Complete interface acceptance: BLOCKED. The 20-second duration constraint: FAIL.** The one run completed without a safety-monitor exception, but its physical duration was 20.004 s. There was no retry.

| Criterion | Result | Actual evidence |
|---|---|---|
| A: 50 Hz COM truth producer | PASS | 1,001 finite generated samples; every source interval exactly 20,000 µs |
| B: MAVLink send | PASS | 1,001 successful ODOMETRY sends; 5,001 HIL_SENSOR and 201 HIL_GPS |
| C: direct `vehicle_visual_odometry` publication | BLOCKED | Topic absent from ULog; logger explicitly failed to subscribe |
| D: EKF EV horizontal position fusion | PASS | 985 aid records, 891 accepted; final three fused/unrejected with advancing fuse times |
| E: EKF EV 3D velocity fusion | PASS | 944 aid records, all accepted; final three fused/unrejected with advancing fuse times |
| F: GNSS retained and fused | PASS | Position/velocity aid sources each have 198 records; final flags and fuse times remain active |
| G: unarmed operation | PASS | 41 recorded `vehicle_status` samples all DISARMED=1; no flight command sent |
| H: safety queries/communication | PASS | 29 completed queries, no monitor fault, no backlog beyond one outstanding query |
| I/J: zero motors, cleanup and history | PASS | Final four controls zero, PX4 exit 0; physics, runtime-during-run and eight checkpoints unchanged |
| K: at most 20 s | FAIL | 20.004 s; one extra 4 ms sensor/physics cycle |
| L: yaw allocation row | PASS | All 29 query completions carry the valid allocator row |

Actual ULog parameters are EV=5, GPS=7, height reference=1, logging profile=131. EV yaw fusion remains disabled. Both EV aid sources end with `time_last_fuse` **20,796,000 → 20,820,000 → 20,836,000 µs**, corresponding to 19.796 → 19.820 → 19.836 s after the simulator's 1,000,000 µs epoch. Their final innovation-test ratios remain below 1. GNSS last-fuse time is 20,788,000 µs. The final EV/GNSS position and velocity flags are all true, and local position/velocity validity remains true. The94 early EV-position records from0.308–2.252s had `fused=false`; none had `innovation_rejected=true`. Those records remain counted.

### Direct receipt evidence gap

The pinned logger has `MAX_TOPICS_NUM=255`. Its console recorded `Too many subscriptions, failed to add: vehicle_visual_odometry 0` twice; 369 subscription-add failures were reported in total. The ULog contains no visual-odometry topic and has no ULog dropouts. Source code and actual EV aid-source observations support that the receiver/EKF path processed data, but they cannot supply the missing direct uORB publication/quaternion/covariance/timestamp trace. No direct receive-time or visual-position comparison figure is fabricated. This prevents complete three-layer interface acceptance.

The static local estimate was compared separately with nearest-prior COM truth: raw coordinate-component position RMSE 0.04910 m, relative displacement RMSE 0.00623 m, NED velocity RMSE 0.001375 m/s. The initial EKF-minus-truth origin offset is approximately `[0.00014, -0.01514, 0.08252]` m. These metrics preserve the local-origin difference and do not validate moving-state accuracy or source-time synchronization.

### Stop boundary deviation and final version

The unchanged historical loop tests accumulated floating-point `d.time < 20`. After 5,000 cycles its value was `19.99999999999979`, so it executed cycle 5,001. The raw summary/source/console/ULog remain untouched. Only evidence within the original 20 s window is used for fusion acceptance; the extra source sample at 21,000,000 µs is separately counted.

The new Phase3C runner now passes a one-microsecond comparison guard (`19.999999` s) to that unchanged loop. A deterministic offline regression reproduces the old 5,001-cycle outcome and verifies that the guard stops after exactly 5,000 four-millisecond cycles, or 20,000,000 µs of physical progression. **This runtime correction was made after the sole SITL test and is NOT TESTED in SITL.** The report includes separate executed/final runtime hashes. No extra run was performed to validate it.

### Safety, performance and verification

Query response P50/P95/P99/max: **1.947 / 4.780 / 4.842 / 4.865 ms**, below 150 ms. Maximum query-start interval was 500.096 ms, below the 650 ms service deadline. Maximum actuator wall interval was 10.518 ms, below 250 ms; simulation actuator intervals stayed 4 ms. Parser peak buffer was 1,836 characters; maximum parsing/validation cost per chunk was 0.798 ms. EKF, source leases and communication monitors reported no fault. Sender packet loss cannot be counted from the missing direct receiver topic; aid-source/log sampling differences are not treated as packet loss.

One PX4 process and one MuJoCo instance were used. Bridge peak RSS was 771,653,632 bytes (735.91 MiB); sampled PX4 peak RSS was 9,863,168 bytes (9.41 MiB). Minimum system available memory was 10,549,301,248 bytes (9.83 GiB). Bridge/PX4 CPU time was 2.239/1.230 s over 20.006 s of measured runner wall time. No OOM was observed and swap was unchanged.

Final relevant Bridge tests: **371 passed** (356 inherited + 15 new), two inherited invalid-angle warnings; suite peak RSS 1,080,576 KiB. Phase3B's 30 Shadow tests also passed in preflight, with its three inherited package-import warnings. No whole-repository CAD/ABI repair or full-suite claim is made. The independent reviewer required and rechecked fixes for proof-kind separation, stale/future evidence, and complete unarmed-state accounting before giving GO; a separate read-only result review confirmed the fusion evidence and both limitations above.

Raw identity: summary SHA256 `35877f9117c836f13563530659eb9964d2c9ed33fea987f401c6ce71d5ffb84d`; ULog `40f159beaec05c69a7ef0a4b2839a05f02f9278a5554d5be0dee81e7b9b0930a`; console `136efc4a80b63f4bde0541784517789766707ac8bf64a094c4a12caf0b003ea4`. Source-manifest identity and all protected checkpoint hashes are in the JSON report. Raw files remain under `mujoco/reports/px4_odometry_fusion_parts/disarmed_01/`, local-only.

## Remaining gates and one next recommendation

GNSS-denied entry is BLOCKED: direct receiver evidence and the final runtime version require another separately authorized verification, and the existing flight safety gate still requires GNSS position. FAST-LIO2 capture-time synchronization, delay compensation, reset/jump behavior and actual sensor uncertainty remain NOT TESTED. Learned-policy PX4 flight additionally needs estimator-fed observation validation, a bounded Offboard command/failsafe path and an independently authorized control-response test; Shadow inference supplies none of those flight outcomes.

The single next recommendation is a newly authorized unarmed verification using an explicit bounded ULog topic list and the offline-fixed 20 s stop boundary, while retaining GNSS. No GNSS removal, arming, policy flight or additional SITL was started.
