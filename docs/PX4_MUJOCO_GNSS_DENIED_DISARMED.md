# PX4–MuJoCo Phase 3D — GNSS-Denied Disarmed EKF2 Fusion Verification

Date: 2026-10-13. Branch: `feat/px4-mujoco-gnss-denied-disarmed`.
Base: `8468dfac1137fb7f79ded1718c24eaace09df835`.

## Result and scope

**PASS: one cold-start, unarmed, GNSS-denied local-estimation run.** Physical simulation ended at 19.999999999999794 s, after exactly 5000 sensor cycles / 10000 physics steps. HIL_SENSOR=5000, ODOMETRY=1000, HIL_GPS=0. No restart, ARM, Offboard command or airborne maneuver occurred. PX4 exited with code0 and all four physical motor controls were zero throughout and after shutdown.

This validates **SIMULATED GROUND TRUTH ODOMETRY** with synthetic covariance in a static/settling ground state. It does not validate FAST-LIO2, moving-state odometry, source-clock synchronization, GNSS-denied flight, or BC/PPO control. Ground `heading_good_for_control=False` and `cs_mag_aligned_in_flight=False` remain recorded; this report grants no flight authorization.

Machine-readable evidence: [report](../mujoco/reports/px4_mujoco_gnss_denied_disarmed.json). Raw ULog, console, source/send arrays, private PX4 startup tree and single-attempt manifest are local-only under `mujoco/reports/px4_gnss_denied_disarmed_parts/`.

## Pinned identities and source audit

MuJoCo3.14.0; unchanged7kg model, motor geometry, kf/km, inertia and actuator mapping. Isolated PX4 commit `142c24cffb0cdfd57ab0a9f2506d3203610ce737`, based on `8f9b91b2bdd3a65952601a084c3aa35e96fee59d`. Approved binary SHA256:

`2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004`

Frozen protocol SHA256: `4ae0b280ca9d98ead2cf9da7e9b0536b3068ed7cbeab041ac6807819e45f2296`. Runtime, physical-model, historical-checkpoint, source-definition and old raw-evidence SHA256 maps are in the report and local frozen protocol. Pre-run, post-run and current identities agree. No operational code changed after the sole run; subsequent edits concern read-only analysis/tests/documentation.

Source references below are relative to `/home/keyousuke/PX4-Autopilot-mineuav-yaw` at the pinned commit, not current online documentation:

| Source | Verified behavior |
| --- | --- |
| `src/modules/ekf2/params_external_vision.yaml:5` | EV_CTRL bits0 horizontal position,1 height,2 3D velocity,3 yaw. Value5 enables horizontal position+3D velocity only. |
| `src/modules/ekf2/params_gnss.yaml:5` | GPS_CTRL bits0 Lon/lat,1 altitude,2 3D velocity,3 dual-antenna heading. Value0 disables all. |
| `src/modules/ekf2/module.yaml:87` | HGT_REF0 barometric pressure;1 GPS;2 range;3 vision. |
| `src/modules/ekf2/params_barometer.yaml:5`, `params_magnetometer.yaml:5` | BARO_CTRL1 enabled; MAG_TYPE0 AUTO uses ground heading fusion before in-flight3D fusion. |
| `src/modules/ekf2/EKF2.cpp:270–355` | Aid-source advertisements depend on enabled sources. GPS_CTRL0 can leave all GNSS aid topics absent. Barometer and magnetometer aid topics are advertised separately. |
| `src/modules/ekf2/EKF/aid_sources/magnetometer/mag_control.cpp:180–207` | Ground magnetic heading fusion does not require final in-flight magnetic alignment; fault and alignment conditions still apply. |
| `src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp:339,389–470` | HIL_GPS receiver creates simulated GPS publications. New transport forbids this and all other outgoing packets except HIL_SENSOR/ODOMETRY. |
| Same file `:862–1065` | ODOMETRY publishes visual odometry; `timestamp_sample=hrt_absolute_time()` replaces source time. NED pose, BODY_FRD velocity, diagonal covariance retention. |
| `ROMFS/px4fmu_common/init.d-posix/rcS:247–261`, `px4-rc.simulator`, `px4-rc.mavlinksim` | Parameter environment overrides apply before simulator startup. Existing mineuav/10016/SIM_GZ_EN0 selects simulator_mavlink; no SIH or Gazebo source is launched. |
| `msg/EstimatorStatusFlags.msg` | GNSS height flag in this version is `cs_gps_hgt`; other flags are `cs_gnss_pos`, `cs_gnss_vel`, `cs_gnss_yaw`. |

Actual ULog parameters, not merely requested environment values:

| Parameter | Actual |
| --- | ---: |
| EKF2_EV_CTRL | 5 |
| EKF2_GPS_CTRL | 0 |
| EKF2_HGT_REF | 0 |
| EKF2_BARO_CTRL | 1 |
| EKF2_MAG_TYPE | 0 |
| SDLOG_PROFILE / SDLOG_MISSION | 0 / 0 |

## Isolated disarmed execution and safety

New modules live in `mujoco/px4_bridge/gnss_denied/`. Original `TelemetryFreshness`, `TelemetryYawProtocol`, `AllocationYawProtocol`, old flight proofs and root runtime files are unchanged. The new protocol permanently has `active=False`, rejects every flight proof/setpoint request, and returns `flight_authorized=False` / `allow_new_maneuver=False`. It requires fresh ground EV position/velocity, barometer and magnetic heading evidence, rejects unexpected GNSS/EV-yaw fusion, airborne/armed state, sensor faults, failsafe and invalid local estimates. It never rewrites PX4 health flags.

Original timing limits remain:150ms query response,250ms actuator watchdog,650ms query service; original local/status/vehicle source leases and irreversible fault latch remain active. Query events, scheduler counts, logical/wall deadlines, final source leases, final decision and instrumentation completeness are independently checked in the report. The new run kind cannot serve as a historical flight smoke proof.

Fresh empty runtime prevents parameter/cache reuse. A fixed campaign `protocol.attempt.json` is created using O_EXCL before startup; changing the supplied protocol filename does not grant another attempt. Sensor generation suppresses GPS even on formerly scheduled10Hz ticks; outgoing transport permits only HIL_SENSOR and ODOMETRY. The summary explicitly records GPS frequency0. ODOMETRY retains the existing50Hz COM adapter, NED/FRD conversions and covariance/quality validation.

Logger:27 explicitly configured instance0 topics, profile0/mission0. Actual ULog contains21 topic instances including the dedicated event stream. Optional GNSS/GPS/global-position topics are configured but never publish in this run. No subscription overflow, corruption, partial ULog record or dropout. The source PX4 startup tree and binary are unchanged.

## Direct receipt, fusion and absence evidence

| Acceptance | Evidence | Result |
| --- | --- | --- |
| A: cold start | Empty runtime, one attempt, no origin command, no prior GNSS priming | PASS |
| B: no GNSS input | HIL_GPS0 from first tick; sensor_gps/vehicle_gps_position0 records | PASS |
| C: GPS assistance disabled | Actual GPS_CTRL0 | PASS |
| D: direct visual receipt | 984 valid records,20ms maximum gap, final sample40ms before endpoint | PASS |
| E: EV horizontal position | 984 aid records,890 accepted, no innovation rejection; last3 fused/unrejected and advancing | PASS |
| F: EV3D velocity | 942 aid records,all accepted; last3 fused/unrejected and advancing | PASS |
| G/H: GNSS pos/vel/height/yaw | All4 flags false in every recorded sample; all4 aids absent; no GPS input | PASS |
| I: local position/velocity | xy/z/vxy/vz valid at endpoint; first recorded valid state1.1s | PASS |
| J: non-GNSS sources | Barometer357 records/337 accepted; magnetometer270/245 accepted; current final3 fusion timestamps advance | PASS |
| K: communication safety |28 completed queries; no watchdog, query or source-lease fault | PASS |
| L: unarmed/zero motors | Recorded arming state DISARMED; no Offboard; physical controls and actuator_outputs_sim all zero | PASS |
| M: strict duration |19.999999999999794s;5000/0/1000 messages | PASS |
| N: identities/log/exit/history | Frozen/post/current hashes agree; intact ULog; PX4exit0; history preserved | PASS |

EV position and velocity final fusion times are **20,756,000 → 20,780,000 → 20,796,000µs**, in PX4's logical clock with1,000,000µs initial epoch. They are not host wall times. Last publication times20,908,000 →20,932,000 →20,948,000µs; last fusion age204ms at21,000,000µs endpoint. Direct receipt final timestamp20,960,000µs.984 exact source/receiver logical-clock pairs have maximum position difference3.17e-9m, velocity6.22e-14m/s, quaternion component1.11e-8. Synthetic diagonal variances are0.01, quality100 and reset counter0.

Absence is corroborated independently: GPS input count0, effective GPS_CTRL0, all GNSS control flags false, no aid-source fusion publication, no sensor GPS publication, no global-position publication. Local `ref_timestamp`, `xy_global`, `z_global` remain zero in all975 local samples. No global origin was injected. The only recorded vehicle command ID is211; it is not ARM400 or an Offboard setpoint. Incoming COMMAND_LONG count1 is a PX4-to-Bridge message, not an outgoing ARM command.

Startup transients are retained: EV position94 nonfused records, barometer20 nonfused records; magnetometer25 rejected/nonfused observations occur at publication times0.244–2.084s. Final barometer and magnetometer fusion is current, accepted and un-rejected. Their mere early rejection does not establish a fault or causal failure mechanism. Ground final flags: tilt/yaw aligned, EV pos/vel and baro/mag_hdg true; mag_3d, EVyaw, in_air, mag_aligned_in_flight false. `heading_good_for_control=False`; flight readiness remains **NOT TESTED**.

Local estimate versus COM: raw position RMSE0.01139m, velocity RMSE0.001417m/s; initial local-origin offset `[0.000136,-0.015144,0.082516]m` is retained. Relative displacement RMSE0.04681m includes initial settling/alignment. These are descriptive static-run statistics, not moving-flight accuracy thresholds.

Receiver timestamp replacement remains unchanged. Exact logical sensor-epoch correspondence validates payload pairing but cannot establish physical host-to-PX4 latency or FAST-LIO2 source synchronization. Those items remain **NOT TESTED**.

## Timing, resources and validation

28 queries; response P50/P95/P99/max **1.965/5.019/5.136/5.154ms**. Maximum start interval500.065ms; maximum actuator wall gap10.790ms (logical4ms). Parser maximum0.756ms, peak1838chars, bounded recent archive65536chars, pending queries at most1, dropped instrumentation `{}`. Scheduler maximum lateness0.080ms. Simulation20.000s versus host20.00391s; host time is not the physical20s cap.

Single Bridge worker, one PX4 and one MuJoCo. Bridge peakRSS771,309,568bytes (735.58MiB); PX4 sampled peak9,838,592bytes (9.38MiB); sampled minimum MemAvailable9,348,911,104bytes (8.71GiB). Bridge CPU2.197s, PX4 sampledCPU1.27s. No observed OOM or resource safety stop; swap unchanged. Resource values are sampled/process metrics, not hard upper bounds for all system allocations.

Pre-run:437 inherited+73 new=510 Bridge tests PASS;30 Shadow tests PASS. Two inherited invalid-angle warnings and three Shadow package warnings disclosed; unrelated full-repository CAD/ABI tests not run. Highest observed Bridge-suite RSS across preflight checks1,264,140KiB. RED→GREEN covered missing GPS count, campaign-lock filename bypass, summary GPS identity, missing/late/truncated safety evidence and fusion evidence. Independent pre-run review gave GO after two Important corrections (campaign lock and incomplete safety-result predicate) and one metadata correction. All fixes preceded runtime freeze/execution.

Final verification: **514 Bridge tests (437 inherited+77 new) and30 Shadow tests PASS**. Final independent read-only review **GO**,0 Critical/0 Important remaining; reviewer independently passed77 new-package tests and recomputed all A–N results from raw evidence. Reviewer did not independently rerun the full514+30 suites or certify the original87-file inventory. Final Bridge-suite RSS1,115,608KiB. The report's verification section records this workflow metadata separately from recomputable measured results.

Only analysis/tests/docs changed after the operational freeze. The original worktree's87 historical untracked files remain present. All learned checkpoints, prior reports/raw traces and both PX4 builds remain intact.

## Reproduction and artifacts

The sole executed command (do not rerun without new authorization; campaign guard is consumed):

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco \
/home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.gnss_denied.runner \
  --px4-root /home/keyousuke/PX4-Autopilot-mineuav-yaw \
  --compatibility-manifest mujoco/px4_bridge/compat/px4_yaw_build_identity.json \
  --protocol mujoco/reports/px4_gnss_denied_disarmed_parts/protocol.json \
  --runtime mujoco/reports/px4_gnss_denied_disarmed_parts/disarmed_01 \
  --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

Read-only recomputation:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco \
MINEUAV_PYULOG_SEARCH_PATH=/home/keyousuke/.local/lib/python3.10/site-packages \
/home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.gnss_denied.analyze \
  --runtime mujoco/reports/px4_gnss_denied_disarmed_parts/disarmed_01 \
  --report mujoco/reports/px4_mujoco_gnss_denied_disarmed.json \
  --figures mujoco/reports/figures/px4_gnss_denied_disarmed
```

Figures under that directory: EV fusion times, GNSS flags, local position/velocity validity, height/heading sources, query timings and strict stop boundary. Discrete states use step plots. ULog SHA256 `34363cf0d5ddfdbc880e47b096497971ef8db577f76148092886346d47c7f485`; immutable summary SHA256 `1fa07e435595251d085f222b325fdb9e9ec91d86cc8b9755253c8f49f99d9934`.

## Next

Only recommend reviewing a separately authorized GNSS-denied flight-safety and estimator-readiness protocol before any flight. This phase stops here; it does not authorize execution of that next protocol.
