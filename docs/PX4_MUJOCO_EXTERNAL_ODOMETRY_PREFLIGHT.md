# PX4–MuJoCo Phase 3A — External Odometry Interface Preflight

Status: **offline interface PASS; PX4 receipt, EKF2 fusion, GNSS-denied and flight NOT TESTED**. This phase did not start SITL, arm, change any PX4 parameter, or connect an odometry producer to the live Bridge. The new `external_odometry.py` is intentionally not imported by `simulation_runner.py` or `allocation_runner.py`.

## Pinned identity and evidence

- Research base `fb3c82ea3a7fdb8b7becf1b0226a9b0b3a3462b0`; branch `feat/px4-mujoco-external-odom-preflight`.
- Isolated PX4 source `/home/keyousuke/PX4-Autopilot-mineuav-yaw` HEAD `142c24cffb0cdfd57ab0a9f2506d3203610ce737` (v1.18.0-beta1-580-g142c24cffb). Its existing SITL binary SHA256 is `2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004`. Neither was changed or run here. Build config `build/px4_sitl_mineuav_yaw/px4_boardconfig.h:57` enables `CONFIG_EKF2_EXTERNAL_VISION`.
- Existing Phase 2F ULog `mujoco/reports/px4_yaw_closed_loop_parts/yaw_01/log/2026-10-12/07_36_24.ulg` SHA256 `a3bf79d84d4548eed3714f7f7ac45f1f73992aafad6bdafb6cd8dfed408fb754`. Its recorded parameters: `EKF2_EV_CTRL=0`, `EKF2_HGT_REF=1` (GPS), `EKF2_GPS_CTRL=7`, `EKF2_BARO_CTRL=1`, `EKF2_EV_QMIN=0`, `EKF2_EV_DELAY=0`, `EKF2_EV_NOISE_MD=0`. These are historical run values, not a fresh SITL measurement. EV fusion was disabled there; ODOMETRY transmission alone cannot prove fusion.
- Existing MuJoCo model `mujoco/models/mine_uav_dynamics_v2.xml` SHA256 `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`.

## Actual pinned PX4 data path

The Bridge's loopback TCP simulator connection (`mujoco/px4_bridge/transport.py`) enters `src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp:327,862–1064`. `handle_message_odometry` converts MAVLink `ODOMETRY` to `vehicle_odometry_s`; estimator types `UNKNOWN`, `NAIVE`, `VISION` and `VIO` publish `vehicle_visual_odometry` (unless `_vio_blocked`), MOCAP goes to another topic, and LIDAR/GPS types are not accepted on this path. EKF2 subscribes to `vehicle_visual_odometry` at `src/modules/ekf2/EKF2.hpp:341`. `EKF2.cpp:2399–2537` checks frames, finite position/velocity, unit quaternion, derives variances, and calls `setExtVisionData`. `EKF/aid_sources/external_vision/ev_control.cpp:42–112` gates quality, inter-sample interval and recency before EV position/velocity/yaw/height controllers. `EKF2.cpp:1139–1149,2087–2129` publishes EV/GNSS aid statuses and control flags.

**Important route difference:** `SimulatorMavlink.cpp:870` sets `odom.timestamp_sample = hrt_absolute_time()` and comments out synchronization of `odom_in.time_usec`. In contrast, the separate ordinary MAVLink receiver at `src/modules/mavlink/mavlink_receiver.cpp:1525–1533` uses `_mavlink_timesync.sync_stamp(odom_in.time_usec)`. The planned TCP simulator link therefore discards source sample time for EKF2; our packet still carries capture time for auditing, and the offline adapter rejects stale/duplicate source samples, but this does **not** make delayed measurements correctly time-aligned inside PX4. A later latency-aware or real FAST-LIO2 path needs an explicitly verified timestamp route, not a silent delay setting.

## Frames and MAVLink fields

`frame_conversions.py` defines MuJoCo world ENU to PX4 NED as `(E,N,U)→(N,E,−U)` and body FLU to FRD as `(F,L,U)→(F,−L,−U)`. MuJoCo quaternions are wxyz; `attitude_to_px4` applies both proper rotations. The offline adapter emits **`frame_id=MAV_FRAME_LOCAL_NED(1)`, `child_frame_id=MAV_FRAME_BODY_FRD(12)`**, rather than using PX4's partially converted LOCAL_ENU route. The source quaternion is checked unit/finite, then converted to NED←FRD. World ENU velocity is first rotated into the instantaneous body FLU frame, then converted to body FRD. Body angular rate is FLU→FRD. This matches `msg/versioned/VehicleOdometry.msg`, in which angular velocity is always body FRD. Tests include a 90° yaw, where an ENU +X velocity becomes FRD +Y.

The later live truth producer should sample the **body COM**, not an arbitrary `qpos` origin: reuse the existing `SensorAdapter.sample` convention (`data.xipos[body]`, `mj_jacBodyCom @ qvel`, `data.xquat[body]`, body angular rate from rotational Jacobian) at a named post-integration physics instant. The Bridge's synthetic HIL_GPS already uses the same COM-based NED position and velocity. This phase has **not** implemented a live truth producer or proven its local origin exactly coincides with the EKF origin after initialization; offset and frame alignment must be measured before fusion is interpreted.

| MAVLink ODOMETRY | Offline adapter | Pinned SimulatorMavlink / EKF2 use |
| --- | --- | --- |
| `time_usec` | monotonically increasing capture time from the existing simulation clock | ignored for `timestamp_sample` on TCP simulator route; PX4 stamps receipt time |
| `frame_id`, `x/y/z` | LOCAL_NED, ENU→NED | pose frame NED, copies position |
| `q` | wxyz NED←FRD, unit/finite | copied; EKF2 validates norm within `1e-5` |
| `child_frame_id`, `vx/vy/vz` | BODY_FRD, `R_enu_fluᵀ v_enu` then FLU→FRD | body-FRD velocity frame; EKF2 accepts it |
| `rollspeed/pitchspeed/yawspeed` | body-FRD angular rate | copied to uORB; not itself an EV fusion axis in the inspected controller |
| `pose_covariance[21]` | transformed full 6×6 `[ENU position; FLU attitude error]`, row-major upper triangle | only position diagonals `[0,6,11]` and attitude diagonals `[15,18,20]` copied; off-diagonals ignored |
| `velocity_covariance[21]` | transformed full 6×6 `[ENU world velocity; FLU body angular rate]` | only linear-velocity diagonals `[0,6,11]` copied; angular/cross covariance ignored |
| `reset_counter` | uint8, explicit reset event increments modulo 256 | copied; EKF compares against previous sample |
| `quality` | explicit 1–100 synthetic source metadata; zero rejected by adapter | copied; `EKF2_EV_QMIN=0` disables quality gate; raising QMIN needs separately validated semantics |
| `estimator_type` | `UNKNOWN(0)` to avoid falsely claiming VIO/SLAM | accepted and published as visual odometry, covariance parsed (unlike NAIVE) |

The covariance inputs are **declared simulation uncertainties**, not measured sensor performance. Every matrix must be finite, symmetric within the current absolute `1e-10` tolerance, positive diagonal and positive semidefinite (eigenvalue tolerance scaled to its magnitude); diagonals must survive conversion as normal positive `float32` values. Coordinate transforms cover all 6 axes and cross terms, although this PX4 receiver uses only the listed diagonals. No covariance or quality parameter was calibrated in this phase. Invalid position, quaternion, velocity, covariance or quality prevents packet construction. Independent review notes a nonblocking limitation: an extremely small covariance could be materially asymmetric relative to its own scale while still passing this absolute symmetry tolerance; the pinned receiver ignores off-diagonal entries, but future sensor covariances need a scale-aware tolerance before use.

## Parameters in this PX4 revision (not changed)

| Parameter | Pinned source | Actual meaning / current Phase 2F value |
| --- | --- | --- |
| `EKF2_EV_CTRL` | `src/modules/ekf2/params_external_vision.yaml:5–18` | bit0 horizontal position, bit1 vertical position, bit2 3D velocity, bit3 yaw; current **0** |
| `EKF2_EV_QMIN` | same file:40–52 | optional minimum quality; current **0** (gate disabled) |
| `EKF2_EV_DELAY` | same file:19–28 | EV relative IMU delay in ms, reboot-required; current **0**; cannot compensate for simulator receive-time substitution without verification |
| `EKF2_EV_NOISE_MD` | same file:29–39 | 0 message variance floored by noise params, 1 parameters only; current **0** |
| `EKF2_HGT_REF` | `src/modules/ekf2/module.yaml:87–108` | 0 baro, 1 GPS, 2 range, 3 vision; current **1**. GNSS can still initialize height-sensor biases when GPS reference/control remain enabled. |
| `EKF2_GPS_CTRL` | `src/modules/ekf2/params_gnss.yaml:5–19` | bit0 lon/lat, bit1 altitude, bit2 3D velocity, bit3 dual-antenna heading; current **7** |

The EKF EV controller requires adjacent valid samples **strictly less than** `EV_MAX_INTERVAL=200 ms` (`EKF/common.h:71`) and a recent source. The adapter proposes **50 Hz** captures (20 ms), source age ≤100 ms at Bridge emission, gap **<200 ms** (200 ms itself fails), monotonic simulation time and source timestamps, with a latched outage. It does not retransmit a cached healthy pose. These Bridge-side limits are a preflight design, not proof of PX4 end-to-end sample-time correctness. `SimClock` starts at 1,000,000 µs and advances 2,000 µs per 500 Hz physics step; PX4 boot clock mapping remains unverified.

## Staged GNSS-denied verification (future work, NOT TESTED)

1. **GNSS retained:** in a separately authorized disarmed SITL test, first verify MAVLink wire receipt, `vehicle_visual_odometry` publication with correct frames/values and valid `timestamp_sample`, then explicitly configure selected EV fusion axes (e.g. horizontal position and 3D velocity) and inspect EKF behavior. Merely seeing ODOMETRY is not fusion. Do not infer fusion from `cs_ev_*` alone: also require advancing `estimator_aid_src_ev_pos/vel` `time_last_fuse`, valid/recent `fused` samples, nonrejected innovations, and valid local position/velocity. Height/yaw are separate decisions; do not silently enable them.
2. **Only after Stage 1 passes:** separately authorize simulated GNSS unavailability. Verify HIL_GPS is actually absent and `EKF2_GPS_CTRL=0`; choose a justified height reference and assess yaw source. Require `cs_gnss_pos`, `cs_gnss_vel`, `cs_gps_hgt`, `cs_gnss_yaw` false and GNSS position/velocity/height/yaw aid-source `time_last_fuse` no longer advancing, while EV aid timestamps continue to advance and PX4 reports valid position/velocity. Check GPS bias/reference behavior and all preflight/failsafe responses. Existing `telemetry_freshness.py:39,83` and `yaw_validation.py:64` **require** `cs_gnss_pos=True`; they must be separately redesigned and fail-closed tested before any GNSS-denied maneuver. Do not bypass them in this phase.
3. **FAST-LIO2 interface later:** replace the simulated truth producer with an independently characterized MID360S+IMU→FAST-LIO2 odometry source. Establish extrinsic calibration, timestamp/time-sync route, local map/reset semantics, covariance and quality calibration, latency/dropout/jump handling, and frame alignment. This adapter does not claim SLAM integration or mine-wind/real-vehicle validation.

The `fusion_evidence` offline helper checks only a narrow pair of successive EV position/velocity fuse timestamps plus selected control flags, and optionally two GNSS-status snapshots and unchanged GNSS aid timestamps. It is **not** wired to a flight gate and is insufficient alone: actual uORB identity, freshness, valid estimator output, parameter identity, source health and no-GNSS input must be checked in a future authorized run.

## Offline validation and risks

`test_external_odometry.py` covers ENU/NED and FLU/FRD pose/twist/covariance, byte-level MAVLink roundtrip, stale/duplicate/future/outage samples, exact 200 ms EKF boundary, monotonic inspection time, invalid quaternion/NaN/Inf/covariance/quality/float32 overflow/underflow, reset-counter wrap, and EV/GNSS source indicators. The full relevant Bridge suite passed **356 tests (336 inherited + 20 added)**. Two inherited `yaw_validation.py` RuntimeWarnings occur in tests intentionally exercising invalid angle data; no failures. No new SITL episode was run.

The final single-worker test run took 9.94 s and peaked at 1,033,652 KiB RSS (`/usr/bin/time`); no owned command OOM was observed. The original checkout still has its 87 historical untracked files, untouched. This is a scoped Bridge result, not a whole-repository CAD/ABI test claim.

Remaining critical preflight risks: (1) simulator-route receive-time stamping prevents authentic delayed-source fusion assessment; (2) current EV_CTRL=0 and GPS reference/control enabled; (3) current safety monitor requires GNSS; (4) covariance, quality, sensor extrinsics and noise/delay/packet-loss assumptions are unvalidated; (5) no PX4/uORB receipt or fusion measurement this phase. These are **BLOCKED** for GNSS-denied operation, not hidden by offline PASS. No historical model, physics or PX4 binary changed.

Independent read-only review checked the pinned PX4 receiver/EKF/uORB path, frames, covariance diagonals, reset/quality/time semantics and three source identities. It independently reran the 356-test Bridge suite: **0 Critical, 0 Important, 1 nonblocking Minor** (small-scale absolute covariance symmetry tolerance, documented above). No files were edited and no SITL was started by the reviewer.
