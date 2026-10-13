# PX4–MuJoCo Phase 3C-R: external odometry evidence reverification

Base: `f09264edaffb25c4e59e33cb8be4c01326fb2c33`.
Branch: `feat/px4-mujoco-odom-fusion-reverification`.

## Result and scope

**PASS: all twelve necessary A–L criteria have direct runtime evidence.** Exactly one new disarmed SITL run was performed; no retry, ARM, Offboard command, GNSS removal or learning-policy control. This validates a static/settling **SIMULATED GROUND TRUTH ODOMETRY** interface with GNSS retained, not FAST-LIO2, GNSS-denied navigation or flight.

The old Phase3C run, report and logs remain unchanged. Its missing visual-odometry evidence and 20.004-second overshoot are not retroactively reclassified.

## Version and identity

- MuJoCo: 3.14.0; unchanged 7 kg physical model.
- Isolated PX4 source: `142c24cffb0cdfd57ab0a9f2506d3203610ce737`, based on `8f9b91b2bdd3a65952601a084c3aa35e96fee59d`.
- PX4 binary SHA256: `2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004`.
- Model XML SHA256: `cde862f424a86bda3676249859268c69dede9dde2ebadeef819813351684b520`.
- Frozen protocol SHA256: `0164d294df0d15d043c749b62172af49f5906b1d041bea2525d2b5d9825bb47b`.
- Raw summary SHA256: `b03fd4e09ed79794f91e7d9923c4e714690139fb6a6911791a697a4ae134aeae`.
- Raw ULog SHA256: `50b27ccb530737ee4e9f7de7c09289853b226a9a19f81a73a4683a1de8c3df3b`.
- Console SHA256: `33103ea0baaff23b307a658d0c6b7130106eb7ef02691623d18fa51f8f703966`.

The JSON report contains all operational Python hashes, before/after/current model and PX4 identities, protected eight-checkpoint hashes, and historical raw-evidence hashes. Operational code was frozen before this run and remains identical afterward. Subsequent work changed only offline analysis, tests, figures and documentation; it did not change the executed runtime. This is not an old Phase2E arming smoke proof: the summary is created with `kind=disarmed_odometry_fusion_reverification`, `not_flight_proof=true` and the protocol hash.

## Logger root cause and configuration

Pinned PX4 source references, relative to `/home/keyousuke/PX4-Autopilot-mineuav-yaw`:

- `src/modules/logger/logged_topics.h:79`: `MAX_TOPICS_NUM=255`.
- `logged_topics.cpp:565–632`: profiles are added first, then the custom file. A custom file alone does not replace profile subscriptions; profile0 without a successful file falls back to defaults.
- `logged_topics.cpp:414–470`: file interface is `<topic> <interval_ms> <instance>`. Explicit, compiled instance0 entries take `add_topic`; otherwise implicit multi-instance expansion can occur.
- `src/modules/logger/logger.cpp:506–540`: mission subscriptions are initialized separately; this run sets mission0.
- `platforms/posix/src/px4/common/main.cpp:429–475`: an existing private runtime `etc` directory is preserved instead of replaced by the approved rootfs symlink.
- `build/px4_sitl_mineuav_yaw/px4_boardconfig.h`: storage root is `.`; the file is `runtime/etc/logging/logger_topics.txt`.
- `src/modules/logger/logger.cpp:401,994`: a dedicated event subscription is recorded separately from the configurable main topic array.

This run used actual `SDLOG_PROFILE=0`, `SDLOG_MISSION=0` and **19 explicit instance0 main topics**, independently checked against compiled binary uORB symbols. The actual ULog has **20 topic instances**, including the logger's dedicated `event`; all are instance0. No subscription failure occurred, no capacity limit was changed and no PX4 binary was rebuilt. Parameters remain in the ULog header independently of a parameter-update subscription.

Private `etc` is a content-verified copy of the approved startup tree, plus only the logger file. Source rootfs and private startup-file hashes are rechecked; no write follows a rootfs symlink into the approved installation. Logger configuration changes recording only, not the controller/EKF algorithms or force model.

The complete configured and actual lists are in the report. Required entries are `vehicle_visual_odometry`, `estimator_status_flags`, EV/GNSS position and velocity aid sources, `vehicle_local_position`, `vehicle_status`, and `vehicle_gps_position`. Additional topics retain actuator, command/mode, failsafe, attitude, IMU, logger and CPU evidence.

## Exact stop boundary and protocol

The historical `simulation_runner.py` remains unchanged. Its accumulated floating-point time would execute a 5001st cycle with literal20. The already-committed Phase3C guard passes `20 - 1e-6` to that loop. The guard is much smaller than 4 ms: the complete 5000th cycle executes, but the 5001st does not. No raw data were truncated to hide overshoot.

The offline test runs real MuJoCo physics and the existing COM sensor/odometry producer, without PX4. It verifies exactly 10000 physics steps (500 Hz), 5000 sensor cycles (250 Hz), 200 GPS sends (10 Hz) and 1000 odometry sends (50 Hz). Source stamps run from 1,000,000 to 20,980,000 µs for odometry; the final integer clock is 21,000,000 µs.

Actual SITL physical duration: **19.999999999999794 s**, or 20 seconds within floating-point accumulation error, strictly <=20. Host runner time: **20.001637575 s**, a separate wall-clock measure. All expected message counts match. The loop still receives the matching control message before every integrated lockstep cycle and performs the original zero-control/shutdown path.

Actual ULog parameters: `EKF2_EV_CTRL=5`, `EKF2_GPS_CTRL=7`, `EKF2_HGT_REF=1`, `SDLOG_PROFILE=0`, `SDLOG_MISSION=0`. Source rechecked in `params_external_vision.yaml:5`, `params_gnss.yaml:5`, `module.yaml:87`: EV bits0/2 enable horizontal position/3D velocity; GPS bits0/1/2 retain horizontal position/altitude/3D velocity; height reference1 is GPS. EV yaw is not enabled.

## Direct receipt and field agreement

`vehicle_visual_odometry` is now directly present: **985 records**. Publications and receipt-based sample stamps advance from **1,280,000 to 20,960,000 µs**, each interval exactly20,000 µs. Last receipt is 40 ms before the physical endpoint, below the predeclared250 ms limit. Preflight also locked minimum900 samples and initial-record coverage within1 second of the epoch.

All checked positions, velocities, angular velocities, quaternion components and retained covariance diagonals are finite. Pose is uORB NED=1; velocity is BODY_FRD=3, matching MAVLink LOCAL_NED=1/BODY_FRD=12. Quaternion norm is valid, reset counter0 and synthetic quality100. Covariance diagonal values are0.01 within float32 tolerance. The pinned receiver retains diagonal variances; it does not pass all MAVLink cross-covariances to EKF2.

Successful-send payload witnesses and all985 received records have exact matching logical timestamps. Maximum absolute field discrepancies:

| Field | Maximum discrepancy |
|---|---:|
| NED COM position | 3.51e-9 m |
| BODY_FRD linear velocity | 2.11e-13 m/s |
| BODY_FRD angular velocity | 1.81e-13 rad/s |
| Quaternion component | 1.11e-8 |
| Covariance diagonal | 2.24e-10 |

The predeclared absolute field tolerance was2e-6. Logging starts after simulator startup and does not retain every transmitted sample; 985 logged records versus1000 sends is **not** a measured wire-packet drop count. The run provides terminal coverage, not an invented1000-record receipt trace.

## Time semantics and fusion

`SimulatorMavlink.cpp:490–492` advances PX4's simulation clock from HIL_SENSOR time. `handle_message_odometry:870` overwrites the ODOMETRY sample stamp with `hrt_absolute_time`; publication uses that same PX4 clock. For the recorded run, exact source/HIL logical-clock correspondence was verified, and publication-minus-receipt time is0 µs on that frozen lockstep clock.

**This does not mean zero physical transport delay.** Bridge send times are host monotonic times. No calibrated host/PX4 clock offset was measured, so actual one-way transport delay and FAST-LIO2 source-clock synchronization remain **NOT TESTED**. Clock domains are plotted separately; no unrelated timestamp subtraction is called latency.

Fusion requires current source flags and valid local position/velocity **plus** the final three fused, unrejected aid samples with strictly advancing `time_last_fuse`. Flags alone cannot pass. Final EV fuse times are **20,756,000 → 20,780,000 → 20,796,000 µs**; final GNSS times are **20,588,000 → 20,692,000 → 20,788,000 µs**. Last fuses are204 ms and212 ms old at the physical endpoint, respectively, within the predeclared1-second freshness bound.

| Aid source | Logged samples | Accepted samples | Nonfused / innovation-rejected |
|---|---:|---:|---:|
| EV position | 984 | 889 | 95 / 0 |
| EV velocity | 942 | 942 | 0 / 0 |
| GNSS position | 198 | 179 | 19 / 5 |
| GNSS velocity | 198 | 178 | 20 / 0 |

Initialization/nonfused/rejected observations remain counted. GNSS position has five early rejected innovations; they are not silently discarded or called continuous accepted fusion from startup. Final-three samples of all four sources are accepted, their fuse times advance, final EV/GNSS position/velocity flags are true, and local position/velocity are valid.

Static local EKF estimate versus COM truth: raw coordinate-component position RMSE0.04920 m, relative displacement RMSE0.00625 m, NED velocity RMSE0.001399 m/s. The retained local-origin offset is approximately `[0.00014,-0.01514,0.08252]` m. This is not a moving-state accuracy or GNSS-denied result.

## Safety, resources and verdicts

All41 vehicle-status records are DISARMED=1; Offboard-enabled and failsafe flags remain false. No ARM command is recorded. The only logged command ID is211, internal rather than an external Bridge flight command. Incoming PX4 COMMAND_LONG/HEARTBEAT packets are not outgoing ARM commands. All physical motor controls throughout the recorded run and at exit are zero. PX4 exit code0; no process remains.

- Completed safety queries:28; no fault, no response timeout, no queue beyond one outstanding query. Valid yaw allocator row at all28 completions.
- Query response P50/P95/P99/max: **2.116 / 5.032 / 5.114 / 5.138 ms**, below150 ms.
- Query-start interval maximum: **500.076 ms**, below650 ms; mean499.998 ms.
- Actuator host interval P50/P95/P99/max: **3.993 / 4.375 / 4.514 / 12.998 ms**, below250 ms. Matching actuator simulation intervals are4 ms; 4985 returned actuator messages after bootstrap.
- Parse/validate maximum:0.882 ms; parser peak buffer1836 characters; recent console window65536 characters; no measurement drops/backlog.
- Bridge peak RSS: **738.00 MiB**; sampled PX4 peak RSS: **9.52 MiB**; sampled minimum MemAvailable: **8.95 GiB**. One worker/PX4/MuJoCo; no OOM observed; swap unchanged.
- Bridge CPU2.317 s and sampled PX4 CPU1.34 s over20.002 s runner wall time. These are process metrics, not whole-system CPU load or hard real-time guarantees.

| Criterion | Result |
|---|---|
| A producer / B actual MAVLink send | PASS |
| C direct PX4 receipt/publication | PASS |
| D EV position / E EV velocity fusion | PASS |
| F GNSS retained and actually fused | PASS |
| G strict20-second boundary/counts | PASS |
| H final operational-code/model/PX4 identity | PASS |
| I original safety query/watchdogs | PASS |
| J disarmed/no Offboard/zero motor controls | PASS |
| K complete immutable log evidence | PASS |
| L normal cleanup/historical protection | PASS |

ULog:19 configurable topics plus event, zero dropouts, no parser corruption, complete14441-record byte framing. Closed ULog/console hashes prevent subsequent truncation—even at a complete-record boundary—from reusing the proof. Main-worktree87 historical untracked files remain present; all protected physics/models/raw logs remain unchanged.

## Tests and independent review

Relevant final tests: **437 Bridge PASS** (371 inherited +66 new) and **30 Shadow PASS**. Bridge has two inherited angle warnings; Shadow has three inherited package-import warnings. Test-suite peak RSS1,146,308 KiB. No whole-repository CAD/ABI claim is made.

RED→GREEN covered missing implementations and evidence gates, including console-only raw identity and no-Offboard/failsafe corroboration. Tests cover finite Logger topics, counts/cycle bounds, field and time validity, truncated/corrupt logs, stale/flags-only fusion, historical-proof rejection, single-use attempts and immutable saved-run recomputation. Removing visual odometry or either EV aid topic cannot produce overall PASS.

Independent pre-run review gave GO after fixing log-corruption/truncation and final-model identity gaps. All60 then-new targeted tests passed independently. Final independent read-only result review gave **GO**, with zero Critical/Important findings; it rechecked raw identities, all direct/fusion evidence, protocol, safety and report recomputation. It separately ran65 targeted tests, deliberately excluding the one physical-schedule test; it did not independently rerun the entire437/30 suites. Its only Minor wording issue was corrected to avoid implying flight. No additional SITL or physics episode was run for review.

## Actual commands and local-only evidence

Executed from this worktree. The exclusive `protocol.attempt.json` reservation prevents repeating this protocol; do not run another test without separate authorization.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco:/usr/lib/python3/dist-packages \
 /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.odometry_reverification_runner \
 --px4-root /home/keyousuke/PX4-Autopilot-mineuav-yaw \
 --compatibility-manifest mujoco/px4_bridge/compat/px4_yaw_build_identity.json \
 --protocol mujoco/reports/px4_odometry_reverification_parts/protocol.json --freeze-only

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco:/usr/lib/python3/dist-packages \
 /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.odometry_reverification_runner \
 --px4-root /home/keyousuke/PX4-Autopilot-mineuav-yaw \
 --compatibility-manifest mujoco/px4_bridge/compat/px4_yaw_build_identity.json \
 --protocol mujoco/reports/px4_odometry_reverification_parts/protocol.json \
 --runtime mujoco/reports/px4_odometry_reverification_parts/disarmed_01 \
 --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

Offline analysis may be repeated without starting PX4:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco:/usr/lib/python3/dist-packages \
 MINEUAV_PYULOG_SEARCH_PATH=/home/keyousuke/.local/lib/python3.10/site-packages \
 /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.analyze_odometry_reverification \
 --runtime mujoco/reports/px4_odometry_reverification_parts/disarmed_01 \
 --report mujoco/reports/px4_mujoco_odometry_fusion_reverification.json \
 --figures mujoco/reports/figures/px4_odometry_reverification
```

Raw ULog, console, full source/send witnesses, physical records, private rootfs, frozen protocol and exclusive attempt receipt are local-only in `mujoco/reports/px4_odometry_reverification_parts/`. Git contains code/tests, aggregate report, ten figures and documentation—not raw trajectories or binaries.

## One next recommendation

A separately authorized **disarmed GNSS-denied fusion-source verification** is now reasonable before considering external-odometry flight. It must prove GNSS has actually exited fusion while external position/velocity remain valid. Do not execute it in this task. FAST-LIO2 timing, learned-policy PX4 control and flight remain NOT TESTED.
