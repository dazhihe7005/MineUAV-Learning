# PX4–MuJoCo Phase2E: disarmed runtime verification

Date: 2026-10-12. Branch `feat/px4-mujoco-disarmed-verification`, base
`c9174292e3e75cfab43706bc63b5b53b689a136c`.

## Outcome and scope

**PASS for one fresh, 16.000 s disarmed PX4 SITL + MuJoCo communication run.**
The Bridge sent 4000 HIL_SENSOR and 160 HIL_GPS messages, received 3986
HIL_ACTUATOR_CONTROLS messages, completed 20 safety queries and received 20
valid allocator yaw rows. There was no safety fault, watchdog timeout, ARM
command from the Bridge, motor command, crash, or abnormal exit. Four MuJoCo
control channels were zero at shutdown; owned PX4 exited with code 0.

This establishes the communication prerequisites for a *separately authorized*
restricted yaw test. It does **not** test final in-flight magnetic heading
alignment, arming, yaw flight control, or hover. There was one actual SITL run,
one MuJoCo instance, one PX4 instance, zero retries, and zero neural-network
training jobs.

| Acceptance item | Status | Evidence |
|---|---|---|
| Full 16 s, new empty runtime | PASS | 16.000000000002 s simulation; raw summary hash `2dad70007c79d0753bed55eb8d9845f0e30878eb8f470dc0f704fe3245b11407` |
| PX4↔MuJoCo and simulated sensors | PASS | 4000 HIL_SENSOR, 160 HIL_GPS sent; 3986 HIL_ACTUATOR_CONTROLS received; PX4 sensor_combined and magnetometer diagnostics present; source publication timestamps advance |
| Allocator yaw row | PASS | 20/20 queried rows `[-0.017207708,+0.017207708,-0.017207708,+0.017207708]` within declared tolerance |
| Query scheduler and safety monitor | PASS | 20/20 complete; no overlaps, backlog queue, dropped measurements or latched faults; 150/250/650 ms limits unchanged |
| No arming and safe cleanup | PASS | Disarmed CLI path only; sent MAVLink message types HIL_SENSOR/HIL_GPS; all 801 recorded motor vectors and final control vector zero; PX4 exit 0; no owned process remained |
| Historical artifact protection | PASS | MuJoCo model/config hashes unchanged; isolated PX4 binary unchanged; 87 historical untracked files remained in the primary worktree |
| EKF final in-flight alignment | NOT TESTED | A disarmed session cannot establish this flight condition |
| Yaw control, hover, Phase3 | NOT TESTED | No ARM, takeoff or maneuver was run |

An incoming PX4→Bridge COMMAND_LONG count of one appears in the transport
counters. The Bridge→PX4 sent types contain no command message. The incoming
payload was not logged, so its purpose is unresolved; it is not evidence that
the Bridge sent ARM.

## Frozen versions and command

- MuJoCo 3.14.0; unchanged dynamics at 500 Hz, simulated sensors at 250 Hz.
- Isolated PX4 source `142c24cffb0cdfd57ab0a9f2506d3203610ce737`, based on
  original `8f9b91b2bdd3a65952601a084c3aa35e96fee59d`;
  `v1.18.0-beta1-580-g142c24cffb`.
- Isolated PX4 executable SHA256
  `2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004`.
  Original PX4 executable SHA256
  `4f491ce85d6c79fbd5713894112c8911903fcefd1bf45b7da3579cd1812beac4`.
- The exact nine Bridge runtime source hashes are stored under
  `raw_evidence.runtime_source_sha256` in the committed JSON report and match
  the sole run's `streaming_runtime_hashes`. They include the new passive
  measurement module.

Run once from `/home/keyousuke/MineUAV-Learning-px4-sitl`:

```bash
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.allocation_runner \
  --px4-root /home/keyousuke/PX4-Autopilot-mineuav-yaw \
  --compatibility-manifest mujoco/px4_bridge/compat/px4_yaw_build_identity.json \
  --runtime mujoco/reports/px4_disarmed_verification_parts/disarmed_01 \
  --mode disarmed \
  --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

The raw runtime directory is local-only and was new and empty before that
command. Its `smoke_summary.json` (SHA above), `px4_console.bin`
(`890d131c2a892637d413d290d704cb56b4e8d2eb1d64c89ad5d78d59d06b5326`)
and sole ULog (`f8883269556ad1b3d30bd163295424eec610f5f996574ed3216e2b7c44153683`)
were preserved. The summary is written atomically after cleanup.

The report/figures can be regenerated offline from that existing result:

```bash
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.report_disarmed_verification \
  --external-peak-rss-kib 754704
```

The generator only reads the saved run and will reject missing evidence,
unmatched runtime source hashes, invalid rows, or incomplete query coverage.
It has no PX4 launch path.

## Live timing and resource evidence

The host monotonic clock measured each query from `begin_query` until all
three complete safety topic records and the allocator row had arrived, with
the PTY confirmed empty. The monitor separately checked the PX4 publication
timestamps. The 188 parse callback measurements include safety validation but
exclude raw archive I/O; they are not PX4 internal execution times.

| Live measurement | Samples | P50 | P95 | P99 | Max | Limit |
|---|---:|---:|---:|---:|---:|---:|
| Query response, ms | 20 | 3.140 | 4.837 | 4.884 | 4.896 | 150 ms |
| Query start interval, ms | 19 | 500.000 | 500.068 | 500.093 | 500.099 | 650 ms service bound |
| Actuator receipt interval, ms | 3985 | 4.000 | 4.100 | 4.263 | 6.078 | 250 ms watchdog |
| Console parse + safety validation, ms | 188 | 0.0967 | 0.453 | 0.570 | 0.625 | included in 150 ms response budget |
| Console service callback, ms | 12000 | 0.0110 | 0.0188 | 0.0281 | 1.632 | checked within same safety service |

The first 10 query responses averaged 2.60 ms; the last 10 averaged 3.80 ms.
This small rise did not approach the deadline in 16 s. It cannot establish
longer-term absence of latency growth. Query dispatch intervals remained near
500 ms (max 500.099 ms). Scheduler maximum recorded lateness was 0.099 ms.
The query service completed every scheduled request; there was no backlog or
old cached health value counted as a new PX4 publication. Of 20 receipts per
topic, `estimator_status_flags` had 11 distinct source publications, consistent
with its lower publish rate. Duplicate publications did not grant a new lease.

Parser processed 266,958 new characters; peak in-progress parser buffer was
1,837 characters. Recent console text was capped at 65,536 characters; the
raw archive was 267,090 bytes under its 16 MiB fail-closed bound. The largest
queued console write was 215 bytes. No measurement samples were dropped.

Bridge peak RSS: 772,816,896 bytes (external `/usr/bin/time` agreed).
PX4 sampled peak RSS: 9,887,744 bytes. Bridge CPU inside its runtime clock:
1.685 s; sampled PX4 process CPU: 1.32 s from 801 samples. Minimum system
MemAvailable: 11,145,252,864 bytes. Swap remained 0; no OOM was observed in
the owned run. PX4 peaks between the 50 Hz resource samples are unobservable.

## Historical comparison and limitations

Phase2C active SITL stopped at 15.484 s with `safety query service stopped`;
its final five wall query intervals rose from 536.78 to 641.64 ms. The new
disarmed live run completed 16 s with max query interval 500.099 ms and no
monitor fault. That is real runtime evidence for this *disarmed* workload,
not proof the full armed flight path is fixed.

Phase2D's 265 KiB fixture measured 1,465 µs mean for a legacy full-history
parse and 2.84 µs for one synthetic incremental record. Those are offline
microbenchmarks with different work from Phase2E's 165 µs mean live callback;
no direct speedup factor is claimed. Query response latency includes PX4 and
console transport; parser callback time is only one component.

Startup console still reports `SIM_GZ_EN` parameter not found and several PX4
logger subscription-capacity warnings. The actual simulator connection used
MuJoCo loopback and no Gazebo process; no safety fault resulted. The warnings
remain a log limitation. No hard real-time or hardware claim follows from a
single 16 s run. In-flight magnetic alignment and independent yaw response
still require an explicitly authorized flight test.

Figures: [query response](../mujoco/reports/figures/px4_disarmed_verification/live_query_response.png),
[query cadence](../mujoco/reports/figures/px4_disarmed_verification/live_query_interval.png),
[parse cost](../mujoco/reports/figures/px4_disarmed_verification/live_incremental_parse_cost.png),
[simulation vs wall time](../mujoco/reports/figures/px4_disarmed_verification/live_sim_vs_wall.png).

Independent read-only review initially found an Important offline assessor
defect: a query-response wall-time variable shadowed the final simulation
time, so truncated query coverage could pass. The 12-query/stale-decision
case reproduced the defect RED; the fix and regression tests passed GREEN.
The final independent re-review found **0 Critical / 0 Important / 0 Minor**
remaining findings and recommended this scoped commit. The full related
Bridge suite passed **321 tests, zero skipped**; the original 289 tests were
retained. One existing invalid-value runtime warning occurred in a yaw
analysis test. The known unrelated CAD SciPy/NumPy ABI collection failure
was not changed or re-run.
