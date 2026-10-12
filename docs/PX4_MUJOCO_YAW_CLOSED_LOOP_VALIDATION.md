# PX4–MuJoCo Phase 2F: complete bounded yaw validation

Date: 2026-10-12. Branch: `feat/px4-mujoco-yaw-closed-loop-validation` from `10c70d0cfa53234d241d476b79002b675d9422b8`.

## Scope and evidence identity

Exactly **one** newly authorized active SITL run was made. It completed 45.000 s of simulated time; no retry, second active run, or Phase 3 test was performed. The matching Phase 2E disarmed proof was checked before flight, not regenerated. Its saved summary SHA-256 is `2dad70007c79d0753bed55eb8d9845f0e30878eb8f470dc0f704fe3245b11407`. The actual isolated PX4 binary SHA-256 was `2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004`; MuJoCo was 3.14.0. The verified runtime-source hashes, 7 kg physical model hashes, simulated magnetic field, initial state and sensor configuration matched the disarmed proof. Protected historical checkpoints were unchanged.

The raw active-run summary, console, ULog and local flight ledger remain ignored, local-only files in `mujoco/reports/px4_yaw_closed_loop_parts/yaw_01/`. Their SHA-256 identities respectively are `edf5a709c307bc9972f28d27d94b23edc1e1aa08be540d7ede3be49be92278da`, `24efc147409a1c3855d4b4354c7cbb9c5c53ec579dd018a3e37a4dc3fb24e471`, and `a3bf79d84d4548eed3714f7f7ac45f1f73992aafad6bdafb6cd8dfed408fb754`. [The committed machine-readable report](../mujoco/reports/px4_mujoco_yaw_closed_loop_validation.json) holds all measured aggregates and version hashes. This text is not a substitute for the local raw signals.

## Frozen protocol and execution

One MuJoCo model and one isolated PX4 SITL, 500 Hz physics and 250 Hz simulated sensors. Initial target NED yaw 90°, height target NED −2 m (body-origin ENU ≈2.051 m), then +10° to NED 100° no earlier than 32 s; 45 s maximum. The original success criterion was true NED yaw error <2° and absolute body-FRD-z yaw rate <0.05 rad/s, sustained ≥2 s with adjacent physical samples ≤0.15 s apart. No controller gain, EKF parameter, motor coefficient, safety limit or threshold was changed. Preflight and final Bridge suites passed 321 and 336 tests respectively. The flight used the exact verified runtime code; only offline analysis, tests, plots and documentation were added afterward.

Actual launch (once, using a fresh empty runtime):

```bash
/usr/bin/time -v env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco \
  /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.allocation_runner \
  --px4-root /home/keyousuke/PX4-Autopilot-mineuav-yaw \
  --compatibility-manifest mujoco/px4_bridge/compat/px4_yaw_build_identity.json \
  --runtime mujoco/reports/px4_yaw_closed_loop_parts/yaw_01 \
  --mode yaw \
  --smoke-proof mujoco/reports/px4_disarmed_verification_parts/disarmed_01/smoke_summary.json \
  --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

The command exited 0; the run recorded `complete=true`, no exception, and zero four-motor controls on shutdown. This is a SITL simulation only; no real flight controller, ESC or motor was connected.

## Observed closed-loop control chain

The +10° step was actually issued at simulation 32.004 s. PX4 published a changed attitude setpoint, a yaw-rate request (peak +0.4802 rad/s), and a yaw-torque request (peak +0.07825 **PX4-normalized units**, not N·m). The queried allocator yaw row remained `[-0.0172077078, +0.0172077078, -0.0172077078, +0.0172077078]` in all 78 query records. Actual ULog motor/allocator parameters matched the unchanged model. For all 65 sampled post-step allocator statuses, `torque_setpoint_achieved` was true; absolute unallocated yaw torque was at most `4.22e-8` normalized units. No allocator saturation flags were reported.

The executed four-motor outputs split in the expected alternating direction: motors 1/3 decreased and 2/4 increased at positive-yaw onset; reverse differential later braked the turn. All normalized outputs remained within [0,1], with zero command-boundary and zero reported allocator saturation samples. With the unchanged ideal motor model, the derived net body-FRD-z reaction torque reached +0.2132 N·m and −0.1604 N·m. These are **model-derived**, not hardware-measured, forces/torques; PX4 normalized torque requests must not be equated numerically with N·m. The `omega²` actuator mapping, rotor rotation signs and physical coefficients were not changed.

MuJoCo true NED yaw was 90.0365° at the step and 99.9492° at flight end, a +9.9127° physical change in the commanded direction. PX4 last supported estimated yaw was 99.9841° at 44.96 s; this estimate is not substituted for physical truth. Mean post-step estimate-minus-truth error was −0.0218° (max absolute 0.1045°). True yaw overshot by 4.2044°, peak true FRD-z yaw rate was 0.3227 rad/s, and final-two-second mean signed target error was −0.0259°. The original strict settling rule was satisfied **3.360 s after the step**, with a maximum adjacent physical sample gap of 0.020 s and 12.996 s observed after the step.

For a causal response sketch, predefined two-consecutive-sample onset thresholds found: attitude setpoint 32.032 s (>2° change), rate setpoint 32.024 s (>0.02 rad/s), torque request 32.036 s (>0.005 normalized), model-derived motor yaw torque 32.044 s (>0.01 N·m), and measured physical yaw rate 32.124 s (>0.02 rad/s). The apparent 0.120 s command-to-physical threshold crossing includes sampling, response and threshold effects; it is **not** a pure component delay. Publication rates explain why the rate-setpoint sample precedes the attitude-setpoint sample in these logs.

Post-step maximum height error relative to the body-origin target was 0.00835 m, maximum XY excursion 0.02724 m and maximum tilt 0.127°. No abnormal position coupling or envelope violation was observed in this single nominal maneuver. First in-flight magnetometer alignment was logged at 15.816 s and first final `heading_good_for_control` at 16.824 s; all post-step logged health/status samples retained final alignment, healthy positioning, armed Offboard state and no failsafe.

## Safety, timing and resource audit

All 78 safety queries completed with no fault; wall-clock start interval median 500.000 ms, maximum 500.072 ms. Query response median 2.090 ms, P95 4.848 ms, maximum 5.109 ms, below the unchanged 150 ms response limit. The last query and last healthy gate state covered the run tail. 11,250 HIL_SENSOR and 450 HIL_GPS messages were sent; 11,236 HIL_ACTUATOR_CONTROLS were received. Maximum sampled actuator wall interval was 11.561 ms, below the 250 ms watchdog. The 650 ms service check did not trigger, there was no query accumulation beyond one in-flight query, and peak streaming parse buffer was 1,838 characters. All 2,251 sampled physical records covered 0.004–45.000 s, had a maximum adjacent gap ≤0.0241 s and matched the final qpos/qvel and simulator clock. All sampled states stayed within the unchanged flight envelope. At shutdown, PX4 exited normally and MuJoCo motor controls were all zero. The independent review found and the offline RED→GREEN tests closed a prior assessor gap that had accepted a truncated physical trace; the actual frozen flight trace was complete and never modified.

Bridge peak RSS was 773,029,888 bytes (external `/usr/bin/time` maximum 754,912 KiB, same value); sampled PX4 peak RSS 9,973,760 bytes; minimum system MemAvailable 10,897,215,488 bytes. Bridge CPU 5.044 s, sampled PX4 CPU 2.65 s. One worker, one PX4 process, no OOM observed during the owned run, and no swap configuration change. PX4 resource peaks between samples and kernel-wide OOM history are not certified by this record. The console contains a nonfatal `SIM_GZ_EN` parameter-not-found line and logger-capacity warnings; neither is evidence that Gazebo was launched or that a safety gate failed.

## Acceptance and limits

| Question | Result | Evidence boundary |
| --- | --- | --- |
| Step issued and PX4 attitude/rate requests produced | PASS | Setpoint and torque/rate uORB logs |
| Yaw allocator row and torque allocation | PASS | 78 current yaw rows; 65/65 post-step allocator samples achieved |
| Correct motor differential and physical yaw direction | PASS | Executed motor commands, unchanged physical mapping, MuJoCo truth |
| True yaw settles near 100° by original rule | PASS | 3.360 s after step; final 99.949° |
| Final EKF alignment and safety gate through maneuver | PASS | Post-step ULog flags/status plus Bridge query/actuator time series |
| Motor saturation or excessive height/XY coupling absent | PASS | Sampled motor, allocator and physical-state records |
| General robustness or hardware deployment | NOT TESTED | One nominal positive step, idealized ESC/motor model |
| Phase 3 external Odometry/ROS integration | NOT TESTED | Not authorized or run |

The source-level guarded PX4 yaw-row compatibility patch from Phase 2C, already pinned in this isolated binary, is the only modified PX4 control-allocation behavior. It keeps the real row and physical motor coefficients; this run provides closed-loop evidence for that pinned build, not a claim about upstream PX4 or deployment hardware. No negative-yaw, disturbance, sensor-fault, repeated-flight or real-time-certification claim follows from one bounded run. Logged health queries can miss faults entirely between queries; the monitor's fail-closed semantics and finite source leases remain necessary.

Offline reproduction without starting SITL:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=mujoco \
  /home/keyousuke/MineUAV-Learning/.venv/bin/python -m px4_bridge.analyze_yaw_closed_loop \
  --dependency-search-path /home/keyousuke/.local/lib/python3.10/site-packages \
  --external-peak-rss-kib 754912
```

The six linked [evidence figures](../mujoco/reports/figures/px4_yaw_closed_loop/) are generated from local raw data. Discrete PX4 setpoints and EKF flags use step traces. The PX4 continuous yaw estimate is linearly interpolated between valid ULog estimate *sample* timestamps to compare with physical samples, with NaN outside that sample interval and no endpoint extrapolation; the report separately gives the last supported estimate on the physical-sample grid. The report and tests can only be completely recomputed on a machine with the local ignored ULog and summary present. The final related Bridge suite passed 336 tests with two nonfatal pre-existing circular-angle NaN warnings. A whole-repository test pass is not claimed; an unrelated historical CAD NumPy/SciPy ABI collection problem was previously disclosed.

Independent read-only review checked raw identity, coordinate and unit signs, sampled safety coverage, the original settling calculation, graph semantics and report reproducibility. Its Important finding was that the first offline assessor could accept a physically truncated trace; this was reproduced with failing tests and fixed offline without a second flight. Final re-review independently ran 336 tests (zero failures/skips), confirmed the original full trace passes while truncated or corrupted traces fail, and recomputed the JSON report and six figure hashes without differences. It found no remaining Critical or Important issue. The review is limited to saved evidence and source; it is not additional SITL or hardware validation.

Only next recommendation: after inspecting this bounded control-chain evidence, separately authorize an external-odometry interface preflight for Phase 3. It was not executed here.
