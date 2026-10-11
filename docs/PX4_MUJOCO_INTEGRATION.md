# PX4–MuJoCo SITL Integration — Phase 1

2026-10-11 · branch `feat/px4-mujoco-sitl-integration` · base
`89df0fb3b5fc5bcd99db217d0a54c51613655a52`.

## Result and evidence limits

One version-pinned, MuJoCo-only SITL bridge and one bounded low-altitude
position hold were verified. No training, Gazebo, real serial/hardware,
historical controller/physics/reward/checkpoint modification or installation.
The old research worktree and its 87 untracked items remain untouched.

| Acceptance | Status / evidence |
|---|---|
| Current PX4 version | PASS: `v1.18.0-beta1-575-g8f9b91b2bd`, runtime `ver all` matches commit `8f9b91b2bdd3a65952601a084c3aa35e96fee59d` |
| SITL startup | PASS: existing binary, owned parameter/log directory, clean exit |
| Bidirectional bridge | PASS: real `HIL_SENSOR`/`HIL_GPS` plus returned `HIL_ACTUATOR_CONTROLS` |
| MuJoCo receives/applies PX4 outputs | PASS: armed nonzero commands drive unchanged omega² site actuators |
| PX4 receives sensors | PASS: sensor_combined, vehicle_air_data, vehicle_magnetometer and EKF fusion evidence, not merely an open port |
| State estimate | PASS, limited: tilt/yaw alignment, GNSS/baro/mag fusion; valid position/velocity, preflight passed |
| PX4 controls flight | PASS: accepted non-force arm and Offboard; nav_state14; no old PI/attitude/allocator called |
| Controlled closed loop | PASS: finite26s test, abort envelope never exceeded |
| Basic hover | PASS: final2s@10Hz satisfy predeclared sampled hold; not general flight certification |
| Dedicated attitude/yaw step response | NOT TESTED: no extra maneuver after the one position hold |
| Real motor/propeller identification | NOT TESTED: estimated dynamics, no real hardware |
| ROS/MID360S/FAST-LIO2 | NOT TESTED: interface requirements below; not integrated |

Flight target was NED `[0,0,-0.6]` relative to PX4's initialized local origin.
Actual body-origin height ended0.638854m, speed0.006119m/s, maximum horizontal
offset0.024853m, peak speed0.194394m/s, zero saturated10Hz samples. Hold
reference body-origin height0.651m: last2s21samples, height tolerance0.15m,
horizontal radius0.2m, speed<0.2m/s, tilt<10°. This is a **sampled** hold,
not proof of every intermediate physics state or a real flight guarantee.

Important estimator caveat: estimated heading≈93.4° versus physical NED
heading≈90° under the synthetic earth field/declination assumptions.
`heading_good_for_control=False` remained: matching PX4's
`Ekf::isYawFinalAlignComplete()` requires in-flight final magnetic alignment
when using magnetometers. Low0.6m hold did not establish that final alignment.
Initial `cs_yaw_align`, local position/velocity and arming checks did pass.
Do **not** interpret this limited hold as validation of yaw tracking,
geographic heading accuracy, or FAST-LIO2.

## Versions, audit and isolation

Ubuntu22.04.5, kernel6.8.0-138-generic, Python3.10.12, MuJoCo3.14.0,
pymavlink2.4.49. Ryzen9 7940HX,16cores/32threads, about14.84GiB RAM, no swap.
Installed PX4 binary SHA256:
`4f491ce85d6c79fbd5713894112c8911903fcefd1bf45b7da3579cd1812beac4`.
Source/header/binary identity and running `ver all` were checked; the older
dirty v1.14.4 tree was not used or modified. No builds or package installs.

Authoritative interface references:

- [PX4 Simulator MAVLink API](https://docs.px4.io/main/en/simulation/).
- [Pinned SimulatorMavlink source](https://github.com/PX4/PX4-Autopilot/blob/8f9b91b2bdd3a65952601a084c3aa35e96fee59d/src/modules/simulation/simulator_mavlink/SimulatorMavlink.cpp).
- [MAVLink common message definitions](https://mavlink.io/en/messages/common.html#HIL_SENSOR).
- [MuJoCo Jacobian/forward dynamics API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html).

Also inspected matching `PWMSim::updateOutputs`, `FunctionMotors::updateValues`,
`ActuatorEffectivenessRotors::computeEffectivenessMatrix`, `DataValidator::put`,
`rcS`, `px4-rc.mavlinksim`, `px4-rc.simulator`, `10016_none_iris`,
`px4-rc.mavlink`, and `platforms/posix/src/px4/common/main.cpp`.

`10016_none_iris` provides only stock multicopter startup defaults, not Iris
physics. Process-local geometry overrides match MineUAV. `SIM_GZ_EN=0`,
`PX4_SIMULATOR=mujoco`; neither Gazebo nor SIH/jMAVSim is launched. PX4 SITL
connects as TCP client to the bridge server127.0.0.1:4560. Instance0 only.
Its data-path argument is **build/px4_sitl_default/etc**, not the build root.
PX4 makes runtime/etc a symlink to that argument. The runner refuses nonempty
runtime directories, so existing user parameters/logs cannot be overwritten.

## Control/data flow

```text
Loopback external NED setpoint → PX4 EKF/controllers/control allocator
  → PWMSim → HIL_ACTUATOR_CONTROLS → actuator_adapter → MuJoCo physics
  → current COM dynamics → sensor_adapter → HIL_SENSOR/HIL_GPS → PX4

Old BC/PPO → old PI/attitude/allocator → old environment: unchanged/separate
```

Modules in `mujoco/px4_bridge/`: transport, sensor_adapter, actuator_adapter,
frame_conversions, diagnostics, simulation_runner; closed_loop is an explicit
opt-in validation client, analyze_phase1 is offline evidence publication.
No BC/PPO/world-model output is fed to PX4 in this phase.

## Dynamics and actuator contract

Unchanged `mine_uav_dynamics_v2.xml`, `rotor_config.json`, `rotor_actuators.xml`:

- Mass7kg, COM FLU `[0.01513436,0.00008125,0.03602789]`m.
- COM inertia body-frame `[Ixx,Iyy,Izz,Ixy,Ixz,Iyz]` =
  `[.161230,.144040,.251680,-.000868,-.011905,.00002662]`kg·m².
  Mass/inertia/rotor centers are engineering estimates, not measured.
- kf=`4.9977603696661616e-5` N/(rad/s)², static U7KV490+15×5CF fit;
  km=`8.6e-7` N·m/(rad/s)² estimated; maximum8500RPM assumed from old tested
  allocation bound. Static total thrust≈158.4N vs weight68.67N.
- Physics500Hz. Immediate combined site motors, no dynamic ESC/motor state.
  Collision box remains the old coarse placeholder. No battery/drag fitting.

| PX4 channel/function | MuJoCo actuator / ctrl | arm about COM, FRD[m] | model reaction torque along FLU+z |
|---|---|---|---|
| 1 / 101 | motor_1_wrench / 0 | (-.234098,-.219803,+.061453) | + |
| 2 / 102 | motor_2_wrench / 1 | (-.234098,+.219199,+.061453) | − |
| 3 / 103 | motor_3_wrench / 2 | (+.204905,+.219199,+.061453) | + |
| 4 / 104 | motor_4_wrench / 3 | (+.204905,-.219803,+.061453) | − |

These signs are **reaction-torque signs**, not verified physical CW/CCW wiring.
Real spin direction and viewing convention must be measured before deployment.
PX4 upward rotor axis is FRD−z; its torque formula is
`CT*r×axis − CT*KM*axis`, hence `CA_ROTORi_KM=−sign_i*km/kf`.
Offline tests compare this actual formula with the transformed MuJoCo wrench.

Current PWMSim maps nonreversible motors to[0,1]. With `THR_MDL_FAC=0`:

```text
c_i ∈ [0,1] = ideal normalized thrust
omega_max = 8500 * 2π/60
omega_target_i = sqrt(c_i) * omega_max
u_i = omega_target_i²
thrust_i = kf * u_i
reaction_torque_i = sign_i * km * u_i
```

This is an explicit idealized adapter, **not an ESC calibration**. Invalid,
NaN, negative, >1, wrong-site/gear/dynamic-actuator configurations are refused.
Disarmed output is zero. No new PI or allocator follows PX4's output.

## Frames, sensors and time

Model axes are assigned ENU world/FLU body for this bridge; actual vehicle
wiring/body-axis correspondence remains a deployment check. For vectors:

```text
ENU→NED: (x,y,z)→(y,x,−z)
FLU→FRD: (x,y,z)→(x,−y,−z)
R_NED_FRD = R_ENU_TO_NED * R_ENU_FLU * R_FLU_TO_FRD
```

All quaternions wxyz. Identity MuJoCo attitude corresponds to PX4 yaw+90°,
not identity. World velocities refer to virtual COM for sensors; physical
trace velocities refer to free-joint/body origin and are labeled accordingly.

Virtual IMU at COM: current `mj_forward`, `mj_jacBodyCom`, `mj_jacDot` give
`a_COM = J*qacc + Jdot*qvel`; specific force is `Rᵀ(a_COM−gravity)`, then
FLU→FRD. Rest gives FRD−g along z, freefall gives zero. Gyro is body angular
velocity rad/s, not linear qvel. Ground, freefall, tilted and single-rotor
force tests verified these independently. No stale preceding-step qacc used.

Synthetic GNSS10Hz converts local COM ENU to fixed Swiss reference
47.397742°,8.545594°,488m MSL, with m→mm and m/s→cm/s conversions. Barometer
uses standard atmosphere; synthetic magnetometer uses NED `[.215,0,.427]`
gauss. No optical-flow/lidar/visual estimator. `HIL_STATE_QUATERNION` was
requested by PX4 but is not sent: pinned handler only publishes ground-truth
topics, so it is not an estimator shortcut here. `HIL_OPTICAL_FLOW` handler
was audited but no flow measurements generated.

First ideal stationary smoke revealed a real interface pitfall: raw mag/baro
updated with fresh timestamps, but equal values triggered PX4 stale-data
voting. Fixed seed0 Gaussian mag noise std.0002gauss and baro std.01hPa avoid
that behavior without disabling checks. These are **unmeasured synthetic
sensor assumptions**, not physical wind/noise training or real sensor specs.

IMU/HIL_SENSOR250Hz; physics500Hz; actuator messages measured250Hz (4000µs).
Integer simulation clock starts at1,000,000µs and advances2000µs per physics
step. Bounded bootstrap freewheeling until first actuator message; thereafter
one current timestamped actuator response gates each two-step advancement.
Mismatch/stale command/CRC/EOF/watchdog/unsafe physical state zeros ctrl and
stops physics/owned PX4. No automatic reconnect or stale-command hold.
Bootstrap discards older packets only within one absolute receive deadline;
once lockstep starts, even one stale actuator response aborts immediately.

Stock `px4-rc.mavlinksim` applies defaults after env overrides: requested
single-IMU settings were superseded by effective `SENS_IMU_MODE=0`,
`EKF2_MULTI_IMU=3`. Only primary sensorid0 was supplied and one healthy EKF
instance ran; spare slots did not receive fabricated measurements. Commands
and effective values are in the report. UDP offboard14540→14580 only for
the gated test; other stock MAVLink links stay localhost, no real serial.

## Reproducible commands

Run from the isolated worktree. Existing benchmark venv and existing user-local
pymavlink are reused via explicit appended search path; no global installation.
Replace paths only to match an identically audited installation. Version
changes are rejected, not silently treated as compatible.

```bash
cd /home/keyousuke/MineUAV-Learning-px4-sitl
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.simulation_runner \
  --px4-root /home/keyousuke/PX4-Autopilot \
  --runtime "$PWD/mujoco/reports/px4_sitl_phase1_parts/repro_smoke" \
  --seconds 16 \
  --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages
```

The runner starts the TCP server, then this actual PX4 command with its
explicit process-local parameter/environment overrides:

```text
/home/keyousuke/PX4-Autopilot/build/px4_sitl_default/bin/px4
 /home/keyousuke/PX4-Autopilot/build/px4_sitl_default/etc
 -w <new runtime directory> -i 0
```

Do not launch a second PX4 manually beside the runner. Parameter overrides
are listed in the JSON report (`runs.*.px4_environment`), not hidden in a
global startup file. Unmodified controller gains and startup budget remain.

Only after a matching smoke_summary.json confirms real sensors, actuators,
estimator and preflight, one can explicitly opt into the bounded SITL test:

```bash
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.simulation_runner \
  --px4-root /home/keyousuke/PX4-Autopilot \
  --runtime "$PWD/mujoco/reports/px4_sitl_phase1_parts/repro_hover" \
  --seconds 26 \
  --pymavlink-search-path /home/keyousuke/.local/lib/python3.10/site-packages \
  --hover-proof "$PWD/mujoco/reports/px4_sitl_phase1_parts/repro_smoke/smoke_summary.json"
```

No force-arm: normal mode/arm command ACKs required; current estimator checked
again. Safety envelope: speed1.5m/s, angular rate2.5rad/s, tilt20°,
horizontal radius0.5m, body-origin height[-.01,1.2]m. Violation stops,
does not retune PX4 or retry flight. Default command remains disarmed.

Offline tests on this machine (existing system pytest; plugin autoload disabled):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -c 'import sys;sys.path.append("/usr/lib/python3/dist-packages");import pytest;raise SystemExit(pytest.main(["mujoco/px4_bridge","-q"]))'
```

33 tests currently cover frames/specific-force/wrench, encoding/CRC/TCP,
clocks/stale inputs, memory gates, source identity, startup, envelope,
flight proof/hold accounting and metric recomputation. This is the complete
new bridge suite; historical training/large simulation fixtures were not
executed. Original tracked code is unchanged.

One independent read-only review reproduced the raw evidence/figure and
30 original tests. It found no Critical issues and one Important stale-stream
watchdog issue. Three regression tests first failed, then passed with the
absolute-deadline/immediate-rejection fix; the final full suite passed33/33.
The orchestration regression uses fake physics/process/transport and proves
previous nonzero controls are zeroed and the owned process is cleaned up.
No extra PX4 invocation or flight followed this communication-safety fix.

Offline publication (does not start PX4/MuJoCo evaluation episodes):

```bash
PYTHONPATH=mujoco /home/keyousuke/MineUAV-Learning/.venv/bin/python \
  -m px4_bridge.analyze_phase1 \
  --parts mujoco/reports/px4_sitl_phase1_parts --output mujoco/reports
```

## Execution history/resources/artifacts

Four sequential finite invocations, never concurrent: smoke01 wrong initial
data-path, no sensor connection; smoke02 12s, ideal-sensor stale voting;
smoke03 16s, passed communications/estimator; hover01 26.004s, bounded hold.
No second flight attempt. Failed runs retained and disclosed, not erased.
Smoke03:4000IMU/160GPS/3942actuator messages; hover01:6501/261/6445.
Raw summaries, ULogs, PX4 parameters and10Hz physical trace remain local in
`mujoco/reports/px4_sitl_phase1_parts/` and ignored trace JSON. Git contains
summary with console evidence/input hashes and one response figure.

Bridge peak RSS/HWM≈547.5MiB; sampled PX4 maximum≈19.5MiB, system minimum
available≈10.55GiB. No OOM observed. One `ps` CPU probe: bridge25.5%,PX47.1%
of one CPU core during startup/flight; not continuous/full-run CPU peak.
Mean bridge CPU fraction separately follows recorded CPU seconds/wall time.
Startup now requires2.5GiB available (1GiB load budget+1.5GiB OS reserve),
runtime guards1.5GiB; no swap modification, parallel compilation or training.
Raw results are atomic JSON and each fresh runtime has independent identity.

PX4 logger warned about subscription capacity for some unused topics; no
missing topic is treated as measured. Compact console topics and physical
logs substantiate the reported interfaces. No hard real-time guarantee.

## Deployment gaps / next interfaces

Real motor spin/channel wiring, ESC signal→RPM/thrust mapping, lag/deadband,
motor saturation/dynamics, mass/inertia/COM, aerodynamic drag and battery
voltage effects need identification before hardware use. Synthetic GNSS and
mag/baro are not mine localization. No claim of real mine/wind capability.

For future ROS/MID360S/FAST-LIO2: establish one sim-clock/timestamp owner,
ROS ENU/FLU↔PX4 NED/FRD TF/extrinsics (IMU/COM/body/lidar), lidar point-cloud
and IMU streams, and external odometry with covariance/reset handling.
PX4 uXRCE-DDS agent/client exists as a possible transport, or validated
MAVLink ODOMETRY; neither is connected in this phase. Require version-matched
PX4 messages and separate sensor-estimation vs diagnostic truth modes.

**Only next recommendation:** validate synthetic geographic heading/final
alignment before yaw maneuvers or ROS external-vision integration. Do not
retune gains, connect hardware or automatically start that next experiment.

## Review boundaries and decisions

No deferred Minor findings. The Important watchdog finding is fixed above.
The following are explicit decisions, not silently passed acceptance gates:

- Native workspace creation failed before writing anything; use an external
  Git worktree and preserve the old87 untracked files. Cost: separate worktree
  bookkeeping.
- Verify the complete new bridge suite, not old training/large simulations.
  Cost: unrelated historical behavioral regressions are not exhaustively
  excluded, although their tracked sources are unchanged.
- Append existing user-local pymavlink/system pytest paths only per process;
  no installation or global environment edits. Cost: prerequisites are local
  paths, not a portable dependency installation.
- Use explicitly uncalibrated, seeded mag/baro noise to avoid PX4's stale
  value voter, without disabling checks. Cost: sensor noise remains a
  simulation assumption, not measured realism.
- Real wiring/coefficient/ESC/battery/aerodynamic judgments remain untested,
  as hardware is prohibited. Cost: no deployment safety/fidelity assurance.
- General attitude/yaw/recovery is untested beyond one hold. Cost: the hold
  cannot establish maneuver or disturbance robustness.
- Geographic heading/final magnetic alignment remains unresolved. Cost:
  yaw/navigation accuracy must not be inferred from the limited hold.
- ROS/MID360S/FAST-LIO2 behavior is outside Phase1. Cost: no tested end-to-end
  localization stack exists yet.
- Historical training/controller behavioral review remains out of scope.
  Cost: unchanged code is not a new runtime regression certificate.
- Local-process impersonation is outside isolated SITL fault handling;
  loopback/MAVLink IDs are routing checks, not authentication. Cost: a
  malicious local process could impersonate PX4; never expose these ports
  or treat the bridge as hardware/network security infrastructure.
- Resource maxima/hold evidence are sampled or process HWM, not every
  intermediate state. Cost: brief unsampled excursions or OS-wide resource
  peaks cannot be excluded, and there is no hard real-time guarantee.
