# Gust Recovery Attitude-Thrust Response Audit

## Outcome

The strongest observed discrepancy is **requested attitude/thrust direction versus actual attitude/thrust direction**, not scalar thrust allocation or simulated motor lag. Gust P changes to braking the returning motion at4.55–4.56s; opposing I delays the signed PI reversal to4.81–4.82s. Actual body-Z, model-derived rotor force, COM acceleration and body-origin acceleration reverse at5.50–5.51s. The observed three-dimensional speed then peaks1.478594m/s at5.51s. These are descriptive crossing separations, not an identified pure delay or unique causal module diagnosis.

No controller structure change or specific PI gain adjustment is established by these two observational traces. Stop this audit: no tuning, training or follow-up experiment was executed.

## Fixed inputs and execution

- Branch `feat/uav-attitude-thrust-response-audit`, base `c84389c3457bfa88d6def6be517055ec3c8e8161`.
- Reuse exactly the previous Constant-Medium and Gust-Medium internal telemetry, old manifest index000, target `external-final-2026101201-000`, seed950140000.
- Target `[-0.7051126477516996,-1.5917973670742391,1.14849756028413]`m; force direction `[0.9845354832399106,0.1751852797513414,0]` worldXYZ.
- Same saved initial MjData/controller pairing. Constant6.867N persists; Gust6.867N at500Hz ticks1000..1999, `[2,4)`s. Original physics/control/policy500/100/25Hz,15s limit and success gates retained.
- **0new episodes,0new flight executions,0integrated physics steps.** One serialized worker. Full analysis evaluates1264+1500=2764saved100Hz states with `mj_forward`; re-analysis checks those same states, not new trajectories.
- Historical previous stage had3flight executions for2unique episodes, including its disclosed Constant verifier recovery. None of those flights was repeated here.

Input provenance checks validate original checkpoint/source hashes, previous acquisition record/raw hashes, exact paired initial fingerprint, original termination and all12old baseline arrays bitwise. Historical report/model metadata bytes are compared with the base commit. New outputs never overwrite previous report/raw paths. Identity mismatch refuses resume/publication.

## Physics and frame audit

Installed MuJoCo3.14.0; compiled model has one7kg rigid vehicle body, nq7/nv6/nu4/na0, no activation, actuator delay/history, plugins or callbacks. Four site `motor` actuators have unity fixed scalar gain, no bias and no internal dynamics. Original environment writes allocator u directly into ctrl before physics. The compiled attributes and source agree with [MuJoCo direct-drive motor semantics](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-motor). No real motor response claim is made.

World: right-handed,+Zup, gravity `[0,0,-9.81]`m/s². MuJoCo wxyz quaternion rotates body to world. Body+Z is rotor thrust. Freejoint qvel[:3] is world-frame **body-origin** velocity; qvel[3:6] is body angular velocity. Desired wrench is body `[Fz,Tx,Ty,Tz]` about the estimated COM. The force-axis plots project world vectors onto the fixed manifest direction; they are not body-X plots.

Body COM offset is `[.0151343576,.0000812497,.0360278856]`m. Body-frame inertia, kg·m²:

```text
[[ .1612299983, -.0008678571, -.0119047485],
 [ -.0008678571, .1440402368, .0000266219],
 [ -.0119047485, .0000266219, .2516798804]]
```

Mass, COM, inertia, rotor locations and coefficients are engineering model estimates, not measured hardware properties. No aerodynamics or wind-speed conversion was introduced.

### What is observed versus reconstructed?

Recorded100Hz signals: qpos/quaternion, world-origin velocity, body angular velocity, desired attitude, desired thrust/wrench, P/I/control output and actual newly written ctrl. Applied disturbance force is recorded500Hz at the original before-physics hook. Existing cached qacc is a previous-tick quantity and **is not used as current acceleration**.

Each saved state is loaded into a new scratch MjData with matching current ctrl and world COM force. [mj_forward evaluates forward dynamics without integrating time](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-forward). It never invokes a controller in this callback-free model. Saved arrays and model parameters are not changed. Contact/constraint/passive-force states or dynamic/history actuators are rejected, rather than assuming missing hidden states. All2764states are contact-free.

The reconstructed actuator scalar output equals ctrl in omega² units. It is **not Newton-valued rotor thrust**. Site gear `[0,0,k_f,0,0,±k_m]` converts scalar output into force/torque. Site world forces and qfrc_actuator[:3] agree; the full qfrc_actuator is a generalized mixed-frame6D force, not a world Cartesian wrench. Site moments about COM are separately reconstructed and checked against the allocator's requested COM torque. See [MuJoCo actuation model](https://mujoco.readthedocs.io/en/stable/computation/index.html#actuation-model).

Derived current COM resultant:

```text
F_net_COM_world = F_rotors_world + F_external_world + m*g_world
a_COM_world = F_net_COM_world/m
a_COM_world = qacc_origin_world + R*(alpha_body×c + omega_body×(omega_body×c))
```

All these are **model-derived** physical quantities, not independent historical force sensors. The new current qacc is distinct from the historical cache. Newton/Euler balances are internal consistency checks, not validation against real vehicle measurements. Maximum COM-balance residual is3.08e-15m/s²; requested COM torque→Euler angular acceleration discrepancy is<9.73e-14rad/s².

## Fixed event definitions

Focus3–8s. Event search4–8s, same direction for both cases, no new state selection. Find the first opposite-to-new-sign reversal sustained for3consecutive100Hz samples. Report the bracket from the last opposite sample to the first confirmed-run sample, plus the later confirmation timestamp. No interpolation, smoothing or extrapolation. Missing events remain null.

Two predeclared detection descriptions are retained: numerical zero epsilon1e-9; fixed deadbands velocity/relative command.005m/s, acceleration/P/I/PI.01m/s², tilt.001rad, force.07N, speed power.001m²/s³, position.002m. Neutral samples widen the bracket; they are not silently discarded. These are event-detection uncertainty descriptions, not controller tuning or a sweep.

Important: at4s the escaping motion is **already decelerating** (v·a=-.89436m²/s³). It then reverses direction at4.32–4.33s. The later positive P/PI events concern braking that **returning approach motion**; they are not the first braking requests of the entire episode. Speed-power positive→negative detection identifies the subsequent5.51s peak. Desired velocity itself stays toward the target until the position crosses; the relevant braking request is desired-minus-actual velocity, not merely a command sign change.

| Gust event, along fixed force axis | Zero bracket, s | Deadband bracket, s |
|---|---:|---:|
|Velocity reverses into approach|4.32–4.33|4.32–4.33|
|Relative velocity request / P becomes braking|4.55–4.56|4.55–4.56|
|Committed PI / desired tilt reverses|4.81–4.82|4.81–4.83|
|Actual tilt / rotor resultant / COM acceleration|5.50–5.51|5.50–5.51|
|Body-origin acceleration|5.50–5.51|5.49–5.51|
|3D speed power becomes negative|5.50–5.51|5.50–5.51|
|Signed position error crosses|5.67–5.68|5.67–5.68|
|I projection reverses|6.10–6.11|6.09–6.13|

P→PI crossing separation bounds.25–.27s; desired→actual tilt.68–.70s. Actual tilt→rotor force→COM acceleration has no resolved additional separation (same100Hz bracket; difference bounds±.01s). The finite sampling cannot exclude sub10ms separation, but the compiled motor model has no dynamic lag state at all.

## Force and acceleration evidence

For desired body-Z b_d, actual b, allocated scalar thrust T and desired thrust T_d, the following identity separates current **body-origin acceleration mismatch** without treating it as causal attribution:

```text
a_origin - a_PI = (T/m)*(b-b_d) + ((T-T_d)/m)*b_d
                 + F_external/m - origin_to_COM_acceleration
```

Maximum identity residual<2.51e-14m/s². Component RMS values below are norms over fixed3–8s, not independently additive RMSEs.

| Metric | Constant | Gust |
|---|---:|---:|
|Attitude tracking RMS, deg|3.07964|6.97201|
|Attitude tracking max, deg|4.89590|10.76984|
|Orientation contribution RMS, m/s²|.525025|1.200105|
|Scalar thrust magnitude contribution RMS, m/s²|8.66e-15|9.42e-15|
|External-force contribution RMS, m/s²|.981000|.438717|
|Origin/COM offset contribution RMS, m/s²|.004260|.009267|
|Scalar thrust magnitude max difference, N|1.57e-13|1.57e-13|
|COM torque max difference, N·m|1.56e-14|1.38e-14|
|Joint position/speed gates after4s at100Hz|19|0|
|Termination|success12.64s|timeout15s|

The largest non-external discrepancy in both cases is the direction/attitude term. This is an exact geometry decomposition inside the fixed model; it does not prove which upstream loop caused that attitude trajectory. Constant retains external compensation throughout and has a different state/history at4s, so it is not a same-state-at-removal causal intervention.

At5.51s Gust force-axis P=+2.075234, I=-.411827 and PI=+1.663408m/s². I reduces the signed P braking request by about19.84% at that instant. Actual force-axis COM acceleration is only+.014940m/s², and body-origin acceleration+.017735m/s²: actual thrust direction has only just reversed. At4.82s P≈+.7751 and I≈-.7713 nearly cancel while actual acceleration remains strongly toward the approaching motion. This directly supports upstream P/I opposition **and** a subsequent desired/actual direction mismatch, not a sole PI cause.

Recorded velocity's100Hz centered finite difference is an additional descriptive check. Constant difference RMS is.000828m/s²; Gust.022073m/s², max.491685m/s² at the force-removal discontinuity because a centered±10ms derivative spans both sides. Do not mistake that boundary estimate for an aligned sensor acceleration or force-timing defect. Away from the discontinuity the curves are close; no forward-reconstructed acceleration was taken from the stale cache.

## Interpretation and stopping decision

- **Confirmed:** scalar motor input→site force is immediate in this model; magnitude and torque allocation are accurate to floating precision; the actual attitude/thrust direction follows its changing demand with substantial observed phase separation. No simulated motor dynamic state, allocator saturation or force-buffer residue explains the measured.69s separation.
- **Confirmed:** the model-derived current acceleration follows the actual thrust direction; position/speed remain out of phase within the original time budget. The analysis never changes the success rule.
- **Supported hypothesis:** bounded integral opposition plus finite attitude/outer-cascade response contributes to insufficient transient braking and oscillatory settling. The strongest measurable downstream difference is desired→actual orientation, rather than thrust amplitude or an actuator-delay stage.
- **Unresolved:** unique causal responsibility of PI, attitude gains or outer position loop; controller structure necessity; optimal PI adjustment; asymptotic stability/eventual recovery; generalization to High/BC/other targets. No isolating intervention was performed.

Therefore do not tune PI solely from this audit or automatically redesign the controller. Evidence is sufficient to localize the dominant observed chain discrepancy, but not to prove a unique cause or specify a safe gain change. This is the final bounded validation, not an invitation to keep subdividing experiments. No follow-up experiment is started.

## Verification and artifacts

66/66relevant tests pass:21new offline physics/event/analysis/publication tests plus45previous passive/saved-data/controller tests. No skips. Tests construct scratch models or verify controller computations, but integrate no flight. Bare global tests containing unrelated simulation/training were excluded by the minimal scope.

One worker, guarded serial phases, atomic JSON/NPZ writes, frozen input identity and raw hashes. NoOOM or swap changes. Resource peak values and every guarded phase are in the report; historicalVmHWM sum is distinct from simultaneous sampled process-treeRSS. A pre-final fixture failure from decimal.02 vs accumulated physics timestamp differed by3.47e-18; corrected to require bitwise saved-clock preservation plus1e-12decimal-lattice tolerance. No trajectory or timestep was rounded/changed.

Fresh independent read-only review found no Critical/Important issues and passed44hash/index checks. Coordinator's final66/66tests passed without skips or flight integration. One deferred Minor: PNG regeneration writes directly to final filenames, unlike atomic JSON/NPZ. An interrupted regeneration can leave a partial figure or break the prior report's figure hash; current five images were visually checked and hash-verified, and finalization rejects altered figures. This affects regeneration resilience, not the verified current results. No second review or broader experiment was performed.

Tracked: analysis, figures source, tests, report, this document,5figures andEXPERIMENT_LOG. New model-derived arrays, manifests/identity anchors, test/resource evidence and process ledger remain local-only in `mujoco/reports/uav_attitude_thrust_response_parts/`. Previous telemetry, reports, checkpoints and87historical untracked files are preserved.

```bash
env PYTHONPATH=mujoco/rl .venv/bin/python -m unittest test_uav_attitude_thrust_response test_uav_pi_internal_telemetry test_uav_gust_recovery_audit test_velocity_command_controller -q
# Re-analysis of saved states only, zero time integration; refreshes NEW audit report.
.venv/bin/python mujoco/rl/uav_attitude_thrust_response.py --finalize
```

Raw input and derived local artifacts are required for re-analysis; a clean checkout without them cannot reproduce raw checks. The audit is deliberately limited to this compiled stateless model, not a generic arbitrary-MuJoCo reconstruction API.
