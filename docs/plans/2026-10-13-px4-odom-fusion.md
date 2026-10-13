# Phase 3C execution plan

Spec: user-supplied PX4–MuJoCo Phase 3C protocol, base `f202d9d2985312e80cf180a2f239077e3734aa81`.

1. Freeze preflight identities and run 356 Bridge + 30 policy tests. Confirm PX4 EV/GNSS bit meanings, ULog topics and receiver timestamp behavior from the pinned source.
2. RED→GREEN: fix relative-scale covariance symmetry; add 50 Hz COM-based ground-truth odometry production and strict source-clock checks. Preserve Phase 3A adapter and frame transforms.
3. RED→GREEN: add disarmed-only runner wiring, explicit EV=5/GPS=7/height=1 startup parameters, send accounting, actual-parameter/ULog fusion-evidence analysis. Keep existing safety watchdogs unchanged.
4. Run full offline regression; independent read-only safety/code review; resolve Important/Critical findings offline and rerun tests.
5. Only if all gates pass, run one new empty-runtime, unarmed SITL test for at most 20 simulated seconds. No retry. Analyze immutable raw summary, console and ULog; no inferred fusion PASS.
6. Produce report, figures, documentation and experiment log; verify tests, diff, identities and repository status; commit/push the feature branch and stop.

Rulings: the user explicitly authorized direct execution and exactly one disarmed SITL run, so no intermediate approval gate is required. Any identity, safety or evidence precondition failure blocks that run rather than being bypassed. Historical proof is not reused to claim a flight pass after runtime code changes.

Execution record: historical runner edits were reverted because they invalidated archived runtime-proof tests. Scoped single-process hooks reuse the unchanged safety loop; the new summary kind is rejected by old flight gates. Independent review issues were closed before the one run. The sole run yielded actual EV/GNSS fusion, missing direct visual-odometry logging due PX4 subscription capacity, and a4ms floating-loop stop overshoot. The new runner stop boundary was then fixed offline; no second run. Final371 Bridge tests and30 inherited Shadow tests pass. Complete fusion-interface acceptance remains BLOCKED and duration compliance FAIL; final runtime SITL validation NOT TESTED.
