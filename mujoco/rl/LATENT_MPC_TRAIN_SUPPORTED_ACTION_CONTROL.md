# Paired Train-supported action constraint control

Branch: `feat/latent-mpc-action-support`. Base:
`7d360778a0e401a091c77811a78968805e813154` from
`feat/latent-random-shooting-mpc`; no main merge.

## Question and immutable protocol

Does a hard Train central95% per-dimension action box improve prediction validity
and closed-loop control? Frozen canonical seed0 v3 checkpoint, E/T/D eval and
requires_grad=False. No training/optimizer/adaptation/correction or policy.

Original random-shooting implementation remains the unconstrained baseline.
The only core change is an overridable candidate-generation hook; its default
calls the original sampler. Supported overrides that hook, not latent recursion,
cost, argmin, action execution, environment, PI, success or timeout.

- N512, H10 (0.4s), original Train statistics and model; previous executed action
  seeds the same Gaussian random walk, std0.5×archived Train action std.
- Candidate0 allzero; candidate1 repeats previous command. Other510 eachstep
  `legal=clip(previous+epsilon,envbounds)`. Supported additionally uses
  `action=clip(legal,train_q025,train_q975)` before the next increment.
  Supported anchors also project; no new heuristic candidate.
- Full original `train.npz` executed actions provide the support quantiles,
  including terminal commands. The model's original noise normalization excludes
  unpaired terminal commands; its mean/std is intentionally **not recomputed**.
  No Val/Test/benchmark/holdout action enters support computation.
- Box is a **Train central-95% support proxy**, not a complete in-distribution
  manifold or strict OOD boundary. Selected-action departure uses1e-7 tolerance.
- Actions normalized[-1,1], physical scales[1.5,1.5,1,1]. Smoothness uses normalized
  commands. Terminal error/velocity/yaw are physical observation components.
- `J=||error_H||² + .5||velocity_H||² + .1yaw_H² + .05mean||Δaction||²` unchanged,
  hand-designed planning cost, not RLreward. Pure T recursion, terminal D only;
  decoded observations never feed E. Argmin, first action only, real E nextstep.
- Same nominal RewardV2 PI task, success distance<.1m AND speed<.15m/s held5steps,
  timeout15s, unchanged physical failure conditions. Archived100benchmark and
  100holdout coordinates/reset seeds/order for both conditions.

## Pairing and first decision

Use the original episode seed:
`SeedSequence([0,split_index,episode_index]).generate_state(1)[0]`. Each planner
decision draws exactly one510×10×4 Gaussian tensor. Instrumentation consumes
no RNG. Matching episode/decision indices therefore have identical raw scaled
epsilon, verified by SHA256; no new RNG definition changes the baseline.

Both conditions run afresh; scientific unconstrained summaries must exactly
reproduce the archived experiment. At decision0, true observation, encoder latent
and raw epsilon hashes must match for every target pair. First selected action,
cost, candidate index, support status and standardized norm are recorded. Later
states and previous actions diverge: this is noise-source pairing, not a claim
of identical-state comparisons after decision0.

## Diagnostics and bounds on interpretation

All400episodes remain in success/final-distance/failure/timeout metrics.
Completion time success-only, N/A if none. Near-target speed uses true pre-action
distance<.1m; missing visits N/A. Crossing and selected-command-element saturation
reuse the previous definitions; neither is rotor saturation.

Projection frequency counts candidate timesteps changed beyond1e-7; magnitudes
are normalized action L2 differences AFTER env clipping, BEFORE support clipping.
Report exact mean/counts and empirical-rank median/p95 upper estimates in fixed
1e-4 bins, maximum quantile error1e-4. Conditional projected-only statistics are
separate. Anchor projections, boundary element mass and unique first-action
fraction expose constraint-induced candidate pileup; no raw candidate arrays.

Prediction windows use first3fixed episode identities per split regardless of
outcome, legal10-step starts stride25. Forecast from saved decision latent using
the **actually executed next10actions**; true future observations only targets.
This is not accuracy of original selected future candidates after replanning.
Both rule-defined and common-available episode/decision windows are reported:
termination lengths may differ. Ordinary wrapped-yaw RMSE retains branch-cut
limitations. Initial target-response correlation is descriptive, not proof of
model causality or a deployable planning method.

CPU oneTorchthread, all512candidates batched. Planning includes sampling,
projection/epsilonhash, T, terminalD, cost and argmin. Full decision additionally
includes causal E and read-only diagnostic aggregation. Excludes simulation/disk
writes/figures. No hard real-time guarantee. Caches bind full executable and
environment/model/data/target source hashes plus original hardware/runtime;
changed identities are rejected, not relabelled.

## Reproduction

```bash
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_latent_mpc*.py'
.venv/bin/python mujoco/rl/run_latent_mpc_action_support.py
.venv/bin/python mujoco/rl/latent_mpc_support_verification.py
```

Original local Train archive and archived target-source traces required, read
only. Report commits complete target/window manifests and hashes; local parts
contain four completed condition fragments, fixed12episode traces, four small
aggregate arrays/noise hashes/projection histograms. No raw candidate tensors,
dataset copies, models, new checkpoints or training logs are committed.

## Results (2026-10-08)

Support archive147219executed commands; normalized lower/upper:

| Axis | Lower2.5% | Upper97.5% |
|---|---:|---:|
| vx command | -0.8820207834 | 0.8821067661 |
| vy command | -0.8766989708 | 0.8859273791 |
| vz command | -0.8880657554 | 0.8921637535 |
| yaw-rate command | -0.9250337780 | 0.9251185060 |

MetadataSHA256`fd137e5a27b98d831fafab39216579b641a2749f38d7e7d108f84956ac61dbde`.
CheckpointSHA256`42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9`;
all three parameter hashes exactly unchanged, no optimizer. All scientific
Unconstrained summaries exactly reproduce the prior experiment.

| Metric | UN benchmark | Supported benchmark | UN holdout | Supported holdout |
|---|---:|---:|---:|---:|
| Success /100 | 0 | 0 | 1 | 0 |
| Mean final distance,m | 4.2033 | 3.9883 | 4.3860 | 4.5357 |
| Physical failures | 78 | 61 | 75 | 69 |
| Timeouts | 22 | 39 | 24 | 31 |
| Near-target actual speed,m/s | .8176 | .7955 | .4076 | .4474 |
| Near-target command speed,m/s | 1.7603 | 1.5792 | 1.6540 | 1.5480 |
| Crossing fraction | .11 | .18 | .09 | .07 |
| Command-element saturation | 39.15% | 0% | 40.24% | 0% |

Supported has no successful completion times; Unconstrained1.24s represents only
one success. Total physical failures153→130, excessive tilt72→38, outside area
81→92, timeouts46→70; no success improvement. Zero saturation is mechanical:
all support bounds are below.95, not evidence of successful braking.

Selected-action support departure85.98%→0%, mean standardized norm3.4447→3.0751.
Small difference from prior85.96% is the explicit full-Train rather than paired-
row support statistics; baseline actions/physics unchanged. Supported candidate
timestep projection56.20/56.62%, magnitude mean/median/p95
.05365/.0346/.1443 and .05426/.0369/.1444; projected-only mean≈.0955/.0958.
Anchor projections0(actual previous actions already inside box). Candidate
boundary-element mass19.03/19.28%; selected boundary mass32.49/33.70%; unique
first-action fraction99.44/99.37%. Some boundary pileup but not global collapse.

Fixed identity prediction windows UN/S:26/32benchmark,11/37holdout. Normalized
H1/H10 RMSE .2224/1.2212→.2839/.4922benchmark; .2508/.4107→.7207/1.2050holdout.
For common-available identities21benchmark and11holdout, respectively:
.1835/.4276→.2355/.5148; .2508/.4107→.2069/.3011. Mixed, not consistent prediction
improvement. Common windows also have different visited states/actions; legal
termination-based availability and yaw wrap limit attribution. All106windows
independently reproduced, perturbed future observations leave predictions equal.

200clean first-decision pairs share initial real observation/latent/epsilon;
38083common decision-index epsilon hashes match across the whole episodes.
First action changes15.5%of pairs, mean command L2 difference.07432. Both initial
selected commands have0support departure. Mean selected cost1.10219→1.11350,
standardized norm1.0020→1.0161. Target xyz correlations .1818/.0508/.2842→
.1904/.0731/.3476 remain weak. Future candidate projection can change ranking
even when the first selected action itself is already inside the box.

CPU oneTorchthread,44229UN and50556Supported decisions. Planning mean/median/p95/
max6.356/6.362/6.436/12.158→6.613/6.611/6.671/12.411ms. Full decision6.742/6.750/
6.832/12.527→7.289/7.284/7.404/13.136ms. Mean overhead4.04%planning/8.12%full
decision (includes diagnostics); all measured decisions<40ms, not a guarantee.

Conclusion: **Negative task-control result.** This hard support proxy removes
its defined action departure and reduces some aggressive failure modes, but
does not restore successful control or consistent prediction validity. It does
not provide specific causal support for marginal action OOD/model exploitation
as a sufficient explanation. State/action-sequence support, decision-cost
fidelity and short horizon remain unseparated; no more MPC parameter tuning.

One next recommendation only: offline action-sequence ranking/decision-cost
fidelity benchmark, comparing predicted versus actual simulated costs under
identical executed sequences with the same frozen model/cost. **Not executed.**

Tests:16new/31related pass; full465run,461passed,0failed,4known optional/legacy
skips using existing read-only Git fixtures. No experimental training/model/task
modification. Eight figures visually inspected; immutable sources/model hashes
and complete400target cohorts independently verified.

Final review identified an omitted executing Transition/Decoder module and
checkpoint-loader dependency in cache identity. A RED→GREEN source-binding
regression fixes this; previous fragments remain local as a recoverable backup.
The identical400episode protocol was rerun, rather than relabelling old caches.
All scientific summaries/actions/costs/noise/projection histograms and all106
prefix predictions are bit-identical; only freshly measured latency differs.

Deferred diagnostic minors: projected-only histogram population uses >0 while
projection counts use >1e-7 (current median/p95 unchanged; tolerance-edge
population inconsistency); verifier covers cohorts/support/hash/noise/prefixes,
not every derived projection/common-window/pooled/first-decision summary.
The fresh reviewer independently recomputed those current derived values.
