# Objective-matched one-step control

Branch: `feat/latent-objective-matched-control`.
Base: `feat/latent-multistep-training`, commit
`63545a60396bbbc9421d87f1f28548104739cfbd`.

Question: with the same loss family/scaling, architecture, initialization and
optimizer, does K10 autoregressive training improve long-horizon prediction
over K1? Only the new matched K1 is trained. Existing K10 and historical
delta-loss K1 models remain immutable; no policy or simulator training occurs.

## Matched controls

The same 23,815-parameter `GRU(11,64,1)` and
`Linear(68,64)-Tanh-Linear(64,64)-Tanh-Linear(64,7)` are used. Encoder inputs are
`[observation7, previous executed normalized action4]`; the head takes
`[latent64,current action4]`. No true PI input or auxiliary loss.

Random initialization (not fine-tuning), Python/NumPy/Torch seed0, one CPU
thread, deterministic Torch operations. Initial parameter hash must match K10:
`6e2ac94b53edfecb8794e23f3224c789add1d1ab81d4cea1a8678a7e9d158cf8`.

Adam lr=.001, default betas/epsilon,16 episodes per batch,60 epochs. The exact
K10 `default_rng(0)` episode-order rule and37 optimizer updates per epoch are
reused. No new clipping, scheduler, early stopping or hyperparameter search.

Dataset `pi_hidden_state_seed0` is not regenerated:84/18/18 target groups and
588/126/126 episodes in Train/Val/Test. All saved Train-only normalization
statistics are reused byte-for-byte in meaning (verified canonical hash).
Test timestep values are not opened until Validation selection is finished.

## Exact start matching and recurrent warm-up

K1 intentionally uses the exact **K10 Train/Val start identities**:
`t in range(N-10+1)`, rather than adding nine terminal-tail starts per episode.
There are141339 Train and29542 Val starts, matching K10. Thus both conditions
have the same starting examples, minibatch weighting and optimizer update count.
Compact manifests retain original episode identity/length/source-row provenance
and reference the existing K10 manifest byte/semantic hashes.

The shared trainer receives views ending just after each episode's last used
start; no original array/file is changed. Every used unidirectional prefix
`x_0..x_t` is identical to K10. Unused future prefix values cannot affect `z_t`.
Tests compare the K1 predictions with the first K10 predictions at all matched
starts. Episodes have zero initial hidden state and no inter-episode memory.
Prefix encoding remains differentiable, but has no separate prefix loss.

K1 predicts physical `o_hat[t+1]` from the real start and recorded action. The
sole loss is

```
mean_over_windows,dimensions (
    (predicted_next_observation - recorded_next_observation) / train_obs_std
) ** 2
```

This is precisely K10's uniform observation-normalized loss family with only
the horizon reduced to1. The head still inverse-transforms its delta output
using the same saved delta mean/std: output parameterization is **not** loss
weighting. K1 does not return to the historical delta-std loss.

The minimum Validation K1 MSE checkpoint is selected after the full60epochs.
Test is neither a training input nor a selection metric. K10 uses its already
selected K10 Validation-best model; this horizon-specific selection rule is
prescribed, not Test tuning.

## Frozen evaluation and yaw audit

Reuse the existing Test start manifest without changing any starts: H1/5/10/
25/50 have32296/31792/31162/29272/26122 windows. True-prefix warm-up, fixed
recorded future actions, autoregressive predicted observations and saved
normalization match the previous evaluator exactly. The old one-step and K10
metric trees are independently reproduced, not merely copied into the report.
The common26122-start cohort, physical dimension RMSE/MAE, distance/PI regions,
latent drift and finite-value checks are also retained.

The environment's yaw observation is
`atan2(sin(target_yaw-actual_yaw),cos(target_yaw-actual_yaw))`, within[-pi,pi].
Dataset delta and prediction errors are raw coordinate differences, as before.
Neither predictions nor residuals receive new circular wrapping or clipping.
The audit records adjacent wrap-jump counts and yaw range for each split; it
does not modify the representation. Models share exactly this convention.

## Reproduction and artifacts

```
.venv/bin/python mujoco/rl/run_latent_objective_matched_control.py
.venv/bin/python mujoco/rl/run_latent_objective_matched_control.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_latent_objective_matched_control.py' -v
```

The training command refuses to overwrite existing control artifacts. The
verification command is read-only: hashes, initialization, manifests,
Validation-best loss, frozen and new Test results are independently checked.
Tests use temporary synthetic datasets, not extra experiments on the real data.

- Best model: `mujoco/rl/models/history_latent_gru_matched_one_step.pt`.
- Report: `mujoco/reports/latent_dynamics_objective_matched_control_seed0.json`.
- Matched manifests: `latent_matched_k1_{train,val}_windows_seed0.json`.
- Three figures: `matched_k1_vs_k10_{rmse_vs_horizon,velocity_rmse,yaw_rmse}.png`.
- Ignored local log: `.superpowers/latent-objective-matched-control/training.log`.

## Interpretation limits

The matched K1-vs-K10 comparison removes the observation-vs-delta scaling
confound and matches starts as well. Training horizon still necessarily changes
future labels, BPTT length and optimization trajectory. Validation uses each
condition's own horizon. This is a single-seed nominal recorded-action dynamics
experiment, not planning/control performance or a uniform-per-dimension claim.

Historical delta-loss K1 also included terminal-tail starts. Therefore historical
K1-to-matched-K1 differences are not a pure scaling-only intervention, and the
historical13.34% aggregate gain cannot be decomposed into exact additive causal
shares. Explicitly report yaw/horizontal/vertical tradeoffs. A possible next
latent-transition experiment is only a recommendation, never executed here.

## Measured result (seed0)

Best epoch60/60. Best-checkpoint Train MSE=.0001587700; Validation K1
MSE=.0004516415 (normalized RMSE=.021252; matched Val starts only).
Training112.19s; full run including frozen evaluation138.57s. Initial hash and
23,815-parameter structure match K10; only the new K1 model was trained.

| Horizon | Historical delta K1 | Matched obs K1 | Obs K10 | K10 reduction vs matched K1 |
|---|---:|---:|---:|---:|
| 1 / .04s | .02142777 | .02116048 | .02224521 | −5.13% |
| 5 / .20s | .06049126 | .05950496 | .06193697 | −4.09% |
| 10 / .40s | .09831910 | .09741101 | .09760580 | −.20% |
| 25 / 1s | .19567178 | .19990681 | .17837163 | +10.77% |
| 50 / 2s | .27962966 | .29601142 | .24231579 | +18.14% |

Positive reduction denotes improvement. K10 improves long horizons even after
loss scaling is matched; it does not improve1/5/10step aggregate accuracy.
The identical26122-start cohort gives the same long-horizon conclusion. The
matched K1 scaling change did not reproduce the historical aggregate long-
horizon gain, so that gain cannot be explained mainly by this scaling change
under the present matched experiment. No exact causal-share percentage is
assigned to the old13.34% result.

At50steps, matched K1→K10 physical RMSE/MAE:

| Dimension | K1 RMSE | K10 RMSE | K1 MAE | K10 MAE |
|---|---:|---:|---:|---:|
| error_x (m) | .195625 | .176015 | .118390 | .099726 |
| error_y (m) | .219652 | .182686 | .148272 | .102278 |
| error_z (m) | .056994 | .061456 | .029540 | .032076 |
| vx (m/s) | .159342 | .136903 | .096241 | .074489 |
| vy (m/s) | .184192 | .149237 | .124376 | .076171 |
| vz (m/s) | .044854 | .046744 | .022473 | .022470 |
| yaw_error (rad) | .119938 | .067869 | .075142 | .041676 |

Horizontal-velocity RMSE .172216→.143203m/s improves16.85%, while vz
RMSE worsens4.21%. Yaw K10 improves43.41% versus matched K1, at every measured
horizon, but remains121.72% worse than historical delta-loss K1 at2s.
Matched K1 is itself291.82% worse than historical K1 in that coordinate. Thus
yaw regression cannot simply be attributed to the longer horizon: a loss-
family/dimension-weighting sensitivity is supported, with the historical
tail-coverage and single-seed caveats. No yaw change/fix is made here. All
three stored splits contain zero adjacent ±pi wrap jumps.

At2s, distance-region normalized RMSE K1→K10 is:
<.1m .087852→.048713 (80 windows), .1–.2m .146705→.125772 (727),
.2–.5m .156569→.105602 (5790), >=.5m .330342→.273222 (19525).
PI-small .098950→.049913; medium .223510→.159288; large .437147→.373594
(9188 large windows,14.54% improvement). Other horizons and every physical
dimension's RMSE/MAE are in the JSON; no region is silently excluded.

No NaN/Inf. Training max gradient norm .007306, max hidden norm5.056416;
Test max hidden4.468021 and physical observation absolute max6.328069.
No added clipping was needed. Independent reload reproduces selection,
Validation loss and every frozen/new Test metric tree.

Verification:7 new control tests pass; the complete tracked-branch suite has
295 passed,0 failed,1 optional desktop-X11 skip (296 total). The prior unrelated
untracked historical-branch tests are not imported as this branch's test suite.
The read-only `--verify` passes all8 verification checks. Three figures were
visually inspected. No algorithm-related failure remains.

Conclusion: matched-objective evidence supports a real training-horizon
benefit for1–2s aggregate prediction, with short-horizon/vertical tradeoffs.
A separately authorized latent-transition formulation is justified as the
next experiment, while retaining the unresolved multivariable loss-balance
issue. It is not started here.
