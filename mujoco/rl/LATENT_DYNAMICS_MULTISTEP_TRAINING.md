# Multi-step latent dynamics training

Scope: a supervised dynamics-only experiment, not policy training/planning.
Base: `feat/latent-multistep-eval` at
`6cd1d3ab01a54ab7304ca52a5c6b92d4ac53ce35`.

## Frozen controls

Exact `pi_hidden_state_seed0` source trajectories, target-disjoint splits and
saved Train-only statistics are reused. There is no resampling or statistics
fitting. Training reads only Train/Validation timestep values. Test is opened
after validation checkpoint selection. Dataset/model/controller/report hashes
are audited and frozen reference metrics are recomputed, not just copied.

Architecture is the existing 23,815-parameter Full-History model:
`GRU(11,64,1)` followed by `Linear(68,64)-Tanh-Linear(64,64)-Tanh-Linear(64,7)`.
Encoder input is normalized `[observation7,previous executed normalized action4]`.
The head receives `[latent64,current normalized action4]`. PI state is never
read by this model or its objective. Random initialization uses seed0 with the
same parameter hash as the original History model; no fine-tuning.

## Window/graph protocol

For N usable transitions in an episode, every start in `range(N-10+1)` is used.
There are 141,339 Train windows in588 episodes/84 target groups and 29,542
Validation windows in126 episodes/18 groups. The unavailable terminal next
observation is excluded exactly as in the existing dataset adapter.

Each minibatch consists of16 complete episodes in the original
`default_rng(0).permutation(episode_count)` epoch schedule. All legal windows in
that minibatch have equal loss weight; padding only belongs to prefix encoding
and never contributes a window or loss. No window crosses an episode.

The unidirectional GRU encodes the true episode prefix from its zero initial
hidden state. Gathered `z_t` contains the start input exactly once and no future.
This computation remains differentiable but has no prefix loss. First head
call uses `z_t,a_t`. For subsequent steps, predicted physical observation is
re-normalized with saved Train statistics and concatenated with recorded
previous action. GRU hidden state is carried through the ten-step horizon.
Future ground truth is used only in the loss. **No teacher forcing or detach**
occurs inside the prediction horizon; later losses backpropagate through earlier
predictions, hidden states and prefix encoding.

The delta head still outputs in the original normalized-delta coordinate
system: `delta_physical = output * saved_delta_std + saved_delta_mean`.
The prediction is accumulated in physical units before the next input is
normalized. The sole loss is

```
mean_over_windows,k=1..10,dimensions=1..7 (
    (predicted_observation[t+k] - recorded_observation[t+k]) / train_obs_std
)^2
```

All horizons are equally weighted. This prescribed loss changes both horizon
length and scaling relative to the original delta-normalized one-step objective;
the experiment cannot isolate their individual contributions.

## Optimization/selection

CPU, Python/NumPy/Torch seed0, deterministic Torch operations, one Torch thread.
Adam defaults (`betas=.9,.999`, `eps=1e-8`), lr=.001,16 episodes per batch,
60 epochs as before. No gradient clipping, new scheduler, auxiliary objective,
or early stopping. All legal validation windows are evaluated after each epoch.
Only the minimum Validation ten-step observation-normalized MSE checkpoint is
saved. Test never influences selection. Per-epoch online Train loss, Validation
loss, gradient norm, hidden norm and physical prediction maximum are recorded;
non-finite prediction/hidden/gradient/loss stops rather than being clipped.

## Frozen evaluation

Reuse `latent_multistep_start_points_seed0.json` exactly, H1/5/10/25/50 with
32296/31792/31162/29272/26122 legal Test starts. Recorded future actions, exact
true-prefix warm-up and predicted observations thereafter. Endpoint RMSE is
scaled by Train observation std (not Train delta std). Physical per-dimension
RMSE/MAE, horizontal/vertical velocity errors, starting distance/PI-region groups,
and a common26122-start cohort are retained. The PI-State reference alone receives
teacher-forced recorded PI input and is **not** a full-state Oracle.

H1/5/10 are within the training horizon; H25/50 are extrapolation beyond it.
Latent drift compares true-history vs autoregressive encoder states at
`z_(t+H-1)`. Different trained models have different latent coordinates/scales;
L2/norm cross-model differences are descriptive, not physical state errors.
Selected plots reuse the prior fixed example, not a favorable newly chosen one.

## Commands/artifacts

```
.venv/bin/python mujoco/rl/run_latent_multistep_training.py
.venv/bin/python mujoco/rl/run_latent_multistep_training.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_*latent_multistep_training.py' -v
```

The runner refuses to overwrite an existing experiment/model. Only one full
experiment was run; small temporary synthetic test fixtures are not experiments.

- Best model: `mujoco/rl/models/history_latent_gru_multistep10.pt`
- Report: `mujoco/reports/latent_dynamics_multistep_training_seed0.json`
- Window manifests: `mujoco/reports/latent_multistep10_{train,val}_windows_seed0.json`
- Seven comparison/loss figures are listed in the report.
- Local-only ignored log: `.superpowers/latent-multistep-training/training.log`.

See the report and `docs/EXPERIMENT_LOG.md` for measured results and conclusion.
Limits: single seed, nominal recorded trajectories, overlapping windows, .4s
training horizon, at most2s evaluation, no policy/planning/robustness claim.
No next-stage training or World Model integration is performed here.

## Measured result (seed0)

Validation-best epoch57/60; best Validation MSE .0042489122, best-checkpoint
Train MSE .0021357584. Training took401.32s (full run/evaluation430.82s).

| Horizon | One-step-trained RMSE | Multi-step-trained RMSE | Change |
|---|---:|---:|---:|
| 1 / .04s | .02142777 | .02224521 | +3.81% |
| 5 / .20s | .06049126 | .06193697 | +2.39% |
| 10 / .40s | .09831910 | .09760580 | −.73% |
| 25 / 1s | .19567178 | .17837163 | −8.84% |
| 50 / 2s | .27962966 | .24231579 | −13.34% |

The identical26122-start common cohort gives the same trend. At2s, horizontal
velocity RMSE falls .175199→.143203m/s, vz .054442→.046744m/s. However, yaw
RMSE rises .030611→.067869rad (+121.72%); overall improvement is not uniform
physical-dimension improvement. Starting-distance regions all improve in
aggregate at2s, including<.1m .072944→.048713 (only80 legal windows). Large-PI
normalized RMSE falls .428217→.373594. One-to50-step absolute RMSE growth falls
.258202→.220071; substantial compounding error remains.

At2s, latent drift L2 .259558→.228146 and cosine .983603→.987607 improve
descriptively, with the cross-model coordinate-system caveat above. No NaN/Inf;
new Test prediction absolute max6.19463, latent norm max4.63950. Across training,
max gradient norm .40748, max hidden norm5.10656; no new clipping needed.

Conclusion: partially supports improved long-horizon consistency and beyond-K
generalization, at the cost of small one-step/5-step regression and material yaw
regression. It does not establish complete dynamics, causal attribution solely
to horizon length, robust generalization, or planning readiness. A separately
scoped latent-transition formulation is a possible next experiment, not executed.

Verification:13 new tests pass; repository288 passed/0 failed/1 optional GUI skip
(289 total). `--verify` independently recomputes Validation-best selection and
all five Test metric trees, frozen artifacts, exact manifests and normalization.
