# Frozen Multi-Step Latent Dynamics Evaluation — seed 0

Research question: Does the learned recurrent latent state remain useful during frozen multi-step autoregressive UAV dynamics prediction?

Branch `feat/latent-multistep-eval`, based on `feat/latent-memory-ablation` commit `9913006fe4f96a8be75a3bbc3a320be2744d0021`. No model training, simulator sampling, controller/reward change, or policy rollout is performed.

## Frozen inputs

Exactly reuse `pi_hidden_state_seed0` Test split: 18 held-out target groups, 126 complete episodes, 32422 original observations and 32296 usable adjacent transitions. Train/Validation timestep values are not loaded. The unavailable terminal next-observation transition stays excluded, as in the previous experiment. No split or dataset changes.

Four saved models are frozen (`eval`, `requires_grad=False`), with no optimizer:

| Model | Existing checkpoint | SHA256 |
|---|---|---|
| Markov MLP | `markov_dynamics_mlp.pt` | `81af03aa1bf5148a5504d2151623df77eecb12cca6d520dbc4e9db3e9a3ca5c1` |
| No-Memory GRU | `history_latent_gru_no_memory.pt` | `ca4cbf9c8f3e4309569659f3239250ec6fa70eb67bc1d45a255b8f8065b528d1` |
| Full-History GRU | `history_latent_gru.pt` | `76a7ed89851c3a10e01b96d19bab1ac4c421c27007937aebbc98951ac1cc65cf` |
| PI-State Baseline | legacy filename `oracle_dynamics_mlp.pt` | `a8324da347e47e852d2781c0caf99b6c27056fb03c37d356854850a4275deb6f` |

The last model is **not a full-state oracle**. It additionally receives only PI integral acceleration, not complete attitude, angular rates or motor dynamics. Architecture and saved Train-only normalization are unchanged. Report contains model/data/statistics/parent-artifact hashes and the original PI-magnitude thresholds.

## Autoregressive protocol and warm-up

Each window begins at a real Test observation `o_t`. Future commands are the recorded **executed normalized actions** `a_t,a_(t+1),...`, not actions recomputed from predicted observations. Subsequent states are exclusively `o_hat_(t+k+1)=o_hat_(t+k)+inverse_normalize(delta_hat_(t+k))`. Physical predictions are normalized again with the same saved Train stats before every model call. There is no prediction clipping or yaw wrapping beyond the original raw observation/delta convention.

- Markov: current predicted observation plus current recorded action.
- No-Memory: current predicted observation plus previous recorded action; zero GRU state at every step; current action enters the head.
- Full-History: encode the real episode prefix `x_0..x_t`, where `x=[o,previous_action]`, to obtain `z_t`. The first head uses `z_t` exactly once. Thereafter, the GRU sees only predicted observations and recorded previous actions. The single-layer unidirectional encoder permits caching all teacher-forced `z_t`; selecting `z_t` is identical to encoding only its prefix, with no future information in the selected state.
- PI-State Baseline: predicted observation plus **teacher-forced true PI state** at every future step and current action. It is a privileged reference, not an autonomous dynamics model.

All legal starts are enumerated: for episode with N usable transitions, horizon H uses `t in range(max(0,N-H+1))`; endpoint truth is `next_obs[t+H-1]`. No cross-episode windows. A compact, lossless manifest records these ranges, episode identity and hash. No model-performance-based window selection.

Main metric is endpoint normalized **observation** RMSE:

`sqrt(mean_windows,dimensions(((o_hat_(t+H)-o_(t+H))/Train_obs_std)^2))`.

It is not average error inside a window, nor normalized delta RMSE. Previous one-step normalization used **delta std**, so its normalized numbers are intentionally different. Physical per-dimension RMSE/MAE are also preserved at every horizon. A secondary curve uses the identical 26122 starts eligible for all horizons, controlling changing window composition.

## Main results

| Steps / time | Windows | Markov | No-Memory | Full-History | PI-State |
|---|---:|---:|---:|---:|---:|
| 1 / .04s | 32296 | .033368 | .033196 | .021428 | .029523 |
| 5 / .20s | 31792 | .136742 | .134917 | .060491 | .110510 |
| 10 / .40s | 31162 | .256472 | .251416 | .098319 | .196031 |
| 25 / 1.00s | 29272 | .547800 | .531941 | .195672 | .376339 |
| 50 / 2.00s | 26122 | .782929 | .757752 | .279630 | .517573 |

All four accumulate error. Full-History remains best at every horizon, including the common-window comparison. Relative normalized RMSE reduction versus No-Memory is 35.45%,55.16%,60.89%,63.22%,63.10%; versus Markov 35.78%,55.76%,61.66%,64.28%,64.28%. The PI-State reference falls between History and the memory-free models in the overall metric, but has the lowest vz RMSE at 10/25/50 steps.

Common 26122-window normalized RMSE curves:

| Model | H1 | H5 | H10 | H25 | H50 |
|---|---:|---:|---:|---:|---:|
| Markov | .034504 | .141234 | .264142 | .559277 | .782929 |
| No-Memory | .034299 | .139109 | .258406 | .542264 | .757752 |
| Full-History | .022238 | .062766 | .101636 | .197576 | .279630 |
| PI-State | .030549 | .114190 | .201811 | .382701 | .517573 |

Horizontal velocity RMSE is `sqrt(mean((vx_error^2+vy_error^2)/2))`, in m/s:

| Steps | Markov | No-Memory | Full-History | PI-State |
|---|---:|---:|---:|---:|
| 1 | .019218 | .018860 | .007206 | .015825 |
| 5 | .090520 | .088715 | .026037 | .070317 |
| 10 | .175718 | .171403 | .050945 | .130409 |
| 25 | .380249 | .368149 | .123199 | .254011 |
| 50 | .515912 | .497372 | .175199 | .329847 |

Vertical velocity RMSE, in m/s:

| Steps | Markov | No-Memory | Full-History | PI-State |
|---|---:|---:|---:|---:|
| 1 | .011332 | .011543 | .010813 | .011116 |
| 5 | .029576 | .030256 | .027531 | .028013 |
| 10 | .041125 | .042320 | .038444 | .038029 |
| 25 | .053327 | .054473 | .049826 | .048568 |
| 50 | .059645 | .060100 | .054442 | .052388 |

50-step physical endpoint RMSE (m for errors xyz, m/s for velocities, rad for yaw):

| Dimension | Markov | No-Memory | Full-History | PI-State |
|---|---:|---:|---:|---:|
| error_x | .639607 | .606242 | .186380 | .404508 |
| error_y | .595680 | .590690 | .208829 | .378719 |
| error_z | .068470 | .071591 | .063869 | .058940 |
| vx | .538556 | .506978 | .163919 | .336470 |
| vy | .492227 | .487578 | .185795 | .323088 |
| vz | .059645 | .060100 | .054442 | .052388 |
| yaw_error | .145615 | .148298 | .030611 | .144739 |

Full per-dimension physical RMSE and MAE for **every** horizon and model are under `models.<kind>.horizons.<H>.physical` in the JSON report. Overall physical metrics mix units; normalized metrics are primary.

## Starting-state regions

50-step normalized RMSE, grouped by true state at the window start:

| Region | Windows | Markov | No-Memory | Full-History | PI-State |
|---|---:|---:|---:|---:|---:|
| distance<.1m | 80 | .434934 | .440057 | .072944 | .155229 |
| .1–.2m | 727 | .363853 | .376248 | .148263 | .243603 |
| .2–.5m | 5790 | .438317 | .418176 | .139454 | .251371 |
| >=.5m | 19525 | .870293 | .842776 | .313057 | .580816 |
| Small PI | 7790 | .228986 | .245720 | .082338 | .104570 |
| Medium PI | 9144 | .782196 | .753644 | .182615 | .384824 |
| Large PI | 9188 | 1.043730 | 1.007968 | .428217 | .777787 |

PI norm thresholds remain Train tertiles .0507170173 and .2328452269m/s². PI is used for grouping, never supplied to History/No-Memory/Markov. Full-History advantage persists in every distance and magnitude region, but near-target 50-step coverage is only80 correlated windows, so do not treat it as broad statistical generalization.

## Latent drift and numerical stability

Compare autoregressive latent to teacher-forced encoder latent on the same recorded episode. The endpoint-predicting latent is `z_(t+H-1)`, not an extra latent computed after the endpoint.

| H | Mean L2 drift | Mean cosine |
|---|---:|---:|
| 1 | 0 | 1.000000 |
| 5 | .020676 | .999705 |
| 10 | .052793 | .998786 |
| 25 | .157582 | .993030 |
| 50 | .259558 | .983603 |

Drift grows while latent norms remain bounded: global autoregressive norm max4.03753. At H50, mean norm1.68635 versus teacher-forced1.70253. No NaN/Inf in predictions/latents, and no observed numerical explosion. Maximum absolute predicted observation across all evaluated steps: Markov6.41084, No-Memory6.17569, Full-History6.30670, PI-State5.86928 (mixed physical units).

Teacher-forced delta sanity exactly reproduces the previous four models: physical RMSE .011240/.011092/.005649/.009594 and delta-standardized RMSE .605748/.598015/.247433/.540332 in Markov/No-Memory/History/PI order. Warmed H1 autoregressive error matches the same teacher-forced observation-scaled result.

## Interpretation and limitations

The result supports useful recurrent predictive state beyond one step: advantages persist through .2/.4/1/2s and are strongest for horizontal velocity and yaw. It most closely matches case A in the experiment brief, with remaining compounding error. It does not prove that the latent is PI state, complete physical state, or reliable for control/planning.

Limits: nominal simulation; one dataset/training seed; overlapping windows are correlated; saved future actions rather than closed-loop policy; privileged teacher-forced PI reference; one-step-trained models only; horizons at most2s; arbitrary window starts have real prefix warm-up. The deterministic example is first eligible episode0/target7/start10, not cherry-picked, and some individual dimensions are worse even though aggregate History performance improves.

Next recommendation only: a separately specified multi-step dynamics training experiment is justified to reduce compounding error. No multi-step training, World Model, policy training or planning is started here.

## Reproduction and artifacts

```bash
cd ~/MineUAV-Learning
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*latent*multistep*.py' -v
.venv/bin/python mujoco/rl/run_latent_dynamics_multistep_eval.py --verify
```

Initial evaluation uses the same script without `--verify`; existing report/manifest paths are protected against overwrite. Read-only verify reloads frozen artifacts and re-evaluates every main/common/region/velocity/stability/drift metric. It never fits any model or statistics.

Report: `mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json`.
Start manifest: `mujoco/reports/latent_multistep_start_points_seed0.json` (compact ranges, all exact identities).
Figures under `mujoco/reports/`: `multistep_normalized_rmse_vs_horizon.png`, `multistep_velocity_rmse_vs_horizon.png`, `multistep_error_vs_target_distance.png`, `multistep_error_vs_pi_magnitude.png`, `selected_autoregressive_rollout_prediction.png`, `latent_drift_vs_horizon.png`.

No raw prediction arrays/duplicate dataset/new weights are saved. Evaluation/test logs and caches remain ignored/local. The first run encountered a plotting-function/boolean name collision after numerical evaluation; a default-plot-path regression reproduced it, the naming bug was fixed, and identical frozen evaluation completed. No predictor/controller/normalization changes were made to obtain these results.

Verification: 16 focused tests pass, including real-prefix warm-up, first-future-truth counterfactual, recurrence versus zero-memory, recorded PI forcing, endpoint normalization, episode boundaries, A→B→A reset, frozen parent hashes, reload/recompute and manifest tampering. Final tracked repository regression plus new tests: 276 run, 275 passed, zero failed, one optional desktop GUI skip. Read-only review found no blocking issues. All four models' main/common/region/velocity/stability/drift metrics were reproduced by `--verify`.
