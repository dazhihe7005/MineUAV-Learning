# Multi-Step Latent Transition Training (seed0)

Branch `feat/latent-transition-multistep`, base `feat/explicit-latent-transition`
at `126a3dbe20a01a2fd591c8e019560d64ef5e33a8`. Only a fresh Transition is trained.
No Encoder/Decoder/PPO/policy update, resampling, joint model, planning or new loss.

## Frozen coordinate system

Encoder: GRU11→64(one layer) from `history_latent_gru_multistep10.pt`, only its
encoder module. Decoder: `latent_observation_decoder.pt`, Tanh MLP64→128→128→7.
Both eval/`requires_grad=False`, no gradients, never owned by the optimizer.
Before/after parameter hashes checked, including after evaluation:

- E: `25a95644f2a2a7b1fc69cd336bd13cfe771f9dca65ee2c9686f64106ccf71e75`
- D: `1e10b256a3f8f2879e6bcffb2e0db47b22f37804d0e8fa3bfb7516c0a0974bb4`

Teacher latent is an encoder representation, not physical ground truth. Frozen
E runs each real `[observation,previous executed action]` episode from zero
hidden. Include all N+1 saved frames for N action transitions; no episode joins.
Original NPZ data, target splits and teacher manifests are hash-verified. All
teacher arrays remain in memory; no large raw copies committed.

Load existing Train latent mean/std verbatim, do NOT refit. Hash
`10bf0867cf36eb8401e48d848dec6608657c418a33ac8f0df336841e81c666c6`;
floor1e-6, original Train-only observation/action statistics also unchanged.

## Objective and training

Same33600-parameter Tanh residual MLP68→128→128→64. Fresh seed0 initialization,
not fine-tuning. Expected initial parameter hash matches original one-step T:
`69ac557a7d00b221613977dc6c20ee186591a8c8e16273ccf2da34755a29685b`.

At window start use teacher z[t] once, then for k=0..9:

```
q_hat = (z_hat-Train latent mean)/Train latent std
delta_q_hat = T(q_hat,normalized recorded current action[t+k])
z_hat_next = z_hat + Train latent std * delta_q_hat
L = mean_windows,steps,dimensions(((z_hat[t+k+1]-teacher z[t+k+1])/latent_std)^2)
```

All10steps equal weight. No teacher latent re-injection, detach, observation,
decoder, PI, reward, cosine, auxiliary or regularization loss. E/D outside the
training graph. Physical latent carry float64, network inputs/parametersfloat32,
same normalization/inverse residual rule as prior frozen evaluator.

Adam defaults betas.9/.999 epsilon1e-8,lr.001,seed0,60epochs, no added gradient
clipping. Same `default_rng(0)` complete-episode permutations,16episodes/batch.
Every legal start in each episode batch contributes equally. Train141339 and
Validation29542 K10 windows (84/18 target groups,588/126episodes). Prediction
windows never cross an episode; saved teacher targets are read only for loss.
Validation uniform K10 normalized latent MSE selects best checkpoint. Test values
are first opened AFTER selection. Per-epoch losses/gradient norms/predicted norms
recorded. K10 omits last9K1starts per episode: this prescribed horizon comparison
is not perfectly sample-matched, despite identical architecture/initialization/
normalization/batching principles. No new K1 control is trained in this stage.

## Evaluation and Decoder Floor

Exact prior Test126episodes/18target groups and start manifest reused.
H1/5/10/25/50 at25Hz; starts32296/31792/31162/29272/26122. Primary plots use each
horizon's all-legal starts, identical across models at that horizon. The shared
26122-start cohort is retained separately in JSON to check cohort effects, not
used as the primary figure cohort. Initial latent comes from real causal
prefix, thereafter only predicted latent and recorded actions enter T. D reads
predicted latent, never feeds E/T. PI used only offline for grouping.

Teacher-Latent Decoder Floor is `D(teacher z[t+H])` against real o[t+H], at the
same starts as both transitions. It is an offline reconstruction reference,
NOT deployable/autonomous and NOT a mathematical lower bound. Decoder is nonlinear:
do not subtract RMSE values to claim isolated transition error.

Recompute unchanged one-step T and DirectK10 metrics against previous report,
then compare new T using SAME E/D/coordinates/actions/starts. Latent endpoint
normalized RMSE, physical latent MAE, cosine/norm and endpoint incremental delta
error; observation endpoint normalized RMSE, physical per-dimension RMSE/MAE,
horizontal/vertical velocity. Distance groups<.1/.1–.2/.2–.5/>=.5; PI bins reuse
Train thresholds .050717/.232845m/s². Yaw conventions remain unchanged.

Correctness tests cover10step residual recursion/action indexing/full BPTT,
future teacher deletion/corruption independence, boundaries, frozen hashes,
normalization, Validation selection/reload, deterministic repetition, Test
manifest reuse and exact Decoder Floor endpoints/counts. Real selected Test
window must also stay bit-identical after future teacher/observation removal.
Finite predictions and norms are checked; no clipping hides numerical failures.

## Reproduction and artifacts

```
.venv/bin/python mujoco/rl/run_latent_transition_multistep.py
.venv/bin/python mujoco/rl/run_latent_transition_multistep.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*latent_transition_multistep*.py' -v
```

Runner refuses overwrite. `--verify` is read-only: it checks immutable components,
saved normalization, teacher hashes, window identity, Validation-best selection
and recomputes all old/new/floor/Direct metrics without optimizer updates.

- New final model `mujoco/rl/models/latent_transition_multistep10_mlp.pt`.
- Report `mujoco/reports/latent_transition_multistep_training_seed0.json`.
- Train/Val manifests `latent_transition10_{train,val}_windows_seed0.json`.
- Eight required PNGs under `mujoco/reports/`, listed in report.
- Source NPZ/cache/teacher arrays/intermediate traces/logs stay local/ignored.

Limitations: single seed, nominal simulation, recorded future actions,
correlated overlapping windows, frozen representation/decoder, deterministic
transition and horizon-specific validation criterion; no control/planning or
physical state sufficiency claim. Suggestions in conclusions are not executed.

## Measured results

Completed all60epochs. Validation-best epoch57: saved-checkpoint Train K10 MSE
.0163215666, Validation K10 MSE .0182317400. Final epoch60 Validation MSE
.0184571922. No batch/clipping/optimizer change was required. New best-model
SHA256 `824bbc10bddb7903364bb1283506d09dd417224035c4ac3fa2466c16fd827afa`;
both E/D parameter hashes above remain bit-identical after training/evaluation.

All-legal-window endpoint metrics, paired across models at each horizon:

| H / time | Windows | One-Step latent RMSE | K10 latent RMSE | One-Step cosine | K10 cosine |
|---|---:|---:|---:|---:|---:|
| 1 / .04s | 32296 | .044726 | .066444 | .999391 | .998529 |
| 5 / .20s | 31792 | .125679 | .135875 | .996227 | .995418 |
| 10 / .40s | 31162 | .191303 | .179689 | .991637 | .992529 |
| 25 / 1s | 29272 | .334250 | .274156 | .974858 | .983307 |
| 50 / 2s | 26122 | .512454 | .367854 | .936980 | .969525 |

| H | One-Step T+D observation RMSE | K10 T+D observation RMSE | Teacher D (offline) | Direct K10 (reference) |
|---|---:|---:|---:|---:|
| 1 | .086475 | .084842 | .077850 | .022245 |
| 5 | .126038 | .116118 | .078132 | .061937 |
| 10 | .175715 | .153719 | .078703 | .097606 |
| 25 | .313664 | .254360 | .080508 | .178372 |
| 50 | .511910 | .355959 | .081897 | .242316 |

Both tables use the saved Train normalization. K10 latent RMSE worsens at H1
by .0217177 (+48.56%) and H5 by8.11%; improves at H10 by6.07%, H25 by17.98%
and H50 by28.22%. Observation RMSE improves at all five horizons, including
18.91%/30.46% at H25/H50. Thus gains extend beyond training horizon10, with a
substantial local latent-accuracy tradeoff. The common26122-start cohort also
preserves these tradeoffs: H1 latent .044839→.066836; H25 .333987→.274778;
H50 unchanged from table. Cohort changes do not explain the ordering.

At H50 physical RMSE/MAE:

| Dimension | One-Step RMSE | K10 RMSE | K10 MAE |
|---|---:|---:|---:|
| target_error_x (m) | .483690 | .292601 | .199205 |
| target_error_y (m) | .843514 | .377395 | .239606 |
| target_error_z (m) | .159637 | .099504 | .072379 |
| vx (m/s) | .219013 | .165624 | .097690 |
| vy (m/s) | .247711 | .194403 | .113127 |
| vz (m/s) | .075618 | .056913 | .032542 |
| yaw_error (rad) | .197865 | .164219 | .139931 |

Horizontal velocity RMSE .233803→.180588m/s; Teacher D .026748m/s. All seven
physical dimensions improve at H50, but the direct recurrent reference remains
better in aggregate. Per-dimension RMSE/MAE at all horizons are retained in JSON.

H50 grouping uses true start distance/PI only for offline evaluation:

| Group | Windows | One-Step latent→K10 | One-Step observation→K10 |
|---|---:|---:|---:|
| distance<.1m | 80 | .234205→.155088 | .267889→.202252 |
| .1–.2m | 727 | .295982→.215878 | .319098→.244948 |
| .2–.5m | 5790 | .304185→.187117 | .341759→.217494 |
| >=.5m | 19525 | .566050→.410878 | .558454→.391265 |
| small PI | 7790 | .262814→.124121 | .325771→.170122 |
| medium PI | 9144 | .401050→.269360 | .413261→.265600 |
| large PI | 9188 | .726623→.547215 | .696478→.515258 |

The <.1m subset is small; overlapping windows are not independent trials.
No NaN/Inf or explosion: Train max gradient norm2.74710, Test max latent norm
5.02051, max predicted observation magnitude6.34736. K10 H50 mean latent norm
1.73646 and cosine .969525 indicate reduced but nonzero drift.

## Verification and conclusion

New related tests9passed. Complete tracked branch suite plus new tests:
315total,314passed,0failed,1optional X11 viewer skip. Independent read-only
verification reproduces all model/floor/reference metrics, saved-best Validation
loss, manifests and immutable hashes. Actual selected Test rollout stays
elementwise identical after corrupting or deleting future teacher latent and
observations. Eight generated figures were visually checked. Read-only code
review found no Critical/Important issues.

Result partially supports caseA: explicitly training latent autoregression
reduces compounding drift at10steps and generalizes to25/50steps under this
frozen representation. It does not establish complete Markov state sufficiency
or uniformly better local predictions. Teacher D error near .08 is material,
but K10 H50 observation error .356 remains well above that reference: both
propagation and reconstruction bottlenecks remain. Do not subtract their RMSEs
or attribute all remaining error to the decoder. The prescribed horizon also
changes terminal-tail window support and Validation criterion, so this is not
a perfectly sample-matched causal effect estimate.

Next suggestion only: retain this frozen transition baseline and consider a
decoder-only reconstruction control before a separately authorized joint latent
formulation. No further training, branch, planner or World Model is started.
