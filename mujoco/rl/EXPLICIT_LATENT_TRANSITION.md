# Explicit latent transition (seed0)

Scope: frozen representation, independent supervised transition and decoder,
recorded-action pure latent autoregression. No encoder/policy/joint training,
simulation resampling, planning, reward/terminal prediction or World Model framework.

Branch `feat/explicit-latent-transition`, base
`feat/latent-objective-matched-control` at
`259000a455956217318a242794f6e9e2e5481f22`.

## Frozen teacher and data

Encoder is ONLY the GRU from `history_latent_gru_multistep10.pt`:11→64,
one layer. The original checkpoint hash is
`c2f6ca5cf729430b89e392e1748962f36284d6215225771111409d76a4b9302c`.
All encoder parameters have `requires_grad=False`; no learner optimizer owns
them. Parameter hashes are checked before/after extraction/training/evaluation.

Teacher latent `z_t` is the frozen encoder output after normalized
`[recorded observation_t,previous executed normalized action]`. It is not a
physical ground-truth state. Every episode resets hidden toNone/zero.

Reuse immutable `pi_hidden_state_seed0`, target-disjoint Train/Val/Test
84/18/18groups,588/126/126episodes. Original saved observation frames are
147219/30802/32422; adjacent transition pairs146631/30676/32296.
The existing adapter exposesN transitions andN+1 saved observations: append its
last `next_obs` and corresponding previous action to the teacher sequence.
No unavailable terminal observation is invented; no pair crosses an episode.

Teacher arrays stay in memory, not in Git. Compact manifests preserve episode
IDs, source rows, per-episode pair/frame counts and teacher/action SHA256.
Source NPZ/hash/split manifest are unchanged. Test timestep values are opened
only AFTER both independently selected best checkpoints are saved.

## Normalization and residual transition

Reuse exact saved Train-only observation/action statistics. Compute latent
mean/std from all147219 saved Train teacher frames only, population std with
floor1e-6. Validation/Test never affect these statistics.

Let `q=(z-mean_z)/std_z`. Transition is a deterministic Tanh MLP:
`[q_t,normalized a_t]68 ->128 ->128 ->64`, output `Delta_q_hat`.

```
z_hat_next = z_current + std_z * Delta_q_hat
loss_T = mean((Delta_q_hat - (teacher_z_next-teacher_z_current)/std_z)^2)
```

There is no separately fitted delta-z std, auxiliary loss, observation loss,
PI/reward loss or multi-step loss in T training. Teacher delta-z magnitude
mean/std/quantiles/max are recorded. PI only groups offline evaluation results.

## Independent decoder and optimization

Decoder is Tanh MLP:`normalized z64 ->128 ->128 ->7`. It predicts normalized
current observation, reconstructed using the original Train observation mean/std.
Its only loss is observation-standardized reconstruction MSE on teacher latents.
No decoder loss reaches T or encoder; no T loss reaches encoder or decoder.

Each learner independently resets Python/NumPy/Torch seed0 and a fresh Adam
optimizer, lr=.001/default betas/epsilon.16complete episodes/batch, shuffled
with independent `default_rng(0)` schedules,60epochs, no new gradient clipping.
All real pairs/frames in each episode minibatch contribute; padding is absent
from these feedforward learners. T selects minimum Val normalized latent-delta
MSE, equivalent to next-latent MSE at a teacher start. D independently selects
minimum Val normalized observation reconstruction MSE. Test is not a selector.

## Pure latent rollout

Use the prior exact Test start manifest and H1/5/10/25/50. A true causal prefix
provides `z_t`. Thereafter ONLY predicted latent and recorded current action
enter T. Decoder reads out each predictednext latent; decoded observations are
NEVER sent back to the GRU. Future teacher latents/observations are accessed
only by the offline metric accumulators.

The low-level `latent_rollout` API has no observation or encoder argument.
The single-window prefix API works unchanged with corrupted or completely
deleted future observations. Both unit tests and a fixed real Test-window check
verify bit-identical predicted latent/observation arrays under that intervention.

Observation metrics are endpoint RMSE/MAE using original Train obs std;
physical per-dimension metrics and horizontalvelocity RMSE are included.
Latent metrics use the SAME frozen coordinates and Train latent std: endpoint
RMSE/MAE, cosine, predicted norm, plus the endpoint incremental delta-z error
`(pred_z[k]-pred_z[k-1])-(teacher_z[k]-teacher_z[k-1])`.
This incremental error is not cumulative displacement error. H1 matches the
separate one-step evaluator exactly. Identity transition is an untrained
reference, not an additional learned model.

Distance/PI groups use true starting values and previous fixed Train PI tertiles.
The common largest-horizon start cohort is also retained. Direct K10 metrics
are independently reproduced with the old observation-feedback evaluator.
Yaw keeps the original wrapped observation/raw-coordinate-error conventions;
no new wrapping or clipping is introduced to predictions.

## Measured results

Transition best epoch57/60: Train/Val MSE .001736334/.001860247;
decoder best epoch60/60: Train/Val MSE .004503187/.005742569. Their parameter
counts are33600 and25735; training took24.09s and18.87s, respectively.
Encoder parameter SHA256 stayed
`25a95644f2a2a7b1fc69cd336bd13cfe771f9dca65ee2c9686f64106ccf71e75`.
Latent statistics semantic hash:
`10bf0867cf36eb8401e48d848dec6608657c418a33ac8f0df336841e81c666c6`;
Train latent std range .145670–.256167. Train physical latent-delta norm
mean/std/max .201120/.238576/2.046242 (median .118252).

Decoder Test normalized observation RMSE=.078243; per-dimension physical
RMSE/MAE/R² follow. Position units m, velocity m/s, yaw rad.

| Dimension | RMSE | MAE | R² |
|---|---:|---:|---:|
| error_x | .088904 | .053952 | .995235 |
| error_y | .109455 | .065291 | .993165 |
| error_z | .032202 | .019406 | .994358 |
| vx | .026576 | .015270 | .995628 |
| vy | .023866 | .013540 | .996368 |
| vz | .018177 | .009872 | .991969 |
| yaw_error | .037994 | .023241 | .990576 |

The independent residual transition's one-step normalized latent RMSE=.044726,
raw latent/Delta-z RMSE=.008510, MAE=.004687, mean cosine=.999391.
Identity transition normalized RMSE=.204087: learned transition reduces it
78.08%. This supports local predictability, not long-horizon accuracy.

| Horizon | Windows | Latent normalized RMSE | Latent cosine | Explicit obs normalized RMSE | Direct K10 obs normalized RMSE |
|---|---:|---:|---:|---:|---:|
| 1 / .04s | 32296 | .044726 | .999391 | .086475 | .022245 |
| 5 / .20s | 31792 | .125679 | .996227 | .126038 | .061937 |
| 10 / .40s | 31162 | .191303 | .991637 | .175715 | .097606 |
| 25 / 1s | 29272 | .334250 | .974858 | .313664 | .178372 |
| 50 / 2s | 26122 | .512454 | .936980 | .511910 | .242316 |

At2s explicit/direct horizontal velocity RMSE=.233803/.143203m/s,
vz=.075618/.046744m/s and yaw=.197865/.067869rad. Every physical dimension's
RMSE/MAE at every horizon is retained in the JSON report.

At2s observation-normalized RMSE by true starting distance:

| Distance | Windows | Explicit | Direct K10 |
|---|---:|---:|---:|
| <.1m | 80 | .267889 | .048713 |
| .1–.2m | 727 | .319098 | .125772 |
| .2–.5m | 5790 | .341759 | .105602 |
| >=.5m | 19525 | .558454 | .273222 |

Large-PI region (Train threshold>.232845m/s²,9188windows): explicit/direct
observation RMSE=.696478/.373594; latent RMSE=.726623, cosine=.905100.
These are offline groups, not inputs. Overlapping windows are correlated;
especially the80 near-target2s windows are not independent trials.

All predictions remained finite: zero NaN/Inf, maximum physical predicted
observation absolute value6.237134, maximum predicted latent norm4.908174.
Finite values do not imply accurate dynamics: substantial latent drift remains.
All seven figures were visually inspected. Full tracked-branch suite plus new
tests:305passed,0failed,1optional X11 Viewer skip; ten new experiment tests
passed. Independent read-only artifact verification passed all eight checks.
Future observations were both corrupted and deleted; results stayed
bit-identical, including after removal of future teacher latents.

Conclusion: partial support for local state-like transition structure, but not
for a reliable autonomous long-horizon model. Closest to caseB, with a material
decoder bottleneck: .078243 teacher reconstruction error already exceeds the
direct predictor's one-step .022245 error. High reconstruction R² is not proof
of precise readout. Rollout error contains both decoder error and accumulated
transition drift; it does not establish that the frozen representation is
fundamentally non-Markovian or lacks current observation information.

Next suggestion only: keep encoder/decoder fixed and separately test a
multi-step latent-transition objective before adding joint representation
learning. This separates propagation consistency from further decoder/encoder
changes. Nothing beyond the present independent one-step experiment was run.

## Reproduction and artifacts

```
.venv/bin/python mujoco/rl/run_explicit_latent_transition.py
.venv/bin/python mujoco/rl/run_explicit_latent_transition.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*explicit_latent_transition.py' -v
```

Runner refuses to overwrite artifacts. `--verify` is read-only and regenerates
teacher sequence hashes/statistics, reloads selected models, recomputes best
Validation losses, fixed starts, all Test metrics and future-leakage checks.

- `mujoco/rl/models/latent_transition_mlp.pt`
- `mujoco/rl/models/latent_observation_decoder.pt`
- `mujoco/reports/explicit_latent_transition_seed0.json`
- `explicit_latent_{train,val,test}_manifest_seed0.json`
- `explicit_latent_normalization_seed0.json`
- Seven required figures listed in the report.
- Local ignored log `.superpowers/explicit-latent-transition/training.log`.

Limitations: single seed/nominal simulation, recorded future actions, frozen
deterministic latent coordinates, independent one-step/reconstruction objectives,
no multi-step training, encoder adaptation, control/planning or causality claim
from prediction correlations. Decoder bottleneck must be separated from latent
transition drift. Suggestions for later experiments are not executed here.
