# Explicit Encoder–Transition Consistency — Deterministic Latent World Model v2

Branch `feat/joint-latent-consistency-v2`; base
`feat/joint-latent-composition-audit` at
`d13541be9a620a55303ab0079eb7a517d5cd1822`. Only one new seed0 model trained.
Existing Joint v1 and all source components/reports remain immutable. No PPO,
lambda sweep, target network, EMA, planning, RSSM, Dreamer or other training.

## Controlled initialization and configuration

Initialize through the SAME original `initialize_joint` function using:

- `history_latent_gru_multistep10.pt` Encoder.
- `latent_transition_multistep10_mlp.pt` Transition.
- `latent_observation_decoder.pt` Decoder.

As in v1, fold pretrained latent affine normalization into T/D weights once,
preserving their functions. Runtime works in raw current latent coordinates.
Do NOT initialize from the optimized `joint_latent_world_model_v1.pt`.
Original post-fold, pre-joint module hashes must match exactly:

```
E 25a95644f2a2a7b1fc69cd336bd13cfe771f9dca65ee2c9686f64106ccf71e75
T 233b0c6d8eac9cbf33b2e3b131784d091a26eca524bd91f54e842f62d924406f
D 18b8e42ec7e78a83149fdc3b6ae1aaab8b26ad40a59b194a2bb443ec5ab2abfd
```

Architecture is unchanged,74119parameters: E GRU11→64(one layer), T residual Tanh
MLP68→128→128→64, D Tanh MLP64→128→128→7. Encoder input is7Dobservation plus
previous executed normalized4Daction; current action enters T. PI only groups
evaluation and never enters any model/loss.

Same seed0 Python/NumPy/PyTorch, CPU one thread/deterministic algorithms; Adam
lr.0003/default betas.9/.999/eps1e-8,16complete episodes/batch,60epochs, no new
clipping. Same `default_rng(0)` episode permutation, all legal K10starts in each
batch, full prefix/autonomous BPTT. Auxiliary forward is deterministic/no dropout,
does not consume training RNG. Initial Validation Lobs independently reproduces
v1's **.013320631866689076** before any optimizer step.

## Loss and gradient boundary

Original `predict_windows` observation path is reused unchanged:
E(real causal prefix through t)→z[t]; then10T-only predicted-latent transitions
with recorded actions and D readout. Future observation, future reference latent
and decoded observation never feed autonomous T. Current reconstruction k0 is
included. No predicted state is detached within this graph.

```
Lobs = mean_over_windows,k=0..10,7dims
       ((D(z_pred[t+k]) - o[t+k]) / Train_obs_std)^2
```

A distinct training-only real-history Encoder forward supplies local pairs:

```
z_source = E(real history through t+k)                 # gradient retained
z_target = E(real history through t+k+1).detach()      # target stop-gradient
z_local  = z_source + T_residual(z_source, normalized a[t+k])
Lcons    = mean_over_windows,k=0..9,64dims
           ((z_local - z_target) / sigma_z_initial_Train)^2
Ltotal   = Lobs + 0.1 * Lcons
```

`sigma_z_initial_Train` is population per-dimension std from147219initial Encoder
Train frames, floor1e-6; fixed throughout, no mean subtraction. Its range is
.145670–.256167. Save statistics/hash/provenance. This is different from each
trained model's post-training Train scale used only for diagnostics.

The same encoded state can be a stopped target for one local pair and a trainable
source for a later pair. Stop-gradient blocks the **target contribution**, not
all gradients through real-history frames that also serve as other sources.
Consistency gives Encoder(source) and T gradients; D is absent from that graph.
No additional head/loss/regularization or target encoder is introduced.

All minibatches record total E/T/D gradients. Weighted-consistency gradients are
measured via `autograd.grad` before the unchanged `Ltotal.backward()`/Adam step.
Observation gradient is inferred as total minus weighted-consistency by linearity,
subject to FP32 subtraction roundoff. Report norm ratios and fraction of batches
where consistency exceeds observation; these diagnostics do not rescale gradients.

## Dataset, selection and evaluation

Original `pi_hidden_state_seed0`, target-disjoint84/18/18groups and588/126/126episodes;
no trajectory generation. Exact original K10Train/Val manifests141339/29542windows.
Train/ValNPZ only opened before training. Test bytes/hashes may be checked but
Test timestep arrays FIRST opened after validation-only selection completes.

Choose best checkpoint strictly by Validation **Lobs**, not Ltotal/Lcons/Test.
Log all loss components, objective-gradient diagnostics, finite prediction/latent
maxima and component hashes. Independently reload and recompute selected Train/Val
losses and observation-only best-epoch rule.

Original Test starts/windows/actions/norms reused: H1/5/10/25/50 has
32296/31792/31162/29272/26122windows. Pure inference contains no consistency helper:
E(prefix) once→T(predicted latent, recorded action) repeatedly→D, no re-encoding.
Delete/corrupt future observations or remove reference branch: predictions remain
elementwise identical. Current reconstruction uses full saved real histories.

Repeat original local/periodic/distribution/Decoder diagnostics. For each model,
fit its **own** post-training Train mean/std/covariance/PCA95% basis; stdfloor1e-6,
ridge `max(1e-12,1e-4*trace(cov)/64)`. Compare predicted and reference scores; PCA
and covariance are linear distribution proxies, not a proven nonlinear manifold.
Native latent errors/rawL2/Decoder ratios are NOT cross-model physical quantities.

Relative local residual:

```
R = ||T(z_enc[t],a[t])-z_enc[t+1]||
    / (||z_enc[t+1]-z_enc[t]|| + 1e-6)
```

Retain every sample; report mean/median/q01/q05/q25/q75/q95/q99 and count of
reference increments<=epsilon. Also disclose moving-reference-only results.
Stationary denominators may dominate the mean; this ratio is not strictly
coordinate-invariant. Local evaluation averages all legal adjacent transitions,
not repeated window-weighted training samples.

Diagnostic correctionC1/5/10/25/never uses identical26122H50windows. Correct current
input before the next step, never final endpoint; C50 equals never. This injects
real-history information and is not deployable. Report absolute never−C1 gap and
relative sensitivity, not just never performance.

Distance bins<.1/.1–.2/.2–.5/>=.5m; PI Train tertiles.050717/.232845m/s². Executed
normalized4Daction-norm tertiles.1137995/1.0390758 use Train only. Observation RMSE
uses unchanged Train observation std; per-dimension physical RMSE/MAE, horizontal
velocity and yaw included. Latent norm/std/effective-rank diagnostics guard collapse;
no regularization or clipping silently added if failure occurs.

## Reproduction and artifacts

```
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*consistency*.py' -v
.venv/bin/python mujoco/rl/run_joint_consistency_v2.py
.venv/bin/python mujoco/rl/run_joint_consistency_v2.py --verify
```

Runner refuses existing final artifacts, preventing unintended retraining.
`--verify` has no training/optimizer step and independently rebuilds initialization,
initial scale, selected losses, manifests, all Test diagnostics and comparison.
Hash checks cover baseline/source checkpoints and all original data/manifests.
Verification recomputes selected-checkpoint losses and evaluation metrics, not
historical training gradients. Recorded epoch/loss/gradient summaries receive
internal consistency checks only; exact historical replay would require retraining.
Read-only review found no Critical/Important correctness issue. Both Minor
verifier-hardening suggestions were addressed with omission/tampering tests.

Commit only new code/tests, best model `joint_latent_world_model_v2_consistency.pt`,
report, initial/post-trained Train statistics/hash JSON,12required PNGs, this methods
file and EXPERIMENT_LOG. Reuse original manifests rather than duplicating them.
No raw NPZ/teacher arrays/prediction traces/intermediate epochs saved. Execution
logs and cache remain local/ignored; historical untracked experiments preserved.

Limitations: Single seed, nominal recorded actions, fixedlambda.1, moving online
Encoder reference, local-only constraint, correlated overlapping windows, mutable
latent geometry, diagnostic corrections, no policy/planning/control evaluation.

## Measured results — seed0

Best epoch **33**, selected only by Validation Lobs; all60epochs completed. Fixed
selected-checkpoint Train/Val Lobs=.003489722/.006143659, Lcons=.004045833/.004018441,
weighted Lcons=.000404583/.000401844, Ltotal=.003894306/.006545503.
Epoch60 Train/Val Lobs=.008323067/.012338830, Lcons=.014862809/.015088647,
Ltotal=.009809348/.013847694: late training deteriorates, rather than converging
monotonically. These online epoch summaries differ from fixed best-model losses.

E/T/D hashes match v1 pre-joint hashes exactly, and all three best-model hashes
change: `4471b82f…`/`fb2389ad…`/`4adbf820…` (full hashes in report).
Best-epoch total gradient mean=.088796/.581221/.086609; whole-run maxima
1.654832/19.526739/1.928749. Weighted-consistency/observation norm ratio averaged
over all minibatches=.095833/.014744/0. **No batch** has consistency norm greater
than observation norm. No new clipping or optimizer setting was used.

| Horizon | Joint v1 observation RMSE | Consistency v2 | v2 worsening |
|---|---:|---:|---:|
| 1 | .043320 | .055747 | 28.69% |
| 5 | .070828 | .080166 | 13.18% |
| 10 | .102727 | .112402 | 9.42% |
| 25 | .230850 | .250077 | 8.33% |
| 50 | .510667 | .564054 | 10.45% |

Local own-Train-scale RMSE improves .109955→.072728 (33.86% native-scale
reduction), MAE=.047426, meanL2 .119334→.083674, cosine .996376→.998375.
Reference increment meanL2 changes .150880→.166088. Relative residual median
1.570927→1.072854, moving-only median1.371499→1.028195. All-sample means
5210.81→1506.92 are **not** robust evidence: near-stationary counts2624→1094
and epsilon denominators dominate them. Full quantiles are retained in JSON.

Autonomous v2 latent own-Train RMSE H1/5/10/25/50:
.072728/.220159/.333632/.519541/.767816; cosine
.998375/.985976/.967183/.924342/.864783. Improvement in local discrepancy does
not persist as improved H50 observation prediction. These latent metrics are
not strictly comparable physical quantities across learned coordinate systems.

Current reconstruction **.052330**, versus v1 .038286 (**36.68% worse**);
per-dimension R² .996359–.997817. High explained variance does not negate this
absolute-error regression. Physical H50 RMSE/MAE:

| Dimension | RMSE | MAE |
|---|---:|---:|
| error_x (m) | .398534 | .319273 |
| error_y (m) | .387618 | .266324 |
| error_z (m) | .124132 | .088243 |
| vx (m/s) | .346247 | .235822 |
| vy (m/s) | .364441 | .232801 |
| vz (m/s) | .082446 | .050425 |
| yaw_error (rad) | .134531 | .095916 |

H50 horizontal velocity .318011→.355460m/s (**11.78% worse**); yaw
.156039→.134531rad (**13.78% better**). All other horizons/dimensions are in JSON.

Diagnostic H50 C1/5/10/25/never observation RMSE:
.059020/.085550/.119386/.259356/.564054; latent discrepancy
.074654/.225331/.340689/.526430/.767816. Still monotonic recovery with more
frequent real-history corrections. Absolute never−C1 gap **increases**
.464503→.505034. Ratio11.062→9.557 declines because C1 also gets worse; this
does not establish lower correction sensitivity or autonomous stability.

Own-Train v2 covariance ridge2.92408e-6; PCA14components retain95.9994%.
Predicted Mahalanobis RMS H1→H50 .844935→2.728598 versus reference
.811766→.839565. v1 H50 prediction3.441289 is higher, but v2 PCA residual
.219485→.559404 versus reference .216502→.219824; v1 H50 .535272 is lower.
Thus geometry evidence is **mixed**, not uniformly reduced off-manifold drift.
Empirical Decoder ratio .369464→1.206192 (v1 .211348→1.198374); H50 decoded
deviation1.162944 versus v1 1.051632. Both are descriptive, not Jacobian analysis
or an additive decomposition of transition/decoder errors.

H50 observation RMSE by start distance <.1/.1–.2/.2–.5/>=.5:
.262841/.272131/.285981/.631159 (counts80/727/5790/19525); all worse than v1.
PI small/medium/large .203044/.455239/.814454; small improves slightly while
medium/large worsen. Action small/medium/large .177447/.594130/.721461; all worse.
Original Train-only PI/action thresholds reused; no test-based regrouping.

Train/Test latent norm mean1.562620/1.550820,std .590602/.581364,max4.121899/4.199501.
Per-dim std min/median/max Train .111917/.163566/.241993, Test
.110180/.170417/.222783. Effective rank12.237907/12.072160;0near-zero dimensions.
No NaN/Inf, global variance collapse or rollout norm explosion; autonomous max
latent norm4.817581 and physical observation absolute max6.244331. No predictions
were clipped. Future-observation corruption/deletion leaves rollout bit-identical.
Nine new related tests pass; final tracked-branch+new repository suite runs341tests:
340passed,0failed,1optional X11skipped. Selected-checkpoint losses and all Test
diagnostics independently reproduce. Historical training summaries are checked
internally, not replayed. All12figures inspected after layout-only correction.

### Conclusion and boundary

The fixed local auxiliary objective **does reduce local Encoder–Transition
discrepancy**, but **does not support the proposed long-horizon remedy**. All
observation horizons and current reconstruction regress despite healthy latent
variance. Do not adopt this v2 as an improved baseline. Local consistency is not
a sufficient condition for useful autonomous composition under this formulation;
there is a representation/prediction trade-off, not a proven unique module cause.
Single seed and coordinate-dependent diagnostics cannot establish necessity of
stochastic dynamics or reject deterministic formulations generally.

Only recommended next experiment: a separately authorized objective-matched
**autonomous multi-step latent-consistency** control. No such training, lambda
sweep, next branch, RSSM/Dreamer/planning/PPO experiment was executed.
