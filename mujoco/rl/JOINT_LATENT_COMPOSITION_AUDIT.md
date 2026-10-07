# Deterministic Latent Composition / Consistency Audit

Branch `feat/joint-latent-composition-audit`, base `feat/joint-latent-world-model-v1`
at `59c0a28468c81a28c90283200c6acef27eb3362c`. Read-only diagnosis, no model
training, optimizer or backward in the audit path; no policy, planning or RSSM.

## Frozen model and immutable data

Load `joint_latent_world_model_v1.pt` SHA256
`c3cfb55b80ee9f388a766a8d3b157c28e80705f8c7dad97071b7a83d31ddb2e1`.
E GRU11→64(one layer), residual T68→128→128→64 and D64→128→128→7, Tanh heads.
All eval/`requires_grad=False`, no parameter gradients. Record/recheck every module
parameter hash and checkpoint/source file hash. No checkpoint output is produced.

Original `pi_hidden_state_seed0`: Train84target groups588episodes used ONLY to fit
diagnostic geometry, Test18groups126episodes used for evaluation. Validation values
not needed. Exact recorded observation/current/previous actions and split hashes;
no generation, no new normalization. Original Train observation/action statistics
and original Test start manifest are reused and compared by hash/identity.

Windows H1/5/10/25/50 have32296/31792/31162/29272/26122legal starts. No cross-episode
transitions or post-terminal invented observations. Original final saved frame is
retained for encoder references. Current action a[t] drives z[t]→z[t+1]; previous
action a[t-1] is only the Encoder's history input. Frequency25Hz.

## Four distinct layers

1. **Encoder reconstruction:** E(real history)→D for current saved observations.
   Offline reconstruction, not a prediction model or strict decoder floor.
2. **Local consistency:** each real-history z_enc[t] independently→T(a[t]); compare
   encoder reference z_enc[t+1]. Also report identity T reference, encoded movement
   and local residual norms: high cosine alone does not imply accurate increments.
3. **Autonomous composition:** E(real prefix) provides initial state ONCE; then
   repeated T(predicted latent,recorded action) and D readout. No future reference,
   observation or decoded observation enters T. Same baseline metrics independently
   reproduce the previous Joint report at rtol1e-7/atol1e-10.
4. **Periodic correction:** explicitly teacher-assisted diagnostic, NEVER deployable.
   All correction intervals use the same26122 H50windows/actions. Before step k+1,
   at0<k<50 divisible byC, replace CURRENT latent with E(real history through t+k).
   Do not replace prediction at endpoint k50. Thus C1 still predicts last step;
   C5/10/25 have last autonomous segments5/10/25steps; C50/never are identical.

Encoder-reference latent is a representation, NOT physical ground truth. Encoder
runs causally from episode start, hidden reset per episode, `[obs7,prev_action4]`.
Real-history future references are only evaluation targets or explicitly labelled
corrections. Future terminal reference cannot affect C1 output. Pure predictions
remain bit-identical after deleting/corrupting future observations/references.

## Metrics and Train geometry

Latent deviation: raw64D RMSE/MAE/meanL2/cosine and scale-aware RMSE/MAE divided by
post-training Joint Train latent std (floor1e-6), ONLY a diagnostic scale. Observation
endpoint RMSE uses original Train observation std; physical7D RMSE/MAE/velocity
included. Decoder sensitivity compares D(predicted z)−D(reference z) in normalized
observation coordinates, not observation ground truth: L2 deviation divided by raw
latent L2+1e-12 per sample. Mean/median/quantiles reported; ratio is unit-dependent,
not Jacobian spectral norm. No optional Jacobian/autograd is executed.

Train encoder frames147219 fit mean/population covariance/std. Regularized covariance
adds `max(1e-12,1e-4*trace(cov)/64)*I`. Mahalanobis-like RMS is
`sqrt((z-mean)'inv(cov_reg)(z-mean)/64)`. Standardized distance is
`sqrt(mean(((z-Train_mean)/Train_std)^2))`. PCA uses minimum components for95% Train
variance, with raw orthogonal residual L2 and residual/centerednorm fraction.
Covariance/PCA are linear distribution proxies, NOT a demonstrated nonlinear
encoder manifold; reference and paired predicted-minus-reference scores are
reported so high-magnitude real states are not silently labelled off-manifold.

Distance bins fixed<.1/.1–.2/.2–.5/>=.5m. PI bins reuse original Train norm tertiles
.0507170173/.2328452269m/s², offline only. Action bins use Train executed normalized
4D action L2 tertiles; not commanded physical velocity units. Additional velocity
grouping uses Train PCA-residual tertiles. No Test threshold fitting.

Pearson correlations use all legal windows, not scatter-display subsampling:
initial local residual vs later latent/observation/horizontalvelocity errors, and
horizontalvelocity error vs predicted/reference/paired-excess distribution proxies.
Constant/insufficient pairs return null. Correlations are descriptive, not causal;
overlapping windows are correlated. Selected trace is the SAME prior fixed Test
episode0/target7/start10/H50, not chosen by result; scatter uniformly subsampled only
for plotting. No single latent dimension gets physical interpretation.

## Reproduction and test boundary

```
.venv/bin/python mujoco/rl/run_joint_latent_composition_audit.py
.venv/bin/python mujoco/rl/run_joint_latent_composition_audit.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*joint*composition*.py' -v
```

Runner refuses existing report/statistics output. Verifier reloads the original
model/data and rebuilds all Train statistics, manifests and A–F metrics, checks
parameter/source hashes and correction/leakage/reset invariants. Tests use saved
fixed-weight fixtures without any training; optimizer creation is forbidden on
the audit path. Prior repository regression fixtures are independent tests of
existing trainers and do not modify the audited model or produce study training.

Commit diagnostic code/tests, summary report, Train statistics/hash, nine PNGs,
this document and EXPERIMENT_LOG. Existing manifests referenced, not duplicated.
No raw latent arrays/large traces/dataset copies/model files/intermediate training
artifacts are saved by this audit. Internal test/progress logs stay ignored.

Limitations: Single seed, nominal recorded actions, no control/planning, diagnostic
real-history corrections, linear geometry proxies, unit-dependent sensitivity,
correlated overlapping windows. No model was trained or algorithm modified.

## Measured results

The checkpoint and all three component hashes match before/after. The independent
read-only verifier rebuilds every diagnostic and reproduces the original Joint
metrics. Current encoder reconstruction remains normalized RMSE **0.038286**.

Local consistency uses all32296 legal transitions: Train-scale latent RMSE
**0.109955**, MAE0.073665, meanL2 **0.119334**, cosine0.996376. Identity transition
RMSE0.192827; the learned transition lowers that RMSE42.98%. Mean encoder-reference
movement is0.150880. Consequently the
local residual is material: high cosine is not evidence of almost exact increments.

| Horizon steps | Windows | Autonomous latent RMSE | Mean L2 | Cosine | Observation RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 32296 | .109955 | .119334 | .996376 | .043320 |
| 5 | 31792 | .274032 | .304307 | .980297 | .070828 |
| 10 | 31162 | .377245 | .417974 | .963863 | .102727 |
| 25 | 29272 | .541702 | .597176 | .929051 | .230850 |
| 50 | 26122 | .749678 | .809398 | .878943 | .510667 |

H50 latent error is6.82times H1. Horizon-specific cohorts differ; the correction
comparison below instead holds the full26122-window H50cohort fixed.

| Correction interval | H50 observation RMSE | H50 latent RMSE | Horizontal velocity RMSE m/s |
| --- | ---: | ---: | ---: |
| C1 | .046164 | .111703 | .017011 |
| C5 | .076160 | .275864 | .032412 |
| C10 | .109667 | .378474 | .055518 |
| C25 | .237967 | .541904 | .146467 |
| C50 / never | .510667 | .749678 | .318011 |

The monotonic recovery is strong evidence for accumulated propagation discrepancy;
C1 is90.96% lower than autonomous in observation RMSE. These corrections also inject
real-history state information, so do not uniquely identify a root cause or provide
a deployable fix. No endpoint re-encoding artificially improves the score.

### Distribution and Decoder diagnostics

Train147219frames fit13PCA components retaining95.1493% variance; covariance ridge
2.390175e-6. Train std range .111669–.215781. Action norm tertiles .1137995/1.0390758.
Full mean/covariance/basis and hashes are in the statistics JSON, not raw latents.

| Distribution score, mean | H1 prediction / reference | H50 prediction / reference |
| --- | ---: | ---: |
| Standardized RMS | .818582 / .828419 | .973672 / .860376 |
| Regularized Mahalanobis RMS | .948641 / .810252 | 3.441289 / .839478 |
| PCA orthogonal residual L2 | .234264 / .227364 | .535272 / .233595 |

Marginal standardized distance changes modestly, but covariance-aware distance and
PCA residual increase markedly. This supports drift along low-variance/out-of-main-
subspace directions, NOT a mathematical proof of departure from a nonlinear manifold.
Latent maxnorm4.629386 and physical prediction absmax5.774427 remain finite.

Empirical Decoder ratio mean .211348→1.198374, median .171558→1.161877. Mean decoded
normalizedL2 .030707→1.051632, raw latentL2 .119334→.809398. Sensitivity along actual
deviation directions grows, but the ratio's units/geometry prevent a strict gain or
Jacobian interpretation. This does not isolate Decoder causality; reconstruction
itself remains good, and correction substantially restores prediction.

Initial local residual versus future latentL2 Pearson r atH5/10/25/50:
.719460/.616529/.541382/.493170. H50 observation/horizontal-error r:
.443245/.414008. H50 horizontal error versus predicted Mahalanobis/PCA residual:
.635503/.656474; versus paired excess over reference: .446859/.278058. Reference
scores are also correlated, so state/action confounding remains. Train-PCA residual
tertiles place every H50prediction in the large bin: this grouping is saturated and
cannot estimate a graded within-H50 relationship.

### Regions and velocity

| True start region | H50 windows | Autonomous obs RMSE | C1 obs RMSE |
| --- | ---: | ---: | ---: |
| distance<.1 | 80 | .243606 | .008890 |
| distance .1–.2 | 727 | .264566 | .025594 |
| distance .2–.5 | 5790 | .283025 | .028687 |
| distance>=.5 | 19525 | .567705 | .050818 |
| PI small | 7790 | .211972 | .011567 |
| PI medium | 9144 | .400007 | .038003 |
| PI large | 9188 | .737618 | .067143 |
| action small | 7568 | .176569 | .009426 |
| action medium | 9461 | .536654 | .055525 |
| action large | 9093 | .650815 | .053296 |

Local latent RMSE similarly increases with PI .062172/.112440/.138705 and action
.053235/.103796/.150420. These are descriptive, not independent causal factors.
H50 physical RMSE(error_xyz,v_xyz,yaw):
.283699/.431516/.094485m, .302906/.332431/.064655m/s, .156039rad.
C1vx/vy RMSE .017023/.017000m/s: horizontal drift is largely rescued diagnostically.
All per-dimension MAE and region-level latent/cosine results are preserved in JSON.

## Conclusion and only recommended next experiment

Most supported: **non-negligible local E→T inconsistency plus repeated composition**,
accompanied by drift into low-variance/out-of-principal-subspace coordinates and
increased empirical Decoder sensitivity. This is caseA-like accumulation with a
caseB local-consistency caveat and caseC geometry association, not caseD failure of
correction. No unique module-level causal root cause has been proved.

The deterministic formulation remains worth a controlled repair; this does not
justify jumping to RSSM. Recommend only a separately authorized objective-matched
**explicit encoder–transition consistency-loss control**, keeping the current K10
observation loss/architecture/data/optimizer fixed, to test local discrepancy and
H50 drift. No such intervention or training was performed here.

## Verification evidence and deferred review hardening

Seven new tests pass; tracked-branch plus new tests:331passed/0failed/1optional X11
skip. Independent read-only full metric/hash verification passes. All nine PNGs
visually inspected. Code review:0Critical/0Important, two Minor hardening suggestions
deferred: mutate temporary Test source data in the Train-independence fixture, and
include `selected_window` metadata in independent verifier comparisons. Actual Train
fit and formal original selected window are correct; no measured diagnostic changes.
