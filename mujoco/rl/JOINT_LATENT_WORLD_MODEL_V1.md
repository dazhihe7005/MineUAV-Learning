# Deterministic Latent World Model v1

Base `feat/latent-transition-multistep`, commit `4fe448576b7df321778291e414c98788c13a8e99`.
Branch `feat/joint-latent-world-model-v1`. This is a deterministic latent dynamics
core, not Dreamer, a complete World Model agent or model-based RL.

## Pretrained joint initialization and raw coordinates

E: GRU11→64(one layer) from `history_latent_gru_multistep10.pt`.
T: residual Tanh MLP68→128→128→64 from `latent_transition_multistep10_mlp.pt`.
D: Tanh MLP64→128→128→7 from `latent_observation_decoder.pt`.
Total74119parameters. All three modules trainable and owned by one optimizer.
No random model reset and no source checkpoint writes.

Saved T/D had normalized-latent inputs. To preserve their learned functions while
removing old latent normalization, fold their input affine map into first-layer
weights once: latent columns W/std; bias b-W*(mean/std). T output rows/bias are
multiplied by old std once to output raw Delta-z. Action columns remain unchanged.
Record original, converted-before-training and selected-after-training hashes;
check numerical function equivalence. Joint forward uses NO latent mean/std or
latent normalization buffers. Observation/action/previous-action normalization
strictly reuses original Train statistics; no Val/Test fitting.

## Data, objective and differentiable prefix

Exact original `pi_hidden_state_seed0`,84/18/18target groups,588/126/126episodes.
No trajectory regeneration. Reuse saved K10 Train141339/Val29542 starts and their
manifests. No cross-episode windows. Every episode begins hidden=None; cache its
unidirectional normalized `[obs,previous_action]` GRU prefixes with gradients.
Select z[t] once; causal prefix is mathematically equivalent to separately encoding
frames0..t (FP32 matrix-shape roundoff tolerance). No padding output is selected.

From z[t], repeat z'=z+T(z,normalized recorded current action), D(z') without
observation or future encoder feedback. Full10step/prefix BPTT, no detach. D(z[t])
provides current reconstruction. Loss is mean over all legal windows, k0..10 and
7dimensions of normalized observation squared error. Current reconstruction has
the same weight as each future step. No latent teacher, latent consistency, PI,
reward, terminal, auxiliary, norm, cosine, contrastive or dimension-specific loss.

Adam lr.0003/default betas.9/.999 eps1e-8, seed0,60epochs,16completeepisodes/batch,
prior default_rng0 episode-permutation rule. Fresh joint optimizer, no assumed
recoverable optimizer state from independently supervised components. No added
gradient clipping. Track per-module gradients, finite predictions/norms each
epoch. Validation all-window k0..10 MSE selects best. Test timestep values only
opened after selection; no Test-based tuning or early stopping.

## Frozen evaluation and offline diagnostics

Reuse all legal Test starts: H1/5/10/25/50 at25Hz and exact prior manifest.
Each model uses the same starts at each horizon; primary curves use perH legal
cohorts, commonH50start cohort retained separately to check cohort effects.
Joint initializes from real prefix then propagates only latent+recorded actions.
Frozen Explicit K10 and Direct K10 are independently recomputed and checked
against prior metrics/hash. Deleting/perturbing future observations must leave
Joint rollout predictions elementwise identical.

Physical per-dimension RMSE/MAE, normalized observation RMSE, horizontal/vertical
velocity. Start-distance bins<.1/.1–.2/.2–.5/>=.5 and saved Train PI thresholds
(.050717,.232845m/s²), PI never input. Raw yaw convention/metric unchanged.

Current reconstruction D(E(realhistory)) uses all saved N+1frames. It is an offline
reconstruction diagnostic, NOT a Joint decoder floor or a deployable autonomous
future model. Old Teacher-Latent Decoder Floor is only historical, never a Joint
lower bound. Errors cannot be linearly decomposed by subtracting RMSEs.

Offline latent consistency compares predicted z[t+H] with Joint E(real future
history)[t+H] in the SAME joint coordinate system; raw L2/cosine and Train-scale
distance reported. Post-training Train latent std used only for that diagnostic,
never training/input normalization. Old teacher-latent RMSE is not a primary metric.
Train/Test latent norm mean/std/max, per-dimension std min/median/max and zero-
variance count (threshold1e-6); covariance entropy effective rank. Correlated latent
coordinates/lower rank alone do not prove collapse. No diagnostic feeds a loss.

## Reproduction, correctness and artifacts

```
.venv/bin/python mujoco/rl/run_joint_latent_world_model.py
.venv/bin/python mujoco/rl/run_joint_latent_world_model.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*joint_latent_world_model.py' -v
```

Runner refuses overwrite. Read-only verifier checks original immutable hashes,
converted initialization, all three updates, Val argmin/reloaded loss, manifest,
normalization, all metrics and future-input independence. Tests cover literal
residual recursion/current-action indexing, uniform k0weight, all three gradients,
prefix and later-step BPTT, causal prefix/reset/boundaries, normalization/reload/
determinism, original Test manifest, variance diagnostic and real fixture end-to-end.

Only best joint model (~300KB), report,9requestedfigures, code/tests/docs and
EXPERIMENT_LOG are committed. Existing manifests referenced by immutable hash,
not duplicated. Raw data/latent arrays/logs/cache/intermediate traces local-only.

Limitations: Single seed, nominal simulation, recorded future actions, overlapping
windows, deterministic residual dynamics, .4s training horizon; no reward/policy/
planning/control evaluation. Joint loss and all modules adapt together, so no
component-specific causal attribution or complete-state sufficiency claim.

## Measured seed0 results

Prescribed60epochs completed, Validation-best epoch60. Selected-checkpoint Train
MSE=.00250791497, Validation MSE=.00483894154 (initial=.01332063187). Final online
training average=.00252666500; distinguish this changing-parameter average from
recomputed selected-checkpoint Train loss. Training565.98s, total599.36s. Module
gradient maxima over the whole run: E1.653262/T19.539579/D1.930154, all finite;
epoch60 means E.071874/T.317156/D.074633. Allthree hashes changed; originals remain
immutable. No new clipping, early stopping or subsequent training.

Old latent affines were folded into trainable weights before optimization. Fixed
probe equivalence errors1.79e-7(next latent)/5.36e-7(normalized observation) satisfy
3e-6 tolerance; no old latent statistics participate in joint forward or loss.
The source/converted/selected full SHA256 hashes are retained in the JSON report.

Normalized observation endpoint RMSE:

| Steps / seconds | Windows | Joint v1 | Frozen Explicit K10 | Direct K10 | Joint reduction vs Frozen |
|---|---:|---:|---:|---:|---:|
| 1 / .04 | 32296 | .043320 | .084842 | .022245 | 48.94% |
| 5 / .20 | 31792 | .070828 | .116118 | .061937 | 39.00% |
| 10 / .40 | 31162 | .102727 | .153719 | .097606 | 33.17% |
| 25 / 1.0 | 29272 | .230850 | .254360 | .178372 | 9.24% |
| 50 / 2.0 | 26122 | .510667 | .355959 | .242316 | **−43.46%** |

Negative reduction means regression. H1/5/10 are within training horizon; H25/50
are outside. The identical26122-start cohort gives the same ordering: Joint
.042466/.072306/.105383/.232400/.510667. Results are not a changing-cohort artifact.
AtH10 Joint closes90.87% of the Frozen→Direct gap, but H50 gap closure is−136.14%
under `(Frozen-Joint)/(Frozen-Direct)`, not positive progress.

Current Test reconstruction RMSE=.03828649 versus historical Frozen E/D=.078243.
Per-dimension R²=.997834–.998782. This is offline D(E(realhistory)), not a Joint
decoder floor. H50 physical metrics:

| Dimension (unit) | Joint RMSE | Joint MAE | Frozen RMSE | Direct RMSE |
|---|---:|---:|---:|---:|
| error_x (m) | .283699 | .172044 | .292601 | .176015 |
| error_y (m) | .431516 | .341742 | .377395 | .182686 |
| error_z (m) | .094485 | .062674 | .099504 | .061456 |
| vx (m/s) | .302906 | .204296 | .165624 | .136903 |
| vy (m/s) | .332431 | .210564 | .194403 | .149237 |
| vz (m/s) | .064655 | .038004 | .056913 | .046744 |
| yaw_error (rad) | .156039 | .121955 | .164219 | .067869 |

Horizontal velocity RMSE H1/5/10/25/50=.015826/.029862/.051524/.141707/.318011m/s.
AtH50 Frozen/Direct=.180588/.143203; horizontal motion drives much of the aggregate
regression. Yaw improves modestly versus Frozen but remains worse than Direct.
Every horizon's physical per-dimension RMSE/MAE is in the report.

H50 start-region normalized observation RMSE:

| Region | Windows | Joint | Frozen | Direct |
|---|---:|---:|---:|---:|
| distance <.1m | 80 | .243606 | .202252 | .048713 |
| .1–.2m | 727 | .264566 | .244948 | .125772 |
| .2–.5m | 5790 | .283025 | .217494 | .105602 |
| >=.5m | 19525 | .567705 | .391265 | .273222 |
| small PI | 7790 | .211972 | .170122 | .049913 |
| medium PI | 9144 | .400007 | .265600 | .159288 |
| large PI | 9188 | .737618 | .515258 | .373594 |

No PI input; fixed prior Train thresholds and distance bins. Small near-target
sample count and overlapping windows preclude treating these as independent trials.

Offline Joint-encoder consistency H1/5/10/25/50:

| Steps | Mean L2 | Train-std distance | Cosine |
|---|---:|---:|---:|
| 1 | .119334 | .109955 | .996376 |
| 5 | .304307 | .274032 | .980297 |
| 10 | .417974 | .377245 | .963863 |
| 25 | .597176 | .541702 | .929051 |
| 50 | .809398 | .749678 | .878943 |

Consistency drift accompanies observation error; this is descriptive, not its
identified cause. Do not interpret single latent dimensions or compare old latent
RMSE against a newly organized coordinate system as a physical-state error.

Train/Test latent norm mean/std/max=1.546784/.485370/3.885851 and
1.538013/.478950/3.912348. Per-dimension std min/median/max=.111669/.151095/.215781
and .109378/.154213/.206408. Zero near-zero-variance dimensions out of64; covariance
effective rank12.2857/12.0732. No evidence of global latent collapse. NoNaN/Inf or
observed explosion; max autonomous latent norm4.629386 and physical prediction
absolute max5.774427. Neither finite norms nor cosine alone establish useful dynamics.

### Conclusion and next suggestion

Joint optimization improves reconstruction and H1–H25 versus Frozen Explicit,
but materially worsens H50 and never beats Direct K10. This is a **mixed/negative
long-horizon result**, not the requested successful long-horizon World Model v1.
It does not match the all-horizons failure or latent-collapse cases: short-term
adaptation is useful and representation reconstruction improves, while autonomous
out-of-horizon consistency remains inadequate. Do not force one success taxonomy.

First retain the deterministic formulation for a separately authorized read-only
K10→H50 composition/latent-consistency diagnosis. These data do not isolate a
component-specific cause or prove a stochastic state-space transition is necessary.
No next experiment, policy, reward model or planning was started.

### Verification evidence

Full read-only artifact verifier independently reproduces converted initialization,
Train/Val losses/selection, baselines, Test metrics, manifests, normalization, all
hashes and strict future-observation corruption/deletion independence. Nine plots
visually reviewed. Newtests10pass/0fail; tracked-branch repository plus newtests
325total/324pass/0fail/1optionalX11GUI skip (`MINE_UAV_GUI_TEST=1`). Historical
untracked experiments are preserved, not silently added to this branch or suite.

Read-only review found no remaining critical/important issues. Verifier provenance
checks were strengthened with four tamper regressions before final verification.
Two additional minor regressions remain deferred (earliest-prefix-input gradient,
mixed-length padding vs isolated prefix); independent review checks passed and
existing prefix/weight/BPTT/reset/boundary tests pass. No observed algorithm failure.
