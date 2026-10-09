# Controlled On-Policy State-Distribution Model Adaptation

Branch: `feat/world-model-onpolicy-adaptation`

Base: `db532ac9b40d9fc49a1c2e18b55090aba4f6b19a`
Date: 2026-10-09

## Research question and scope

Does frozen-MPC visited-state coverage improve later-state prediction and
decision-cost fidelity beyond equal-budget scripted/replay-state adaptation?
This experiment changes training **data source**, not the planner, loss or
architecture. Counterfactual actions are not called actually executed on-policy
controller actions. No adapted model controls a closed-loop episode here.

Three models:

- Original: canonical frozen v3 seed0.
- Replay control: fine-tune canonical v3 using scripted-controller visited states
  plus true MuJoCo counterfactual candidate branches.
- MPC-state adaptation: the same initialization/budget, with states visited by
  the original frozen unconstrained MPC plus the same branch-generation rule.

Canonical checkpoint:
`mujoco/rl/models/joint_latent_world_model_v3_autonomous_consistency.pt`;
SHA256 `42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9`.
The original file and original in-memory E/T/D are never updated.

## Data isolation and matched budget

Fresh generation seed202710091; reset seed710090000+global target index. Targets
use the unchanged nominal x/y[-2,2],z[0.7,1.5] distribution. Train60,Val20,Final100
groups have unique IDs/seeds and no coordinates within1e-8 of one another, the
original120 PI dataset target groups or previous200 benchmark/holdout targets.
Old Train/Val/Test identities are read only for exclusion, never recomputed stats.

Each source has800 Train snapshots and200 Val snapshots. Per-target quotas are
14 for the first20 Train targets,13 for the remaining40, and10 per Val target.
Each source/target runs the entire original episode, including failures.
Decisions are searched in a predetermined seeded permutation, not selected by
cost, prediction accuracy or success. A snapshot is valid only if all8 original
candidate branches execute10 actual finite steps. Early terminal absorbing
padding is rejected for training; termination exactly on step10 is legal. All
rejected attempts/outcome reasons and failed source episodes remain in metadata.
No quota was reduced or target discarded. This valid-window filter induces
selection bias; these are not every state visited by either controller.

Eight branches: original candidate indices0(zero),1(previous-action repeat), and
six uniformly chosen distinct indices2..511, seeded by target and decision ID.
Both sources use the same generation/selection law. The original generator is
unchanged:512 sequences,H10,Gaussian increments std=.5 originalTrain action std,
original environment bounds, no support clipping. Source differences include
real histories/previous actions, not just positions.

Resulting datasets:6400 Train and1600 Val K10 windows **per source**. True prefixes
are stored once per snapshot; each branch supplies its own real10step future.
Each training batch has16 branch windows with exactly one explicit start each,
not all starts in the saved prefix. Previous action uses the original real
prefix up to the branch, then that branch's commands. No cross-episode window.

Source trajectories: Replay80/80 success. MPC80/80 failed or timed out:
32 excessive tilt,33 outside flight area,15 time limit. None was removed.

## Exact simulator/controller state

Reuse `decision_fidelity_snapshot`: copy all MjData, including qpos/qvel/act/ctrl,
integration/solver warmstart and derived fields; all mutable task/environment
state/counters/RNG; PI integral/yaw state; allocator state; previous action.
Restore with `mj_copyData`, without recomputing solver caches via `mj_forward`.
At every accepted training snapshot and every Final audit state, repeat
candidate0 and assert observation arrays and final snapshot fingerprint equal.
The independent verifier also replays the first fixed Final target, all stages,
and repeats candidate0 and scripted sequences against archived true arrays.

## Model and training

Architecture unchanged,74119 parameters: GRU11→64×1; residual Tanh transition
68→128→128→64; Tanh decoder64→128→128→7. Both adaptations start from canonical v3
weights, not its original pre-joint components. Fresh Adam defaults,lr.0003,
seed0,batch16,1000 optimizer updates. Paired NumPy index-permutation stream has
the same hash in both sources. No new clipping, auxiliary losses or weights.

Original v3 loss is literally reused through `loss_batch(..., starts=...)`:

```
z0 = E(real prefix)
z[k+1] = z[k] + T(z[k], normalized recorded branch action[k])
L_obs = mean(k=0..10, 7D): ((D(z[k])-o_true[k])/original Train obs std)^2
L_auto_cons = mean(k=1..10, 64D):
    ((z[k]-stopgrad(E(real prefix + real branch history through k)))/sigma_init)^2
L_total = L_obs + 0.1 L_auto_cons
```

The decoder returns normalized observations internally. All original
observation/action means/stds and saved original fixed initial-Encoder latent
std are reused unchanged; no new fitting. The full10step graph and source prefix
retain gradients. Future real branch observations enter only observation loss
targets and the detached consistency reference branch, not rollout state input.
E/T/D all update; Decoder receives no direct consistency gradient.

Validation checks every50 updates; select only the lowest own independent
Validation L_obs among steps50..1000. Initial validation is diagnostic, not a
selection candidate. Never use nominal or Final evaluation to choose a model.
Replay best step950; MPC-state best step700; both complete1000 updates.

## Shared unseen evaluation

After both best checkpoints and equal budgets are finalized, run the **original
frozen** MPC on all100 new Final targets. Select S0/S2/S3 nearest0%/50%/75% of
each realized episode's decision length, deduplicating very short episodes.
Generate exactly the candidates originally generated at each such decision.
All three models use these same physical/controller snapshots, real prefixes,
512H10 action sequences, true MuJoCo trajectories and true costs. No adapted
model replans or selects the evaluation state distribution.

True candidate rollout uses unchanged physics500Hz/control100Hz/policy25Hz.
Retain every evaluation candidate, including early failures, with the original
absorbing terminal rule and no newly introduced failure penalty. Scripted H10
feedback behavior from the same exact snapshot is a reference, not an extra
candidate. Feed its recorded actions to each model for pure latent prediction.

Cost unchanged:
`||error_H||² + .5||velocity_H||² + .1 yaw_H² + .05 mean||Δaction||²`,
including the actual previous action in the first smoothness increment.
Ranking/regret use the original tie-aware metrics; true-best26 candidates define
the best tail. Predicted-best-in-true5 and true-best-in-predicted5 are distinct.
Normalized regret divides selected-minus-oracle true cost by true cost spread.
Random expectation is the exact mean true candidate cost minus oracle cost.

Every model's encoder reads the same real prefix, then only its transition
propagates latent under candidate actions. No future observation feedback.
Original Train OOD geometry is only a shared state-distribution proxy; old raw
latent coordinates are not compared to adapted-model raw coordinates.

Nominal regression uses all legal starts in original PI Test with the original
unchanged manifest, at H1/10/25/50. H1 is the original one-step frozen prediction
protocol. The exact endpoint frame, including each episode's final observed
frame, is preserved. Adaptation never trains on these targets.

## Reproduction and artifacts

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python mujoco/rl/run_guarded_onpolicy_adaptation_collection.py --phase adaptation --workers 2
.venv/bin/python mujoco/rl/onpolicy_adaptation_training.py --source replay
.venv/bin/python mujoco/rl/onpolicy_adaptation_training.py --source mpc_state
.venv/bin/python mujoco/rl/run_guarded_onpolicy_adaptation_collection.py --phase final --workers 4
.venv/bin/python mujoco/rl/run_world_model_onpolicy_adaptation.py
.venv/bin/python mujoco/rl/verify_world_model_onpolicy_adaptation.py
```

The original local PI dataset is required; no duplicate raw dataset is committed.
Raw branch/Final arrays, partial results, per-update records and runtime logs
stay in ignored `mujoco/reports/world_model_onpolicy_adaptation_seed0_parts/`.
Only best adapted models, reports, manifests/hashes, necessary figures, source,
tests and this documentation are tracked. Validation checkpoints overwrite the
same best path; no intermediate epochs are kept.

The commands above are for **fresh collection** in a clean parts directory.
The supported guarded entrypoint writes runtime/execution-dependency signatures
before the first cache entry and refuses changes to Python/MuJoCo/NumPy/PyTorch
or relevant dynamics/normalization/physics code. It refuses collection reuse of
existing unstamped caches. Original low-level collectors are retained unchanged
as hash-bound historical generation implementations, not supported safe-resume
entrypoints; calling them directly bypasses this guard.

Historical per-cache runtime stamps were **not recorded** for this completed
experiment; no retrospective stamp is fabricated and original cache identities
are preserved. Same-environment exact physics repeats pass, with no evidence of
mixed runtimes, but they cannot establish historical per-cache runtime versions.
Existing artifacts remain available for read-only evaluation/verification.

Eight-worker data collection exited with BrokenProcessPool; kernel logs were
unavailable. Resume with2workers completed the same quota/hashes. Final physics
resumed immutable per-target caches with4workers after observing~2GB RSS per
worker. Parallelism alone changed; no scientific budget or planner changed.

## Results, verification and conclusion

All100 unseen targets complete, with100 S0,100 S2,100 S3 states and153600 true
candidate rollouts. All900 model-state comparisons use the same archived truth.
The original source controller has1 success,32 excessive-tilt failures,50 outside
flight-area failures and17 timeouts. All are included at every stage; there is
no stage attrition or success-only filtering. These are **not** adapted-model
closed-loop results.

### Matched training

E/T/D initial parameter hashes are identical in B/C and equal to canonical v3:

```
E 693981f23e2ea2b10588a4afddbcccd5aa951058156e5e4f771437f35d2f71a0
T 16faf70b486cf67bcdba89cd5037853731c269f662a4d870b59bf8d70a74c010
D 3fe33c5ccd5096224978af87aabb9376fa8b7b5d3bee0b9de2c77fd6891bc580
```

Paired1000-batch order SHA256:
`2777b3c645d43733835de6661f2a0cd365fbe1a9e34543bf11cd0c0dc407f2c6`.
Each best model updates all3 components; final/best hashes are in the report.

| Source | Best step | Best Train L_obs | Best Val L_obs | Best Train L_auto | Best Val L_auto |
|---|---:|---:|---:|---:|---:|
| Replay | 950 | 0.000335 | 0.000563 | 0.001940 | 0.002406 |
| MPC-state | 700 | 0.150018 | 0.349465 | 0.164487 | 0.209627 |

These losses belong to **different source distributions**; the absolute B/C
validation loss comparison is not the causal test. The common Final cohort is.
Max E/T/D total-gradient norms: Replay0.782/5.727/1.048;
MPC-state10.940/13.559/2.632. Maximum training predicted-latent norm2.404/6.443.
No NaN/Inf, clipping or numerical remedy was introduced.

### Prediction on common candidate sets

Values below are **mean per-state endpoint NRMSE**, not the pooled RMSE of all
states. Whole candidate sets use all512 candidates per state. Different models'
selected-candidate errors refer to their own selected indices in that same set.
The JSON retains physical7D RMSE/MAE for each state/selection/horizon.

| Stage | Model | Whole H1 | Whole H5 | Whole H10 | Selected H10 | True-best H10 | Scripted H10 |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | Original | 0.113889 | 0.152726 | 0.179442 | 0.204868 | 0.181085 | 0.176763 |
| S0 | Replay | 0.092433 | 0.129830 | 0.143105 | 0.160850 | 0.139908 | 0.139400 |
| S0 | MPC-state | 0.107724 | 0.185070 | 0.223501 | 0.244241 | 0.232078 | 0.206538 |
| S2 | Original | 0.209189 | 0.302845 | 0.418474 | 0.464510 | 0.363986 | 0.355981 |
| S2 | Replay | 0.282865 | 0.425802 | 0.591800 | 0.623900 | 0.547254 | 0.534007 |
| S2 | MPC-state | 0.159292 | 0.288102 | 0.451411 | 0.462171 | 0.428128 | 0.436643 |
| S3 | Original | 0.480542 | 0.693396 | 0.890673 | 0.951701 | 0.889385 | 0.823901 |
| S3 | Replay | 0.579230 | 0.812470 | 1.020330 | 1.057961 | 1.008990 | 0.953707 |
| S3 | MPC-state | 0.319700 | 0.567357 | 0.799155 | 0.806570 | 0.793113 | 0.813840 |

C reduces B's mean whole-H10 error23.72% atS2 and21.68% atS3, but is56.18%
worse atS0. AtS2 C is still7.87% worse than Original; atS3 C improves Original
by10.27%. The benefit is state-dependent, not universal prediction superiority.

### Decision fidelity

Correlations/regret are per-state then averaged. Rank percentile below is the
median; lower is better. True-best tail is26/512 candidates, not predicted tail.
Top5 columns separate the two directions; all counts use100 states.

| Stage | Model | Spearman | Kendall | Best-tail rho | Mean norm regret | Selected true rank median | Top1 | pred-best∈true5 | true-best∈pred5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | Original | 0.585764 | 0.429727 | 0.069926 | 0.129592 | 23.19% | 2 | 5 | 3 |
| S0 | Replay | 0.503914 | 0.364947 | 0.025067 | 0.148668 | 26.91% | 0 | 5 | 7 |
| S0 | MPC-state | 0.280735 | 0.196036 | 0.074541 | 0.221141 | 48.63% | 0 | 3 | 1 |
| S2 | Original | −0.121720 | −0.085790 | −0.196923 | 0.563292 | 64.38% | 2 | 4 | 4 |
| S2 | Replay | −0.064821 | −0.049791 | −0.126003 | 0.519367 | 47.55% | 1 | 1 | 4 |
| S2 | MPC-state | −0.021567 | −0.016821 | −0.066721 | 0.503737 | 55.48% | 3 | 9 | 9 |
| S3 | Original | −0.052498 | −0.044796 | −0.165176 | 0.548246 | 39.43% | 2 | 6 | 6 |
| S3 | Replay | −0.013394 | −0.013367 | −0.108533 | 0.529725 | 51.08% | 3 | 9 | 5 |
| S3 | MPC-state | −0.052754 | −0.041297 | −0.050537 | 0.512777 | 56.65% | 3 | 7 | 8 |

Within the common cohort, C−B normalized-regret mean is−0.015630 atS2 and
−0.016948 atS3, but paired SD0.295913/0.289258 and median0 atboth. Regret improves
in only48/100 and46/100 states. C−B best-tail rho is+0.059282/+0.057997, yet
the resulting tail correlation remains negative. S3 whole rho worsens by0.039360
and median selected true rank also worsens. This is not robust decision-ranking
recovery. No statistical-significance claim or outcome-based model selection.

Random expected normalized regret:S0.249414,S2.521868,S3.529644. C's small
later-stage mean advantage over random is insufficient to establish dependable
best-tail selection; exact state-level deltas remain in the report.

### Nominal regression and representation health

Original Test dataset/start manifest are unchanged. All legal endpoint windows
are evaluated; the original v3 curve reproduces independently.

| Model | H1 | H10 | H25 | H50 |
|---|---:|---:|---:|---:|
| Original | 0.044057 | 0.104600 | 0.201413 | 0.308050 |
| Replay | 0.092331 | 0.175406 | 0.309185 | 0.458162 |
| MPC-state | 0.098742 | 0.276733 | 0.659729 | 1.172470 |

Replay H50 worsens48.73%; MPC-state worsens280.61%. C's nominal H1 also worsens
124.12%. **Both adaptations fail nominal retention; C's forgetting is severe.**
Neither best model replaces canonical v3.

This is not global collapse: all64 dimensions have std>1e-6. Nominal Train/Test
effective rank: Original11.96/11.74,Replay11.87/11.61,MPC-state12.22/12.17.
MPC-state Test latent std min/median/max and norm mean/std/max are recorded in
JSON; Test mean/max norm1.624/3.578. All nominal error accumulators have zero
nonfinite windows. Latent variance alone does not guarantee semantic retention.

### Shared task-progress ceiling and state shift

Oracle true-cost candidates improve distance in96/100 S0,21/100 S2,29/100 S3
states; mean distance reductions+0.0102,−0.2572,−0.2874m. Later H10 recovery is
limited even without model ranking error. These common oracle outcomes cannot
change when only the evaluated prediction model changes.

Scripted true-cost median percentiles:S0.1953%,S2.6836%,S3 2.8320%. C's predicted
percentiles28.91%,65.72%,89.75% still miss scripted sequence quality, especially
later. Scripted is a feedback H10 reference, not proof of long-term recovery
from every failed state; no new cost/failure penalty was added.

Original-Train observation-standardized RMS grows0.5282→2.0375→2.6544. The
report retains state-OOD/ranking/regret correlations. These are associations,
not sole-root-cause proof; adaptation changes history and action-context data
along with visited-state coverage.

### Verification, limitations and decision

Independent verifier rechecks all rawfile hashes/target exclusion/labels, both
1000-update budgets and order hashes, all post-update selection points, each
best Validation loss, all153600 true cost formulas, all900 model-state metrics
and aggregates, model hashes, repeated exact-physics snapshots and scripted
references. A separately executed nominal run matches all final curves/latent
statistics. All parameters are frozen during evaluation.

Review fixes add recursive provenance checks (including full training records),
exact comparison of compact training reports to archived updates, the complete
50/100/.../1000 Validation schedule, independent reconstruction of all state OOD
fields from original-model prefixes and hash-bound original Train geometry,
source-episode summary equality, and numeric conclusion percentage checks.
Regression tests reject corrupted best Val loss, OOD latent norm, nominal
regression percentage, source summaries and nested hashes. Future cache reuse
has separate unstamped/runtime/dependency-drift rejection tests.

New tests:24 passed,0 failed,0 skipped. Full filesystem suite with the unchanged
read-only historical fixtures:531 run,527 passed,0 failed,4 known platform or
optional-source skips. Bare filesystem discovery:469 run,30 pre-existing missing
historical dependency errors,1 skip. Missing dependencies and all30 names are
listed in `WORLD_MODEL_LATER_DECISION_RANKING_AUDIT.md` (verification section).
No historical files are restored/edited. The same in-memory fixture command
there applies; no new algorithm test is skipped. A sole fresh read-only review
found no invalid current scientific values; both Important verification/cache
findings were fixed in one RED-to-GREEN pass, without retraining or new policies.

Deferred review minors (current results unaffected): a quota-failure worker
does not save its rejection log before raising (no quota failed); separate fresh
adaptation Train/Val reconstruction and latent-variance summaries are not
reported (health evidence is nominal Train/Test plus finite training batches);
very-short-episode stage deduplication can omit S2 (all current Final episodes
have at least61 decisions). These are not silently generalized health/coverage
claims outside the completed cohort.

Conclusion: MPC-state coverage is relevant to later prediction, but this
fixed-objective, fixed-budget data adaptation does **not** robustly repair
decision-critical ranking and **does not preserve nominal prediction**. This
is a mixed/negative result, not an accepted upgraded world model. Keep canonical
v3 unchanged. No adapted-model closed-loop success improvement was measured.

Limitations: one adaptation seed/model, nominal simulator,800 valid-window
filtered snapshots per source,8 sampled original branches per snapshot,
different source-specific Val distributions, relative realized-episode stages,
fixed K10/H10/N512/cost, no replay mixing or retention mechanism, and no adapted
policy closed-loop evaluation. Train/Test latent coordinates may change.

Next (one suggestion only): an offline **best-tail decision-cost-supervised
modeling control**, with nominal regression retained as an acceptance gate.
Do not execute it or modify MPC from this experiment.
