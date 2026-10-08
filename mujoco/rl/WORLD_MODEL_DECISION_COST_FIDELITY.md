# Offline World Model Decision-Cost Fidelity Audit

Branch: `feat/world-model-decision-cost-fidelity`.
Base: `feat/latent-mpc-action-support` at
`d565cb8f3f85dccef1da4d5a9f713cf2f4142433`.
Date: 2026-10-08. No model training, new closed-loop controller, or MPC changes.

## Protocol and integrity

Canonical frozen seed0 v3:
`models/joint_latent_world_model_v3_autonomous_consistency.pt`, SHA256
`42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9`.
All modules eval/requires_grad=False; no optimizer. Component hashes before/after:

- E: `693981f23e2ea2b10588a4afddbcccd5aa951058156e5e4f771437f35d2f71a0`.
- T: `16faf70b486cf67bcdba89cd5037853731c269f662a4d870b59bf8d70a74c010`.
- D: `3fe33c5ccd5096224978af87aabb9376fa8b7b5d3bee0b9de2c77fd6891bc580`.

All100 benchmark +100 holdout first-decision reset identities retained, original
target order/reset seeds. N=512, H=10 (0.4s), physics/control/policy500/100/25Hz.
Both original generators reused without modification: unconstrained and fixed
Train central95% marginal box. Original epsilon scale remains0.5×Train action
std, zero/current anchors unchanged. SeedSequence([0,split_index,episode_index])
and first draw match the previous control experiment.

Historical raw candidate arrays/physical snapshots were not archived. This is
**deterministic reconstruction**, not a claim of matching nonexistent old raw
hashes. For each of200 states, immutable original reset/controller/sampler/loader
sources and saved observation/latent/epsilon hashes, chosen index/action,
predicted terminal, min/median/mean cost reproduce exactly. New candidate-set and
snapshot hashes are in the manifest. Both conditions share initial latent/noise.

Each candidate restores an independent full MjData copy via mj_copyData, including
qpos/qvel/ctrl/act/integration/history/solver-warmstart/derived data. All mutable
Python env/task state, target/yaw, RNG, episode/done/success/previous-distance,
full PI and nested attitude/yaw controller, allocator, and previous executed
action are restored. The compiled model/XML/config remain immutable. No7D-only
state restoration and no mj_forward recomputation on restore. Snapshot/restore
tests also cover nonzero PI/yaw/previous action and candidate-order independence.

204,800 true candidate rollouts and200 scripted references completed. Model uses
only initial encoder latent and the exact same recorded candidate commands for
pure latent recursion; no future observations or decoded-observation feedback.
Scripted uses true feedback to GENERATE its10 commands, then the identical fixed
command sequence is evaluated by the model and replayed open loop in MuJoCo.
It is a reference, never added to the512 candidates.

Original hand-designed cost, NOT the RL reward:

`J=||error_xyz_H||² +0.5||velocity_xyz_H||² +0.1*yaw_error_H² +0.05*mean||a[k]-a[k-1]||²`.

Actions/smoothness use original normalized command units; `a[-1]` is previous
executed action. Cost function reused directly. No failure penalty exists and
none was added. On original termination, terminal observation is held absorbing
without post-done physics; smoothness still uses the original full sequence.
Physical/invalid/termination flags accompany raw cost, no candidate exclusion.
In this first-decision cohort: **0 failures,0 invalid,0 early terminations**.
The invalid-state regression explicitly exposes the original sanitized zero
observation/raw-zero-cost behavior, rather than hiding it with a new penalty.

## Ranking and regret

Coefficients below are means of per-state512-candidate correlations, not pooled
across targets. Spearman uses average ranks, Kendall is tau-b with exact ties;
NumPy implementation with literal hand-checked tie/reversed/constant fixtures.
Undefined coefficients would be null with counts, never dropped silently.

| Set / split | Spearman | Kendall | Mean normalized regret | Random expected | Model better than random |
|---|---:|---:|---:|---:|---:|
| UN benchmark | .608633 | .447334 | .129662 | .252033 |94/100|
| UN holdout | .624986 | .460769 | .125566 | .253346 |93/100|
| Supported benchmark | .593505 | .435537 | .144566 | .275133 |92/100|
| Supported holdout | .610970 | .449945 | .137080 | .275990 |94/100|

`regret=J_true(argmin J_pred)-min J_true`;
`normalized regret=regret/(max J_true-min J_true+1e-12)`.
Random expected regret analytically averages ALL512 choices; no random draws or
selected successful states. Support changes each set's cost range, so normalized
regret alone is not a fixed-denominator cross-set causal effect.

| Set / split | Absolute mean / median / P75 / P90 / max | Normalized median / P75 / P90 / max |
|---|---|---|
| UN benchmark |.030523/.027615/.039793/.055525/.132043|.124067/.180289/.236625/.418875|
| UN holdout |.029800/.026381/.040055/.061099/.094280|.109623/.179893/.226810/.399254|
| Supported benchmark |.029406/.025771/.040828/.055017/.105225|.137536/.203487/.266615/.384481|
| Supported holdout |.028186/.026222/.039174/.049337/.092853|.133774/.192892/.244688/.443943|

| Set, all200 | Top1 exact | Pred-best in true Top5 | True-best in predicted Top5 | Selected true rank median |
|---|---:|---:|---:|---:|
| UN |7/200|15/200|14/200|26.22%|
| Supported |7/200|19/200|18/200|25.05%|

Rank percentile0=best,1=worst; tied costs get average rank; Top5/argmin use stable
indices. Model-selected candidate is in the true worse half in35/200 UN and34/200
Supported states. Mean overall rho .616810→.602238; tau .454052→.442741; absolute
regret .030161→.028796, normalized .127614→.140823. Thus support modestly changes
selection but **does not clearly improve ranking fidelity**. Model has partial
first-decision utility versus random, not reliable optimal-tail fidelity.

Calibration mean per-state Pearson .658982/.637449; MAE1.466649/1.467374,
RMSE1.468575/1.469081; mean predicted-minus-true bias−1.466647/−1.467374 (UN/S).
Large calibration bias is distinct from ranking quality; it does not by itself
prove that model selections are useless.

## Oracle candidate utility and scripted reference

All200, mean values:

| Sequence | True cost UN / S | Distance reduction UN / S [m] | Terminal speed UN / S [m/s] | Distance improved |
|---|---:|---:|---:|---:|
| True-cost best candidate |2.704298/2.704117|.011286/.011131|.098128/.093942|193/200 both|
| Model-selected candidate |2.734459/2.732913|.003297/.003421|.130510/.125512|123/200,127/200|
| Scripted reference |2.707006 both|.005400 both|.060778 both|200/200|

No oracle/model/scripted physical failures in H10. True-best speed<.15m/s in
175/200 UN and177/200 Supported states. Oracle candidates generally make task
progress, not uniformly harmful, but the progress is only~1.1cm over0.4s from
rest. This is NOT an oracle closed-loop success experiment.

Scripted true cost median percentile **0.1953%** for both sets (better than99.8%
of candidates at the median). Mean better-than fraction99.5576% UN /99.5322% S.
It beats every candidate in78/200 states for both sets and beats model-selected
cost in189/200. Conversely candidate true-best beats scripted in122/200; mean
scripted-minus-oracle cost .002709/.002889 is small. Hence script-quality proposal
coverage is incomplete, but it is wrong to say all original candidates are bad.

Scripted predicted median percentile **11.9141% UN /11.3281% Supported**;
mean12.6309%/12.2305%. Model recognizes it as better than most random sequences,
but does not recognize its actual near-best quality. It predicts scripted better
than all candidates in1/200 UN and0/200 Supported states (vs78/200 true).
Scripted prediction mean H10 NRMSE .171566.

The historical scripted controller achieved100/100 on both splits. This audit
only examines its first0.4s, so it does not establish the complete mechanism of
that closed-loop success. It demonstrates coherent low-speed positive progress
and near-best original-cost behavior that model ranking systematically underrates.

## Cost alignment, error, margins

Mean within-state Spearman J_true versus distance: .507229/.542430; versus
distance reduction:−.507229/−.542430; versus speed: .776360/.751408. The reduction
correlation is algebraically the negative of distance within a fixed state, not
independent evidence. Original cost favors closer/slower terminal states in this
cohort; this neither proves nor disproves alignment with15s closed-loop success.
No evidence here that scripted is rejected by the TRUE cost.

Mean whole-set H10 observation NRMSE .176283/.175471; selected-candidate
.200388/.196549. Across200 states, prediction error vs rho Pearson−.566/−.537,
Spearman−.546/−.519; vs normalized regret Pearson .517/.517, Spearman .535/.526.
These are descriptive associations, not causes.

Mean true best-second margin .002755/.002746, spread .233698/.202108. Small
margins explain why exact Top1 can be brittle, but regret is not only adjacent
ties: true margin versus normalized regret Spearman **+.257/+.252**, cost spread
versus regret .087/.103. Largest errors are not confined to tiny-margin states.
Report includes all margin/spread quartiles and per-state values, no filtering.

## Conclusion and one next recommendation

Most supported: **incomplete best-tail decision-cost ranking fidelity**, with
partial proposal-coverage limitations. Model selection beats random expectation
in93%+first states, yet usually misses near-best sequences and undervalues
scripted. Marginal support clipping does not repair this. Oracle candidates and
scripted generally improve true H10 task progress; pure candidate-uselessness or
strong true-cost misalignment is not supported by these first-state observations.

This audit cannot uniquely explain the later near-total closed-loop failure.
It does not test later visited states, oracle receding-horizon control, or
replanning temporal consistency. Do not infer a unique model/cost/generator root
cause or automatically tune MPC/CEM/MPPI.

Only next recommendation: reuse saved failed MPC trajectories for an offline
**exact-state later-decision ranking audit**, keeping candidate generator/cost
unchanged, to test degradation under receding-horizon state-distribution shift.
Not executed in this stage.

## Reproduction and artifacts

From repo root, existing `.venv` and original local dataset:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python mujoco/rl/run_world_model_decision_fidelity.py --workers 8
.venv/bin/python mujoco/rl/finalize_decision_fidelity.py
.venv/bin/python mujoco/rl/decision_fidelity_verification.py
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_decision_fidelity*.py'
```

Cache identities bind original/new execution sources, forward/loader, checkpoint,
normalization, support, dataset/target hashes and runtime. Stale caches reject;
do not relabel results. NumPy-only correlations require no new dependency.

Reports: `../reports/world_model_decision_cost_fidelity_seed0.json` and
`../reports/decision_cost_fidelity_manifest_seed0.json`; ten requested PNGs under
`../reports/`. Candidate endpoint/true-cost arrays (~36MB with per-state
fragments) remain local/ignored in `world_model_decision_cost_fidelity_seed0_parts/`;
hashes/config/summary/manifests committed, not raw candidate trajectories.
Independent verifier recomputes all204800 endpoint costs/model outputs, all
derived ranking/split metrics,200 candidate manifests, source/figure hashes and
postprocessing; fixed states have8 additional exact physics replays. All200
states already have selected/oracle/scripted exact restore/replay checks.

21 new tests pass; full filesystem suite: **482 passed,0 failed,4 skipped**
(486 run). Full filesystem tests use unchanged historical helpers/report
read in memory from Git `8879754a43302e9c55b27de4ce3c629c7e4e1a53`, no old/tmp
dependency and no algorithm mutation. Four known skips: GUI requires X11 and
three optional absent recurrent-source checks. Counts and exact fixture provenance
recorded in JSON. Historical untracked artifacts left untouched; final status
contains82pre-existing untracked entries, none staged with this experiment.

Sole fresh read-only review found one Important edge case: paired postprocessing
subtracted undefined correlation values. Fixed with RED→GREEN tests for either
or both conditions having null Spearman/Kendall, including final verification;
missing counts retained. All actual200 coefficients are finite, and scientific
state/results/analysis values remained bit-identical. No deferred review issues.
