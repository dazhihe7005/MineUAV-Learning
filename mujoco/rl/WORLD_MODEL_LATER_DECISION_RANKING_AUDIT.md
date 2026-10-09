# Later-decision exact-state ranking audit

Branch: `feat/world-model-later-decision-ranking`.
Base: `feat/world-model-decision-cost-fidelity`,
`e21e7c0cff7e18ce1e4e8074bd9a323e387b30e5`.

This is a read-only audit of the **original** failed MPC trajectories, not a
new closed-loop controller. It asks whether ranking degrades at later states,
whether whole-set ranking hides true-best-tail failures, and whether the
oracle proposal set or scripted reference still makes real task progress.

## Fixed model and protocol

Canonical seed0 v3:
`models/joint_latent_world_model_v3_autonomous_consistency.pt`.
SHA256: `42571249195a7ab396dbbcdec9764127ac4aae47eb069efb88e3465168790fa9`.
Encoder, Transition and Decoder are frozen/eval; there is no optimizer,
parameter update, new cost, support proxy, sampler, controller or horizon.

N=512, H=10, physics/control/policy=500/100/25Hz. Cost is unchanged:

`J = ||error_H||² + 0.5||velocity_H||² + 0.1yaw_H² + 0.05mean||a_k-a_(k-1)||²`.

Actions in the last term are the original normalized commands, with the actual
previous executed action as `a_-1`. Candidate0 is zero, candidate1 repeats the
previous action; other510 sequences use the original clipped Gaussian random
walk, std=0.5×saved Train action std. Supported candidates use the original
Train central95% box. No resampling, retuning or replacement candidate occurs.

All original100 Benchmark and100 Holdout targets are retained **in each
condition**:400 original controller episodes, not400 distinct reset targets.
The single successful Unconstrained Holdout episode is retained and a failed-only
paired analysis is provided. Post-S0 condition comparisons are not identical-state
comparisons: original closed-loop states and realized episode lengths differ.

## Exact reconstruction and simulator restore

`later_ranking_replay.py` adds snapshot recording to the original
`InstrumentedSupportMPC` and uses the unchanged `evaluate_episode`. Every saved
decision action, cost(min/median/mean) and epsilon hash is checked elementwise.
All original scientific episode summaries are checked;12 archived representative
episodes additionally check full observations, previous actions, encoder latents,
selected indices and predicted selected terminal observations. Timing is excluded
from exact equality, since wall-clock execution is not a deterministic signal.

Snapshots use full `MjData` copy and `mj_copyData`, including qpos/qvel/act/ctrl,
warmstart, applied forces, time, derived fields and all other MjData buffers.
All mutable environment/task/target/counter/RNG state, full PI plus nested
attitude/yaw controller state, allocator state and previous command are deep
copied. The compiled immutable MjModel is shared. Restore does not call
`mj_forward` or reconstruct a simulator from the7D observation.

Planner RNG state hashes before/after the original draw, epsilon hashes,
candidate-set hashes and snapshot fingerprints are recorded. At every audited
state, reconstructed selected candidate index/action must match the original
decision. Model-selected, oracle and scripted rollouts are restored/repeated
exactly in the main audit; independent verification repeats fixed episode
identities across all stages.

## Stage and composition rules

For an episode with L actual decisions, stage indices are
`floor(p*(L-1)+0.5)` for p=0/.25/.5/.75. Duplicate rounded indices are omitted,
retaining the earlier stratum. Every stage reports eligible count, actual
decision indices, realized progress, original termination-type composition,
initial and current distance distributions. No absent stage is fabricated.

Main trend summaries include within-episode S3−S0 changes only for episodes with
all four distinct stages. Failed-only sensitivity excludes the one success.
Even without stage attrition, selecting relative progress of realized lengths
is a state-selection protocol, not a randomized intervention on elapsed time.

## Frozen predictions and true rollouts

At each original decision the real-history encoder latent initializes all512
sequences. `prediction_trajectory(model,stats,z,actions)` has no future-observation
argument. Ten residual Transition steps autonomously propagate latent; Decoder
outputs are never fed back to Encoder or Transition.

MuJoCo executes the same10 commands from the exact snapshot, with original PI
and task dynamics. If original termination occurs, later observations are
absorbing, no additional physics runs, and all10 commands still enter the
original smoothness cost. Physical/invalid failure flags remain separate: there
is **no invented failure penalty** and no candidate is deleted.

Scripted reference uses true feedback from the same snapshot to generate its H10
sequence. That identical sequence is supplied open-loop to the frozen model;
it is never inserted into the512 candidate set.

## Metrics and definitions

- Whole512 Spearman(midrank) and Kendall tau-b, stable argmin/Top5.
- Both directed Top5 tests: predicted-best in true Top5, true-best in predicted Top5.
- Regret=`J_true(model choice)-min J_true`; normalized by full true-cost range+1e-12.
- Random expected regret is the exact average of all512 true costs minus the minimum.
- Selected true rank percentile=`(average one-based true rank-1)/511`;0=best.
- Scripted percentile is empirical mid-CDF among candidate costs;0=best.
- True best tail is exactly26 stable lowest true-cost candidates(ceil5% of512).
  Tail correlations/MAE and pairwise ordering accuracy are reported; true ties
  are excluded from ordering accuracy, prediction ties get half credit.
- Undefined constant-cost correlations are null, with missing counts, not zero.
- H10 terminal NRMSE is distinct from10-step trajectory NRMSE. Both use saved
  Train observation std. Whole-set pooled RMSE and mean candidate RMSE are
  separately named. Selected/oracle/scripted errors are not hidden by averages.
- True best-second margin, full spread and true best26 spread accompany rankings.
- Task utility reports distance reduction, terminal distance/speed and failure.

Train-only geometry uses all147,219 recorded Train observation/action frames,
588 complete episodes with encoder reset at every boundary. It does not use
Validation/Test/Benchmark/Holdout. Observation/action standardized RMS, latent
standardized RMS, regularized covariance Mahalanobis RMS and95% PCA residual
are descriptive proxies, not strict OOD/manifold boundaries. Covariance ridge is
`max(1e-12,1e-4*trace/64)` and std floor1e-6, matching the existing audit primitive.
Action OOD uses the decision's selected command; the previous command's distance
is also recorded separately. Rank/OOD/error/margin correlations are descriptive,
not causal.

## Reproduction and artifacts

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python mujoco/rl/finalize_later_ranking.py --collect --workers 8
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python mujoco/rl/later_ranking_verification.py --workers 8
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_later_ranking*.py'
```

Requires existing local dataset, original aggregate archives and representative
traces. Source/runtime/model/geometry identities bind cached execution; stale
results are rejected, never relabeled. Candidate true/predicted costs, endpoints,
per-candidate trajectory error summaries and scripted10-step traces are local-only
in `mujoco/reports/world_model_later_decision_ranking_seed0_parts/`. No512-candidate
raw trajectories or full snapshot dumps are committed.

The entrypoint's report-only aggregation wrapper treats differing candidate
termination-reason keys as categorical counts. It replaces only the collector's
final aggregation alias for the scoped call, restoring it afterward. Original
physics/encoder/candidate sources and cached execution identities are unchanged;
reporting sources are independently hashed. The low-level collector is not the
recommended report entrypoint because its generic numeric summary assumes an
identical nested-key schema.

Tracked results:

- `mujoco/reports/world_model_later_decision_ranking_audit_seed0.json`
- `mujoco/reports/later_decision_ranking_manifest_seed0.json`
- `mujoco/reports/later_decision_train_distribution_seed0.json`
- Ten aggregate PNGs listed in the report with hashes.

Limitations: single canonical model, original finite H10/N512 cost/proposals,
nominal MuJoCo, first and relative later decision strata; no retraining, causal
uniqueness proof or modified-controller experiment. Train action support remains
a marginal box rather than joint state/action/history support.

## Results — complete cohort

All400 original episodes were exactly replayed; all1,600 audited states and
819,200 candidate H10 rollouts completed. Every stage has100 Benchmark+100
Holdout states **per condition**. All400 episodes have four distinct stages;
there is no stage attrition in this realized cohort. UN original decision lengths
span31–375, Supported73–375. Mean absolute indices S0/S1/S2/S3:
UN0/55.23/110.27/165.33; Supported0/63.21/126.06/189.10.

Termination composition is unchanged across stages: UN81 outside-flight-area,
72 excessive-tilt,46 timeout,1 success; Supported92 outside-flight-area,38
excessive-tilt,70 timeout. All stages share the original target-distance
distribution: mean1.549m, median1.616m, range0.269–2.700m. Current mean distance
S0→S3: UN1.549/1.707/2.292/3.038m; Supported1.549/1.643/2.380/3.325m.
Actual-state selection and differing absolute durations still constrain causal
interpretation even though eligibility/target composition is constant.
The Train95% PCA basis retains13 components in this canonical encoder coordinate.

| Condition / stage | Spearman | Kendall | Top1 | pred-best∈true5 | true-best∈pred5 | Mean normalized regret | Random expected | Selected true rank median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| UN S0 | .616810 | .454052 | 7/200 | 15/200 | 14/200 | .127614 | .252689 | 26.22% |
| UN S1 | .176074 | .136063 | 3/200 | 8/200 | 11/200 | .365686 | .437524 | 31.80% |
| UN S2 | −.111023 | −.082975 | 3/200 | 14/200 | 7/200 | .534100 | .512194 | 52.05% |
| UN S3 | −.103874 | −.082498 | 7/200 | 14/200 | 11/200 | .548530 | .516323 | 59.88% |
| Supported S0 | .602238 | .442741 | 7/200 | 19/200 | 18/200 | .140823 | .275561 | 25.05% |
| Supported S1 | .144396 | .115850 | 1/200 | 9/200 | 7/200 | .415527 | .444851 | 36.59% |
| Supported S2 | −.084374 | −.062479 | 5/200 | 10/200 | 6/200 | .518107 | .483643 | 61.25% |
| Supported S3 | −.159296 | −.123437 | 5/200 | 11/200 | 10/200 | .598313 | .524813 | 77.69% |

Mean absolute regret S0/S1/S2/S3: UN .030161/.185175/.285876/.430929;
Supported .028796/.198137/.252253/.398007. Model beats analytical random
expectation in UN187/137/96/96 states and Supported186/121/94/81 states.
Thus partial first-decision ranking utility is not preserved at later states.

Paired S3−S0 Spearman: UN −.720684±.602891(sample SD), median−.774122,
177/200 episodes deteriorate; Supported −.761533±.588433, median−.883210,
174/200 deteriorate. Regret paired increase: UN+.420916,164/200;
Supported+.457490,172/200. Selected true-rank mean changes:+25.20/+33.77
percentage points. Failed-only UN199 episodes: mean rho change−.723830,
consistent with the retained-success analysis. Supported has200 failed episodes.

### Whole-set versus best tail, margins

| Condition | Best26 Spearman S0/S1/S2/S3 | Pairwise accuracy S0/S1/S2/S3 |
|---|---|---|
| UN | .081414/−.134745/−.160995/−.185979 | .5284/.4525/.4432/.4345 |
| Supported | .090800/−.103785/−.157456/−.142841 | .5310/.4616/.4446/.4497 |

Tail cost MAE S0/S1/S2/S3: UN1.4866/.7749/2.1003/4.2136;
Supported1.4858/.6588/2.3120/4.7405. Whole-set moderate S0 ranking hides weak
best-tail discrimination; at later stages both whole and tail ranking degrade.
Median true best-second margins grow .001872→.013226(UN), .001889→.014267
(Supported). Median best26 cost spread grows .014278→.099377 and
.014159→.096949. Whole spreads also grow .233549→.612021 and
.201665→.567081. Tail spread versus normalized-regret pooled Spearman is
.6797/+.7330; worsening is not confined to ever-smaller near-tied costs.

### H10 prediction validity

Numbers below are means of per-state terminal NRMSE. The JSON additionally
contains mean candidate NRMSE and full10-step trajectory NRMSE, separately named.

| Condition / stage | Whole512 pooled | Model selected | True-best | Scripted |
|---|---:|---:|---:|---:|
| UN S0 | .176283 | .200388 | .180092 | .171566 |
| UN S1 | .265360 | .292217 | .248442 | .216510 |
| UN S2 | .482519 | .515430 | .432184 | .409625 |
| UN S3 | .807521 | .874517 | .791896 | .738929 |
| Supported S0 | .175471 | .196549 | .179212 | .171566 |
| Supported S1 | .228406 | .245571 | .206589 | .189650 |
| Supported S2 | .421495 | .452912 | .398574 | .383550 |
| Supported S3 | .685596 | .732984 | .685659 | .653266 |

Supported has lower later observation error than UN, but does not restore
decision ranking. Prediction accuracy and decision-cost fidelity are distinct.
Pooled selected-error versus rho/regret Spearman: UN−.3650/+.3382;
Supported−.4262/+.3990. Corresponding Pearson correlations are weaker:
UN−.0980/+.1105; Supported−.1650/+.1772. These are correlations, not proof
that a single prediction-error scalar explains ranking failure.

### State distribution shift

| Condition | Observation standardized RMS S0/S1/S2/S3 | Actual encoder latent Mahalanobis RMS | PCA residual L2 |
|---|---|---|---|
| UN | .5154/1.3925/2.1355/2.6385 | 1.4242/2.0633/3.0674/3.7377 | .5421/.5183/.7318/.8450 |
| Supported | .5154/1.4097/2.2441/2.6661 | 1.4242/1.9712/3.0185/3.4584 | .5421/.5031/.7228/.7851 |

Selected-action standardized RMS: UN .5012/1.6782/1.7989/1.8497;
Supported .5082/1.5164/1.5654/1.6384. The manifold queries are current
real-history encoder latents, not the autonomous future latent states from a
previous composition audit. Diagnostic scales use all recorded Train frames;
model-input normalization remains the unchanged saved Train normalization.

Observation OOD versus rho/regret pooled Spearman: UN−.5802/+.5386;
Supported−.5854/+.5379. Latent Mahalanobis versus rho/regret: UN−.5173/+.4662;
Supported−.5445/+.4969. Latent OOD versus selected prediction error:
UN+.7681, Supported+.7402. All Pearson and per-stage correlations are in JSON.
Pooled correlations combine progress effects and repeated episodes;800 states
are not claimed to be800 independent trials.

### Oracle progress and scripted reference

| Condition | Oracle positive progress S0/S1/S2/S3 | Oracle mean distance reduction(m) | Scripted mean reduction(m) |
|---|---|---|---|
| UN | 193/90/62/71 of200 | .011286/−.060279/−.223197/−.259532 | .005400/−.068211/−.229679/−.274396 |
| Supported | 193/70/59/78 of200 | .011131/−.075582/−.237448/−.191619 | .005400/−.083948/−.244238/−.199455 |

Oracle terminal-speed means S0/S1/S2/S3: UN .0981/1.0419/1.2618/1.2965m/s;
Supported .0939/1.1010/1.2068/1.2065m/s. All819,200 candidate rollouts have
zero physical-failure and invalid flags. UN S3 includes156 original successful
early terminations, retained with absorbing observations and unmodified cost.

| Condition | Scripted true-cost median percentile S0/S1/S2/S3 | Predicted-cost median percentile |
|---|---|---|
| UN | .1953/2.6367/1.3672/11.3281% | 11.9141/90.0391/97.4609/98.6328% |
| Supported | .1953/3.3203/3.2227/2.3438% | 11.3281/80.4688/94.9219/97.2656% |

Scripted remains much better in true cost than the model recognizes; at later
states the model often places its actions near the worst predicted tail.
Scripted is not uniformly better than all candidate actions, so proposal coverage
alone is not established as the dominant mechanism. Immediate H10 progress of
both oracle and scripted worsens: the states also become harder to recover from
within0.4s. This does not demonstrate that scripted could not eventually recover
with a longer real trajectory; no new recovery controller was evaluated.

## Conclusion and one next recommendation

Most supported: **closed-loop state/latent distribution shift accompanies strong
later decision-fidelity degradation**, on top of already weak S0 best-tail
ranking. There is also a short-horizon recovery limitation: oracle candidate
progress deteriorates after entering these states. Neither a uniquely isolated
module cause nor simple proposal-coverage causality is established. Marginal
action support does not repair later ranking; stop treating it as a sufficient
remedy. Do not upgrade the planner or change its cost/H/N from this audit.

Only next recommendation: one controlled **on-policy model-data robustness**
experiment, using new training targets/seeds disjoint from this audit cohort,
holding architecture/planner/cost/H/N fixed. Not executed.

## Verification and test environment

Independent verification reconstructs all400 original trajectories and1,600
candidate sets again; all819,200 endpoint costs and all derived/paired statistics
are recomputed. Train geometry is independently regenerated.1,693 additional
exact-state physics repeats pass, including scripted feedback references at all
1,600 states. Episode/state outcome metadata is bound to original replay;
candidate termination counts are rederived from arrays and scripted task utility
is independently checked. Canonical checkpoint and all E/T/D hashes
are unchanged. Required figures and source/manifests/statistics have hashes.

Final related suite:21 passed,0 failed,0 skipped. Complete filesystem suite with
the fixed historical read-only fixtures:507 run,503 passed,0 failed,4 known skips.
No algorithm-related failure remains in the verified test environment.
The final read-only review's one Important verification gap was fixed in one pass:
five real archived S3 corruption regressions failed before the fix and passed
afterward. Scientific outputs and frozen execution sources were unchanged;
there are no deferred review findings.

The user's82 historical untracked artifacts are preserved. Bare filesystem
discovery encounters pre-existing missing historical `ppo_reward_success_alignment`
and report dependencies(445 tests,30 errors,1 skip in the final bare run). Full verification uses the
same **read-only in-memory** Git fixtures as the previous handoff, from commit
`8879754a43302e9c55b27de4ce3c629c7e4e1a53`: archived
`reward_v2_alignment_audit.py`, `ppo_reward_success_alignment.py`, and the absent
`ppo_reward_success_alignment_diagnosis.json`. No historical code/report is
restored or edited in the working tree. Four known skips concern X11 viewer and
optional unavailable recurrent-source cases; none skips this audit's algorithms.

To reproduce the full filesystem suite without restoring old files:

```bash
.venv/bin/python - <<'PY'
import sys, types, subprocess, unittest
from pathlib import Path
root=Path.cwd(); sys.path.insert(0,str(root/'mujoco/rl'))
commit='8879754a43302e9c55b27de4ce3c629c7e4e1a53'
def archive(path):
    return subprocess.check_output(['git','show',commit+':'+path]).decode()
for name in ('reward_v2_alignment_audit','ppo_reward_success_alignment'):
    m=types.ModuleType(name); m.__file__=str(root/'mujoco/rl'/f'{name}.py')
    sys.modules[name]=m
    exec(compile(archive(f'mujoco/rl/{name}.py'),m.__file__,'exec'),m.__dict__)
original=Path.read_text
missing=root/'mujoco/reports/ppo_reward_success_alignment_diagnosis.json'
fixture=archive('mujoco/reports/ppo_reward_success_alignment_diagnosis.json')
def read_fixture(path,*args,**kwargs):
    if path.resolve()==missing and not path.exists(): return fixture
    return original(path,*args,**kwargs)
Path.read_text=read_fixture
result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('mujoco/rl'))
sys.exit(not result.wasSuccessful())
PY
```

The30 bare-discovery error names were: `DatasetTests.test_fresh_targets_safe_disjoint_and_reproducible`;
`ActorMaskTests.test_baseline_loss_is_bitwise_identical_to_stock_expression`,
`test_candidate_mask_uses_stored_distance_and_raw_not_normalized_sign`,
`test_mask_keeps_full_valid_denominator_and_padding_has_no_contribution`,
`test_random_controls_are_fixed_nonoverlapping_with_candidate_and_equal_count`,
`test_trace_validation_rejects_changed_denominator_usage_or_actor_mask`,
`test_train_patch_restores_native_method_after_normal_or_exceptional_exit`;
`RecurrentInterventionTests.test_candidate_actor_mask_preserves_full_minibatch_normalization_and_critic`;
`EpochReuseContracts.test_both_conditions_restore_same_original_fifty_k_policy_and_adam`,
`test_one_rollout_validator_allows_only_condition_specific_optimizer_steps`,
`test_paired_summary_preserves_seed_identity_not_performance_sorting`,
`test_schedule_stops_after_twenty_four_complete_rollouts`,
`test_stability_measures_do_not_confuse_final_and_peak_performance`;
`EpochReuseResultContracts.test_descriptive_arrays_follow_seed_identity_even_when_branches_reordered`,
`test_evaluation_rng_restore_preserves_next_training_draws`,
`test_final_three_seed_subset_reuses_saved_rows_without_new_sampling`,
`test_first_rollout_pair_audit_rejects_mismatch`,
`test_missing_near_target_speed_is_not_nan_or_a_zero_speed_success`,
`test_no_entered_target_region_yields_undefined_speed_statistics`,
`test_serialized_constant_clip_schedules_compare_values_not_identity`,
`test_speed_denominator_distinguishes_entered_episode_and_step_means`;
module import errors `test_ppo_recurrent_epoch_local_gradient_dynamics`,
`test_ppo_recurrent_fixed_order_replay`, `test_ppo_recurrent_gradient_attribution`,
`test_ppo_recurrent_one_update_multiseed`, `test_ppo_recurrent_rollout_continuation`,
`test_ppo_recurrent_same_rollout_multi_order`, `test_ppo_stochastic_deterministic_diagnosis`,
`test_run_pi_hidden_state_observer`, `test_verify_pi_hidden_state_observer`.
