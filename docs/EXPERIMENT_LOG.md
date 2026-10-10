# Experiment Log

## 2026-10-05 — One-Step Latent Dynamics

Branch: `feat/latent-dynamics-one-step`

Research Question: Can observation-action history produce a latent state that improves one-step UAV dynamics prediction over a current-state-only Markov baseline?

Setup:

- Reused immutable `pi_hidden_state_seed0` mixed sequence dataset; 120 target groups, 840 episodes; no regeneration.
- Train/Val/Test: 588/126/126 episodes, 146631/30676/32296 valid transitions. Excluded each unavailable terminal next-observation transition.
- Seed0; 60 epochs; identical episode minibatches; train-only normalization; normalized delta7 MSE; validation-best selection.
- Markov MLP11→64→64→7; History GRU11→64 plus head68→64→64→7; PI-state Oracle MLP14→64→64→7. PI never enters History inputs/loss.

Key Results:

- Physical delta Test RMSE: Markov .0112404, History .00564894, Oracle .00959443 (mixed observation units).
- Dimensionless standardized RMSE: .605748 / .247433 / .540332. History improves physical RMSE 49.7% versus Markov.
- Gap closure 3.397 physical / 5.477 standardized; History beats this PI-state-only Oracle, which is not a full physical-state lower bound.
- Distance<.1m RMSE: .0040000 / .0004270 / .0015227; large-PI RMSE: .0166522 / .0083929 / .0146036.
- Frozen Train-fitted latent linear probe: Test PI RMSE .140976m/s², MAE .0737793, R² .525441.
- Latent norm mean/std/max 1.670/.475/4.033; finite, exact A→B→A reset, no memory leakage.

Conclusion: History-only dynamics supervision forms useful predictive latent state. It improves one-step prediction markedly, while only partially linearly encoding PI state. Improvement cannot be uniquely attributed to PI recovery.

Limitations: Nominal simulation, one seed, one-step only; larger recurrent parameter capacity; PI-only Oracle omits attitude/rates; near-target Test coverage is 431 transitions. No multi-step, policy optimization, robustness or planning experiment.

Artifacts: `mujoco/reports/latent_dynamics_one_step_seed0.json`; four representative PT models; five one-step/region/latent-probe PNGs under `mujoco/reports/`; detailed methods in `mujoco/rl/LATENT_DYNAMICS_ONE_STEP.md`. Raw NPZ and derived latent arrays stay local; manifests/hashes are versioned.

Next: Consider a separately authorized multi-step dynamics experiment to test whether the predictive state remains useful beyond one step. Do not automatically start it.

## 2026-10-05 — No-Memory GRU Ablation

Branch: `feat/latent-memory-ablation`, based on the pushed one-step experiment `ad2a7a2`.

Research Question: Does the one-step latent-dynamics gain depend on cross-time recurrent memory when GRU/head architecture and parameter initialization are held fixed?

Setup:

- Reused exact `pi_hidden_state_seed0` dataset/splits/statistics and raw delta7 target; no regeneration. Three previous models/report frozen by hash.
- Same GRU11→64, one layer, head68→64→64→7; identical seed0 parameter tensors/count23815. Each timestep starts with hidden zero in the No-Memory branch; previous executed action remains an explicit input.
- Same shared trainer: seed0, 60epochs, normalized masked MSE, episode minibatches/order, validation-best checkpoint. Only new No-Memory experimental model trained.

Key Results:

- No-Memory best epoch57; Test RMSE .01109201, MAE .00445811, R² .676758; normalized RMSE .598015.
- Markov RMSE .01124036; Full History .00564894; PI-state MLP .00959443.
- Full History improves49.07% physical /58.62% normalized RMSE versus No-Memory; No-Memory improves only1.32% /1.28% versus Markov.
- Distance<.1m RMSE: No-Memory .00409706 vs Full History .00042697. Large-PI RMSE: .01636837 vs .00839287.
- Zero cross-timestep input gradient and no influence of altered past inputs; all frozen-baseline metrics/hashes and train-only normalization reproduced.

Conclusion: Fixed architecture without recurrent memory loses nearly all of the previous gain. This supports cross-time memory as an important contributor under this seed/split, not architecture/parameter count alone.

Limitations: Single seed, nominal one-step only, explicit previous action still present, temporal paths naturally inactive with zero hidden, incomplete PI-only Oracle. No long-horizon or physical-generalization claim.

Artifacts: `mujoco/reports/latent_memory_ablation_seed0.json`; `mujoco/rl/models/history_latent_gru_no_memory.pt`; three `memory_ablation*.png`; methods in `mujoco/rl/LATENT_MEMORY_ABLATION.md`. No new raw dataset.

Next: A separately authorized frozen multi-step rollout evaluation is justified. Do not automatically execute it.

## 2026-10-06 — Frozen Multi-Step Latent Dynamics Evaluation

Branch: `feat/latent-multistep-eval`; base `9913006fe4f96a8be75a3bbc3a320be2744d0021` from memory ablation.

Research Question: Does the learned recurrent latent state remain useful during frozen multi-step autoregressive UAV dynamics prediction?

Setup:

- Four existing models frozen; exact `pi_hidden_state_seed0` Test only, 126 episodes/18 target groups; saved Train-only normalization, no fitting or resampling.
- Horizons1/5/10/25/50 at25Hz; all legal starts32296/31792/31162/29272/26122; same-model shared manifest. Endpoint observation-scaled RMSE, physical per-dimension RMSE/MAE, region and latent diagnostics.
- True initial observation, recorded future actions, predicted observation thereafter. Full-History uses true-prefix warm-up once then predicted inputs. PI-State reference receives teacher-forced PI, not full physical state.

Key Results:

- H1→H50 normalized observation RMSE: Markov .033368→.782929; No-Memory .033196→.757752; Full-History .021428→.279630; PI-State .029523→.517573.
- Full-History improves H50 RMSE63.10% versus No-Memory and64.28% versus Markov; identical26122-window curves preserve the ordering.
- H50 horizontal-velocity RMSE: .515912/.497372/.175199/.329847m/s in the above model order. PI-State has slightly lower vz error than History at long horizons.
- H50 large-PI normalized RMSE:1.043730/1.007968/.428217/.777787. Distance<.1m:.434934/.440057/.072944/.155229 (only80 eligible windows).
- History latent L2 drift grows0→.259558, cosine1→.983603; max norm4.03753. No NaN/Inf/explosion. Previous teacher-forced one-step metrics reproduced exactly.

Conclusion: Recurrent predictive latent remains useful beyond one step and slows error accumulation through2s; one-step superiority does not disappear in autoregression. Remaining compounding errors motivate a separately scoped multi-step training objective.

Limitations: Recorded future actions, nominal simulation, teacher-forced PI reference, single dataset/seed, correlated overlapping windows, at most2s and no model retraining. No closed-loop planning/control or complete-state claim.

Artifacts: `mujoco/reports/latent_dynamics_multistep_frozen_eval_seed0.json`; `latent_multistep_start_points_seed0.json`; six rollout/region/velocity/drift PNGs; methods in `mujoco/rl/LATENT_DYNAMICS_MULTISTEP_EVAL.md`. Existing model files/data stay immutable; no large prediction traces saved.

Next: Consider separately authorized multi-step latent dynamics training; do not automatically execute it or start a World Model/planner.

## 2026-10-06 — Multi-Step Latent Dynamics Training

Branch: `feat/latent-multistep-training`; base `feat/latent-multistep-eval` at `6cd1d3ab01a54ab7304ca52a5c6b92d4ac53ce35`.

Research Question: Does autoregressive multi-step training improve long-horizon UAV dynamics prediction while preserving useful short-horizon accuracy?

Setup:

- Exact original dataset/splits/Train-only normalization; 84/18/18 target groups. Train141339/Val29542 legal K10 windows; fixed prior Test starts reused.
- Same23815-parameter GRU11→64(one layer), Tanh head68→64→64→7; identical original seed0 random initialization, not fine-tuning. No PI input.
- K10/.4s, pure autoregression, true-prefix warm-up, recorded actions, full differentiable BPTT, uniform observation-normalized MSE. Adam lr.001,16 complete episodes/batch,60epochs, no new clipping. Validation-best epoch57.

Key Results:

- Test normalized observation RMSE H1/5/10/25/50: .022245/.061937/.097606/.178372/.242316 versus one-step .021428/.060491/.098319/.195672/.279630.
- H1 +3.81%, H5 +2.39% regression; H10 −.73%; beyond-training-horizon H25/H50 reductions8.84%/13.34%. Identical26122-start cohort preserves this trend.
- H50 horizontal velocity RMSE .175199→.143203m/s (−18.26%); vz .054442→.046744. **Yaw RMSE worsens .030611→.067869rad (+121.72%).**
- H50 distance<.1m normalized RMSE .072944→.048713 (80 windows); large-PI .428217→.373594. Absolute H1→H50 growth .258202→.220071.
- H50 latent drift L2 .259558→.228146, cosine .983603→.987607 (descriptive different-coordinate comparison). No NaN/Inf; training max grad .40748, hidden5.10656; Test hidden max4.63950.
- Best-checkpoint Train/Val MSE .00213576/.00424891; training401.32s. Independent artifact/metric/selection verification passed;13 new tests and repository288pass/0fail/1optional GUI skip.

Conclusion: Partially supports modest long-horizon consistency gains and beyond-K generalization. Short-horizon and especially yaw regressions prevent a claim of uniform improvement; compounding error remains.

Limitations: Single seed, nominal simulation, recorded future actions, .4s training horizon, overlapping windows, no policy/planning evaluation. The prescribed new objective changes observation-vs-delta scaling as well as horizon; latent coordinates differ. PI-State reference teacher-forces future PI and is not full-state Oracle.

Artifacts: `mujoco/reports/latent_dynamics_multistep_training_seed0.json`; `history_latent_gru_multistep10.pt` (~101KB); Train/Val window manifests; seven required loss/rollout/velocity/region/drift PNGs under `mujoco/reports/`; `LATENT_DYNAMICS_MULTISTEP_TRAINING.md`. Raw dataset and logs remain local/ignored; no intermediate epoch checkpoints or raw prediction traces saved.

Next: A separately scoped latent-state/latent-transition formulation is a reasonable candidate, with yaw/short-horizon tradeoffs retained as constraints. Do not automatically start it, planning, MPC or World Model training.

## 2026-10-06 — Objective-Matched One-Step Control

Branch: `feat/latent-objective-matched-control`; base `feat/latent-multistep-training` at `63545a60396bbbc9421d87f1f28548104739cfbd`.

Research Question: Does K10 autoregressive training improve long-horizon prediction over K1 under the same architecture, initialization, optimizer and observation-standardized loss?

Setup:

- Only new K1 trained; historical delta-loss K1 and existing K10 frozen/hash-verified. Exact original target-disjoint dataset and Train statistics; no regeneration.
- Same23815-parameter GRU11→64+Tanh head68→64→64→7; identical seed0 initial hash. Adam .001,16 episode minibatches,60 epochs, no added clipping.
- Same observation-std normalized physical next-observation MSE loss family as K10; exact K10 Train/Val start identities141339/29542, identical episode-order/update-count rule. True-prefix warm-up; no PI input.
- Best epoch60 by Validation K1 MSE=.0004516415; best Train MSE=.0001587700. Frozen Test evaluator and manifest reused at H1/5/10/25/50.

Key Results:

- Matched K1 normalized RMSE:.021160/.059505/.097411/.199907/.296011; K10:.022245/.061937/.097606/.178372/.242316.
- K10 vs matched K1: H1/H5/H10 worsen5.13%/4.09%/.20%; H25/H50 improve10.77%/18.14%. Common-start cohort confirms long-horizon ordering.
- At2s horizontal velocity RMSE .172216→.143203m/s (−16.85%); vz .044854→.046744 (+4.21%).
- Yaw .119938→.067869rad (−43.41%) versus matched K1, yet K10 remains +121.72% worse than historical delta-loss K1 .030611. All splits have zero adjacent wrap jumps; representation/evaluation unchanged.
- H50 distance<.1m .087852→.048713 (80 windows); large-PI .437147→.373594 (−14.54%). All measured distance/PI regions improve at2s.
- 112.19s training; no NaN/Inf or added clipping; max Train gradient .007306. Frozen models/data/stats/manifests and Test metrics independently reproduced. Tests:295 passed/0 failed/1 optional X11 skip;7 new control tests pass.

Conclusion: Removing the loss-scaling confound does not remove the long-horizon gain: K10 retains18.14% lower2s aggregate error. Evidence supports horizon-related consistency benefits, not uniform dimension accuracy. Yaw regression relative to the historical model is not simply caused by longer horizon; matched K1 is worse still.

Limitations: Single seed, nominal recorded actions, overlapping windows, horizon-specific Validation selection. Historical delta-loss K1 also used terminal-tail starts, so historical A→B is not a pure scaling-only experiment; no exact additive causal-share attribution. Short-horizon and vertical tradeoffs remain; no policy/planning evaluation.

Artifacts: `mujoco/reports/latent_dynamics_objective_matched_control_seed0.json`; `history_latent_gru_matched_one_step.pt`; matched Train/Val manifests; three `matched_k1_vs_k10*.png`; methods in `LATENT_DYNAMICS_OBJECTIVE_MATCHED_CONTROL.md`. Raw NPZ, cache and training logs remain local/ignored; no intermediate checkpoints/raw traces saved.

Next: A separately scoped latent-transition formulation is justified; retain explicit dimension-level loss balance as an unresolved issue. Do not automatically start it or tune weights/horizons.

## 2026-10-06 — Explicit Latent Transition

Branch: `feat/explicit-latent-transition`; base `feat/latent-objective-matched-control` at `259000a455956217318a242794f6e9e2e5481f22`.

Research Question: Can the frozen recurrent latent representation support action-conditioned latent-state propagation without future observation feedback?

Setup:

- Frozen K10 GRU11→64; immutable checkpoint/encoder parameter hashes. Exact original target-disjoint84/18/18groups,588/126/126episodes; no resampling. Teacher latent is an encoder representation, not physical ground truth.
- Independent residual Tanh transition68→128→128→64 and decoder64→128→128→7. Train-only latent statistics; delta-latent MSE and observation reconstruction MSE trained separately, seed0/Adam.001/16episodes/60epochs, no new clipping. Validation-best epochs57/60.
- Train/Val/Test146631/30676/32296 adjacent pairs; all saved final frames retained. Prior Test manifest/horizons reused; recorded actions, prefix initialization, T-only latent propagation, D readout, no future observation/GRU feedback.

Key Results:

- Decoder Test normalized observation RMSE .078243; per-dimension R² .9906–.9964, but a material absolute reconstruction bottleneck remains.
- One-step latent normalized RMSE .044726 vs identity .204087 (−78.08%); raw latent/Delta-z RMSE .008510, MAE .004687, cosine .999391.
- H1/5/10/25/50 latent RMSE .044726/.125679/.191303/.334250/.512454; cosine .999391→.936980.
- Observation RMSE .086475/.126038/.175715/.313664/.511910 vs Direct K10 .022245/.061937/.097606/.178372/.242316. Explicit model is worse at every horizon.
- H50 horizontal velocity .233803 vs .143203m/s; yaw .197865 vs .067869rad. Large-PI observation error .696478 vs .373594; distance<.1m .267889 vs .048713 (80 windows).
- Zero NaN/Inf, latent max norm4.90817. Deleted/corrupted future observations and removed future latent arrays leave rollout bit-identical. Independent metric/hash/selection verification passed;305tests passed/0failed/1optional X11 skip.

Conclusion: Frozen history latent supports learnable local action-conditioned transitions, but autonomous propagation accumulates substantial drift. Long-horizon state-like sufficiency is not established; decoder reconstruction also limits observation accuracy. This is a useful negative result, not evidence that history information is absent.

Limitations: Single seed, nominal simulation, overlapping recorded-action windows, frozen coordinates, deterministic one-step transition and independent finite-capacity decoder; no joint/multi-step training or control/planning evaluation. Cannot uniquely separate representation insufficiency, transition approximation and decoder error.

Artifacts: `mujoco/reports/explicit_latent_transition_seed0.json`; `latent_transition_mlp.pt` (~140KB), `latent_observation_decoder.pt` (~109KB); three teacher manifests, latent normalization/hash, seven required PNGs; methods in `mujoco/rl/EXPLICIT_LATENT_TRANSITION.md`. Teacher arrays stay in memory; source raw NPZ/cache/logs remain local/ignored.

Next: Consider a separately authorized multi-step transition-only experiment with encoder/decoder frozen before joint World Model training. Do not automatically execute it.

## 2026-10-07 — Multi-Step Latent Transition Training

Branch: `feat/latent-transition-multistep`; base `feat/explicit-latent-transition` at `126a3dbe20a01a2fd591c8e019560d64ef5e33a8`.

Research Question: Does pure autoregressive multi-step latent-transition training reduce long-horizon state drift while keeping the encoder and decoder frozen?

Setup:

- Exact existing dataset, splits, Train latent/observation/action normalization and Test window manifest; no trajectory regeneration. Train141339/Val29542 legal K10 windows; Test126episodes,18target groups.
- Frozen K10 GRU encoder and independent decoder, before/after parameter hashes identical. Fresh seed0 residual Tanh T68→128→128→64,33600parameters, same initial hash as old one-step T.
- K10/.4s teacher-latent targets, initial teacher latent only, full pure autoregressive BPTT, uniform latent-normalized MSE. Adam .001,16episode batches,60epochs; Validation-best epoch57, Train/Val MSE .01632157/.01823174. No decoder/observation loss, clipping addition or joint training.

Key Results:

- One-Step→K10 latent RMSE H1/5/10/25/50: .044726→.066444 / .125679→.135875 / .191303→.179689 / .334250→.274156 / .512454→.367854.
- H1 latent worsens48.56%, H5 worsens8.11%; H10 improves6.07%, beyond-training H25/H50 improve17.98%/28.22%. H50 cosine .936980→.969525. Common-start cohort confirms the same pattern.
- Observation RMSE .086475/.126038/.175715/.313664/.511910→.084842/.116118/.153719/.254360/.355959; H25/H50 improve18.91%/30.46%.
- Teacher-Latent Decoder Floor (offline reconstruction reference, not a strict lower bound): .077850/.078132/.078703/.080508/.081897. Direct K10 remains better at H50 .242316; do not linearly subtract RMSE components.
- H50 horizontal velocity .233803→.180588m/s; yaw .197865→.164219rad. Distance<.1m latent .234205→.155088, observation .267889→.202252 (80windows). Large-PI latent .726623→.547215, observation .696478→.515258.
- No NaN/Inf/explosion; max Test latent norm5.02051. Future teacher/observation corruption and deletion leave predictions bit-identical. Full metric/hash/selection verifier passes;314tests passed/0failed/1optional X11 skip,9new tests pass.

Conclusion: Multi-step transition training improves long-horizon propagation beyond K10, with a substantial one-step latent-accuracy tradeoff. Drift is reduced, not eliminated. Both transition and decoder bottlenecks remain; the decoder alone does not explain remaining2s error.

Limitations: Single seed, nominal simulation, recorded future actions, overlapping windows, frozen representation/decoder, deterministic transition and horizon-specific Validation selection. K10 excludes9terminal-tail starts/episode used by old K1, so not perfectly sample-matched. Teacher latent is not physical ground truth; no policy/planning or state-completeness claim.

Artifacts: `mujoco/reports/latent_transition_multistep_training_seed0.json`; `latent_transition_multistep10_mlp.pt` (~140KB); two window manifests; eight required rollout/floor/velocity/region PNGs; `LATENT_TRANSITION_MULTISTEP_TRAINING.md`. Teacher arrays remain in memory; dataset/cache/logs local/ignored; no intermediate checkpoints or raw prediction traces saved.

Next: Consider a decoder-only reconstruction control before a separately authorized joint latent formulation. This is a recommendation only; do not automatically begin further training, World Model, planning or control experiments.

## 2026-10-07 — Deterministic Latent World Model v1

Branch: `feat/joint-latent-world-model-v1`; base `feat/latent-transition-multistep` at `4fe448576b7df321778291e414c98788c13a8e99`.

Research Question: Can joint optimization of the history encoder, latent transition, and observation decoder learn a latent state that is more suitable for autonomous long-horizon dynamics prediction?

Setup:

- Exact existing `pi_hidden_state_seed0`, target-disjoint84/18/18groups and588/126/126episodes. Reuse Train141339/Val29542 K10windows, all prior Test starts/statistics/hash; no trajectory generation.
- Pretrained E from K10 GRU11→64, T from multistep residual MLP68→128→128→64, D64→128→128→7. All74119parameters trainable. Fold old latent affine normalization into weights once with numerical equivalence; runtime raw joint coordinates, no latent teacher target.
- Joint fine-tuning seed0, Adam.0003,16episode batches,60epochs; true causal prefix WITHgradients then10step pure latent autoregression with recorded actions. Uniform k0(current reconstruction)..k10 observation-standardized MSE, no added loss/clipping. Validation-best epoch60.

Key Results:

- Selected Train/Val MSE .002507915/.004838942 versus initial Val .013320632. E/T/D whole-run gradient maxima1.653262/19.539579/1.930154; all finite, all three parameter hashes changed, source checkpoints immutable.
- Test normalized observation RMSE H1/5/10/25/50:.043320/.070828/.102727/.230850/.510667. Frozen Explicit:.084842/.116118/.153719/.254360/.355959; Direct:.022245/.061937/.097606/.178372/.242316.
- Versus Frozen: H1/5/10/25 improve48.94%/39.00%/33.17%/9.24%, **H50 worsens43.46%**. H10 closes90.87% Frozen→Direct gap; H50 is110.74% worse than Direct. Common-start cohort preserves the mixed ordering.
- Current reconstruction RMSE .038286 versus old Frozen E/D .078243; per-dimension R² .997834–.998782. This is offline reconstruction, not a Joint decoder floor.
- H50 horizontal velocity RMSE .318011 versus Frozen .180588/Direct .143203m/s. Yaw .156039 versus .164219/.067869rad. All H50 distance/PI groups worse than Frozen; large-PI .737618 versus .515258/.373594; distance<.1m .243606 versus .202252/.048713 (80windows).
- Offline same-Joint-coordinate consistency cosine .996376→.878943, Train-scale distance .109955→.749678. Train/Test effective rank12.2857/12.0732;0/64near-zero-variance dims, all std>.109. No NaN/Inf/explosion; autonomous max latent norm4.629386.
- Independent full artifact/metric/loss/hash verifier passes, future observations corrupted/deleted predictions bit-identical,9plots visually checked.10newtests pass; trackedbranch repository324pass/0fail/1optionalX11skip. Two optional extra regression fixtures deferred, not observed algorithm bugs.

Conclusion: Joint adaptation improves reconstruction and short/medium prediction, but does **not** support successful long-horizon autonomous dynamics:2s error is worse than both references. Pure observation-space joint K10 fine-tuning is insufficient to establish a reliable long-horizon latent dynamics core. No global latent collapse observed.

Limitations: Single seed, nominal simulation, recorded future actions, deterministic dynamics, overlapping windows, .4s training objective, no reward/policy/planning. Joint geometry/loss/all modules adapt together; cannot isolate module causality, prove state completeness or infer that stochastic dynamics are necessary.

Artifacts: `mujoco/reports/joint_latent_world_model_v1_seed0.json`; `mujoco/rl/models/joint_latent_world_model_v1.pt` (~304KB); nine requested joint-model PNGs; five modules/two test modules and `mujoco/rl/JOINT_LATENT_WORLD_MODEL_V1.md`. Existing manifests reused by hash; raw dataset/latent arrays/logs/cache remain local/ignored, no intermediate epoch checkpoints or large rollout traces saved.

Next: Recommend a separately authorized read-only deterministic K10→H50 composition/latent-consistency diagnosis before choosing a stochastic state-space formulation. No further training, next branch, RSSM/Dreamer/reward/policy/planning experiment was started.

## 2026-10-07 — Joint Latent Composition Audit

Branch: `feat/joint-latent-composition-audit`; base `feat/joint-latent-world-model-v1` at `59c0a28468c81a28c90283200c6acef27eb3362c`.

Research Question: Why does Joint Latent World Model v1 perform well at short horizons but degrade during 50-step autonomous composition?

Setup:

- Frozen Joint v1, all E/T/D eval/no gradients; checkpoint and component hashes identical before/after. No study training, optimizer or autograd. Original dataset/splits/normalization/Test manifest, recorded actions; no regeneration or selection.
- Local reference consistency, autonomous composition and diagnostic C1/5/10/25/never correction on identical26122H50windows. Corrections replace current input before next Transition; never reset endpoint. Encoder-reference latent is not physical ground truth.
- Train147219encoder frames only fit covariance/PCA95%/action tertiles; predicted vs reference and paired-excess scores. Empirical Decoder sensitivity and all-window Pearson correlations, no causal attribution.

Key Results:

- Current reconstruction .038286. Local latent RMSE .109955 vs identity .192827 (42.98% lower), cosine .996376; mean residualL2 .119334 vs encoder movement .150880, so local discrepancy is not negligible.
- Autonomous H1/5/10/25/50 latent RMSE .109955/.274032/.377245/.541702/.749678; observation .043320/.070828/.102727/.230850/.510667. H50 cosine .878943, meanL2 .809398.
- H50 observation RMSE with C1/5/10/25/never: .046164/.076160/.109667/.237967/.510667, monotonic diagnostic recovery; C1 horizontal velocity .017011 vs never .318011m/s.
- Train PCA13components retain95.1493%. H1→H50 predicted MahalanobisRMS .948641→3.441289 vs reference .810252→.839478; PCA residual .234264→.535272 vs reference .227364→.233595. Marginal standardized RMS grows only .818582→.973672, indicating low-variance/subspace drift rather than simple norm explosion.
- Empirical Decoder ratio mean .211348→1.198374, unit-dependent/not Jacobian. Local residual vs H50latent/obs/horizontal error r=.493170/.443245/.414008. Horizontal error vs paired Mahalanobis/PCA excess r=.446859/.278058, descriptive only.
- H50 observation error by distance<.1/.1–.2/.2–.5/>=.5: .243606/.264566/.283025/.567705; PI small/medium/large .211972/.400007/.737618; action norm small/medium/large .176569/.536654/.650815. C1 improves every group; near-target H50group only80windows.
- No NaN/Inf/explosion; max latent norm4.629386. Exact previous Joint metrics and full diagnostics independently reproduced; all nine plots inspected. Seven new tests pass; full tracked-branch+new suite331passed/0failed/1optional X11skip. Read-only review0Critical/0Important; two Minor verification-hardening suggestions deferred, documented in methods/report.

Conclusion: Non-negligible local encoder-transition inconsistency accumulates during autonomous composition, with departure along low-variance/out-of-principal-subspace directions and increasing empirical Decoder sensitivity. Diagnostic corrections substantially recover accuracy. Strongest evidence is for composition/drift, not a uniquely isolated module root cause or a need for stochastic dynamics.

Limitations: Single seed, nominal simulation, recorded actions, overlapping windows, diagnostic teacher corrections, linear distribution proxies rather than proven nonlinear manifold, unit-dependent decoder ratios, correlations not causality, no retraining/control/planning. Train-PCA tertile bins saturate at H50; do not infer a within-horizon graded trend from them.

Artifacts: `mujoco/reports/joint_latent_composition_audit_seed0.json`; Train statistics/hash JSON; nine requested composition audit PNGs; four diagnostic modules/two test modules; `mujoco/rl/JOINT_LATENT_COMPOSITION_AUDIT.md`. Existing Test manifest reused by hash. No raw latents/large traces/new models/data copies saved; internal logs/cache and source NPZ stay local/ignored; historical untracked experiments preserved.

Next: Only recommend a separately authorized objective-matched explicit encoder–transition consistency-loss control, keeping the existing K10 observation objective and other settings fixed, to test local discrepancy and H50 composition drift. Do not execute it or start RSSM/Dreamer/planning/PPO.

## 2026-10-07 — Explicit Encoder–Transition Consistency

Branch: `feat/joint-latent-consistency-v2`; base `feat/joint-latent-composition-audit` at `d13541be9a620a55303ab0079eb7a517d5cd1822`.

Research Question: Does an explicit local Encoder–Transition consistency objective reduce composition drift in the deterministic latent world model?

Setup:

- Same pre-joint E/T/D initialization and hashes as Joint v1,74119parameters, original GRU11→64/residual Tanh68→128→128→64/Tanh64→128→128→7.
- Original `pi_hidden_state_seed0`, target-disjoint84/18/18groups, Train141339/Val29542K10windows, exact original Test starts/normalization; no dataset regeneration or baseline retraining.
- Same seed0/Adam.0003/16episode batches/60epochs/K10uniform k0..10 observation objective; add only lambda.1 local consistency with source E/T gradients and next-E target stop-gradient. Fixed initial Train64Dstd, no mean subtraction; no new clipping/EMA/targetnetwork.
- Validation-best solely by Lobs, epoch33. Selected Train/Val Lobs .003489722/.006143659,Lcons .004045833/.004018441,Ltotal .003894306/.006545503. Late losses rise; epoch60Val Lobs .012338830, not hidden by early stopping.

Key Results:

- Local native Train-scale RMSE .109955→.072728,meanL2 .119334→.083674,cos .996376→.998375; relative residual median1.570927→1.072854. Native coordinates differ; near-stationary denominators dominate ratio means.
- Joint v1→v2 observation RMSE H1/5/10/25/50: .043320→.055747 / .070828→.080166 / .102727→.112402 / .230850→.250077 / .510667→.564054. **Every horizon worsens; H50 +10.45%.** Reconstruction .038286→.052330 (+36.68%).
- Diagnostic C1/5/10/25/never H50 RMSE .059020/.085550/.119386/.259356/.564054; absolute never−C1 gap .464503→.505034. Smaller ratio alone does not mean greater stability because C1 also worsens.
- Own-Train H50 Mahalanobis prediction3.441289→2.728598 improves, but PCA residual .535272→.559404 worsens; reference remains .219824. Geometry evidence mixed. Decoder empirical ratio1.206192, coordinate-dependent/not Jacobian.
- H50 horizontal velocity .318011→.355460m/s (+11.78%),yaw .156039→.134531rad (−13.78%). Distance-group errors .262841/.272131/.285981/.631159; large-PI .814454 vs v1 .737618; large-action .721461 vs .650815.
- Train/Test effective rank12.237907/12.072160,0near-zero-std dimensions; no NaN/Inf/explosion. Weighted-consistency/observation gradient mean E/T/D .095833/.014744/0; no minibatch dominance. All module hashes updated, all original source hashes unchanged.
- Independent selected-model loss/metric/hash/selection verifier passes; future observations deleted/corrupted predictions bit-identical.12figures visually checked. Read-only review0Critical/0Important;2Minor verifier-hardening suggestions implemented with tamper tests.
- Final tracked-branch+new regression suite340passed/0failed/1optional X11skip (341run);9new related tests pass. Test-generated unrelated JSON key reordering restored and excluded from commit.

Conclusion: Local mismatch is reduced but this fixed local-consistency remedy **fails to improve long-horizon observation prediction** and regresses reconstruction/short prediction. Do not adopt v2 as an improved baseline. Healthy variance rules out global latent collapse here; local consistency is insufficient, not proof of a unique cause or a need for stochastic dynamics.

Limitations: Single seed, nominal simulation, recorded actions, fixedlambda, moving online Encoder reference, local-only constraint, overlapping windows, learned coordinate changes, diagnostic teacher corrections, no policy/planning evaluation. Historical gradients are logged, not replayed by the read-only verifier.

Artifacts: `mujoco/reports/joint_latent_consistency_v2_seed0.json`; best `joint_latent_world_model_v2_consistency.pt` (305395bytes); initial/post-trained Train statistics/hash JSON;12requested PNGs;4modules/2test modules; `mujoco/rl/JOINT_LATENT_CONSISTENCY_V2.md`. Original manifests reused; raw arrays/NPZ/cache/logs remain local/ignored, no intermediate epoch models or large rollout traces saved.

Next: Only recommend a separately authorized objective-matched autonomous multi-step latent-consistency control. Do not execute it, sweep lambda, create a next branch, or start RSSM/Dreamer/planning/PPO.

## 2026-10-07 — Autonomous Multi-Step Latent Consistency

Branch: `feat/joint-autonomous-consistency-v3`; base `feat/joint-latent-consistency-v2` at `ad3e13c95bbdbebdc97738c99747c81b89901169`.

Research Question: Does directly aligning autonomous latent rollouts with future encoder-reference states reduce long-horizon composition drift?

Setup:

- Same pre-joint E/T/D checkpoint initialization/hashes and74119parameter architecture as v1/v2; no baseline retraining or fine-tuning from their best models.
- Original `pi_hidden_state_seed0`,84/18/18target groups,588/126/126episodes; exact Train/Val141339/29542K10windows and original Test manifest/normalization reused.
- Same seed0/Adam.0003/16episode batches/60epochs, original uniform k0..10 observation loss. Add only .1×K10autonomous consistency using composed latent states and detached real-history Encoder references; no local consistency loss.
- Fixed v2 initial Train latent std; fullBPTT/sourceE+T gradients, D no direct auxiliary gradient. Best by Validation Lobs only, epoch60. Test values opened after selection.

Key Results:

- Fixed selected-model Train/Val Lobs .002664117/.005034969,Lauto .006454096/.007723126,Ltotal .003309526/.005807282. Whole-run weighted-auto/obs gradient norm ratios E/T/D .058415/.050797/0; no minibatch auxiliary dominance.
- Current reconstruction .038199 vs v1 .038286, essentially preserved. Local native Train-scale RMSE .048581,meanL2 .042709,cos .999355; relative residual median .441244. Native coordinate/near-stationary denominator caveats apply.
- v3 H1/5/10/25/50 observation RMSE .044057/.071549/.104600/.201413/.308050. Relative to v1, short horizons worsen1.70/1.02/1.82%; H25/H50 improve12.75%/39.68%. H50 improves45.39%vs v2 but remains27.13%worse than DirectK10.
- Autonomous native latent RMSE .048581/.106817/.147791/.247459/.356615; cosine .999355/.997659/.995819/.988820/.976299. Different learned geometries are not cross-model physical quantities.
- H50C1/5/10/25/never RMSE .046858/.076610/.111145/.209165/.308050; never−C1gap .261191 vs v1 .464503 (−43.77%). No endpoint correction; identical26122H50cohort.
- Own-Train H50 predicted/reference MahalanobisRMS .754693/.837605,PCAresidual .202782/.233190. No increasing off-distribution proxy departure, but smaller predicted scores may reflect contraction, not exact state agreement.
- H50 horizontal velocity .187326m/s (−41.09%vs v1),yaw .070760rad (−54.65%). H50all distance/PI/action groups improve; large PI still .466219. Decoder empirical ratio grows .649475→1.891351 while decoded H50 deviation falls to .582185; do not infer intrinsic decoder robustness from native-coordinate ratios.
- Train/Test effective ranks11.961706/11.738051,0near-zero-std dimensions; no collapse/NaN/Inf/explosion. All3best hashes changed; source hashes unchanged; future-target perturbation changes loss, not prediction.
- Independent selected-model verifier passes,13figures visually checked,9new tests pass. Full tracked+new suite349passed/0failed/1optionalX11skip. Read-only review verifier/doc fixes did not alter formal training logic.

Conclusion: Qualified positive controlled result: autonomous multi-step consistency reduces long-horizon composition error where local consistency v2 did not, without the v2 reconstruction regression. Deterministic state-space formulation remains promising, but short-horizon trade-off and the gap to DirectK10 remain. No unique root cause or multi-seed robustness established.

Limitations: Single seed, nominal simulation, recorded actions, fixedlambda, moving online encoder targets, K10training horizon, overlapping windows, learned-coordinate changes/distribution proxies, no planning/policy evaluation. Best epoch at60is not proof of convergence. Historical gradients are logged, not replayed.

Artifacts: `mujoco/reports/joint_latent_autonomous_consistency_v3_seed0.json`; best `joint_latent_world_model_v3_autonomous_consistency.pt` (305765bytes); own-Train statistics/hash JSON;13required PNGs;4modules/2test modules; `mujoco/rl/JOINT_LATENT_AUTONOMOUS_CONSISTENCY_V3.md`. Existing scale/manifests reused by hash. RawNPZ/latents/cache/logs remain local/ignored; no intermediate checkpoints or raw rollout arrays saved; historical untracked experiments preserved.

Next: Only recommend an independently authorized multi-seed replication of the fixed v1-vs-v3 contrast, with no new consistency forms or lambda/K sweep. Stop consistency-family expansion; do not execute another experiment, replace baselines automatically, or start RSSM/Dreamer/planning/PPO.

## 2026-10-07 — Joint v1 vs v3 Multi-Seed Replication

Branch: `feat/joint-v1-v3-multiseed`; base `feat/joint-autonomous-consistency-v3` at `fa357af3c1a9fc0f24e635e1b073b180bdca4b64`.

Research Question: Does the long-horizon benefit of autonomous latent-consistency training replicate across multiple paired joint-training seeds?

Setup:

- Paired seeds0–4; seed0 reused unchanged, eight new v1/v3 runs for seeds1–4. **Joint-training seed replication only**, not full end-to-end/random-initialization replication. All start from the same fixed pre-joint E/T/D checkpoints and component hashes.
- Same74119parameter architecture, dataset/splits/normalization/manifests; Train/Val141339/29542K10windows, Test32296/31792/31162/29272/26122H1/5/10/25/50windows.
- Adam.0003,16complete episodes/batch,K10,60epochs,no new clipping. Actual epoch-order hashes identical within each new pair; seed0 order hashes reconstructed and labelled accordingly. No upstream training, v2 condition or new sweep.
- v1 Lobs only; v3 Lobs+.1Lauto with detached online encoder reference, no local term. Both selected only by Validation Lobs; paired best epochs60/59/58/51/60.

Key Results:

- H25 mean±sampleSD v1 .235221±.004865, v3 .200931±.002534; paired deltas[.029438,.033250,.033062,.032314,.043384], mean.034290±.005309; wins5/5. Paired percentage improvement14.55±1.96%.
- H50 v1 .531082±.025303 (median.527100,range.505980–.567516), v3 .305755±.006403 (median.308050,range.297185–.313745); paired deltas[.202618,.208794,.253771,.219043,.242409], mean.225327±.021962; wins5/5. Per-seed improvements39.68/41.27/44.72/41.56/44.55%, mean42.35±2.20%. Improvement from group means42.43%is a different aggregation.
- H1/H5/H10 v3 worsens5/5 by1.20–1.70%/.31–1.02%/.27–1.82%; stable small short-horizon trade-off. Current reconstruction improves slightly5/5: .038386±.000491→.038167±.000506.
- H50 horizontal velocity .333208±.020359→.188546±.004836m/s; yaw .161008±.019745→.061557±.006383rad; both improve5/5.
- Never−C1 correction gap .484969±.025287→.259062±.006228, mean paired reduction46.50%; never−C10 .421379±.025056→.195025±.005994,53.63%; both shrink5/5. Full C1/5/10/25/never curves retained for every seed.
- H50 within-model mean native discrepancy .752141→.358191, cosine .881170→.976304; coordinate caveat applies. Every Train/Test latent has0near-zero dimensions; effective ranks v1≈12.05–12.42,v3≈11.73–12.12. No failed/abnormal/replaced seed, NaN/Inf/explosion or collapse warning; no heavy extra audit triggered.
- Related13tests pass. Final full local suite with read-only historical Git fixture:430passed/0failed/4explicit GUI/absent-legacy-source skips (434total); no new algorithm skip. Seven final figures visually checked; core read-only review approved.
- Independent saved-model verifier passes: seed0 archived metrics, all new Validation-best Train/Val losses, recorded histories/orders, frozen Test metrics, hashes and paired statistics reproduce without an optimizer.

Conclusion: V3 receives preliminary paired five-training-seed support for long-horizon prediction with fixed upstream initialization. Establish the v3 formulation as the current deterministic latent dynamics baseline, retaining existing seed0 as reference rather than selecting the Test-best seed. This is not overall dominance at all horizons. Formally stop the deterministic consistency-objective family; no additional loss/K/lambda/EMA experiment is run.

Limitations: Same upstream pretrained initialization, only joint-training seeds, n=5, nominal simulation, overlapping windows, recorded actions, fixedλ/K/budget and moving online targets; no policy/planning evaluation or statistical-significance claim. Best-at-cap is not proof of convergence.

Artifacts: `mujoco/reports/joint_v1_v3_multiseed_replication.json`; eight small Validation-best models `joint_v1_seed{1..4}.pt` and `joint_v3_autonomous_consistency_seed{1..4}.pt`; seven required figures; runner/evaluation/plot code and tests; `mujoco/rl/JOINT_V1_V3_MULTISEED_REPLICATION.md`. Existing manifests/statistics referenced by hash. Raw arrays, per-branch report fragments, logs/cache remain local/ignored; original untracked diagnostics preserved.

Next: Only recommend paired end-to-end replication across independently pretrained upstream initializations; do not execute it or start any next experiment here.

## 2026-10-08 — Latent Random-Shooting MPC

Branch: `feat/latent-random-shooting-mpc`; base `feat/joint-v1-v3-multiseed` at `e3ce35a0c736b3332c70dddd9c7d03cc20627b0d`.

Research Question: Can the frozen deterministic latent dynamics model support useful model-based action selection in closed-loop waypoint control?

Setup:

- Fixed v3 seed0, unchanged E/T/D and checkpoint/module hashes, no training of models/policies/reward. No Test-best seed selection.
- Random shooting512candidates/H10, bounded Gaussian random walk with0.5×Train action std, explicit zero/repeat-last candidates; pure latent transitions, real-history E updated each real step, only first action executed.
- Hand-designed physical terminal error/speed/yaw cost plus normalized-command smoothness weights1/.5/.1/.05. No cost/N/H tuning or alternative planner.
- Unchanged nominal PI waypoint RewardV2 environment, success hold5steps and15s timeout. Archived100benchmark+100holdout seeds/coordinates for zero/scripted/MPC (600episodes total).

Key Results:

- Success benchmark/holdout: zero0/0, scripted100/100, latentMPC0/1 (each denominator100). Final distance1.568/1.530 vs .07144/.06874 vs4.203/4.386m. MPC153physical failures,46timeouts,1success.
- MPC near-target actual/command speed .818/1.760 and .408/1.654m/s, only29/81near-target steps; crossing11%/9%, command-element saturation39.15%/40.24%. Low crossing is not stable stopping evidence. MPC completion1.24s comes from only1success; scripted5.538/5.813s comes from100successes/split.
- CPU Ryzen9 7940HX, one Torch thread,44229decisions: planning mean/median/p95/max6.535/6.494/6.765/13.422ms; including E6.704/6.654/6.991/14.445ms. All measured decisions below40ms, no real-time OS guarantee. Final same-protocol rerun reproduces all six scientific cohort summaries exactly.
- Selected actions outside at least one per-axis central Train95% interval85.96%, outside elements45.35%, mean standardized norm3.445. Actions highly variable, not a constant/scripted equivalent; first-command target correlations x/y/z .182/.051/.284 are weak.
- First3fixed episodes/split,37realized-action-prefix windows: normalized forecast H1/H10 .2224/1.2212benchmark and .2508/.4107holdout. These compare actual action prefixes, not the original candidate sequence; yaw branch cuts can inflate ordinary RMSE.
-15new tests pass; independent read-only verifier reproduces target/count/summary/hash/timing/OOD/forecast data and future-observation perturbation leaves predictions bit-identical. Full filesystem suite with the unchanged read-only historical Git fixture:445passed/0failed/4explicit GUI-or-absent-legacy-source skips (449run). Eight final figures visually checked. Fresh read-only review found2integrity issues (cohort completeness/count and cache environment/runtime provenance), fixed with RED→GREEN regressions; same-protocol evaluation repeated without any model/control-path change.

Conclusion: Negative decision-usefulness result for this fixed planner: good recorded-data prediction and fast runtime do not translate into safe/useful action selection here. Action-support departure and prediction mismatch are compatible with OOD/model-exploitation risk, but do not isolate it from cost/horizon/candidate coverage/state OOD. Do not promote this MPC or automatically escalate to CEM/MPPI.

Limitations: Single planner configuration, nominal simulation, recorded-action model training, deterministic dynamics, hand-designed cost/short horizon, no uncertainty estimate. Marginal95% action ranges are not joint sequence-support tests; small fixed prediction subset; ordinary yaw RMSE wrap caveat; no causal model-use ablation.

Artifacts: `mujoco/reports/latent_random_shooting_mpc_seed0.json` (includes complete target manifest and source hashes); planner/evaluation/runner/read-only verification/plot code,3test modules,8requested figures; `mujoco/rl/LATENT_RANDOM_SHOOTING_MPC.md`. Existing model/dataset unchanged. Raw candidate arrays are not saved;6fixed-episode traces,2aggregate arrays and condition fragments stay local/ignored. Historical untracked experiments preserved.

Next: Only recommend a separately authorized paired Train-supported-action constraint control, holding frozen model/N/H/cost fixed versus current environment-bound sampling, to test action-support departure. Do not execute here or train/tune any model/planner.

## 2026-10-08 — Train-Supported Latent MPC Control

Branch: `feat/latent-mpc-action-support`; base `feat/latent-random-shooting-mpc` at `7d360778a0e401a091c77811a78968805e813154`.

Research Question: Does constraining random-shooting MPC to the action support observed in training reduce model exploitation and improve closed-loop waypoint control?

Setup:

- Frozen canonical v3seed0, N512/H10, original cost/noise/normalization/replanning/task/PI. Only intervention: per-step env clipping followed by central95% Train action-box projection, including anchors.
- Full147219executed Train commands determine bounds; original model's146631paired-row action std remains the noise scale. Box is a coarse support proxy, not joint state-action/sequence support.
- Both conditions rerun on100benchmark+100holdout; original episode RNG/fixeddraws preserve paired rawnoise without redefining baseline.200initial state/latent pairs and38083common decision noise hashes agree; later states diverge.

Key Results:

- UN success0/100benchmark,1/100holdout exactly reproduces prior scientific summaries; Supported0/100both. Finaldistance4.203→3.988benchmark,4.386→4.536holdout. Physicalfailures153→130, timeouts46→70, no task recovery.
- Selected-action proxy departure85.98%→0%; standardizednorm3.445→3.075. Saturation39.15/40.24%→0%mechanically, because bounds<.95. Near-target command1.760/1.654→1.579/1.548m/s remains high; actual .818/.408→.796/.447m/s.
- Supported projection56.2–56.6%candidate timesteps; L2mean≈.054,p95≈.144;0anchor projections. Selected boundary-element mass≈33%, but unique firstactions≈99.4%, no global candidate-set collapse.
- Fixed-rule prefix H1/H10: benchmark .2224/1.2212→.2839/.4922, holdout .2508/.4107→.7207/1.2050, with26/32and11/37windows. Common-available identities21benchmark/11holdout: .1835/.4276→.2355/.5148 and .2508/.4107→.2069/.3011. Mixed, not stable validity improvement; state/window/yaw-wrap caveats.
- First selectedaction changes15.5%; target xyz correlations .182/.051/.284→.190/.073/.348 remain weak. First selectedcost1.1022→1.1135; both firstcommands already within box.
- CPU planningmean6.356→6.613ms, full decision6.742→7.289ms, allobserved<40ms.16new/31related tests pass; full461passed/0failed/4known skips (465run), read-only historical Git fixture. Independent verifier400cohorts/106prefix windows/supportquantiles/frozenhashes passes;8figures visuallychecked.
- Fresh review found an executable-forward/checkpoint-loader cache dependency omission; fixed with a RED→GREEN regression. Preserved old fragments locally and reran the identical400protocol instead of relabelling caches: scientific summaries, action/cost/noise/projection arrays and106prefix forecasts bit-identical, only latency remeasured. Deferred minors: projected-only histogram/count tolerance-edge populations differ (published median/p95 unchanged), and verifier does not itself cover every derived summary (fresh reviewer independently checked current values).

Conclusion: Negative task-control result. Central95%action projection eliminates defined departure and changes aggressive-failure mix but does not restore success or consistent prediction validity; not specific causal support for action OOD/model exploitation as the sufficient/unique mechanism. Stop MPC parameter tuning.

Limitations: Central95%box is only a marginal support proxy; nominal MuJoCo, fixed cost/H/N/std, single canonicalmodel, diverging closed-loop visitation, small fixed prediction subset/termination-based commonwindows, ordinary yaw wrap, diagnostic CPU timing, no CEM/MPPI.

Artifacts: `mujoco/reports/latent_mpc_train_supported_action_control_seed0.json`; `train_action_support_central95.json`; planner/evaluation/runner/verifier/plot code and3new testmodules;8requested PNGs; `mujoco/rl/LATENT_MPC_TRAIN_SUPPORTED_ACTION_CONTROL.md`. Raw12episode traces/4aggregate arrays/noise hashes/condition fragments remain local/ignored; dataset/model unchanged and historical untracked files preserved.

Next: Only recommend an offline action-sequence ranking/decision-cost fidelity benchmark under identical executed sequences, same frozen model and cost. Not executed; no further support/N/H/std/cost sweep or stronger MPC method.

## 2026-10-08 — World Model Decision-Cost Fidelity Audit

Branch: `feat/world-model-decision-cost-fidelity`; base `feat/latent-mpc-action-support` at `d565cb8f3f85dccef1da4d5a9f713cf2f4142433`.

Research Question: Does the frozen deterministic world model correctly rank the same candidate action sequences used by the failed MPC controller, and are the candidate set and fixed planning cost themselves capable of supporting useful decisions?

Setup:

- Frozen canonical v3seed0; N512/H10, original noise/cost/support/task/PI and100benchmark+100holdout first-decision states. No training/new closed-loop controller or MPC tuning.
- Old raw candidate/snapshot arrays were not archived: deterministic immutable-source reset/sampler reconstruction verifies all200 saved initial obs/latent/noise/action/index/min/median/mean cost signals. New candidate/snapshot hashes disclosed, not invented historical hashes.
- Full MjData and Python PI/yaw/allocator/task/RNG/previous-action snapshot restore per candidate;204800 MuJoCo candidate rollouts plus200 scripted10step references. Exact same actions to frozen pure latent model; scripted not injected into candidate set. No failure penalty added.

Key Results:

- All200 mean UN/S Spearman .616810/.602238, Kendall .454052/.442741. Model normalized regret .127614/.140823 vs analytic random .252689/.275561; model beats random expectation187/200 and186/200. Exact Top1 both7/200; predicted-best in trueTop5 15/200 and19/200. Selected true median rank26.22%/25.05%;35/34states in worse half.
- True-cost best candidate improves distance193/200 both, mean .011286/.011131m, zero physical failures; model .003297/.003421m with123/127positive-progress states. Scripted .005400m, .060778m/s terminalspeed,200positive-progress states. These small0.4s gains do not prove oracle closed-loop success.
- Scripted true median cost percentile .1953% both, beats99.56%/99.53% on average and every candidate78/200; better than model selected189/200. Predicted percentile median11.91%/11.33%, only1/0predicted wins over all candidates. Candidate oracle beats scripted122/200, tiny mean cost gap .0027/.0029: coverage incomplete, not uniformly bad candidates.
- True cost aligns with H10 distance/speed (meanrho .507/.776 UN, .542/.751 S). Whole-set H10 NRMSE .1763/.1755; selected .2004/.1965. Error correlates negatively withrho and positively with regret; small-margin ties are not the sole ranking issue. Cost prediction is optimistic (mean bias≈−1.467), distinct from partial useful ranking.
- Allcheckpoint/E/T/D/source hashes unchanged, allcandidate sets and endpoint costs independently verified; no failures/invalid/earlyterminations in this cohort. Ten figures;21new regression tests, complete derived-statistic tamper checks and exact physics replays. Full filesystem tests482passed/0failed/4knownskips (486run), unchanged read-only historical Git fixture. Sole fresh review's undefined paired-correlation edge case fixed RED→GREEN; allactual scientific values unchanged, no deferred issues.

Conclusion: Partial first-decision decision utility, but poor best-tail ranking/scripted-quality recognition; central95%action support does not repair it. Proposal coverage is incomplete, yet oracle candidates generally progress and true cost ranks scripted well. First-decision evidence does not uniquely explain subsequent near-total closed-loop failure or establish cost-horizon/replanning causality. Do not change MPC from this audit.

Limitations: First-decision offline audit only, fixed H10/candidate sets/cost, nominal MuJoCo, original raw firstarrays unavailable (explicit reconstructed evidence), no oracle closed-loop/control modification, no causal uniqueness claim.

Artifacts: `mujoco/reports/world_model_decision_cost_fidelity_seed0.json`; `decision_cost_fidelity_manifest_seed0.json`; diagnostic/runner/verifier/postprocessing/plot code,7testmodules,10requiredfigures; `mujoco/rl/WORLD_MODEL_DECISION_COST_FIDELITY.md`. ~36MB local cost/endpoint/perstate fragments ignored; no raw candidate trajectories/dataset/model copies committed. Historical untracked experiments preserved.

Next: Only recommend reusing saved failed MPC trajectories for an exact-state later-decision offline ranking audit under unchanged candidate/cost, testing receding-horizon state-distribution shift. Not executed.

## 2026-10-08 — Later-Decision Exact-State Ranking Audit

Branch: feat/world-model-later-decision-ranking
Base: feat/world-model-decision-cost-fidelity / e21e7c0cff7e18ce1e4e8074bd9a323e387b30e5

Research Question: Does frozen-model decision ranking deteriorate as the original failed MPC trajectories visit later states?

Setup: Frozen canonical v3 seed0, original UN/Train-supported trajectories;100 Benchmark+100 Holdout per condition; exact full simulator/controller/RNG restore; distinct nearest0/25/50/75% stages. N512/H10, cost/proposal/noise/PI unchanged. No training or new controller. All400 episodes retained including one UN success, failed-only paired sensitivity reported.

Key Results:
- All1600 states /819200 candidate rollouts complete; each stage200 states per condition. No stage attrition, constant target/failure-type composition. All perdecision original action/cost/noise signals exact;12 archived full representative traces exact. S0 metrics/candidate/snapshot hashes match prior audit.
- UN Spearman S0/S1/S2/S3=.616810/.176074/−.111023/−.103874; Supported=.602238/.144396/−.084374/−.159296. Paired S3−S0=−.720684/−.761533,177/174 of200 worsen. Failed-only UN199 gives−.723830.
- Normalized regret UN .127614→.548530, Supported .140823→.598313; S3 random expectation .516323/.524813. Model beats random187→96 and186→81 of200. Selected true-rank median26.22→59.88% /25.05→77.69%.
- True best26rho .0814→−.1860 /.0908→−.1428; pairwise accuracy .5284→.4345 /.5310→.4497. True margins/tail spreads grow, not merely tiny-margin ambiguity.
- Selected H10 NRMSE .200388→.874517 /.196549→.732984; actual observation OOD RMS .5154→2.6385/2.6661; actual encoder Mahalanobis RMS1.4242→3.7377/3.4584. OOD/rank/regret associations are descriptive, not unique causal proof.
- Oracle distance-progress193→71 /193→78 of200; S3 mean reduction−.2595/−.1916m. Scripted S3−.2744/−.1995m immediate progress, despite true-cost median percentile11.33%/2.34%; predicted percentile98.63%/97.27%. Short0.4s recovery is also limited; this does not prove long-term scripted recovery impossible.
- Zero physical-failure/invalid candidate flags;156 UN successful early terminations retained. Independent second replay validates all400 trajectories/1600 candidate sets/allcosts/derivedstatistics/Train geometry plus1693 exact physics repeats, including all1600 scripted references and independent outcome metadata/count checks. Checkpoint/E/T/D unchanged. No MPC change.21new audit tests pass; full filesystem suite503passed/0failed/4knownskips(507run), same read-only historical fixtures; final bare run's30 pre-existing dependency errors disclosed in experiment documentation. Sole final review's verification gap fixed RED→GREEN in one pass; no deferred findings.

Conclusion: Strong later-state decision-fidelity degradation accompanies closed-loop state/latent distribution shift, beyond initial best-tail weakness. H10 oracle recovery also worsens. Marginal central95% support does not restore later ranking; it is not a sufficient remedy. Do not attribute everything to one module or upgrade the planner from this audit.

Limitations: single frozen canonical model, fixedH10/N512/cost/proposal, nominal MuJoCo; relative realized episode stages, differing post-S0 condition states, repeated-state correlations, no causal uniqueness or long-horizon recovery test.

Artifacts: `mujoco/reports/world_model_later_decision_ranking_audit_seed0.json`, `later_decision_ranking_manifest_seed0.json`, `later_decision_train_distribution_seed0.json`; audit/replay/metrics/reporting/verifier/plots/tests,10requiredfigures; `mujoco/rl/WORLD_MODEL_LATER_DECISION_RANKING_AUDIT.md`. Local candidate costs/endpoints/errors/scripts remain ignored, no raw512trajectories/snapshot dumps/dataset copies committed.

Next: Only recommend one controlled on-policy model-data robustness experiment with new training targets/seeds disjoint from this audit cohort and architecture/planner/cost/H/N fixed. Not executed.

## 2026-10-09 — On-Policy State-Distribution Model Adaptation

Branch: feat/world-model-onpolicy-adaptation
Base: feat/world-model-later-decision-ranking / db532ac9b40d9fc49a1c2e18b55090aba4f6b19a

Research Question: Does MPC-visited state coverage improve later prediction and decision fidelity beyond equal-budget scripted/replay-state adaptation without losing nominal prediction?

Setup: Original canonical v3 frozen; Replay and MPC-state adaptations start from identical v3 E/T/D hashes. Fresh isolated60Train/20Val/100Final targets, seed202710091; no overlap with original PI120 or benchmark/holdout200. Per source800/200 accepted snapshots×8 actualK10 counterfactual branches=6400/1600 windows. All80 source episodes retained: Replay80success; MPC32tilt/33outside/15timeout. Adam.0003/seed0/batch16/1000updates, original L_obs+.1L_auto and normalization/scaling unchanged. Identical batch-order hash. Best by own ValL_obs only: Replay950,MPC700. Original frozen MPC/cost/N512/H10/proposal/physics/PI unchanged.

Key Results:
- All100unseen Final episodes /300S0/S2/S3 states /153600 true candidate rollouts complete; three models share exact snapshots/candidates/truth. All100 participate at each stage; no adapted-controller closed-loop evaluation.
- Candidate-mean H10 NRMSE S0/S2/S3: Original .179442/.418474/.890673; Replay .143105/.591800/1.020330; MPC-state .223501/.451411/.799155. C vs B improves S2/S3 23.72%/21.68%, worsens S0 56.18%. C is still worse than Original atS2.
- Whole rho S2/S3: Replay−.064821/−.013394; C−.021567/−.052754. Best-tail rho improves−.126003/−.108533→−.066721/−.050537, still negative. Regret .519367/.529725→.503737/.512777; paired median delta0, only48/46 of100 wins. S3 selected true-rank median worsens51.08%→56.65%. Decision-critical fidelity not robustly recovered.
- Nominal H1/H10/H25/H50: Original .044057/.104600/.201413/.308050; Replay .092331/.175406/.309185/.458162; C .098742/.276733/.659729/1.172470. H50 regression48.73%/280.61%; strong forgetting. No collapse/NaN/Inf;64 nonzero-variance dimensions, C Train/Test effective rank12.22/12.17.
- Shared oracle distance progress96/21/29 of100 S0/S2/S3; later mean progress negative. Scripted true-cost medians remain good (.20/.68/2.83%), but C predicted percentiles28.91/65.72/89.75%. Model improvement cannot remove the H10 recovery ceiling by itself.
- Independent verification recomputes all true costs,900 model-state metrics/aggregates and state OOD, both best Val losses, compact training/complete selection schedule, numeric conclusion fields, source summaries, nested provenance, pairedbudget/order, exact fixed physics/scripted replay and nominal reproducibility. Original SHA42571249… unchanged.24new tests passed; full read-only-fixture suite527passed/0failed/4knownskips(531run). Bare-discovery30 historical dependency errors disclosed in the experiment documentation; no current algorithm failure. Future collection has fail-closed runtime/dependency guards; historical cache runtime stamps were not recorded or retroactively fabricated.

Conclusion: State coverage helps later prediction relative to Replay, but this fixed dynamics objective does not reliably restore best-tail decisions and severely damages nominal prediction. Mixed/negative result; keep canonical v3 as baseline. No claim of improved closed-loop success; no MPC modification.

Limitations: single adaptation seed, nominal simulation, valid-window selection bias,8branches/snapshot, different source-specific Validation distributions, fixedK10/H10/N512/cost, counterfactual actions not actually on-policy actions, no retention mixing, no adapted-controller evaluation.

Artifacts: `mujoco/reports/world_model_onpolicy_adaptation_seed0.json`, isolated target/data/Final manifests; two best299KB adapted models; collector/trainer/evaluator/verifier/analysis/plots/cache guard,7testmodules,8figures; `mujoco/rl/WORLD_MODEL_ONPOLICY_ADAPTATION.md`. Raw branch/rollout NPZ, per-update records, caches/runtime logs stay local-only in ignored experiment parts; original historical untracked files preserved.

Next: Only suggest an offline best-tail decision-cost-supervised modeling control with nominal regression as an acceptance gate. Not executed; do not automatically rerun MPC.

## 2026-10-09 — UAV Behavior Cloning Policy Baseline

Branch: feat/uav-behavior-cloning-baseline
Base: feat/world-model-onpolicy-adaptation / 4858ce65a29806f37a813f0225b6aa6adfa6cd15

Research Question: Can a simple supervised 7D-observation neural policy imitate the existing Scripted controller and independently complete nominal MuJoCo waypoint tasks?

Setup: Reuse original PI120 target coordinates/grouped84Train/18Val/18Test split, freshly collect one full expert episode per target (seeds2026100900+ID), 8974/1888/2199 labelled commands. All120 experts succeed; no outcome filtering. Archived benchmark100/holdout100 target/seeds checked disjoint from all120 dataset targets. Expert uses only current position error/yaw, no privileged state, and ignores velocity. Unchanged PI RewardV2, physics500/control100/policy25Hz, action bounds and success/failure/timeout. MLP7→128ReLU→128ReLU→4,18052params; Train-only obs/action scaling, action-std normalized MSE; seed0/Adam.001/batch512/exactly100epochs/Val-only selection, no tuning or extra loss. No World Model/PPO training or use in BC execution.

Key Results:
- Best epoch100; Train normalized MSE .0000812689, Val .0002641008. Gradient norm best mean/max .0110725/.0171335, all-training max .837776, no clipping/nonfinite values.
- Independent2199sample Test: normalized action MSE .0004307048; raw-command MSE3.510915e-6, RMSE .001873744, MAE .001037225. Axis RMSE vx/vy/vz/yaw .00239668/.00273742/.000897631/.0000187293; correlations≥.999614. Distance/velocity/action regions saved with Train-only thresholds.
- BC benchmark100/100 and holdout100/100; Scripted also100/100 each. No physical failure, timeout, action saturation or deployment clipping. Mean finaldistance BC .066642/.063369m vs Scripted .071441/.068744m; meancompletion5.7244/6.0464s vs5.5376/5.8132s. BC is slightly slower, not globally superior. Near-target actual speed .192324/.193490m/s averages transit steps, not success-terminal speed; unchanged hold criterion. Crossings .43/.43.
- BC visited-state posthoc expert RMSE .003925/.003213 exceeds offline error. BC standardized obsRMS .9989/.9631 vs Train .8719, Scripted1.0344/.9905. Mild distribution differences without observed catastrophic accumulation; marginal box departures are not strict OOD proof. No privileged-input deficit for this expert; no general physical-state sufficiency claim.
- Independent verifier checks all120expert labels/indices,400complete closedloop records, target isolation, scaling, epoch/Valselection, frozenmodel identity and metric recomputation; two real complete episodes bitwise reproducible. Canonical v3 SHA42571249… unchanged and unused.20newtests pass; fresh post-fix read-only historical-fixture suite547passed/4knownskips(551run), no current algorithm failure. Pre-review bare455passed/30knownlegacyimporterrors/1skip(486run) is disclosed separately, not claimed green. Sole fresh review's SIGTERM child cleanup and outcome-dependent publication fixes each RED→GREEN in one pass; no deferred Minor findings and no scientific metric change.
- Serialized guarded CPU phases,1MuJoCo worker/max2allowed, reserve1.5GiB/abortavailable<2GiB/RSS>4GiB; atomic fsync/replace/cache identity. No new OOM or swap modification. Core experiment peakHWM approximately1.07GiB; final full-suite sampled process-tree peakRSS2.851GiB / summedHWM3.908GiB, minimum available8.258GiB. These sampled/aggregated values are not hard OS guarantees. No concurrent large simulation/training.

Conclusion: Accept a nominal independent learned-policy baseline: BC matches Scripted200/200 success using the same nonprivileged observation. Offline errors remain small enough for this fixed nominal task. This does not establish noisy/reset/physics/recovery robustness or replace canonical world-model v3. No failures exist for the requested failed-trajectory plot; it truthfully states this instead of fabricating one.

Limitations: one training seed,120 expert trajectories, simple directly representable expert, fixed nominal task/resets, narrow yaw/action coverage, ceiling-success evaluation, no perturbation/recovery/real flight/visual control; historical raw target archives remain local dependencies.

Artifacts: best75KiB `mujoco/rl/models/uav_bc_mlp_seed0.pt`, `mujoco/reports/uav_behavior_cloning_seed0.json`, target/dataset/closedloop manifests/hashes, source/tests/verifier/runner, seven scientific figures, `mujoco/rl/UAV_BEHAVIOR_CLONING_BASELINE.md`. Raw NPZ, training/runtime logs, caches/scratch remain local-only; historical untracked files preserved.

Next: Only recommend one fixed-manifest perturbed-initial-state BC robustness evaluation. Not executed. No DAgger/PPO/SAC/world-model/MPC/visual training is started.

## 2026-10-09 — BC Initial-State Perturbation Robustness

Branch: feat/uav-bc-initial-state-robustness
Base: feat/uav-behavior-cloning-baseline / 11d3e6bed47d87e86956349db207e89727d08bef

Research Question: Does the frozen nominal BC policy retain waypoint-control performance under real initial position, world-velocity and yaw offsets, compared with its nonprivileged Scripted expert?

Setup: No training or tuning. Frozen `uav_bc_mlp_seed0.pt` SHA7d7cf248… and parameter hash77138275… unchanged; standalone7→128ReLU→128ReLU→4,18,052params, original Train-only scaling. Scripted uses the same7D observation, position/yaw only, no privileged state. Nine predeclared conditions: nominal; position norm .10/.30m, velocity .15/.40m/s, yaw balanced±10/30°, combined small/large. Deterministic Gaussian directions reused across severity/combined, seed2026101001; original benchmark100/holdout100 targets/resetseeds. All1,800 initial states valid before smoke, minimum true collision clearance .651823m. Real qpos/qvel/unit quaternion + mj_forward; full MjData/PI/yaw/allocator/env/RNG snapshots, previous-distance correction, otherwise original reset hidden state. Unchanged500/100/25Hz, PI RewardV2/success/failure/15s timeout. All3,600 paired controller episodes retained.

Key Results:
- Nominal, position-small/large and velocity-small: BC and Scripted100/100 benchmark and holdout. All400 nominal trajectories bitwise match previous action/observation arrays.
- Velocity-large: BC98/99 vs Scripted95/96; pooled197/200 vs191/200, +3pp.191 both successes,6 BC-only,0 expert-only,3 shared timeouts. BC mean max speed .826913m/s vs expert .837736; no velocity-only saturation.
- Yaw-small: BC73/76 vs expert100/100, pooled149/200, −25.5pp and51 expert-only successes. Yaw-large: BC1/1 vs expert100/100, pooled2/200, −99pp and198 expert-only successes.
- Combined-small: BC75/72 vs expert100/100 (147/200, −26.5pp). Combined-large: BC1/3 vs expert94/90 (4/200 vs184/200, −90pp);180 expert-only successes and16 shared timeouts, so not all failures are BC-specific.
- All1,800/controller: BC1,299successes/501timeouts, expert1,775successes/25timeouts. No physical/nonfinite/tilt/flight-area failures. All failed completion times null. Paired completion difference only both-success pairs: nominal+.210s; position small/large+.259/+.240; velocity small/large−.014/+.0329; yaw small/large+6.7611/+7.0(n2); combined small/large+5.9238/+2.58(n4).
- Pooled mean final distance BC/expert: nominal .065006/.070092m; position .065322/.071305 and .066912/.071857; velocity .064436/.066016 and .073652/.074315; yaw .084500/.069444 and .258092/.066152; combined .086447/.065879 and .256575/.078205. Full per-split outcomes, near-target speed, excursion, recovery, saturation, paired distributions and exploratory signed direction groups retained.
- BC mean max speed yaw-large2.189m/s vs expert .789532, combined-large2.183338 vs .866008. BC final absolute yaw nevertheless .314°/.422° at yaw10/30 vs expert3.077°/7.228°; poor waypoint success is not simply failure to correct yaw. Task does not require yaw recovery. BC saturation yaw-small/large .0135%/1.0638%, combined .0111%/1.0858%; expert0. Nominal Train yaw std .00090545rad implies193/578-std initial yaw extrapolation; coverage is a plausible explanation, not unique causal proof or evidence of privileged-input insufficiency.
- Wilson95% success intervals and fixed-seed descriptive paired bootstrap reported.200/200 is not a zero-risk guarantee. All1,800 equal paired snapshots/3,600 unique tasks/hash-checked resume verified.15new+18existingBC tests pass; final full read-only archived-fixture suite562passed/4knownskips(566run). Final bare suite473passed/30 historical missing-dependency import errors/1skip(504run), no assertion failures; all error names disclosed. No legacy restoration.
- Fresh review found an Important nonfinite-terminal placeholder/JSON failure-accounting edge case. Real-environment injection reproduced it RED; validity-mask/nullable unavailable measurements fixed it GREEN without changing physics/controller/finite formulas. Same3,600 tasks rerun under new source-bound manifest: every original observation/action/physical array bitwise equal, all original scalar outcomes identical. First-run artifacts recoverably archived; no extra independent samples implied.
- One worker, atomic fsync/replace/cache verification, no OOM or swap changes. Final evaluation720.20s, sampled peakRSS0.998GiB/HWM1.146GiB; both-run all-phase peakRSS2.802GiB/summedHWM3.728GiB, minavailable7.248GiB (whole-suite tests dominate). Sampled telemetry, not hard OS safety guarantee. Canonical World Model remains unchanged/unused.

Conclusion: BC is a useful nominal/position/small-velocity policy baseline, not a generally initial-state-robust policy. Strong yaw/combined robustness gap relative to the same-information expert; both controllers also have some large-combined recovery limitations. Do not replace these negative findings with training or timeout changes.

Limitations: single frozen BC training seed;200 fixed targets, one fixed direction per target/component; estimated nominal simulation/coarse collision geometry; initial offsets only; yaw not a success condition; exploratory direction analysis; no causal identification of a unique failure mechanism; no training or recovery policy improvement.

Artifacts: evaluator/snapshot manifest/paired analysis/publication/plots and15tests, `uav_bc_initial_state_robustness_seed0.json`, `uav_bc_robustness_manifest_seed0.json`, seven requested figures, `UAV_BC_INITIAL_STATE_ROBUSTNESS.md`. Both-run raw/cache records/runtime logs/review scratch total about198MiB and stay local-only; all historical untracked files preserved. No model copies or changed checkpoints. Three low-priority review items deferred/disclosed: imported metric-source identity coverage, absolute checkout path in resume manifest, cosmetic completion-plot count-label overlap.

Next: Only recommend a separate controlled yaw-coverage expert-data BC experiment with unchanged architecture and strict target isolation. Not executed; no DAgger/PPO/SAC/world-model/MPC changes.

## 2026-10-09 — Controlled BC Yaw Coverage

Branch: feat/uav-bc-yaw-coverage
Base: feat/uav-bc-initial-state-robustness / b05cf751cb776f2a17d21b19f4c4e629de9a8775

Research Question: Does yaw-perturbed expert coverage improve translational recovery and braking beyond equal-budget nominal-expanded BC, without changing architecture, normalization or control interfaces?

Setup: Read-only preflight finds no label/interface error. Observation xyz error/velocity and expert velocity commands are world-frame; yaw error wrapped target−actual, wxyz physical quaternion. Expert uses the same7D observation, no privileged inputs; not a full-Markov sufficiency claim. Original8974Train yaw std.00090545rad, maxabs.28864449°. Original BC SHA7d7cf248… frozen. Fresh seed0 B/C actors7→128ReLU→128ReLU→4,18052params, identical initialhash668f4076…; Original Train-only observation/action stats strictly retained. Adam.001/batch512/exact100epochs/4000updates each, identical shuffle schedule; only common yaw-mixed Val normalized action MSE selects best. No fine-tuning, search, teacher inference or added loss.

Data splits: Original84Train/18Val target groups reused, original18Test excluded.840paired base states/source with common small initial position/velocity variations; B actualyaw0; C40%nominal/30%balancedyaw10/30%balancedyaw30.180commonVal expert episodes. All1860experts succeed, no outcome filtering. Exact20000uniqueTrain transitions/model and4000commonVal; fixed early/braking/highspeed/other quotas10/30/25/35%, per-phase yawmix/signbalance.100newFinal targets seed2026101101 isolated from all original Train/Val/Test/benchmark/holdout and all adaptation targets. Nine previously fixed perturbation conditions ×100targets×4controllers=3600episodes, exact paired full simulator/controller snapshots. Unchanged25/100/500Hz, PI RewardV2, success distance<.1m/speed<.15m/s for5steps,15s timeout. Original is historical reference, not a budget-matched4000update causal control.

Key Results:
- Best B35: Train .000245292/Val1.534668607; best C84: Train .003143932/Val .003336325. Valyaw0/10/30 B .000364/.541202/4.573874; C .002187/.002372/.005834. All gradients finite, no clipping. Shared newTest45337validexpert states: deployed action RMSE A/B/C .181207/.113588/.005668; yaw30 .284085/.179871/.007402; earlylargeyaw .500307/.379369/.015470; nearbraking .182137/.105390/.004408.
- Success Scripted/A/B/C (all /100): nominal/position small/large/velocity small all100; velocitylarge97/98/98/94; yaw10 100/74/92/100; yaw30 100/0/6/100; combinedsmall100/74/93/100; combinedlarge92/2/6/98. C meets predeclared nominal≥98/yaw30≥90, not a population-safety guarantee. All failures retained; no physical/nonfinite failures, only timeouts. C has8timeouts:6largevelocity BC-specific relative to Scripted,2largecombined shared with Scripted.
- Paired C−B success yaw10+8pp/yaw30+94pp/combinedlarge+92pp, no B-only successes in those three; velocitylarge−4pp (2 C-only,6 B-only). B−A yaw10+18pp demonstrates an extra-data/coverage effect but cannot explain C−B yaw30+94pp.100/100 Wilson95% lower bound96.30%, not zero population risk.
- Yaw30 mean peak translational speed Scripted/A/B/C .780119/2.173531/1.563390/.777216m/s; distance rebound .258167/2.176317/.948151/.256673m; mean final distance .063311/.262867/.127629/.059448m. C near-target actual/command speed .202678/.021008m/s versus B .263256/.028330; A has no near-target samples, not zero speed. C saturation0 in everycondition, Originalyaw30 about1.00%. Near-target metrics include braking samples, not only terminal success-hold states.
- C mean completion nominal/yaw10/yaw30/combinedlarge5.670/6.285/6.616/9.869s; paired C−B yaw30−6.913s only n6 both-success episodes. C yaw30 finalyaw6.95° vs Original .42° reflects earlier successful termination; yaw is not a success condition. Transient/braking repair, not faster yaw correction, is the relevant improvement.
- No nominal closed-loop success regression; nominal offline RMSE C .004492 vs B .002745 is worse, and mean finaldistance .065769 vs .061963m slightly higher. Position/velocitysmall preserved; velocitylarge94 vs98 remains a trade-off. No universal superiority claim.
- Independent data/label/phase/model/Val-selector/snapshot/trace-scalar/paired integrity verification passed; complete same-cohort resume verified, no duplicates/omissions.56BC-related tests passed (23new); read-only historical-fixture full suite585passed/4knownskips(589run),0failures/errors. Bare whole suite496passed/1skip/30historicalmissing-dependencyerrors(527run),0assertionfailures; bare suite not green, all error names/commands disclosed in JSON/documentation. No historical files restored. Wilson0/100 endpoint-rendering roundoff regression fixed RED→GREEN without changing statistics/results, fresh suite repeated.
- One MuJoCo worker, serialized guarded phases, no OOM/swap changes. Formal evaluation756.49s/RSS .934GiB; completed phases including retained earlier test-guard telemetry sampledtreeRSS2.986GiB/summedHWM5.132GiB/minavailable8.362GiB (tests dominate). Summed per-process historical HWM is not a simultaneous RSS peak;4GiB sampledRSS guard respected. Phase lock refused one premature test launch before any child; later serial run completed. Cache/checkpoint/content hashes intact, canonical World Model unchanged/unused.
- One fresh read-only whole-branch review: Important physical-failure partial-interval verification fixed via real .042s second-action outside-flight-area injection RED→GREEN; scientific cohort onlysuccess/timeouts, no outcomes/data/model changes. Two minor disclosures deferred: mid-epoch optimizer-interruption scenario untested; early/braking action diagnostics clipped-only despite broad documentation wording. No extra research run or second review.

Conclusion: The matched expert-data intervention supports yaw coverage as an important source of the prior yaw-conditioned translational failure. C is a useful yaw-covered simulation policy baseline, but the remaining large-velocity timeout gap and nominal imitation-precision trade-off are retained. Canonical World Model and Original BC remain unchanged; no DAgger/PPO/SAC/MPC/controller modification.

Limitations: single training seed,100fixednewtargets, initial offsets only, nominal simulation, fixed phase-stratified imitation frequencies, common small xyz/velocity coverage expansion, Original not budget matched, expert ignores velocity and7D not guaranteed fully Markov, yaw not in success criterion, no real-world safety claim.

Artifacts: two bestBC models, train/data/evaluator/verification/plots code, tests, `uav_bc_yaw_coverage_seed0.json`, target/state manifest and hashes, eight figures, `UAV_BC_YAW_COVERAGE.md`. Rawexpert/finaltraces, selectedNPZ, recovery optimizer checkpoints/cache/logs stay local-only; historical untracked files preserved.

Next: Only recommend one fixed-policy persistent external-disturbance recovery evaluation of Yaw-Augmented BC. Not executed; ordinary supervised yaw coverage repaired yaw30, so DAgger is not the immediate required next step.

## 2026-10-10 — Frozen BC External Disturbance Robustness

Branch: feat/uav-bc-external-disturbance
Base: feat/uav-bc-yaw-coverage / e30984ad879c2263081c0f9244b3ad1dc1bea8d0

Research Question: Can unchanged Scripted, Original BC and Yaw-Augmented BC recover waypoint control under sustained lateral COM forces and short gusts, beyond initial-yaw generalization?

Setup: Evaluation only, no optimizer/model/PI/reward/action/termination changes. Both checkpoints eval/requires_gradFalse and before/after file/parameter hashes unchanged (Original7d7cf248…/77138275…; Yaw-Augmentedd29b9355…/b1cdac0d…). Real7kg compiled single-body nominal model, gravity9.81m/s², worldXY force at moving bodyCOM through xfrc_applied assigned before every500Hz physics tick, zero direct torque. Low/Medium/High=.05/.10/.20mg=3.4335/6.867/13.734N. Constant from0to termination, gustMedium/High ticks1000–1999/[2,4)s, nominal0;100/25Hz PI/policy unchanged. StaticHigh70.029934N/11.31° feasible below158.391N thrust/3m/s² horizontal command limit, but1.962m/s² exceeds unchanged1.5m/s² integral compensation cap; no force retuning. Not a wind-speed/aerodynamic model.

Data:100fresh full-task target groups, seed2026101201/reset950140000+i; isolated from600original/adaptation/yaw-final targets byIDs/seeds/coordinates.100deterministic permuted angular bins cover worldXY circle, same directions/conditions/full MjData+PI/yaw/allocator/RNG/previous-action snapshots across controllers. All600paired initial states valid, all1800episodes retained, full hash-checked same-cohort resume without duplicates/omissions. Ground-truthforce/time trace per physics tick independently verified; nominal subclass bitwise matches original PI trajectory in tests.

Key Results:
- Success Scripted/A/C (/100): Nominal100/100/100; ConstantLow100/100/100; ConstantMedium100/97/94; ConstantHigh0/0/0; GustMedium0/0/0; GustHigh0/0/0. Medium C−A−3pp (91both success,3C-only,6A-only), C−expert−6pp (6expert-only). High/gust failures are same-episode shared across allthree, not simply aggregate ties.
- High outside-flight-area failures2/1/2 and timeouts98/99/98; other failures alltimeouts. Across600/controller success/physical/timeout:300/2/298,297/1/302,294/2/304. No nonfinite/tilt/below-ground failures; all failed completion times null. No catastrophic numerical explosion, but real flight-area safety failures retained.
- Mean final distance m S/A/C: Nominal .073668/.067980/.066201; Low .078710/.077321/.078369; Medium .055920/.061910/.067928; High1.007850/.988912/1.040618; GustMedium .235642/.226688/.233598; GustHigh .114990/.097200/.113910. Original GustHigh small finaldistance does not imply success: velocity/5step hold still fail.
- Mean peakspeed m/s S/A/C: Medium1.103025/1.096313/1.088197; High1.793786/1.795975/1.790496; GustMedium1.506488/1.496542/1.478409; GustHigh2.100862/2.089564/2.065649. C gust near-targetactual .488080/.623060m/s, well above success threshold when crossing/braking; High near sample only1episode/2steps percontroller, explicitly sparse. C mean distance-rebound High/GustHigh2.234408/2.312690m. Command and allocator saturation0everycondition; finalyaw remains small, so yaw correction/saturation not proximate observed failure explanations.
- Finite terminal1s constantHigh proxy (duration≥5s): eligible98/99/98; distance .956015/.964914/.985876m, velocity .017222/.025372/.043107m/s, targethold fraction0. Settling with positionbias is not target recovery. Low/Medium eligible95/95/98 and100/100/100; proxies/uncertainty/selection counts disclosed, not asymptotic holding claims because unchanged task stops on5step success.
- All100targets/controller receive full2s gust,100survive4s, no early censoring in actual cohort, every post4s physics tick force0. No qualifying five-sample position+velocity recovery by15s; recoverytime null/right-censored, not15s or permanent unrecoverability. Fixed real traces show target crossing with speed and continued oscillation; common Scripted/BC limitation, not BC-only.
- Independent complete/hash/action/index/physics/scalar/aggregate/paired verification passed;84BCtests (28new) passed. Fresh full read-only archival-fixture suite617run/613passed/4knownskips/0errors/failures. Bare555run/524passed/30pre-existing missing-dependency import errors/1skip/0assertion failures; notgreen, exact same error-name set and commands disclosed, no legacy file restoration.
- Oneworker, guarded serialized phases, atomic cache/fsync/replace, noOOM orswap changes. Formal simulation585.36s/sampledtreeRSS1.101GiB; allcompleted-phase peakRSS3.122GiB, maximum sumhistoricalHWM3.934GiB (not simultaneouspeak), minimumavailable7.545GiB; fullsuite dominates memory. No unbudgeted simulation/training concurrency.

Conclusion: Yaw-Augmented BC retains nominal and Low sustained-force task success but is not broadly external-disturbance robust. Yaw coverage does not fix Medium expert-relative timeout gap or shared High/gust failures. Current evidence supports a sustained-position-bias/shared transient-braking limitation, not unique BC failure or real mine wind resistance. No policy replacement/training or PI intervention.

Limitations: fixed100target cohort/single frozen policy seed; nominal engineering-estimated mass/inertia/coarse collision; external resultant rather than aerodynamic wind; no force observation;15s recovery budget;25Hz sampledpeaks; success stop prevents indefinite stationholding; unequal terminal-window eligibility; exploratory paired uncertainty; no real-world guarantee.

Artifacts: four new source modules/three testmodules, JSONreport/target+force+initialhash manifest, eight namespaced figures, `UAV_BC_EXTERNAL_DISTURBANCE.md`. Rawpolicy/physics traces, cache/runtime logs and review scratch remain local-only; existing checkpoints/worldmodel/historical untracked files untouched.

Final review: two Important issues fixed in one RED→GREEN pass (partial/failed terminal steps cannot confirm gust recovery; retain one public acquisition identity with separately hashed revised processors/publication). Full2s gust exposure and consecutive completed25Hz boundaries are required. Original evaluator/source/cache/trace identities are preserved, not relabeled as new evidence; all1800 cached metrics recompute unchanged. One Minor generic `shared_failed_conditions` naming issue is deferred: authoritative same-target evidence is `three_way_paired_outcomes`, which independently reports100/100 shared High/gust failures. No new simulations or second review.

Next: Only recommend one read-only fixed-controller post-gust PI-integral/braking-response audit with paired Scripted/BC traces, no tuning. Not executed; no DAgger/PPO/SAC/worldmodel/MPC next stage.
