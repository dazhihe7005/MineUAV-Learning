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

## 2026-10-10 — UAV Gust Recovery Audit

Branch: feat/uav-gust-recovery-audit
Base: feat/uav-bc-external-disturbance / e273c24eda37a3f06c21d845a4bc5e94be065b98

Research Question: Why do Scripted, Original BC and Yaw-Augmented BC share Gust-Medium/Gust-High failures, despite Scripted100/100success under Constant-Medium?

Setup: Read-only code/trajectory audit;0new MuJoCo episodes,0model execution/training/optimizer steps,0PI/physics/success edits. All1800original task records/trace SHA256 verified,900existing episodes analyzed (3conditions x3controllers x100same targets). Fixed representative indices0/33/66/99 specified before metrics, with full-cohort aggregate and same-target pairing. Sequential source reads, atomic new report/figures, preserve original raw/cache/checkpoints/history.

Key Results:
- Timeline confirmed500Hzphysics/.002s,100HzPI/.01s,25Hzpolicy/.04s,15sunchanged timeout. Gustfull2s atinteger ticks1000–1999/[2,4)s, every post4srecordedphysics tick force0; all600gust episodes have11srecovery window and endtimeout, not physical/numericalfailure.
- Constant-Medium success S/A/C100/97/94; GustMedium/High all0/100. Every600gust episode briefly increases targetdistance after4s then approaches again;599/600enterdistance<.1m,592/600everreduce speed<.15m/s, but0/600evermeet both simultaneously after4s. Five-step dwell alone cannot explain shared failure; permanent unrecoverability beyond15s remains censored.
- ScriptedMedium/High d4=1.18295/2.49616m, d6=.72706/.49134m, final=.23564/.11499m; post4distance growth excessmean .10197/.29625m. Post4peak speedmean1.50649/2.08428m/s atmedian5.4/5.6s, after forcewithdrawal. Final1sspeedmean .29075/.45575m/s. BCcurves/velocity remain similar, with continued damped-looking targetcrossings rather than monotonic divergence.
- Recorded command-element and allocator saturation0across900episodes. LateScriptedmedium/high XYopposing-command fraction47.78%/46.32% among command>=.02m/s &actualXYspeed>=.15m/s, longestcontinuousopposedrun episode-mean1.1012/1.2388s. Opposing direction can be legitimate braking, not proof of wrong-sign control.
- SourcePI Kp[1.5,1.5,2], Ki[.5,.5,.8], KiI XYnorm/Zabs cap1.5m/s², outputacceleration caps3m/s², conditionalantiwindup freeze; episode-reset only, no gust-reset/leak. InitialIzero recorded, but subsequentintegral/P/output/clamp/freezecount/attitude/rotor time series NOTRECORDED. No syntheticPIplot and no directwindup/retention/rootcauseclaim.
- Constant startsforceat0s; Gust startsat2s thenwithdrawsat4s, so histories differ beforewithdrawal. Cannot attribute Constant-vs-Gust difference solely to integral carryover/removal. Success stops Constant earlier, no post-success15scontinuation manufactured.
- 18/18pure analysis regression tests and exact second-pass metrics/identity reproduction; oldscalars/success/recovery agree. Fresh finalreview terminal-failure/completion success-streak issue fixed in one RED→GREEN pass, and completedpublication HWM included rather than earlyRSS only. Checkpoint/source/rawhashes unchanged.0MuJoCoworkers,1serialanalysisprocess, noOOM; final guardedprocess-treeRSSpeak155.19MiB. Fullproject tests deliberately not run because some create episodes/train fixtures.

Conclusion: Confirmed shared failure phenotype is postgust motion/target recrossing without joint position-and-speed settling within the unchanged15swindow. A common underdamped/transienttracking limitation is supported, but single-module cause is unresolved. PIwindup, actualintegralclamping, retained/reversedcompensation, or timingbug are not established. Effective stopping is insufficient in this window; does not prove divergentdynamics or permanent inability.

Limitations: Existing100targetcohort, finite15s/right-censoring,25Hzstate logs vs100HzPI, missinginternaltelemetry, force-onsethistoryconfound, unequalConstantterminationexposure, no causal intervention.

Artifacts: Pure analysis/plots source, fixturetests, `uav_gust_recovery_audit.json`, fivefigures, `UAV_GUST_RECOVERY_AUDIT.md`, thislog. Originalrawtrajectories and newanalysis scratch/resources remainlocal-only. No `pi_integral_diagnostics.png` because no crediblePI timeseries.

Next: One minimal fixed-target Scripted Constant-Medium/Gust-Medium telemetry validation with unchanged gains/criteria, logging100HzP/I/error/raw-and-limitedacceleration/clamp/antiwindup/attitude/allocator. Suggested only, not executed. Stop this stage; no controller/model tuning.

## 2026-10-10 — Minimal PI Gust Recovery Telemetry

Branch: feat/uav-pi-internal-telemetry
Base: feat/uav-gust-recovery-audit / a3e6ee4da8eb437f04d2e2c030f5be7b7b5d4737

Research Question: Which measured internal cascade signals explain phase-separated position/speed recovery after gust removal, without changing controller gains or task criteria?

Setup: Preselect old manifest targetexternal-final-2026101201-000/index000, seed950140000, fixed worldXYdirection[.98453548,.17518528,0]. Scripted Constant-Medium/Gust-Medium6.867N only; original500/100/25Hz and15sbudget. External passiveCPythontrace observer copies realPI intermediates/clampbranch/returnoutputs at100Hz; actualforce500Hz, originaltermination25Hz. No simulation/controller/model/policy/physics/gain changes ortraining.

Key Results:
- Full two retained trajectories match original nontelemetry arrays bitwise, including actions, states, timestamps,force andtermination. Constant316steps/12.64ssuccess; Gust375steps/15stimeout. Fullsnapshotrestore andinitialphysical/controllerpair hashesverified, excluding only deliberateforce-conditionname. Resume0newflights.
- XYI peaknorm1.010090Constant/.809576Gust versus1.5m/s²cap. IntegralclampXY/Z, antiwindupfreezeXY/Z, accelerationlimitingXY/Z andallocatorsaturation all0inboth. Measuredretention is not saturation-drivenwindupproof.
- Gust d4=1.265298m, post4peakd1.368262m@4.34s, thenreturns/crossestarget. Post4speedpeak1.478594m/s@5.51s. Alongforceaxis:P reverses4.55–4.56s; committedPI/desiredbodyZ4.81–4.82s; actualbodyZ5.50–5.51s; I6.10–6.11s. AtpeakP+2.075234/I−.411827/PI+1.663408m/s², actualbodyZonly+.001501. Requested/actualreversal timingseparation≈.68–.70s isdescriptive, notidentifiedpuredelay.
- After4sGust100Hzdistance-only105/speed-only60/joint0;25Hz26/15/0. Near-targetminimumspeed.327396m/s; low-speedminimumdistance.211435m. Fivepost4signedposition/velocityreversals; nojointgates evenwithoutdwell. Timeoutcensors eventualsettling; noasymptoticinstabilityproof.
- No separately storedPvariable: P isexplicitlyderivedfromactualerror/gains. Actualpre-limithelperlocal afterIcommit differsfrompreantiwinduptrialraw. Allocatorcommandedthrust isnotmeasuredrotorthrust. CachedMjDataqacc isfrompriorphysicsstep, notinstantaneousnewcommandresponse. Forceindexedafteractualhookavoidsstalebufferat2/4s.
- Two uniqueepisodes,3flight executions: initialConstantcompletedbutpublicationfailedonnewvalidator's25/100Hznamespacecollision; exactregressionRED→GREEN, onesameepisode technicalrepeatlogged, no newtarget/condition. Allattempts/sourceidentities retained, no hiddenomission. NoOOM. One worker, guardedatomiccache; sampledRSS≈1.178GiB, sumhistoricalHWM≈1.184GiB, exactresourcephasesinreport.
-45/45relevant tests passed(19new,18savedtraceaudit,8controller). No newflighttest/training. Bareglobaltestsnotrun becauseunrelatedsimulation/trainingfixtures violate minimalbudget. Originalsource/checkpointhashesunchanged.

Conclusion: Confirmed phase-separated transient response and requested-versus-actualattitudelag; supported boundedintegral/P opposition plus finiteattitude/cascade response as contributors. No integralclamp/windup orunique modulecausalproof. OneMediumScriptedtarget cannot establish allHigh/BC600failures. No automaticPI adjustment orPPO/SACstart.

Limitations: onefixedtarget/direction, ScriptedMediumonly; Constant/Gustonsethistoryconfound; finite15scensoring; diagnosticderivedP/angle/projections; commandednotmeasuredrotorthrust; no causalI/attitude intervention; no crosscontroller/hightelemetry.

Artifacts: passive telemetry/acquisition/analysis/plots/testsource, `uav_pi_internal_telemetry.json`,8namespacedfigures, `UAV_PI_INTERNAL_TELEMETRY.md`, thislog. Rawtelemetry, selection/attempts/recovery/cache/resource/testevidence local-only; historicaluntracked/rawfilespreserved.

Final review: independent readonlyreview recomputed bothcomplete analyses, all12baselinearrays bitwiseequal, original/processor/figurehashesverified and45/45tests. No blockingproduction/scientificfinding. Raw-corruptiontest originallyfailedtooearlyonmissingrecorddigest; regradedImportant, strengthenedexpectedrawhasherrorRED→correctfixturevalidrecorddigestGREEN,45/45freshsuite. One documentationminor deferred: `--verify-only` is zero-new-flight verification but refreshes localverification.json, not literalfilesystem-read-only. No secondreview/newflight/controlchange.

Next: One separate same-state local attitude-cascade response validation comparing requested braking with actual thrust-direction response, unchangedgains/taskcriteria. Recommendation only; no controller changes, PPO/SAC or othertraining executed.

## 2026-10-10 — Gust Recovery Attitude-Thrust Response Audit

Branch: feat/uav-attitude-thrust-response-audit
Base: c84389c3457bfa88d6def6be517055ec3c8e8161

Research Question: Where do requested braking, desired attitude, actual attitude, model-derived thrust and translational acceleration differ during the saved gust recovery?

Setup: Exactly2saved Scripted Constant/Gust-Medium index000 trajectories. No new flight episodes or integrated steps. Same target/seed/initialstates/6.867N/worlddirection,500/100/25Hz,15stimeout, original success. Scratch mj_forward at1264+1500saved control states; compiled stateless motors/no delays/history/contact, current world COM force, correct body-origin/COM distinction. All historical input/source/checkpoint/report identities unchanged; full previous12-array bitwise reproduction reverified.

Key Results:
- Gust returning-motion P/relative-velocity braking event4.55–4.56s; opposingI delays PI/desiredtilt reversal4.81–4.82s. Actualtilt, model rotor force/currentCOM/origin acceleration and3Dspeed-power reversal5.50–5.51s. Speedpeak1.478594m/s@5.51s. P→PI separation.25–.27s, desired→actualtilt.68–.70s, descriptive crossing bounds notidentifiedpure delays. At4s earlier escapingmotion alreadydecelerates; these later events concern returning approach.
- Model scalar actuatoroutput equals actualctrl exactly; no motor dynamic state/delay/history. Max scalar thrust difference1.57e-13N, COMtorque difference1.56e-14N·m, COMbalance3.08e-15m/s², requestedtorque/Euler angularacceleration discrepancy<9.73e-14rad/s². No additional resolved attitude→force→COMacceleration separation.
- Fixed3–8s attitudeerrorRMS Constant3.07964°/Gust6.97201°, orientation acceleration-error contribution RMS.525025/1.200105m/s²; scalar-magnitude contribution≈1e-14m/s²; body-origin/COM correction.004260/.009267m/s². This exact geometry decomposition is not causal module attribution.
- AtGustspeedpeak force-axisP+2.075234, I−.411827, PI+1.663408m/s²; I reduces signedP by19.84%. Physicalmodel COMaccel only+.014940/origin+.017735m/s² because actualthrust direction has onlyjustreversed. Integralopposition plus downstreamattituderesponse bothremain relevant.
- Constant succeeds12.64s; Gusttime-outs15s. After4s100Hzjointgates19/0. Position/speedphase separation persists; no success criterion changes or eventualsettlingclaim.
-66/66relevant tests pass(21new+45previous). One worker, atomic/frozenidentity/localderivedcache; noOOM/swapchange. Exact sampledRSS/HWM andphaseevidence in newreport. Earlier testfixture decimal-clock discrepancy3.47e-18 wascorrected tosaved-clockbitwise+1e-12lattice checks, not trajectory rounding.

Conclusion: Strongest directly localized chain discrepancy is desired→actual attitude/thrust direction. No evidence for scalar allocator fidelity failure or simulated motor lag. Supports finite attitude/cascade response together with upstreamP/I opposition contributing to transient braking/settling; cannot uniquely blamePI/attitude/outerloop or infer specificgain/structuralfix.

Limitations: oneMediumScriptedtarget/direction, saved100Hzstates, engineering mass/inertia/rotor model, conditional model-derived forces notsensors, finite15swindow, no causalintervention, Constant/Gusthistoriesalreadydiffer, no High/BC/generalcohort telemetry. Bare global simulation/trainingtests excluded. Raw/derivedarrays/identity/process evidence local-only; historicaluntracked preserved.

Final review: fresh independent read-only reviewer found no Critical/Important issues;44hash/index checks passed. Fresh66/66coordinator tests,0skips/no integrated flight. One deferred Minor: PNG regeneration is not atomic; current5figures verified, changed hashes rejectfinalization. JSON/NPZpublication remainsatomic. No controller or scientific-result changes after review, no second review.

Next: Stop this bounded audit. Any controller adjustment requires a separately authorized causal design; no new experiment, tuning or training automatically started.

## 2026-10-10 — PPO vs Behavior Cloning Matched Nominal Evaluation

Branch: feat/uav-ppo-bc-nominal-comparison
Base: feat/uav-attitude-thrust-response-audit / 7e93edda8e1625862dc6dafb4baae58dd62ca3f5

Research Question: Under exactly the same nominal7D observation,4D action, fixedPI dynamics and task protocol, can existing deterministic PPO matchBC closed-loop success, completion time and trajectory efficiency?

Setup: FrozenScripted/OriginalBC/Yaw-AugmentedBC/PI7D PPO seed0, final100k checkpoint predeclared; no training/PI/reward/physics/task changes. New200targets seed2026101501, resetseeds950150000–950150199; disjoint from700known target records inBC/adaptation/yaw/external manifests. CompletePPO trainingtarget stream notpersisted, so fullPPOTrain nonoverlap unprovable/disclosed. ExactfullMjData/controller/task/RNG/allocator pairedinitialstate,500/100/25Hz,15s timeout, distance<.10m ANDspeed<.15m/s for5policy steps.

Key Results:
- Newnominal success S/Original/Yaw/PPO200/200,200/200,200/200,165/200; PPO35timeouts,0physicalfailures; all4command saturation0. Frozencheckpoint/parameter hashesunchanged.
- Meanfinaldistance m .069621/.064944/.064173/.100744; meansuccessful completion s5.4632/5.7060/5.5780/8.773818(PPO n165). Timeoutnotcompletion; common-success165pairs PPO−Original time+3.292848s/path+.264823m, PPO−Yaw+3.394182s/path+.301698m. No faster/shorter-pathPPO advantage. PPOmeanpeakspeed.616836m/s lowerthanBC~.7477, notoverall superiority.
- PPO35timeouts:19everenterdistancethreshold,10everjointgatewithout5stepdwell,5maxstreak4;15crossingepisodes. Timeoutmeanfinaldistance.190992m/speed.065723m/s; residualposition and/or sustainedsettling limitation, notuniversal runaway/saturation; noidentifieduniquerootcause.
- SeparateoldPI7D benchmark/holdout200flights exactlyreproduce74/100and82/100success; meanfinaldistance matcheshistorical .10751775177629046/.09892600587780538m. Trainingreport lacksoriginalcheckpointSHA; currentZIP seed0/budget100352/network/hashfrozen, noinventedhistoricalidentity.
- Historical95.6%mean belongsoldPvelocity-loop7Dfiveindependentseeds20260929/0/8/16/24, notcurrentPI. HistoricalPI10D100/100+100/100 usesextra3KiIaccelerationinputs, notfair7Dmaincomparison. Only1eligiblecurrentPI7Dseed; cross-training-seedSD N/A.642ZIPinventory includesintermediates/diagnosticreplays, not642independentseeds.
- Formal1000uniquerecords(800new+200historical),4technicalexactrepeats=1004experimentflights. Repeatarraysbitwiseequal; allrecord/NPZ/manifesthashes andsnapshotpairschecked; resume0newflights/noomissions/duplicates. All38relevantnon-trainingtests passed(15new); bareglobaltraining/8-envfixtures deliberatelyexcluded. Oneworker, noOOM, sampledprocess-treeRSSpeak1.444GiB; HWMsummary1.572GiB notsimultaneousRSS. AtomicJSON/NPZ/PNG,4GiBcap/systemreserve1.5GiB/no swapchanges.

SourceTeaching: PPO_FROM_SOURCE.md mapsactualSB32.9.0 actor/critic/distribution/GAE/clippedpolicy/value/entropy/Adam sourcefunctions toTensor shapesand formulas, distinguishesBCexpertMSE vs PPOrewardpolicygradient. Model9673parameters,BC18052; currentPPOactual100352transitions. Differenttrainingbudgets/architectures preclude sample-efficiencyclaim. Noactualoptimizerupdateperformed.

Conclusion: ExistingPI7D PPO baselineisreproducible andevaluation-interface compatible, butnominal closed-loop success andpaired efficiency remain belowbothBCpolicies. BCretainscurrentnominalbaseline; noevidence hereofPPOdisturbance superiority. Noobviousaction/observationcompatibilitybugfound;7Dhiddencontrollerstate,finitebudget,rewardalignment arepossible factors, notproven uniquecauses. Noautomaticretraining.

Limitations: singlecompatiblePI7Dtrainingseed; unknowncompletePPOtrainingtargetstream; historicalcheckpointSHA absent;nominalsimulation;deterministicinferenceonly;differentexpert/interactionbudgets andnetworks;fixed15sconditionedcompletionstatistics;noalgorithmcausalorreal-worldclaim.

Artifacts:3source modules/1test module,summaryJSON/fixedmanifest,6atomicfigures,UAV_PPO_BC_NOMINAL_COMPARISON.md,PPO_FROM_SOURCE.md,thislog. Rawtraces/per-episodecache/resources/processledger remainlocal-only; previouscheckpoints/reports/untrackedpreserved.

Final Review: Freshindependent read-only reviewfound noCritical/Importantissues; independentlychecked1000record/rawhashes/metrics/summaries,700knownexclusions,model/runner/SB3identities,figures/tutorial. Minor deferred: frozenacquisition training_source string says ::train, actualentrypoint ::run_training (teachingdoccorrect); preservecacheidentity, noresult/parameterimpact. Coordinator independentlyrecomputed185064formalpolicydecisions; allconsistent. Nosecondreviewornewflights.

Next: Only recommend preregistering one7D PI-PPO independent-seed replication with unchangedinterface/PI/reward/fixedtrainingbudget, completeTrain targetprovenance andindependentVal/Test/checkpointselection. Do notexecute; stopaftercommit/push, noPPO/SACorothertraining.

## 2026-10-10 — Fixed-Protocol 7D PPO Multi-Seed Replication

Branch: feat/uav-ppo-multiseed-replication
Base: feat/uav-ppo-bc-nominal-comparison / 77ce81a99cc6306167a91a9fd4f804417433f655

Research Question: Across five fresh random training initializations under the fixed7D PI PPO protocol, what are nominal success, training-seed variation, failure modes and paired gaps to the existing frozen BC policies?

Pre-registration: uav_ppo_multiseed_preregistration.json, manifestSHA14be61bea7c191eef5259273c0e88f47b0523e9ef624f49d5f9293b1cc2d38df, configSHAe9c1fbb99bdcaad6fed9cf8d837777f7f43fa46863675c120e7a0e101d4aebf3, published before first optimizer step. Seed0–4 fresh random networks, same unchanged historical make_ppo/MineUAVPIEnv/RewardV2/PI/7D/4D; actor/critic64x64Tanh,9673params,log_std_init−2,Adam.0003/eps1e-5,n_steps256,n_envs8 synchronous DummyVecEnv in ONE worker,batch256,n_epochs10,gamma.99,GAE.95,clip.2,entropy0,value.5,gradnorm.5. CPU1thread/deterministic algorithms; sequential seeds. Final checkpoint only, actual100352transitions/seed(49fullrollouts,490optimizationepochs,3920Adamsteps). No Test or Val selection/tuning/extended budget; old models untouched.

Data Isolation: unchanged full continuous Train sampler, passive exact target reset audit; fixed40Val and200newFinal targets, generator2026101601, resetseeds950160000..039 and950170000..199. Val/Final disjoint by coordinates/ID/reset seeds, also excluded900known BC/adaptation/yaw/external/lastnominal target records. Actual Train visits286/316/286/293/293,total1474, coordinate overlap0. Rank/episode identity hashes and aggregate target manifest committed. Historical seed+rank Gym stream rule retained, neighboring training runs share some environment target RNG seeds; network/action/minibatch initial seed differs. Not fully disjoint dataset-stream replication. Val diagnostic at20480/51200/100352 with RNG/mode restored,600Valepisodes, never checkpoint choice.

Key Results:
- All5training runs complete; total501760transitions,19600optimizer steps, all finalZIPseed/budget/Adamcounter/parameter hashes verified, fiveinitialhashesunique. New separate model paths preserve historical seed0 ZIP.
- Final200 success seed0..4=27/77/33/121/45 (13.5/38.5/16.5/60.5/22.5%). Mean30.3%,sampleSD19.4474pp,median22.5%,range13.5–60.5%; exploratory seed-unit t95 CI6.153–54.447%,n5,not200independenttrainingseeds. Current protocol did not reproduce historical checkpoint82.5%; exact historical stochastic trajectory not claimed, original thread/BLAS detail incomplete and initial update has tiny numerical differences.
- FrozenScripted/OriginalBC/YawBC all200/200 on SAME NEW targets and exact initial snapshots, all original checkpoint/source hashes unchanged. BothBC beatPPOsuccess5/5, PPO−BC success gaps−86.5/−61.5/−83.5/−39.5/−77.5pp(mean−69.7pp). On common successes, PPO−Original time+8.087/+3.355/+4.830/+3.977/+3.620s andpath+.680/+.195/+.430/+.271/+.245m; no failed-episode path/time efficiency claim.
- PPO697timeouts,0physicalfailures; all8controllers command saturation0. Failurepatterns heterogeneous:231failedepisodes everdistance<.10,36everjointthresholdwithout5stepdwell; not exclusively near-target settling. Seed0 often slow residualposition(meanfinaldistance.594m),seed1 frequently distancepass/speedfail(76timeouts); no uniquePI/reward/value cause established.
- Val20k/50k/final successes0/9/6,1/24/17,1/14/8,1/9/25,0/9/10. Seed0/1/2 show milestone regression, no checkpoint substitution. Final valueEV.970/.815/.808/.770/.798,finite policy/value/entropy/KL/clip histories; highEVseed0 with13.5%Finalsuccess means value-fit alone is not task success. StochasticTrain return/success and deterministicFinal have different data/conditions, not isolated inference-mode causal evidence.
- FormalFinal1600episodes+8bitwiseexacttechnicalrepeats;600Val,2208evaluationexecutions total (testfixtures separate). Hash/metric/paired-state verification full; cache resume0newflights,trainingresume0newsteps,allseedattempts1.58relevant tests pass0skip. OneMuJoCoworker, noOOM, sampledprocess-treeRSSpeak2.088GiB; monitored4GiBcap/1.5GiBOSreserve,swap unchanged. Rawtraces/intermediateZIP/logs local-only.

Conclusion: Under this fixed7D PI PPO100k protocol, random training runs are highly seed-sensitive and consistently below frozenBC on nominal success and common-success efficiency. CurrentBC remains nominal baseline. This is not proof PPO algorithm universally underperformsBC, not a sample-efficiency comparison, and not evidence for immediately replacing PPO withSAC. Reward/task settling alignment and training stability merit targeted evidence, not an automatic hyperparameter search.

Limitations: fiveinitialization/trainingseeds; shared fixedFinaltargets; partialGymtarget-stream overlap; exacthistoricthread/BLAS identity unavailable; limited100kbudget; deterministicnominalevaluation; differentBCnetwork/trainingdata/budget; no disturbances or newalgorithm contrast; no uniquelyidentifiedreward/value/POMDPcause.

Artifacts:4experiment source modules,16newtests,5necessaryfinalZIPs,preregistration/actual-target-audit/reportJSON,8figures,UAV_PPO_MULTISEED_REPLICATION.md,thislog. Rawtraces/trainingepisodes/intermediatecheckpoints/resource/processlogs retainedlocal-only; historyuntracked/models preserved.

Final Review: Fresh independent read-only reviewer found0Critical/0Important; independently reconstructed1474Train targets andVal/Final generation,verified all1600raw/recordhashes,15pairedcomparisons,5ZIPbudget/Adamcounts and8figures. Fresh guarded58testsPASS0skip, no extra training/flights. Minor deferred: preregisteredsourcehash list omits reused nominal protocol/run/analysis helper modules; historicalhelpers unchanged and current results independently recomputed. Preserve immutablemanifest; broaden dependency coverage only in future preregistrations.

Next: Only recommend ONE preregistered offline RewardV2-versus-task-settling alignment audit on existing trajectories/returns. Do notexecute reward tuning orSAC; stopaftercommit/push.

## 2026-10-11 — Offline RewardV2 and Task-Success Alignment Audit

Branch: feat/uav-ppo-reward-alignment-audit
Base: feat/uav-ppo-multiseed-replication / b36d7539dbf53fa3f1c17a510ac3a12f8b3bcc62

Research Question: Does the actual five-seed PPO reward and time-limit target construction align with sustained nominal waypoint success, based exclusively on existing source and logs?

Setup: Read-only RewardV2/PI environment, factory and ZIP metadata audit; installed SB3 2.9.0 Monitor/DummyVecEnv/collect_rollouts/GAE source. All1000 Final PPO episodes (303success/697timeout/0physicalfailure), sequential raw/record/target/snapshot/manifest verification. No new episode, integrated physics step, optimizer step or changes to reward/PI/model/task. Original2108 input file hashes checked before/after; raw and historical untracked preserved.

Key Results:
- Confirmed RewardV2: progress10*(dprev-d), action−.005*||normalized4Daction||², brake−.5*clip((.5-d)/.4,0,1)*speed², terminal success+10, physicalfailure−10. No dedicated absolute-distance/yaw/intermediate-dwell/timeout term, reward clipping, VecNormalize or dt factor. Near-target speed incentive exists; do not claim otherwise.
- Final acquisition discarded reward and breakdown; internal float64 qvel/Value/Advantage also absent. Final cumulative/discounted/component reward, high-reward Final failures and timeout reward trend are NOT OBSERVABLE. No synthetic reconstruction or substitution with Train return.
- Gamma.99 at25Hz, successbonus hypothetical discounted5/10/15s contribution2.875836/.818773/.233111. Undiscounted progress telescopes10*(d0-dT); discounted progress implicitly prefers early proximity. Actual time-limit bootstrap adds gamma*terminalValue to float32buffer reward after Monitor logging; done masks prevent reset-value leakage. Critic continuation versus evaluation timeout failure confirmed, historical magnitude/causal impact unresolved.
- Final failures231everdistancegate,36everjointgate,10maxstreak4;466neverdistancegate. Pre-action near-speed failures episode-weightedmean.301603m/s(n227), success.145803(n303);4timeouts only enterdistance onfinalframe, missingnearpreaction preserved. Heterogeneous residualposition/braking/dwell failure, not universal settling-only explanation.
- Timeout12–15s mean distance reduction seeds0–4=−.137996/+.062523/−.058431/−.040080/−.074755m. These are state trends, not rewardtrends. Seed/distance/duration strata preserve emptycells. Common-target common2s outcome-discordant pair N42–100,0censored, earlyprogress contrasts vary in direction; overlappingpairs are not independenttrainingseeds.
- Separately labeled actual stochastic TRAIN supplement1434completed episodes (278/308/278/285/285),40MonitorCSVs. All component totals agree with independently logged raw return≤5.01e−7; progress/enddistance identity≤1e−9. Successmeanreturns19.953485/20.956201/18.911856/20.126822/20.815432 exceed failuremeans10.088949/9.710614/8.951618/10.000377/10.417927.144failed Train episodes at/above ownseed all-returnQ75, definitions explicit; not Final failures. Initialdistance and policy-age confounding preclude causal inference.
- Lastrolling100 logged Train return vs frozen Finalsuccess descriptive seed-unit Pearson.600317/Spearman.5,n5; conditions differ, no p-value/significance or Final rewardcorrelation claim.
-26/26 scoped pure tests pass(21new+5ranking); no simulator fixtures. Guarded analysis1process/0MuJoCoworkers, sampledprocess-treeRSS/HWMpeak345591808bytes(~.322GiB),0OOM, swapunchanged. AtomicJSON/PNG; one earlyJSON NumPy-int serialization failure reproduced/fixedRED→GREEN, failedphase retained without incomplete publication. Reanalysis harness initially compared native tuple toJSONlist; JSON-normalized rerun verifies every nonresource aggregate/input/figurehash identical. All7figures visually checked.

Conclusion: Source and actual Train logs support specific reward/task-alignment questions, not proof reward is PPO's primary failure cause. Original Final missing rewards prevent a definitive high-return-failure/timeouttrend analysis. Near-target penalty and terminalbonus do distinguish success on average in recordedTrain; initialdistance, finitebudget,seedvariation,7DhiddenPIstate and dynamics remain alternative explanations. No reward tuning or new algorithm initiated.

Limitations: missing Final reward/Value histories; evolving stochastic Train supplement is not frozen deterministic Final; only5trainingseeds, sharedtargets; unequal episode lengths and initial difficulty, incomplete matched short-duration overlap; fixed15stime censoring; no causal intervention.

Artifacts:2offline analysis modules,1test module,summaryJSON/identityhashes,7atomicfigures,UAV_PPO_REWARD_ALIGNMENT_AUDIT.md,thislog. Existing raw/logs/models and new resource/process records local-only; no dataset copy or model commit.

Final Review: Fresh independent read-only reviewer found0Critical/0Important; independently checked all1000Final metrics,10common-target2s contrasts,1434Train/40Monitor records,2108inputhashes/7figurehashes and allfigures;26puretests passed with no simulation/training/artifactgeneration. One deferred Minor: describing all2108hashinputs as old/original is imprecise because3are newaudit source/testfiles; hashmap explicitly lists them and allidentities/numericalresults arecorrect. Finalreward/timeouttrend,bootstrapcausality andcounterfactualreward-improvement judgments remainunresolved, not silentlyassumed.

Next: One separately authorized minimal passive logging validation preserving per-step environment reward/breakdown, termination/truncation and training terminal bootstrap/values. Recommendation only; do not execute reward changes, PPO/SAC or new flights.

## 2026-10-11 — PX4–MuJoCo SITL Integration Phase 1

Branch: feat/px4-mujoco-sitl-integration
Base: feat/uav-ppo-reward-alignment-audit / 89df0fb3b5fc5bcd99db217d0a54c51613655a52

Research Question: Can the existing estimated MineUAV rigid-body/rotor model be driven by version-matched PX4 SITL through an isolated simulator sensor/actuator bridge, without changing historical research environments or using another physics engine?

Setup: Ubuntu22.04.5, Python3.10.12, MuJoCo3.14.0, existing pymavlink2.4.49; installed PX4 v1.18.0-beta1-575-g8f9b91b2bd, source/runtime ver-all commit8f9b91b2bdd3a65952601a084c3aa35e96fee59d, binarySHA4f491ce85d6c79fbd5713894112c8911903fcefd1bf45b7da3579cd1812beac4. External feature worktree, original status preserved; no installs/builds/training/serial/Gazebo/SIH. Generic10016 startup template, process-local allocation geometry matches unchanged7kg model aboutCOM. PX4 controls/allocates; no oldPI after outputs. Ideal [0,1] normalized thrust with THR_MDL_FAC0 maps through sqrt targetRPM to omega²,8500RPM assumed; kf static fit,km estimated,CW/CCW unverified.

Interfaces: loopbackTCP4560 bridge server/PX4 client, MAVLink2common; physics500Hz,IMU250Hz,GPS10Hz, returnedactuators250Hz/4000µs; integer clock, bounded startup then actuator lockstep. ENU/FLU→NED/FRD proper rotations, virtualCOM specific force Rᵀ(a−g) from current dynamics/Jacobians, no rawqvel-as-IMU. SyntheticGNSS/baro/mag, notFAST-LIO2. HIL_STATE_QUATERNION request observed but no truth estimator bypass; opticalflow notgenerated.

Execution: four finite sequential invocations. Smoke01 wrong data-path failed before connect (current PX4 symlinks supplied build/etc to runtime/etc); regression test fixed. Smoke02 completed12s/3000IMU/2941actuators but idealconstantmag/baro tripped DataValidator stale voting; ULog raw/voted streams plus source verified mechanism. Fixed seeded syntheticmagstd.0002gauss/barostd.01hPa (unmeasured assumptions, checks not disabled). Smoke03 completed16s:4000IMU,160GPS,3942actuators; healthyEKF attitude/local/global, yaw aligned,GNSS/mag/baro fusion,preflight passed. Actual simulator rc overrides requestedsingleIMU values toSENS_IMU_MODE0/EKF2_MULTI_IMU3; one healthy fed instance, no fabricated backups.

Key Results: one opt-in26.004s boundedOffboardhold, non-force mode/armACKs accepted, PX4armed/nav_state14 and real4motor outputs applied. Finalheight.638854m/speed.006119m/s, maxhorizontaloffset.024853m, peakspeed.194394m/s, peaktilt.141875deg, zero saturated10Hzsamples. Final2s21samples meet predefined height.651±.15m,horizontal<.2m,speed<.2m/s,tilt<10deg. Envelope1.2mheight/.5mradius/1.5m/s/20deg never exceeded. No gain tuning, second flight or realhardware. Clean ownedPX4exit0 and ctrlzeroonstop. Model/config hashes unchanged; BC/PPO/worldmodel files untouched.

Validation:33 new interface/transport/safety/analysis tests passed, RED→GREEN recorded; complete newbridge suite only, historicaltraining/largefixtures not executed. Visual responsefigure checked; reports retain failedruns and localrawULG/summary hashes. OneMuJoCo/onePX4, noOOM observed, bridgepeak574103552bytes(~.535GiB), sampledPX4peak20406272bytes, lowestavailable11324428288bytes. Startup1GiBloadbudget+1.5GiBOSreserve; runtime1.5GiBguard, swapunchanged. CPUprobe recorded, notclaimed continuouspeak orhardreal-time.

Final Review: Independent read-only reviewer verified30originaltests, all7rawsummary/ULGhashes, hovermetrics/21samplehold and figure;0Critical/1Important. Stale actuator stream could renew receive timeouts indefinitely; fixed via single absolute bootstrap deadline and immediate stale rejection after lockstep. Three pure regression tests observedRED→GREEN, including fake orchestration proving nonzeroctrlzeroing/ownedprocesscleanup; full33/33PASS. No extra SITL/flight after fix, no deferredMinor. Unmeasured hardware, general maneuvers/heading, ROS/localization, historical behavioral regression, maliciouslocalprocessauthentication and unsampledpeaks remain explicitly outside evidence; detailed decisions/costs documented.

Conclusion: PASS for version/startup/bidirectional real sensor-actuator path and one limitedPX4-controlled positionhold. This establishes a SITL foundation, notdeployment certification, broadattitude/yaw response, ROS orlidar localization.

Limitations: engineeringmass/inertia/rotorpositions, estimatedkm/unverifiedrotationwiring; ideal instantaneous motors/noESC,battery ordrag identification; syntheticearthfield/GNSS/noise. Estimatedheading≈3.4deg biased versusphysicalNED, heading_good_for_control false until required finalin-flightmagalignment (notestablished inlowhold); no yawtracking claim. Logger subscriptioncapacity warnings disclosed. Dedicated attitude-step test/ROS/MID360S/FAST-LIO2 NOTTESTED. Fourfinite runs, oneflight, no statisticalflightrobustness claim.

Next: Only recommend one separately scoped synthetic geographic heading/finalalignment validation before yaw maneuvers orROS externalvision integration. Do notexecute, retune gains orconnecthardware; stopafterfeaturecommit/push.

## 2026-10-11 — PX4–MuJoCo Heading and Yaw Validation

Branch: feat/px4-mujoco-yaw-validation
Base: feat/px4-mujoco-sitl-integration / 3b36e9d77f2f6b89699b1ebd065e1e38310999be

Research Question: Is Phase1's ~3.4degree heading difference caused by sensor/reference inconsistency, can final in-flight EKF alignment be established without health spoofing, and can one independently gated small yaw step be verified?

Setup: Actual pinned PX4 8f9b91b2/prebuilt binary and MuJoCo3.14.0; offline oldULog/math/frame audit; optional NED geomagnetic field from matchingPX4 WMM grid at synthetic47.397742/8.545594 GNSS reference. No gain/EKF-parameter/actuator/frame/model/reward/network changes. Corrected disarmed16s smoke, fixed2m alignment ascent, requested90→100deg NED yaw at32s only with actual fresh heading/health and physicalhold checks. Single MuJoCo/PX4, loopback only, no install/build/realhardware.

Key Results:
- Confirmed old field0declination versus actualEKF grid/parameter3.405605deg. CorrectedNEDfield[.217434407,.012939331,.430409260]gauss; pinnedWMM2020 table dated2024.41257, not2026measuredfield. Oldflight5s estimate-minus-truth+3.413907deg; correctedfinalflight prestep5s+.002979deg(maxabs.010692). Actualfinalinflightmagalignment15.848s/headinggood16.856s, retainedthroughflight. Groundstartupheadingtransient explicitlyexcluded.
- Yawstep actuallyauthorized32.004s with allrealflags/physicalhold, but finalrunstopped32.196s: latestqueriedestimatorflags age1.004s exceededfixed1sgate. SourcePublishStatusFlags~1Hz andquery2Hz make gate publication/querysemantics incompatible; actualULogheading/alignmentremainhealthy. No thresholdrelaxation/no furtherflight.
- Censored0.192s response,9alignedpoststepsamples. Maxobserveddesired-minus-truth10.002422deg reflectsinitialtransient; settlingnull/NOTTESTED. Noclaimyawcontrollergood/bad, yawactuatordynamics orPhase3readiness. Maximumheight2.052484m,XYexcursion.023690m,motorfraction.561665,0saturation.
- FourPhase2invocations retained: disarmed01complete16s; yaw01nostep32.004s passive3instancesheaderparsingfault; yaw02interrupted23.556s aftertoolbatchlaunchedbeforefailedtestoutputread(no step,workflowviolationdisclosed); yaw03authorizedstepbutfreshnessabort. Allctrlzero/ownedPX4exit0/historicalhashesunchanged. Finiteinstrumentationbudgetrevision, no moreretry.
- Initial59completebridge testsPASS; analysis includesclockoffset/sampletimestamps/circularinterpolation/causalsetpointhold/noextrapolation, missingstate/no fabricatedPASS, properprimaryEKFinstance, complete/stale/partialgate, correctedfieldproof, bounds/wirefields preserved. No tests concurrentwithSITL. Rawsummary/ULGidentities and7protectedbase/currenthasheschecked. Originalworktree87untracked/main/models/RL/control preserved.
- NoOOM,1worker. BridgeHWM581996544bytes/PX4sampledpeak20418560bytes,minavailable11299450880bytes; runtime1.5GiB/systemreserve,startup2.5GiB,swapunchanged. Offlineanalysispeakseparatelylogged. Rawphysicalarrays/ULogs/consoles/parameters/processledger remainlocal-only.

Conclusion: PASS for simulatedgeomagneticbias explanation/correction and genuine finalinflightEKFheadingalignment. CompleteyawresponseBLOCKED byconservative live-telemetry freshness mismatch, settling/fullactuatordynamicsNOTTESTED, externalOdometryphaseentryBLOCKED. Censoredyawrecord isnot evidenceofcontroller orEKFfailure. Do notbypasshealthchecks orretunecontrolgains basedonthisresult.

Final Review: Fresh independentread-only review59/59PASS0skip, allraw/protected/figurehashes andactualmetrics/source/gateprefix independentlyreconstructed.0Critical/2Important edgecasefindings: partialnewtopicheadersretainedoldhealth; GNSSfusiontrue didnotrequireexplicitnofault. SingleofflineTDDpass10newtestsRED→GREEN, full69/69PASS0skip; rejecttruncatedheaders/unfinishedbodies, requireexplicitGNSSposition/yaw/heightfaultandvehiclefailsafeFalse. ActualSHA-matchedrecordedgateprefixstillpassesstrictfix; noextraSITL/no secondreview. Known1sfreshnessblocker retained/explicitBLOCKED, no fullphasecompletionclaim. DeferredMinor: lineplot slopes atdiscreteflag/setpointchanges (causalnumericscorrect). Unmeasuredhardware/generalflight/fullhistoricalregression/allcleanupfaults/unsampledresourceevents remainoutsideevidence.

Limitations: boundednominalalignmentascent/one+10deg request; no completeflightresponse, multiseedrobustness orhardwaredeployment; pinnedsyntheticWMM/no calibratedmagnetometer; idealinstantaneousmotor/noESC/dragidentification; existing completebridge suiteonly, no historicaltrainingtests; diagnosticparser/workflowmistakesdisclosed.

Next: Only recommend source-consistent livehealth telemetry publication/receipt timing and offline verification, then the same boundedyawprotocol before externalOdometry. Notexecuted; noPhase3.

## 2026-10-11 — PX4–MuJoCo Telemetry Freshness and Yaw Validation Phase2B

Branch: feat/px4-mujoco-yaw-telemetry-fix
Base: feat/px4-mujoco-yaw-validation / 601f517bc9bc6034197808646931f8214e9596a4

Research Question: Can source-consistent safety telemetry avoid false stopping of the fixed +10degree yaw test while remaining fail-closed, and does the real PX4 control chain produce a valid complete yaw response?

Setup: Matching unchangedPX4 8f9b91b2bdd3a65952601a084c3aa35e96fee59d/prebuiltbinary, MuJoCo3.14.0, unchanged7kg/rotors/coefficients/idealactuator/PX4gains, correctedmagneticreference retained. Independent originalsourcepublication/sample, actualquery/receipt and250Hz lockstepactuator liveness times. Fixed source leases650/1650/1150ms: flags/status nominalperiod+500msquery+150msresponse;125Hzlocal650ms is conservative versus658ms worstsum. No duplicate timestamp renewal. Pendingquery forbids newmaneuvers, malformed/missing/unsafe/backwards/future/failsafe data latch stop; independent250ms wall actuator watchdog. Originalentrypoint retained, new Phase2B opt-in. No health spoofing, extra training, system installs or gain changes.

Key Results:
- RED monitor23failed→GREEN, opt-inrunner and analysis RED→GREEN. Original69tests retained; full bridge103/103passed0skip before independent finalreview. OriginalPhase2 ULog last5.196s replay passes newsemanticmonitor where old1s gate rejects normal queriedstates. Repository-widepytest attempted: CADcollection blocked by systemSciPy/venvNumPy2.2.6 ABI mismatch, no unrelatedinstall/fix.
- Exactly1disarmed16s check then1full45s flight; no retry. Disarmed4000IMU/160GPS/3941actuator; flight11250IMU/450GPS/11187actuator. Bothcomplete/exit0/ctrlzero/historysame. No telemetryfault/falseabort. Finalmagalignment15.864s, headinggood16.872s; yawrequest32.004s. Flagpublicationage median.774s/p95 1.280s/max1.456s;38duplicate topicreceipts independentlylogged.
- YawtrackingFAIL: trueyaw remains≈90deg whiledesired100deg. Maxpoststep targeterror10.001821deg, final2s mean9.996366deg, peakactualFRDzrate.000153485rad/s, settlingnotachieved under unchanged<2deg/<.05rad/s/2s/≤.15s samplingcriterion. Full12.996s responseobserved, notcensored/NOTTESTED. No active4motor saturation; maxphysicalheight2.056629m andXYexcursion.030747m stayinsideunchangedenvelope.
- Confirmed pinnedsource/config with corroboratingULog: existing CT1/KM±.017207708 produces yawrow entriesbelowControlAllocator weakaxis threshold.05, so row iszeroed beforematrixassignment. Requestedrate≈.488rad/s/torque≈.097 butyawtorqueremainsunallocated, torqueachieved0/65poststep status samples. No actuator/PX4source/gainfix because fixedphaseprotocol forbids it. No secondflight.
-1worker/1PX4, noOOMobserved. Bridgepeak584245248bytes, sampledPX4peak20361216bytes, minavailable10733146112bytes; atomicraw summaries andULoghashes preserved. No swapchanges; historical87untracked/main/models/dynamics/oldreports remainunchanged.

Final Review: Independentreadonly103/103PASS,0Critical/2Important/1Minor, declinedjudgmentnone. Oneofflinefixpass:6newregressionsRED→full109/109PASS0skip; independentlycomplete unsafe topics survivefollowingfragmentedtitle, olddeadlineexpiry latches before anyrenewal, blockingreceive servicesconsole in≤8msslices andusesearliestabsolutequery/actuator deadline. Duplicatepublication fixture doubleactuator finding regradedImportant for safetyevidence and corrected; remainingDeferredMinor:none. No secondreview. Final hardening wasnotre-flown; bothrawSITLruns preceded it, rawsummary/ULGunchanged, finalsourcehashes identify currentcode notbyte-exacthistoricalexecution. Fullflightbudget2runs remains exhausted.

Conclusion: PASS for telemetrytime semanticfix, realEKFalignment andcompleteboundedobservation. FAIL for yawcontrol andallocator yawauthority. ExternalOdometry/Phase3entry remainsBLOCKED. Successfullogcompletion isnot successfulmaneuver validation.

Limitations: Onefixedboundedflight; idealmotor/nohardware calibration, sampledresources/nohardrealtimeguarantee, consolepolling may miss fullytransient fault betweenqueries; sustainedfault detection bounded, no unlimitedcachedhealth. Allocatorinternalmatrix is analyticalsource/config evidence, not a sensor; actual unallocatedtorque separatelylogged. HistoricalCAD/RLfullsuite notpassing/claimed. No ROS/Odometryintegration.

Next: Only recommend separately authorized actuator-effectiveness scaling compatibility fix for pinnedPX4 threshold, verify physicaltorque mappingunchanged offline, then authorize a new bounded yawtest. Notexecuted; noPhase3.

## 2026-10-11 — PX4–MuJoCo Yaw Control Allocation Compatibility Phase2C

Branch: feat/px4-mujoco-yaw-allocation-fix
Base: feat/px4-mujoco-yaw-telemetry-fix / fbd273be0f7169f738f6c8024e7ec7be37e14941

Research Question: Can the pinned PX4 weak-axis filter retain physically valid yaw authority without falsifying motor parameters or changing the existing safety/flight protocol?

Setup: MuJoCo3.14.0, unchanged7kg dynamics/COM/inertia/rotors/kf/km, ideal normalized-thrust adapter. OriginalPX4 8f9b91b2/source/binary preserved. Rejected unproved matrix/adapter scalingA; selected local guarded SITL-only source compatibilityB. IsolatedPX4worktree commit142c24cffb0cdfd57ab0a9f2506d3203610ce737, binarySHA2f6bbc24252577513ffb7f078bcb1048e258744f6c482918d3ea29c411e17004. Single-job outer/nested build, no installs, defaultSITL board overlay excludes only3Gazebo modules. No controller gain/EKF parameter tuning or new physics engine.

Key Results:
- Confirmed original ControlAllocator whole-row filter maxabs≤.05 removes actual yaw±.017207708. Local exception retains the identical yaw row only for validated4-motor non-tilting geometry, independent well-separated rows and interior zero-moment trim, known healthy motor detector state/no stop. Raw fault transitions revoke/refresh even under default failure Ignore. Fx/Fy remain zero; RP/thrust untouched.
- Offline original filtered rank3→4, row-normalized condition1.068724. Same physical B and full motor box[0,1], c→omega² map unchanged; actual MuJoCo basis/extreme/mixed actuator wrenches match. Hoverc[.40412044,.40531244,.46297427,.46178227],Fz−68.67N, moments≈0. Native actual SequentialDesaturation/defaultparameters validates zero-yaw nominalhold/RP equality, ±yaw and unreachable-request bounds; no arbitrarytorque feasibility claim.
- One disarmed16.000s complete:4000IMU/160GPS/3985actuator, real live yawrow retained. CLIinitialexit1 was ANSI reset parsing, not bad matrix; real-line RED→GREEN parser correction before flight admits unchanged complete proof, no repeat invocation.
- Only active run stopped at simulation15.484s/wall16.088s, normalarm11s; no32s yawstep. Live yawrow preserved, finalphysicalheight1.461455m, samplemaxXY.011646m/peakspeed1.264570m/s/maxmotor.561637. No physical envelope violation recorded, ctrlzero/PX4exit0, no retry. Yawresponse/direction/settling/poststepallocation NOTTESTED; final in-flightmagalignment/45scompletion/Phase3 BLOCKED.
- Confirmed underlying monitorfault is safetyqueryservice>.65s, not proven actuator loss: lastquery15.435843s→deadline16.085843s, actuator deadline16.333173s, lastsample actuator age.002078s. Generic receiveexception names actuatorwatchdog for the minimum of all deadlines. Last5query wallintervals .53678/.55974/.58786/.60974/.64164s versus.5s simulation. Added high-frequency fullconsole parser265803chars~1.7ms/check offline is a supported overhead hypothesis, not unique causal attribution. No postflight runtime/safety/PX4 rebuild changes.
- No upper motor saturation; zero allocator limit flags after armrequest.20pre-arm lower-bound entries at.3–.9s reflect normalzerooutputs, not active saturation.388MC/MPC/EKF/CA ULogparameters compared:387same; volatileEKF2_MAG_DECL old3.405605/new0 at differentloggerstart1.344/.304s, sameconfigureddefault0/DECL_TYPE3 andexplicitenvironment/field/noise/initialstate. Automatic no-notification EKFdeclinationsave/sourceconfirmed; exactunloggedtime unresolved, not deliberatetuning.
- One independent readonlypreflightreview:134tests,0Critical/4Important/0Minor; allImportantfixedoffline beforeflight.147preflight/150finalBridge testsPASS0skip, original109preserved; nosecondreview. Review fixes author-verified, not independently rebuilt/reviewed. OriginalCADwhole-repo collection still ABI-BLOCKED; no unrelateddependencychanges. Two analysis-only faults (nullprequery plot/statusclassification) regression-fixed after stoppedflight. Fouractualtruncatedplots visually inspected; no syntheticresponse orchangedcriteria.
-14protected historicalfiles hash-equal tobase, originalPX4binaryunchanged, finalruntimehashes equaltoflightfreeze, all87historicaluntrackedfiles preserved. Raw2summaries/ULGs/build/resource/identity logs local-only. OneMuJoCo/onePX4, Bridgepeak772112384bytes(.719GiB), sampledPX4peak10018816bytes, minavailable8817274880bytes; finalincrementalbuildchildpeak406724608bytes. NoOOM observed inownedruns, no kernel-wideOOMcertificate, no swapchanges.

Conclusion: PASS for physically unchanged allocation compatibility, exact build identity, live yawrow retention and fail-closed stopping. Complete +10degree closed-loop yaw validation remains BLOCKED by wall-time query scheduling; no fullmaneuver or Phase3 readiness claim.

Limitations: one truncated active run, no final custom-buildheadingalignment/hold/yawstep; fullconsole diagnostic overhead versushostload not causally isolated; SITL-only localpatch, idealESC/motor/no hardware identification; actualphysicaltorque model-derived ratherthanmeasured; no independentbinaryreconstruction; historicalCADsuiteblocked.

Next: Only a separately authorized offline bounded/incremental allocator-diagnostic and wall-versus-simulation query-scheduling audit, preserving fail-closed deadlines. Notexecuted; no flightretry, Phase3, ROS or FAST-LIO2.

## 2026-10-12 — PX4–MuJoCo Incremental Diagnostics and Query Scheduling Phase2D

Branch: feat/px4-mujoco-query-scheduling-fix
Base: feat/px4-mujoco-yaw-allocation-fix / 26f3df6ed1417c9c9b90696809a2c80d99a43ff5

Research Question: Can bounded incremental diagnostics and causal dual-clock query scheduling remove cumulative hot-path work while preserving fail-closed safety deadlines?

Setup: Offline only; zero new PX4 processes, live MuJoCo runner episodes, arming or yaw-step commands. Retained150 tests, original and isolated PX4 source/binary identities, physics/rotors/adapters/gains/models and immutable Phase2B/2C summaries/ULogs. New-chunk streaming parser, finite record/line/recent/archive/event bounds, nonblocking single-batch console writes, single-flight absolute wall/simulation scheduling serviced during actuator receive slices. No timeout relaxation:150ms response/fragment,250ms actuator,650ms query service/receipt; source leases650/1650/1150ms.

Key Results:
- Confirmed stopped query-service deadline16.085843s versus actuator deadline16.333173s; last actuator age.002078s. Old full-history joins/scans and simulation-only dispatch are source-confirmed defects; unique historical CPU/I/O cause remains unresolved because original PTY arrival/per-call timings were not recorded.
- Five batches×100 calls at1KiB/10KiB/100KiB/265803chars/1MiB: historical yaw-row scan scales from~15us to~6–7ms; new fixed-record ingestion stays~2–3us. At265803chars mean1465.225us versus2.838us, P95 1558.209us versus2.895us. Whole query construction/validation595us versus retained legacy parsing326us; repeated legacy suffix processing1110us. Improvement is bounded incremental work, not universal end-to-end speedup. Query comparison shares the current validator and is not a frozen historical validator CPU measurement.
- Constructed12s slow-simulation replay(.8sim/wall) schedules25 queries at.5s wall intervals, no fault/backlog. Causal4ms-check fixtures stop missing reply at152ms, actuator outage with stalled simulation at252ms, service stop/duplicate publication at652ms; received failsafe latches immediately, late health cannot clear faults. These are offline fixtures, not live RTT/hard-real-time guarantees.
- Primary records and instances, partial bytes/lines, ANSI controls, duplicate fields, malformed bodies/headers, invalid source/sample timestamps and cross-read backlog are fail-closed in tested cases. An actual empty PTY is required before dispatch AND completion; expired timers checked after I/O before renewal. Raw bytes archived before decoding; old nonzero allocator text cannot perpetually renew authority. Native allocator row remains an unauthenticated query snapshot with no source timestamp.
- Author289/289PASS0skip, old150unchanged/new139. Final fresh independent read-only review0Critical/0Important/0Minor,289PASS6.89s plus144 integrated actual-before_step malformed-lifecycle casesPASS. Every prior Important/Critical finding fixed with RED→GREEN and fresh re-review; incomplete interrupted review not accepted. No runtime change after final review.
- Full-repository pytest attempted:3pre-existing collection errors in cad_check/test_inspect_stl_assembly.py, mujoco/scripts/test_baseline.py, mujoco/scripts/test_dynamics_v2.py, systemSciPy/venvNumPy2.2.6 ABI incompatibility. Global suite BLOCKED; no unrelated dependency installation/repair.
- Eight raw identities/twelve protected artifacts hash-equal; both PX4 binaries unchanged;87 original-worktree untracked files preserved. Raw data/ULogs/local ledger preserved, not committed. One analysis worker, zero live instances, no OOM observed in owned jobs, no swap change; final analyzer RSS433184768bytes, recorded author-command maximum1188416KiB(~1.133GiB), final289-test MaxRSS1045868KiB. Per-command/native compiler accounting is not combined process-tree or kernel-wide OOM certification.

Conclusion: PASS for scoped offline bounded parsing/scheduling and final independent review. Historical unique root cause remains BLOCKED; new runtime real RTT/disarmed/flight validation NOT TESTED. Full yaw response and Phase3 admission remain BLOCKED. Changed runtime requires a new matching disarmed proof; old smoke cannot authorize it.

Limitations: Synthetic timing does not reconstruct missing PTY boundaries/host scheduling; no live deadline compliance or hard-real-time guarantee; native console has no authenticated allocation publication timestamp; current pinned log schemas only, unseen formats unverified; repository-wide CAD ABI remains blocked; no new flight or physical deployment evidence.

Next: Only recommend separately authorizing a new matching disarmed SITL communication check before considering any bounded yaw flight. Not executed; no automatic flight, Phase3, ROS, MID360S or FAST-LIO2.

## 2026-10-12 — PX4–MuJoCo Disarmed Runtime Integration Phase2E

Branch: feat/px4-mujoco-disarmed-verification
Base: feat/px4-mujoco-query-scheduling-fix / c9174292e3e75cfab43706bc63b5b53b689a136c

Research Question: Does the Phase2D Bridge maintain real PX4 SITL + MuJoCo communication, bounded diagnostics, and fail-closed safety queries through one complete16s disarmed run?

Setup: One new empty runtime, one isolated verified PX4 binary2f6bbc24…, one MuJoCo3.14.0 instance, 500Hz physics/250Hz sensors. `allocation_runner --mode disarmed`, no smoke proof, OffboardLink, ARM, takeoff or yaw step. Existing150ms response/250ms actuator/650ms service limits, physical model, motor mapping, magnetic reference and protected models unchanged. Added bounded passive timing instrumentation and a strict offline assessor before the sole run.

Key Results:
- Complete16.000s simulation in16.002s Bridge wall time;4000HIL_SENSOR/160HIL_GPS sent,3986HIL_ACTUATOR_CONTROLS received.20/20 safety queries and20/20 actual allocator yaw rows complete; no monitor fault, watchdog or exception; PX4 exit0; all801 recorded motor vectors and shutdown controls zero. The Bridge sent no command message. PX4→Bridge received one COMMAND_LONG whose payload was not logged.
- Live query start intervals P50/P95/P99/max500.000/500.068/500.093/500.099ms. Query response P50/P95/P99/max3.140/4.837/4.884/4.896ms, well inside150ms. Actuator wall intervals max6.078ms, below250ms. First/last ten query means2.60/3.80ms: some rise, no bound approach in this16s observation. No scheduled query accumulation.
- 188 live parse+validation callbacks P50/P95/P99/max0.0967/0.453/0.570/0.625ms. Parser peak1,837chars; recent console cap65,536chars; raw archive267,090bytes under16MiB; no measurement truncation. Phase2D offline2.838µs incremental fixture is a different workload and not equated to live callback latency. Phase2C active-run query interval drift and early stop were absent from this *disarmed* session only.
- Exact raw summary SHA2562dad70007c79d0753bed55eb8d9845f0e30878eb8f470dc0f704fe3245b11407, console SHA890d131c…, ULog SHAf8883269… kept local. Runtime hashes included newly executed measurement module and matched after the run; isolated PX4 binary, physical model, rotor config, historical neural models and original PX4 installation unchanged. Original checkout87 historical untracked files preserved.
- Related Bridge tests: original289 passed before run; RED→GREEN new evidence/identity/clock/cleanup cases. Fresh final related suite321 passed, zero skipped (one existing runtime warning). Independent review found a truncated-query coverage flaw in the offline assessor, reproduced it RED and fixed it GREEN. Final independent re-review:0Critical/0Important/0Minor; scoped commit approved. Known unrelated CAD NumPy/SciPy ABI collection failure was not modified or re-run.
- One worker/no OOM observed; Bridge peak RSS772816896bytes, sampledPX4 RSS9887744bytes, system minimum MemAvailable11145252864bytes, swap0; CPU Bridge1.685s/PX4sampled1.32s. PX4 CPU/RSS peaks between 50Hz samples are unobservable.

Conclusion: PASS for the scoped real disarmed runtime integration. Query timeout did not recur; actual yaw allocation row remains valid. Communication prerequisites for a separately authorized restricted yaw test are met. Final in-flight EKF alignment, yaw response and hover remain NOT TESTED; this result does not authorize or verify Phase3.

Limitations: Single16s run, no active flight path, no long-duration stability or hard-real-time guarantee, no logged payload for one incoming PX4 COMMAND_LONG. Console includes a nonfatal `SIM_GZ_EN` parameter-not-found line and logger capacity warnings. No hardware connection. Raw ULog/console/trace remain local-only.

Next: One separately authorized restricted yaw validation using the new matching disarmed proof, with in-flight EKF and safety gates intact. Not executed in Phase2E.

## 2026-10-12 — PX4–MuJoCo Complete Yaw Closed-Loop Validation Phase2F

Branch: feat/px4-mujoco-yaw-closed-loop-validation
Base: feat/px4-mujoco-disarmed-verification / 10c70d0cfa53234d241d476b79002b675d9422b8

Research Question: Can the pinned PX4 allocator, Bridge and unchanged MuJoCo physical model complete the original restricted +10° yaw response with healthy EKF, safety monitoring and physical-state limits?

Setup: One authorized active SITL run only, using the SHA-verified Phase2E disarmed proof and isolated PX4 binary2f6bbc24…, MuJoCo3.14.0, 7kg model, 500Hz physics/250Hz sensors. Initial NED yaw90°, +10° command at or after32s, target100°, altitude2m, 45s maximum. Original settling condition: true yaw error<2°, absolute FRD body-z rate<0.05rad/s continuously≥2s, adjacent sample gaps≤0.15s. No runtime, PX4 gain, motor, model or safety-threshold changes after proof.

Key Results:
- Sole active run complete45.000s, step32.004s, exit0, zero four-motor controls at shutdown, no retry. True yaw90.0365°→99.9492°, positive9.9127° physical change; 4.2044° overshoot, 0.3227rad/s peak yaw rate, 3.360s strict settling time, final2s mean signed error−0.0259°.
- Actual allocator yaw row ±0.017207708 in78/78 current queries; post-step torque achieved65/65 sampled statuses, unallocated yaw≤4.22e-8 PX4 normalized units, no allocator or motor saturation. Alternating motor differential and model-derived +0.2132/−0.1604N·m reaction yaw torque have physically correct signs; torque is not hardware-measured.
- Post-step PX4 estimated yaw minus MuJoCo truth mean−0.0218°, max absolute0.1045°. Final magnetometer/heading alignment remained healthy; post-step maximum body-origin height error0.00835m, XY excursion0.02724m, tilt0.127°. No sampled flight-boundary violation.
-78 safety queries, max response5.109ms<150ms, max start interval500.072ms, max actuator interval11.561ms<250ms; no 650ms service fault, monitor fault, backlog or unsafe gate continuation. Parser peak1,838chars. Bridge peakRSS773029888bytes, sampledPX4peak9973760bytes, min system available10897215488bytes; one worker, no owned-run OOM observed, swap unchanged.
- Phase2E proof, ULog, console, summary, runtime/PX4/model hashes independently checked; raw evidence and flight ledger local-only. Preflight321 tests passed; final related Bridge suite336 passed, two existing nonfatal NaN-angle warnings. Independent read-only review found and offline RED→GREEN tests closed a truncated-physical-trace assessor gap; actual2,251-sample record covered0.004–45.000s with bounded gaps and exact final state. Three sole-run raw hashes now pinned in offline analysis; runtime was not changed or re-flown.

Conclusion: PASS for the restricted single-run PX4→allocator→four motors→MuJoCo physical +10° yaw control chain, original settling criterion, EKF alignment and sampled safety/time budgets. Control-chain prerequisite for considering Phase3 is met; Phase3 Odometry/ROS integration itself is NOT TESTED. This is not a hardware or generalized robustness result.

Limitations: one nominal positive yaw step; no negative yaw, disturbance, repeatability, real hardware, motor/ESC lag calibration or hard-real-time guarantee. PX4 torque requests are normalized and distinct from model-derived physical Nm. Query sampling may miss brief between-query transients. Whole-repository CAD ABI issue remains unrelated and unmodified.

Next: Only recommend separately authorizing a Phase3 external-odometry interface preflight after review of this evidence. Not executed.

## 2026-10-12 — PX4–MuJoCo External Odometry Interface Preflight Phase 3A

Branch: `feat/px4-mujoco-external-odom-preflight`; base `fb3c82ea3a7fdb8b7becf1b0226a9b0b3a3462b0`.

Research Question: What interface and safety evidence would be required to move MuJoCo truth-derived odometry through this exact PX4 SITL build toward EKF2, before reducing GNSS reliance?

Setup: Source/ULog audit and isolated offline ODOMETRY adapter only. No SITL process, flight, arming, PX4 parameter/binary edit, physics change or model training. Pinned PX4 source `142c24cffb`, binary SHA256 `2f6bbc24…`, MuJoCo 3.14.0. Offline message: LOCAL_NED pose, BODY_FRD twist, full transformed covariance, explicit reset/quality, source-time monotonicity/freshness gate. Existing Bridge runtime remains unchanged.

Key Results: Pinned simulator receiver publishes accepted ODOMETRY to `vehicle_visual_odometry`, EKF2 can consume it, but simulator path replaces MAVLink sample time with PX4 receive time; ordinary MAVLink receiver instead syncs `time_usec`. Phase 2F ULog had `EKF2_EV_CTRL=0`, `EKF2_GPS_CTRL=7`, `EKF2_HGT_REF=1`: no historical EV fusion claim. Existing safety monitor requires GNSS position. Offline MAVLink/frame/covariance/timestamp/reset/source-indicator/float32 range and strict EKF gap tests added; 336 inherited + 20 new = 356 related Bridge tests passed, 0 failed, 2 inherited nonfatal invalid-angle warnings. Zero new SITL runs. No raw flight artifacts changed.

Conclusion: Offline interface preflight **PASS**; live PX4 receipt, actual EKF fusion and GNSS denial **NOT TESTED**. Source-time loss and GNSS-dependent safety gating **BLOCK** immediate latency-aware or GNSS-denied flight claims. Detailed evidence and staged verification design: `docs/PX4_MUJOCO_EXTERNAL_ODOMETRY_PREFLIGHT.md`; machine-readable report: `mujoco/reports/px4_mujoco_external_odometry_preflight.json`.

Independent review: 0 Critical / 0 Important / 1 nonblocking Minor (absolute covariance symmetry tolerance at extremely small scale). Reviewer independently reran 356 related tests and checked pinned PX4/uORB source paths and protected hashes; no SITL launch.

Limitations: Synthetic ground-truth odometry is not FAST-LIO2; quality/covariance not sensor-calibrated; no time-sync or delay verification inside PX4, no live messages or controlled flight in this phase. Offline tests do not establish a real estimator or safety outcome.

Next: Separately authorize a disarmed ODOMETRY receipt/fusion and timestamp-path test with GNSS retained, before any GNSS-denied maneuver. Not executed.

## 2026-10-13 — PX4–MuJoCo BC/PPO Policy Shadow Preflight Phase 3B

Branch: `feat/px4-mujoco-policy-shadow-preflight`; base `63e504eb1835a53a08d975476ba2dc10cc503d5d`.

Research Question: Can the frozen Original BC, yaw-augmented BC and five fixed-protocol PPO checkpoints be identity-verified and mapped from their actual 7D/4D PI-training interface to a non-transmitted PX4-format NED velocity/yaw-rate action?

Setup: Offline-only, one process, one Torch thread. Fixed preselected first 25 historical scripted nominal evaluation traces (3,620 valid observations; all raw trace hashes verified), seven frozen models loaded sequentially. No SITL, Offboard message, new MuJoCo episode, training, parameter change or closed-loop policy test. BC checkpoint Train-only z-score/action de-normalization preserved; PPO uses raw 7D without VecNormalize and deterministic SB3 inference.

Key Results: All seven checkpoint hashes and parameter hashes stable; ENU/NED and yaw-rate sign tests pass; repeated outputs exact and finite. Isolated CPU single-model P95 inference about0.05ms for BC and0.11–0.12ms for PPO, all maxima<0.2ms against the 40ms policy period; peak RSS462MiB, sampled minimum system available around10GiB, no observed OOM. The 961 near-target scripted states show BC mean command speed≈0.04m/s versus PPO≈0.10–0.12m/s, but these are off-policy commands. Synthetic ±30° yaw-error input probes show yaw-rate response slope positive for both BC models and negative for four of five PPO seeds; these probes are not physically replayed trajectories. Nominal replay command saturation at `|a|≥.95` was 0% for all seven models. Per-model details: `mujoco/reports/px4_mujoco_policy_shadow_preflight.json` and `docs/PX4_MUJOCO_POLICY_SHADOW_PREFLIGHT.md`. Separate verification:30 new offline tests and356 existing Bridge tests passed; two inherited Bridge warnings and three package-import warnings in the new suite, no failures.

Conclusion: **PASS** for scoped offline policy inference, identity and interface conversion; PX4 estimator-fed 7D construction, Offboard delivery and closed-loop performance remain **NOT TESTED**. The PI control environment is not equivalent to the PX4 control chain.

Independent review: Two Important findings were fixed (public Shadow gate bypass and local-only replay identity), then final read-only re-review found 0 Critical / 0 Important remaining. Reviewer did not rerun tests.

Limitations: Fixed 25 nominal scripted trajectories only; local raw traces are required to reproduce; synthetic yaw probes change one scalar without physical replay; isolated CPU timing is not a scheduling or hard-real-time guarantee; no SLAM/EV fusion evidence or flight-safety claim. Historical checkpoints and training reports unchanged.

Next: Only recommend a separately authorized disarmed end-to-end estimator-observation and Offboard command-path verification after Phase3A external-odometry fusion prerequisites are met; not executed.

## 2026-10-13 — PX4–MuJoCo GNSS-Retained External Odometry Fusion Phase 3C

Branch: `feat/px4-mujoco-odom-fusion-verification`; base `f202d9d2985312e80cf180a2f239077e3734aa81`.

Research Question: Does simulated MuJoCo COM odometry pass through MAVLink/SimulatorMavlink to real PX4 EKF2 EV position and velocity fusion while GNSS remains enabled?

Setup: One unarmed SITL run, no retry, one MuJoCo/one PX4 process. Pinned isolated PX4 source `142c24cffb`, binary SHA256 `2f6bbc24…`, MuJoCo3.14.0, unchanged7kg model. Reused Phase3A adapter/frame conversions and historical safety loop;50Hz COM truth ODOMETRY,250Hz HIL_SENSOR,10Hz HIL_GPS,500Hz physics. Synthetic covariance diagonal0.01; no EV yaw. Actual ULog params: EV_CTRL5/GPS_CTRL7/HGT_REF1/SDLOG_PROFILE131. Source is SIMULATED GROUND TRUTH ODOMETRY, not FAST-LIO2/SLAM.

Key Results: EKF EV-position985 records (891 accepted) and EV-velocity944 records (944 accepted), final3 samples fused/unrejected with advancing fuse timestamps20,796,000→20,820,000→20,836,000µs. GNSS position/velocity198 records each, final fusion flags and aid timestamps active; local position/velocity valid. Actual ODOMETRY1001/HIL_SENSOR5001/GPS201. Runtime remained unarmed; four motor commands zero on exit, PX4exit0, safety fault null.29 queries: P50/P95/P99/max1.947/4.780/4.842/4.865ms; max start interval500.096ms; max actuator wall gap10.518ms. Bridge peakRSS771653632bytes, PX4 sampled peak9863168bytes, system minimum available10549301248bytes, no observed OOM.

Negative/Incomplete Results: Direct `vehicle_visual_odometry` ULog evidence is BLOCKED: fixed PX4 logger reached its255-subscription capacity and explicitly failed to add the topic twice. Aid-source records prove actual EV fusion but cannot replace the missing direct receiver publication/frame/timestamp trace. The sole run advanced20.004s, violating the20s cap by one4ms cycle: accumulated floating `d.time` remained19.99999999999979 after5000cycles. Raw files remain immutable. The new Phase3C runner stop comparison was fixed offline with a1µs guard; RED→GREEN verifies5000cycles. Final modified runner SITL validation is NOT TESTED; no extra SITL occurred.

Verification:356 inherited+15 new=371 Bridge tests pass, two inherited invalid-angle warnings;30 Phase3B Shadow tests pass in preflight with its package warnings. Suite peakRSS1080576KiB. Independent pre-run review findings (proof-kind reuse, stale/future/empty state evidence) were fixed before GO. Independent result review confirmed EV/GNSS fusion, logger evidence gap and stop-boundary error. Protected runtime/checkpoint/model/PX4 hashes recorded; original checkout retains87 historical untracked files. Historical runtime and experiment reports were not rewritten.

Conclusion: EV/GNSS actual aid-source fusion PASS; complete three-layer interface acceptance BLOCKED; strict duration compliance FAIL. GNSS-denied, FAST-LIO2 timing and learned-policy PX4 flight remain NOT TESTED/BLOCKED. Reports: `mujoco/reports/px4_mujoco_odometry_fusion.json`, `docs/PX4_MUJOCO_ODOMETRY_FUSION.md`.

Limitations: Single static unarmed state; synthetic noise/quality; PX4 receiver replaces source capture time with receive time; direct visual topic not recorded; one post-run offline boundary fix not revalidated live. Static local-origin offset is reported separately from displacement/velocity error; no moving-state accuracy, packet-loss or hardware claim.

Next: Only recommend a separately authorized unarmed verification with an explicit bounded ULog topic list and the offline-fixed20s boundary, keeping GNSS enabled. Not executed.

## 2026-10-13 — PX4–MuJoCo External Odometry Evidence Reverification Phase 3C-R

Branch: `feat/px4-mujoco-odom-fusion-reverification`; base `f09264edaffb25c4e59e33cb8be4c01326fb2c33`.

Research Question: Can a finite Logger configuration and the offline-fixed stop boundary supply direct visual-odometry receipt and actual EV/GNSS fusion evidence in one strictly <=20-second unarmed run, with the final operational code unchanged?

Setup: Exactly one new disarmed SITL attempt, no retry/ARM/Offboard. Same approved isolated PX4 binary SHA256 `2f6bbc24…`, source `142c24cffb`, MuJoCo3.14.0 and unchanged7kg dynamics. Reused500Hz physics/250Hz sensors/50Hz COM truth ODOMETRY/10Hz GNSS, EV_CTRL5/GPS_CTRL7/HGT_REF1, no EV yaw. Verified current Logger API: profile0 plus private runtime logger_topics.txt,19 explicit instance0 topics, mission0; binary and startup files unchanged. Actual ULog has20 instances including separate built-in event. Frozen protocol SHA256 `0164d294df0d15d043c749b62172af49f5906b1d041bea2525d2b5d9825bb47b`. Exclusive attempt record prevents reuse; summary explicitly not a flight smoke proof.

Key Results: Actual physical time19.999999999999794s; counts5000 HIL_SENSOR/200 HIL_GPS/1000 ODOMETRY exactly. Direct `vehicle_visual_odometry`985 records,20ms intervals, final receipt40ms before endpoint; all mandatory fields finite, NED pose/BODY_FRD velocity, valid quaternions/covariance.985 exact logical-clock send/receive pairs, maximum position discrepancy3.51e-9m. EV position984 records (889 accepted), EV velocity942 (all accepted); final3 accepted/unrejected advancing fuse times20,756,000→20,780,000→20,796,000µs. GNSS sources198 each with current final fusion; five early GNSS-position rejected innovations disclosed and retained. Local position/velocity valid. All A–L verdicts PASS; no old evidence substituted, no raw duration truncation.

Safety/Resources:28 healthy queries, P50/P95/P99/max2.116/5.032/5.114/5.138ms; max query interval500.076ms; max actuator wall gap12.998ms.150/250/650ms safety deadlines unchanged. No queue beyond1, parser peak1836chars, no fault. DISARMED and no-Offboard flags, zero physical motor controls throughout and at stop, PX4exit0. One worker; Bridge738.00MiB peak RSS, PX4 sampled9.52MiB, sampled minimum available8.95GiB; no observed OOM, swap unchanged. Original87 untracked files and protected models/raw evidence remain unchanged.

Verification:371 inherited+66 new=437 Bridge tests PASS and30 Shadow tests PASS, inherited warnings only. Bridge suite peak RSS1,146,308KiB. RED→GREEN for finite logger, invalid evidence, exact stop counts, corruption/truncation/hash checks and passive-control corroboration. Independent pre-run review GO after evidence-integrity and final-model identity fixes. Final independent read-only source/raw-evidence/report review GO,0 Critical/Important;65 targeted tests passed independently,1 physical-schedule test excluded, no additional simulation. Minor disarmed-scope wording corrected. Operational source hashes exactly match the sole SITL run; post-run changes are analysis/tests/docs only.

Conclusion: **PASS** for complete GNSS-retained, simulated-truth external-odometry receipt/fusion and corrected final-runtime/20-second evidence. This does not validate GNSS-denied positioning, moving-state odometry, FAST-LIO2, or BC/PPO PX4 control. Previous Phase3C report/raw logs remain unchanged.

Limitations: Single static/settling disarmed run, synthetic covariance/quality,985 retained logs versus1000 sends is not packet-loss measurement. PX4 overwrites source timestamp; exact lockstep-clock correspondence does not measure host physical transport delay or establish FAST-LIO2 time sync. Static local-origin offset retained; no flight authorization proof. Raw ULog/source arrays/private rootfs stay local-only; JSON,10 plots, code/tests/docs are versioned.

Next: Only recommend a separately authorized disarmed GNSS-denied fusion-source verification before any external-odometry flight. Not executed.

## 2026-10-13 — PX4–MuJoCo GNSS-Denied Disarmed EKF2 Fusion Phase 3D

Branch: `feat/px4-mujoco-gnss-denied-disarmed`; base `8468dfac1137fb7f79ded1718c24eaace09df835`.

Research Question: Can cold-start PX4 maintain valid local position/velocity using simulated IMU, magnetometer, barometer and external odometry with no GNSS input from startup?

Setup: Exactly one unarmed SITL attempt; no retry/ARM/Offboard. Approved isolated PX4 binary SHA256 `2f6bbc24…`, source `142c24cffb`, MuJoCo3.14.0 and unchanged7kg model.500Hz physics/250Hz sensors/50Hz COM truth ODOMETRY; HIL_GPS entirely suppressed with an outgoing sensor/odometry-only whitelist. Actual ULog EV_CTRL5/GPS_CTRL0/HGT_REF0/BARO_CTRL1/MAG_TYPE0; no EVyaw. Private27-topic logger/profile0,21 actual topic instances including event; no overflow/dropout/corruption. Frozen protocol `4ae0b280ca9d98ead2cf9da7e9b0536b3068ed7cbeab041ac6807819e45f2296`. Independent disarmed mode preserves original flight gates and150/250/650ms watchdogs; fixed campaign lock prevents alternate-filename retries and no new flight proof is produced.

Key Results: Physical endpoint19.999999999999794s, exactly5000 HIL_SENSOR/0 HIL_GPS/1000 ODOMETRY.984 direct visual-odometry records,20ms maximum gap,40ms terminal age;984 exact logical-clock pairs, maximum position discrepancy3.17e-9m. EVposition984 records/890 accepted, EVvelocity942/all accepted; final3 fusion timestamps20,756,000→20,780,000→20,796,000µs. GNSS pos/vel/height/yaw flags false throughout; all GNSS aids and sensor GPS records absent.975 local records show no global origin; xy/z/vxy/vz valid from first recorded1.1s and at endpoint. Barometer357/337 accepted and magnetometer270/245 accepted with final3 current accepted advancing fusion samples. A–N acceptance PASS.

Negative/Boundary Findings:25 early magnetic innovation rejections at publication0.244–2.084s retained; ground magnetic heading later fuses correctly. Final heading_good_for_control and mag_aligned_in_flight remain false; inflight readiness NOT TESTED. Local position RMSE0.01139m and velocity0.001417m/s describe a static/settling run with retained origin offset, not flight accuracy. True source latency, FAST-LIO2, policy control and GNSS-denied flight remain NOT TESTED.

Safety/Resources:28 queries; P50/P95/P99/max1.965/5.019/5.136/5.154ms; maximum query interval500.065ms, actuator wall gap10.790ms, parser peak1838chars. No fault, no instrumentation drops, no queue beyond1. Zero physical and recorded motor outputs, DISARMED, PX4exit0. Single worker; Bridge735.58MiB peakRSS, PX4 sampled9.38MiB, system minimum8.71GiB available, no observed OOM. Host20.00391s is separate from physical20s maximum.

Verification: Pre-run510 Bridge tests (437 inherited+73 new) and30 Shadow tests PASS; final514 Bridge (437+77) and30 Shadow PASS, inherited warnings only. RED→GREEN covered missing GPS-counter evidence, alternate protocol filename attempts, zero GPS metadata and incomplete safety evidence. Independent pre-review GO after fixes; final independent source/raw/report review GO,0 Critical/Important,77 package tests independently passed and all A–N results recomputed. Reviewer did not independently rerun full514+30 or certify87-file inventory. Operational hashes unchanged after run; all old runtime sources, binary, dynamics,8 checkpoints and old raw evidence preserved. Original87 untracked files preserved; raw run artifacts local-only.

Conclusion: Cold-start GNSS-denied **disarmed local fusion PASS**. This supplies estimator interface evidence, not permission or readiness for flight.

Limitations: One static ground run; ideal/synthetic truth/covariance; PX4 replaces source capture timestamps; no moving-state, delay/dropout or aircraft-control validation. No controller/model retraining or tuning.

Next: Only recommend reviewing a separately authorized GNSS-denied flight-safety and estimator-readiness protocol before any flight. Not executed.

## 2026-10-13 — Real MID360S Interface Audit and MuJoCo LiDAR Design Phase 3E-L1

Branch: `feat/mujoco-mid360s-interface-audit`; base `ebbe6ed2c69ef00d0bb37466f5d8278c9f2dd652`.

Research Question: Which MID360S network, ROS, timing, extrinsic, FAST-LIO2 and simulation interfaces are actually evidenced by the fixed real-project source, and what can be safely reused for future MuJoCo-only LiDAR work?

Setup: Read-only fixed `-frontier-` commit `db897c4e3a1607f5852b484afd0a01523cb08c6b`; fixed upstream reference snapshots for `livox_ros_driver2` (`21445540`), FAST_LIO (`7cc4175d`) and MuJoCo-LiDAR (`e6b879df`). Current MuJoCo model retained SHA256 `cde862f4…`. No PX4/SITL/flight/ROS run, LiDAR scan, or training; no historical model, sensor, dynamics, or controller changes.

Key Results: Historical MID360S type35 and NUC/radar `192.168.1.10`/`192.168.1.157` with `/livox/lidar` ~10Hz and `/livox/imu` ~200Hz. Pinned patch explicitly configures host command/push/point receive UDP ports 56101/56201/56301. Reference ROS1 driver uses `livox_ros_driver2/CustomMsg` (XYZ, reflectivity, tag, line, per-point offset) and `sensor_msgs/Imu`; reference FAST-LIO uses `livox_ros_driver/CustomMsg`, so deployed subscriber compatibility is not proven. The old Gazebo mount pose is not a real calibrated extrinsic. Current MuJoCo `lidar_visual` is only a mesh, not a sensor. Newly found open-source MuJoCo-LiDAR already provides `mid360` angle data and native raycasting; it is a useful reuse candidate, **not** a validated MID360S or FAST-LIO2 emulator. Its ROS1 example emits ordinary PointCloud2 with synthetic constant intensity.

Conclusion: Source/interface audit and design PASS; implementation/physics fidelity/FAST-LIO2/PX4 integration NOT TESTED. The next implementation should evaluate the pinned MuJoCo-LiDAR CPU backend first, then separately validate MID360S scan/timing/IMU/message compatibility and measured installation extrinsics before any localization claim.

Verification/Review: Six new stdlib-only offline tests PASS; original MuJoCo XML SHA unchanged. Independent read-only review GO after narrowing the real-extrinsic evidence claim and clarifying transform directions/tag-line gates; no remaining Critical/Important. Existing MuJoCo-dependent Bridge tests could not be fully run here because Python MuJoCo is not installed; combined invocation showed9 dependency failures/9 passes, not a code-regression claim.

Limitations: No live NUC, installed driver/FAST-LIO2 build identity, verifiable real calibration, wire capture or actual LiDAR scan; historical frequencies are not current-state evidence. The apparent CustomMsg package mismatch is unresolved. No Gazebo dependency introduced.

## 2026-10-13 — MuJoCo MID360-like Point Cloud Integration Phase 3E-L2

Branch: `feat/mujoco-mid360s-pointcloud-integration`; base `5ffd67922eab7e1549ee87bc94184e37b8194700`.

Research Question: Can fixed open-source MuJoCo-LiDAR CPU raycasting produce verifiable 3D point clouds from an isolated UAV scene without touching historical dynamics, and what are the actual local scan costs?

Setup: Reused historical real MuJoCo3.14.0 Python venv, pinned MIT `discoverse-dev/MuJoCo-LiDAR` commit `e6b879df` and exact `mid360.npy` SHA256 `9fa01655…`. In-memory MJCF copy adds a SIM_CONFIG LiDAR site at body FLU `(0.10,0,0.10)m`, walls and box; 7kg original model SHA256 `cde862f4…` unchanged. Single frozen pose per frame, 10Hz simulation timestamp spacing, CPU backend;12 frames each at1k/4k/24k rays plus one rotated frame. No PX4, ROS, flight, training or hardware communication.

Key Results: Actual MuJoCo ray clouds PASS, full24k frame gives15,650–17,068 valid points across repeated pattern phases; vertical extent2.58m. Front/side/ground known-distance max error1.19e-8m, box error1.58e-9m; no-return masked, self-body excluded without removing exterior wall. Yaw+90° range error1.49e-9m; local/world roundtrip2.22e-16m. Full24k static frame CPU total P50/P95/P99/max5.906/6.545/6.613/6.630ms, separately trace P50/P953.435/3.972ms; within this static 10Hz scan-only budget. Explicitly downsampled1k and4k are reported separately. NPZ and four actual-ray evidence figures produced; raw NPZ local-only.

Verification/Resources:31 new real-MuJoCo integration tests PASS; existing Bridge + L1 audit suite520 PASS (one inherited Matplotlib Axes3D warning). One worker, peakRSS845.5MiB, no observed OOM. Historical physical model hash and data retained. Independent read-only review GO,0 Critical/Important; reviewer independently ran the 30 tests present before the final additional package-identity test and matched hashes/report. Minor future hardening: extend upstream source integrity coverage; fixed checkout currently clean.

Conclusion: CPU MuJoCo geometry-to-point-cloud interface **PASS** for static isolated scenes and future PPO observation prototyping. MID360S physical fidelity, true per-point timing, motion distortion, synchronized IMU, Livox CustomMsg, FAST-LIO2, ROS and PX4/SITL **NOT TESTED**; no localization or flying claim follows.

Limitations: Small static scene and12 scan samples per scale; no complex mine, moving targets, full simulation+policy timing or real device calibration. `mid360` is an upstream approximate mode, not validated MID360S truth. Plotting/saving time is separate from scan timing.

Next: Only recommend a separately authorized dynamic-scan/IMU timing and measured-extrinsic validation before any FAST-LIO2 claim. Not executed.

## 2026-10-14 — MuJoCo MID360-like Dynamic Scanning Phase 3E-L3

Branch: `feat/mujoco-mid360s-dynamic-scanning`; base `afd7d0fda3431dedcf180f49e59f0a01dbd940de`.

Research Question: Can the fixed MuJoCo-LiDAR CPU scanner follow deterministic moving UAV poses with 10Hz simulation-time samples, numerically correct world/local geometry and bounded local CPU cost, without suggesting physics-consistent IMU or FAST-LIO2 performance?

Setup: MuJoCo3.14.0, fixed MIT upstream commit `e6b879df` and `mid360.npy` SHA `9fa01655…`; unchanged 7kg historical MJCF SHA `cde862f4…`. Six pre-registered one-second KINEMATIC_POSE_REPLAY paths (X, Y, Z up/down, yaw, combined, uniform shaft). Pose prescribed every 500Hz MuJoCo tick; full 24k rays at 10Hz; 1k/4k/24k additional matched CPU benchmark. Raw 66 trajectory frames local-only, six actual-ray figures tracked. No PX4/ROS/FAST-LIO2/training/flight.

Key Results: 99 real CPU pattern scans in total. Frame times0..1s at0.1s intervals (maximum period error8.33e-17s), all trajectories end after500 actual `mj_step` calls. Measured X/Y wall range maximum errors1.31e-8/1.19e-8m; static world wall point error max7.45e-8m. Uniform shaft front/side horizontal rays invariant with Z over1m motion (range spread0m), a limited geometric observation, not a FAST-LIO2 outcome. Full24k combined-path scan P50/P95/P99/max6.150/6.919/7.012/7.036ms;1k and4k timings reported separately. Peak RSS1193.4MiB, no observed OOM.

Verification: Related suite **575 PASS** (551 inherited, 24 new), one inherited Matplotlib Axes3D import warning; all six figures are 2D. RED→GREEN verified an evidence guard refusing to overwrite existing formal JSON or nonempty figure directories before creating raw output. All 66 local raw NPZ were independently rehashed against their report entries; ray counts, valid masks, timestamps and sensor poses matched. Upstream mode and original physical-model hashes unchanged. Independent read-only final review **GO**, 0 Critical/Important; reviewer also independently reran all 575 tests, checked raw rays and recomputed wall geometry against the JSON. The post-scan guard changes output-file preflight only, not raycasting, trajectory, physical timing or metric computation.

Conclusion: Dynamic-pose MuJoCo CPU raycasting and 10Hz simulation clock PASS for the fixed isolated replay. This is not closed-loop flight or a physics-consistent sensor simulation. Existing 250Hz SITL SensorAdapter depends on qacc/qvel and cannot be treated as an IMU sampled from directly overwritten qpos/qvel; no IMU data were generated. LiDAR per-point timing, motion distortion, measured LiDAR–IMU extrinsic, FAST-LIO2 and PX4 NOT TESTED.

Limitations: Small static geometry and11 frames/path; horizontal-only shaft degeneracy does not encompass floor/oblique rays or estimator observability. Scan-only CPU timing excludes data save/plot and full PX4 workload. `mid360` approximate mode not validated as real MID360S.

Next: Only recommend a separately authorized physically consistent moving-state LiDAR+IMU timing study before FAST-LIO2 integration. Not executed.

## 2026-10-14 — MuJoCo LiDAR Policy Observation Preflight Phase 3E-L4

Branch: `feat/mujoco-lidar-policy-observation`; base `d95552c86c87c4e749a238752082d5ffe00a3e0c`.

Research Question: Can the 66 real MuJoCo ray frames from Phase3E-L3 become finite fixed-size LiDAR inputs, and can 10Hz source observations be read causally by a future 25Hz policy without mistaking old or future frames for new data?

Setup: Read-only SHA256 verification of all 66 local 24k-ray NPZ from the pinned upstream `mid360` approximate mode; the formal analysis used no new raycast, while a separate Roll/Pitch regression test used two fresh 1k-ray MuJoCo scans. No PX4, SITL, ROS, FAST-LIO2 or training. Fixed 24-azimuth ×12-elevation grid over `[0,360)° × [-10,55]°`, chosen to enclose measured `[-7.2075,52.1640]°` elevations; three finite float32 channels: range, sampled mask, hit mask. Offline 10Hz source/25Hz policy schedules, bounded four-frame cache and 150ms *experimental* stale threshold. SIM truth, PX4 estimate and target remain separate typed inputs.

Key Results: 66/66 NPZ hashes matched the prior report; 1,584,000 source rays encoded. Network tensor `[3,12,24]` is 3456 bytes; all per-observation arrays 7488 bytes, maximum four-frame cache 29,952 bytes. Across 66 frames, 438 unscanned-bin instances and 4322 sampled-but-no-hit instances remain distinguishable; four angular bins never sampled. In 156 zero-delay offline policy reads: zero future/stale/missing, 90 repeated-but-not-new valid reads. With explicitly artificial +20ms arrival delay: six initial missing reads, zero future. Data source timestamps remain unchanged. Final-code encoding CPU P50/P95/P99/max `3.483/4.032/4.293/4.375ms`; NPZ load `5.848/7.141/8.340/10.228ms`; isolated cache-read P95 `0.00170ms`. Peak RSS `563.8MiB`, no observed OOM.

Verification/Review: Related suite **603 PASS, 0 FAIL, 0 SKIP** (575 inherited, 28 new), one inherited Matplotlib Axes3D import warning. Independent final read-only review **GO**, 0 Critical/Important. RED→GREEN caught backdated arrival, re-enableable array mutation and the checked-file/reopened-file identity gap; final report was rerun with verified-byte decoding and immutable snapshots. All 66 raw NPZ were hashed against the prior manifest; source pattern and historical physical-model SHA256 unchanged. Six figures are from actual encoded ray frames and measured policy-source times, not analytic point replacement. Cryptographic identity applies to the verified file path, not arbitrary direct `encode_frame` labels.

Conclusion: Offline fixed-size LiDAR observation and causal asynchronous handoff **PASS**. This does not modify the legacy 7D BC/PPO checkpoints, establish PX4-estimated observations, run a learned policy, or validate a real MID360S.

Limitations: Whole-frame frozen-pose rays with no physical per-point timing/IMU sync; fixed small static scenes and only 66 frames; artificial delay is not measured transport latency; no combined scan+encoder+policy+PX4 real-time test. `SIM_CONFIG` extrinsic is not real calibration.

Next: Only recommend a separately authorized physically consistent LiDAR+IMU timing and estimator-state adaptation audit before any LiDAR-policy training. Not executed.

## 2026-10-14 — LiDAR Navigation Environment and Reward Preflight Phase 3E-L5

Branch: `feat/mujoco-lidar-navigation-env`; base `371db05b8e8bcf6db3cbe7fd2c0c7c7a34c2b603`.

Research Question: Can unchanged 7kg MuJoCo/PPO-PI dynamics, 24k-ray approximate MID360 scanning, fixed `[3,12,24]` encoding and causal 10Hz→25Hz handoff form a genuine-contact Gymnasium obstacle-navigation environment without training a policy?

Setup: Five isolated in-memory layouts (`empty`, front box, side pair, detour, channel); disjoint five-layout Train/Validation/Test seed/geometry identities; original MJCF SHA256 `cde862f4…` and pinned `mid360.npy` SHA256 `9fa01655…` unchanged. Existing 500Hz MuJoCo/100Hz PI/25Hz action stack retained, 10Hz CPU LiDAR; Dict observation has LiDAR, original 7D **simulated ground-truth** state and LiDAR age. Candidate Reward V2 extension includes collision/current-step-progress suppression, −0.01 per executed step and −2 timeout; no model training or systematic reward search. Three sequential Train-layout episodes use obstacle-blind state-only Scripted for interface proof, not a LiDAR policy.

Results: Empty and channel reach/stabilize successfully at3.64s; front-box episode contacts the real obstacle at1.696s and terminates collision, with no success/progress credit on that step. The initial `−10` collision extension exposed an undesirable collision (`−5.31`) versus stationary timeout (`−5.75`) return ordering on Train-layout preflight; before any training/Test feedback the frozen candidate was corrected to `−30` for collision/physical failure. For all fixed manifest initial distances <2m, even a best-case 20-point prior progress credit then failure scores at most `−10.01`, below hover timeout `−5.75`. Final front-box return `−25.3092`. Total225 policy steps/4488 actual physics ticks/898 PI updates/91 full24k-ray LiDAR frames. Full Gym step wall-time P50/P95/P99/max `1.090/12.275/12.484/12.531ms`; scan `6.628/7.021/7.069/7.168ms`, encode `4.167/4.374/4.417/4.432ms`; 1 worker, peak RSS1172.5MiB, minimum MemAvailable8769.4MiB, no observed OOM. No PX4/SITL/FAST-LIO2/training.

Verification: Historical related suite603 PASS; final related suite627 PASS (24 new), old physics/controller/reward/PPO-PI targeted25 PASS; Gymnasium `check_env` PASS. New RED→GREEN tests covered wall/box contact, collision-over-tilt priority, causal sensor stop, terminal classification, fixed-layout unsafe-return upper bound, obstacle target refusal, identical seed/step info, controller reset, scene split and historical-model identity. Runner now derives contact/overall acceptance from actual episode evidence; empty-only runs cannot claim contact PASS. Inherited Matplotlib Axes3D and direct-instantiation render-check warnings do not affect 2D figures or physics. Independent final read-only review GO,0 Critical/Important; reviewer independently checked raw SHA, reward sums, 225/4488/898/91 counts, causal timestamps and performance stats, but did not rerun full627+25 tests.

Conclusion: **PASS** as a physically advanced LiDAR navigation environment preflight, not as a trained or deployable collision-avoidance policy. The old central-body contact box is a `TEMPORARY_PLACEHOLDER`, approximate `mid360` rays have no true per-point timing/IMU model, and the 7D state is simulator truth rather than PX4 EKF output. Default SB3 `MultiInputPolicy` feature extraction is not validated for this float32 LiDAR tensor; PPO optimizer and policy success NOT TESTED. Raw NPZ local-only; JSON, seven measured figures and [method document](MUJOCO_LIDAR_NAVIGATION_ENV.md) versioned.

Next: Only recommend a separately authorized LiDAR+PPO training/evaluation protocol with an explicit custom feature extractor, state-provenance transition and scene-split discipline. Not executed.

## 2026-10-14 — LiDAR-CNN-PPO Training Preflight Phase 3E-L6

Branch: `feat/mujoco-lidar-ppo-training-preflight`; base `5205cf568dd2fab107436096d1c6b1a424260111`.

Research Question: Can a heading-aware `[3,12,24]` LiDAR + 7D state + age Dict observation pass through a custom SB3 PPO CNN/actor/critic and a strictly bounded, single-seed training update without changing historical plant, Reward or checkpoints?

Setup: Original 500/100/25/10Hz MuJoCo/PI/policy/LiDAR stack, full 24k CPU rays, `NAV_V2_COLLISION_PREFLIGHT_V1`, new explicit current-yaw sin/cos from SIM ground truth. Independent Train/Validation/sealed Test 12/6/6 seeded scene identities with varied start/goal/yaw/obstacles and nominal-height geometric path check. One pre-registered Train `front_box` scenario only, one worker, seed0, four 128-step PPO rollouts = 512 timesteps, no formal training or Test evaluation. `MultiInputPolicy` with float32 CNN and no image /255 normalization; 170,857 parameters. All source/model/protocol/split hashes saved before optimizer work.

Results: Sole smoke PASS for training mechanics: 512 environment steps, 10,240 physics ticks; actor/critic head hashes changed; finite policy weights, gradients and available train diagnostics; checkpoint reload deterministic action exact. One time-limit episode is not a navigation success claim. Wall2.913s, CPU2.901s, 175.77 steps/s; complete env step P99 11.833ms; full scan P99 6.659ms, encoder P99 4.357ms; RSS peak1325.7MiB, minimum MemAvailable9982.6MiB, no observed OOM. Seven historical BC/PPO checkpoint hashes and original MJCF hash unchanged. The system cv2/NumPy ABI warning remained nonfatal but requires environment cleanup before formal training.

Reward/Limitations: Historical L5 raw traces rehashed; they alone were not same-scene. After the sole training smoke, three one-shot, nontraining, fixed-`front_box` physical episodes with identical qpos/qvel/target identity SHA (not a full controller-state snapshot) produced zero-command timeout −5.750, obstacle-blind direct collision −25.309 and **privileged geometry-waypoint** detour success +22.835. The same-scene `C>A>B` ordering is descriptive for these trajectories only, not evidence of LiDAR-policy learning or global Reward alignment. These post-smoke results did not change Reward, training configuration or checkpoint choice. Only one Train scene sampled for smoke; full 12-scene formal scheduler, Test generalization, learned avoidance, PX4/FAST-LIO2/real-sensor operation are NOT TESTED. The 3-seed/1M-step formal protocol is documented only, not run; direct formal-training readiness remains BLOCKED until scene scheduling and remaining validation machinery are completed.

Verification/Review: Relevant suite 667 PASS (627 inherited, 25 old controller, 15 new), 0 failures, 2 inherited warnings. Independent read-only pre-smoke review GO after correcting cross-reset sim-time accounting, memory fail-closed and NaN diagnostic handling; no Critical/Important remaining. Final evidence generator checked prereg, smoke checkpoint, original L5 raw, the three new paired raw traces and frozen runtime source hashes. Raw smoke checkpoint/summary and Reward trajectories local-only. No PX4, SITL, training beyond 512 steps or historical model changes.

Next: Only recommend a separately authorized formal-training implementation that first verifies the 12-scene scheduler, interrupted-run recovery and sealed evaluation; this stage stops here.

## 2026-10-14 — LiDAR PPO Multi-Scenario Formal Readiness Phase 3E-L7

Branch: `feat/mujoco-lidar-ppo-formal-readiness`; base `3658b9f230fd3b461976ae0f1abad60bf204bd72`.

Research Question: Can the frozen L6 12/6/6 scenario split be scheduled and validated with identity-checked checkpointing, without any new PPO optimizer update or Test episode, and is the 3-seed/1M-step formal protocol ready to run?

Setup: L6 prereg file, manifest and 12 runtime source SHA256 identities revalidated. One active Train MuJoCo scene at a time, independently seeded uniform scene draw on episode reset, deterministic all-12 coverage test and separate six-scene Validation evaluator. Original 500/100/25/10Hz physics/PI/policy/LiDAR, full 24k rays, Reward `NAV_V2_COLLISION_PREFLIGHT_V1`, CNN and PPO hyperparameters unchanged. A random-initialized untrained model exercised the complete Validation interface; a local-only untrained ZIP/sidecar exercised save/load and WARM_START classification. No optimizer update, Test episode, PX4 or real device use.

Results: 12/12 Train scenes initialized and stepped in each of two cycles (24 coverage resets, 24 one-step physics probes); one additional Train reset prepared the untrained model, for 25 total scheduler visits. Registered geometry, start/goal/yaw, qvel, actuator/control, PI integral and LiDAR cache/time reset identities PASS. Fixed-seed 1200 uniform-sampler draws counted 79–120 per scene (descriptive χ²(11)=19.5). Six full Validation episodes using the **untrained** policy yielded 0 success, 0 collision, six 15s timeouts; this is interface evidence, not trained navigation performance. Evaluator left policy, optimizer, train timesteps and scene RNG unchanged. Checkpoint archive SHA256 `5686a020…` contains optimizer state, but no MjData/PI/LiDAR/partial-rollout state: exact resume NOT TESTED; interrupted formal runs require restart from original seed. L7 candidate protocol SHA256 `7c0c0320…` explicitly revises L6's unimplemented scheduler and adds restart semantics, retaining all other budget/hyperparameters and final-checkpoint selection.

Resources: Bounded probe wall23.13s; Train scene reset/compile P50/P99 310.45/384.56ms, one policy step P50/P99 1.24/1.29ms, initial full-ray scan P50/P99 6.52/6.98ms and encoding 3.94/4.29ms; full six-scene Validation wall14.08s. One CPU worker, peak RSS1756.4MiB, minimum MemAvailable9556.1MiB, no observed OOM. Current RSS rose about48MiB between first and second 12-scene passes, so long-run leakage is unresolved. The pre-existing system OpenCV/NumPy ABI import warning remains; no environment installation or swap change.

Verification/Review: 667 inherited related tests + 12 L7 tests = **679 PASS**, 0 fail, 2 inherited warnings. RED→GREEN covered malformed frozen identity, missing scheduler/evaluator/checkpoint/report contracts, source-byte tamper and incomplete Validation aggregate. Independent read-only review found and prompted fixes for two Important integrity gaps (nested optimizer-state comparison and actual L6 source-byte verification) before the bounded probe; final review status is recorded in the L7 report. All historical BC/PPO checkpoints unchanged; raw checkpoint/probe local-only.

Conclusion: **FORMAL_TRAINING_READY = NO.** Bounded scheduler, full six-scene Validation and checkpoint identity are verified; periodic 100k-step checkpoint/Validation orchestration is not connected to a formal-run driver, and million-step resource stability is untested. Next step is a separately authorized offline periodic-orchestration and interruption-contract verification. No formal training was started.

## 2026-10-14 — LiDAR PPO Orchestration and Bounded Pilot Phase 3E-L8

Branch: `feat/mujoco-lidar-ppo-training-orchestration`; base `c4e1166ac2e01f4747201442e07d4fe079261ce3`.

Question: Can the frozen 12-Train/6-Validation PPO protocol execute rollout-aligned, post-optimizer checkpoint and diagnostic Validation events, and complete exactly one 8192-step multi-scenario Pilot without Test access or historical-model mutation?

Protocol: Original L6 1M-step nominal budget and all PPO/CNN/Reward/physics settings retained. Explicit L8 execution-protocol SHA256 `fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6` records first full128-step boundary at/after each nominal100k threshold: first100096, second200064, final1000064. Events occur only after the associated SB3 optimizer update; final-checkpoint selection and interrupted-run `RESTART_REQUIRED` remain. The formal 1M-step run was **not** started.

Real Pilot: One seed0, one worker, 8192 PPO environment steps, 64 rollouts, 128 epoch updates, 256 optimizer steps and 163440 MuJoCo physics ticks. All12 Train scenes visited in47 resets; 46 finished Train episodes: 42 collisions, 4 timeouts, 0 successes. Actor/Critic parameters changed; gradients finite. The independent six-scene deterministic Validation interface completed with 6 collisions/0 successes, no Train optimizer/RNG/weight mutation; Test episodes0. These are mechanics and interface results, **not** evidence of a learned navigation policy. Pilot ZIP SHA256 `c0a11c85e038a312968bf806b040dd64d1007a3e8b8204e2a8d78b237de7aa20`, stored local-only with raw events.

Resources: Pilot wall67.672s including checkpoint/Validation; environment step P99 12.157ms, 24k-ray scan P99 6.940ms, encoder P99 4.502ms, PPO train-update P99 21.888ms. Peak RSS1728.770MiB, minimum MemAvailable9245.742MiB, no observed OOM. This short run does not prove million-step memory stability. System OpenCV/NumPy ABI import warning remains nonfatal in the tested path.

Verification/Review: Historical suite679 PASS; final suite **698 PASS, 0 FAIL, 0 SKIP, 2 inherited warnings**. Independent read-only review **GO, 0 Critical/Important**. RED→GREEN review repair makes readiness fail closed if tests fail/skip, required counts are missing, or read-only review is not GO. Pilot raw hashes, source identity, checkpoint sidecar, 12 Train identities and six Validation identities verified. Raw per-Train-episode trajectories were not separately retained, so summary reward/contact counts cannot be independently recomputed from a second trajectory archive. One Pilot attempt only; checkpoint and full raw evidence local-only. No PX4, FAST-LIO2, Test rollout or historical checkpoint changes.

Conclusion: Training **driver** and single-seed **orchestration** may be marked ready only after final suite and review pass; `LONG_RUN_MEMORY_PROVEN=NO`, learned navigation performance `NOT TESTED`. Next step, only with separate authorization, is the frozen formal seed0 run under continued memory monitoring. Not executed here.

## 2026-10-14 — LiDAR CNN PPO Formal Seed0 Phase 3E-L9 (Interrupted)

Branch: `feat/mujoco-lidar-ppo-formal-seed0`; base `53bb653a428f28467863f1e625f493fb643f4edf`.

Protocol: One authorized CPU worker and seed0 formal attempt, intended actual budget1,000,064; unchanged L8 execution SHA256 `fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6`, 12 frozen Train/6 independent Validation/6 sealed Test, original24k-ray CNN/Reward/500-100-25-10Hz plant. Added explicit formal CLI and passive episode/resource telemetry; no Pilot/Smoke initialization. Preflight714 tests and independent read-only GO preceded the sole launch.

Result: **FAIL, incomplete; no retry or continuation.** Process exited1 with `RuntimeError: invalid or nonexclusive training episode outcome`. Resources persisted through119,552 environmental steps; exact final `model.num_timesteps` and failing transition info were not recorded. The last complete Train episode ended119,370;706 saved terminal episodes contain656 collision,50 timeout,0 success (the failing transition is excluded). One nominal100,000→actual100,096 post-optimizer checkpoint/Validation event survived:782 rollouts,1564 epoch updates,3128 Adam steps, valid reload and finite weights/optimizer. ZIP SHA256 `d71927d83c3f844f035a3059c1329b0df22773a58ad6b258fa1093c2377abff9`. Independent Validation0/6 success,6 collision; Train RNG unchanged. Test episodes0. No final checkpoint, summary or complete marker exists; `.attempt`, raw evidence and checkpoint remain local-only under the fixed run_01 directory.

Confirmed defect: `navigation_env` maps inherited physical causes to the `physical_failure` outcome/flag but retains raw `termination_reason`; frozen L8 TrainingMetricsEnv rejects raw `z_below_zero/outside_flight_area/excessive_tilt/nonfinite_state` instead of accepting the canonical physical-failure class. A controlled zero-optimizer offline boundary test reproduced the same exception with exactly one true failure flag. The actual failed transition's specific physical cause remains **UNRESOLVED**, since its info was not persisted. Original environment/training source and protocol remain unchanged; no bug fix was followed by a new training attempt. Formal readiness is now **NO**.

Resources: Last persisted wall950.486s, CPU945.540s; recorded peakRSS1793.051MiB, minimumMemAvailable8583.656MiB, no observed OOM. Exact exit wall/peak and in-memory environment/scan/encoding/update timing aggregates were lost at exception and are marked NOT RECORDED. Source/model/checkpoint hashes remain consistent. Complete reset visits are saved only through100096; later terminal episode identities are available but the failing reset identity is missing. Nine plots explicitly present partial/interrupted evidence; no full-run learning or generalization claim.

Verification: Completed full regression **719 PASS,0 FAIL,0 SKIP,2 inherited warnings** (698 inherited+21 new); after adding one PNG-header artifact test, final L9 targeted suite **22 PASS** in28.69s. Joint unique case coverage720 is not a single complete720-case run. An additional full recheck (session22540) was terminated by SIGTERM/exit143 near10%, cause unverified; it is BLOCKED/INCOMPLETE and excluded from passing counts. Independent read-only failure/artifact review GO accepts the truthful FAIL handoff and confirms contract mismatch, checkpoint/Validation identities, resource records, unchanged seven historical models/MJCF/runtime sources,0 Test and visible INTERRUPTED headers in all9 original PNGs. Historical raw files were retained; no PX4/FAST-LIO2/real hardware or seed1/2 execution.

Next: Only recommend a separate canonical-termination/exception-evidence repair and protocol re-registration before requesting a new seed0 attempt. This stage stops with the failed attempt preserved.

## 2026-10-14 — PPO Terminal Accounting Repair Phase 3E-L9R

Branch: `fix/lidar-ppo-terminal-accounting`; base `c1c8d4f288a75a539b8721d4f991c613b830605f`.

Scope: Offline repair/tests only. Formal/Pilot/Smoke runs0, optimizer updates0, Test episodes0, PX4/FAST-LIO2/real-device connections0. Original L9 remains FAIL; no resume, overwrite, completion marker or old protocol modification.

Confirmed defect: NavigationEnv retained raw physical causes while TrainingMetricsEnv required the raw string to be a canonical category. Controlled MuJoCo outside_flight_area first reproduced the old accounting RuntimeError (RED). Shared strict classification now preserves terminal_class/raw_termination_reason, checks all explicit mutually exclusive flags and terminated/truncated semantics, and fails closed on unknown/inconsistent causes. Training, Health and Validation share it. NavigationEnv/Reward/physics/Observation untouched; eight forced terminal cases verify exact reward/observation/termination equivalence, plus real refreshed contact confirms collision priority. FORCED_TEST_STATE is not a natural L9 failure.

Evidence: New bounded exception recorder keeps run/attempt/step/scenario/time/action/qpos/qvel/ctrl/flags/reward/resource/traceback; distinguishes RETURNED_TRANSITION from LAST_KNOWN_CONTEXT with NOT_AVAILABLE, explicit nonfinite markers, global serialization budget, atomic fsync+hard-link no-overwrite and failure latch. Validation/Health nonfinite position diagnostics are unavailable, not fabricated zeros or stale final distances. Coverage does not promise initialization/reset/optimizer/SIGKILL evidence or exact resume.

History: All11 original raw files, old report/document and seven checkpoints/model (21 protected files) are unchanged. Base Git objects verify original runtime hashes. 706 saved episodes recompute to656 collision/50 timeout/0 success; specific failing physical trigger remains unrecorded. Only milestone100096: approx_kl0.0000675833, clip_fraction0, value_loss27.6652, explained_variance0.662545, mean action norm1.441488, mean path3.160586m, mean final distance1.754031m. One partial seed/snapshot does not establish learned avoidance or permanent PPO incapability.

Identity: New repair SHA256 `e267406ec4203353b4ee0c7f8349b57759ef64b31bd076c3db57b99b3682678f`; parent `fe07816f48bd4183165db482d8c93de50c637a32d13d86eccb47667d85e8cef6` unchanged. All parent hyperparameters/scene/network/Reward/budget preserved; adds9 source hashes, strict terminal/evidence contracts and formal_retrain_authorized=false. Driver rejects execution before model/attempt creation. No new training authorized. Python float/int JSON types preserved and published protocol roundtrip SHA verified; result review caught preliminary ent_coef=0 rather than0.0, fixed without changing hash algorithm. Reset exceptions do not invent attempted policy steps.

Verification: Final selected suite777 PASS (720 inherited +57 new),0 FAIL/0 SKIP,22 explicitly deselected historical update/version-specific tests,2 inherited warnings; not a full-repository or all-history PASS. Tripwires prevent accidental learning/optimizer updates. First broad run767 PASS/6 FAIL due test-monitor thread violating unchanged Bridge single-thread checks; instrumentation corrected to single-thread lifecycle sampling, then complete selected rerun passed. Final peak RSS1890.613MiB, sampled minimumMemAvailable8297.098MiB, wall102.132s; largest measured regression peak2007.945MiB, no observed OOM. Whole-task peak across earlier separate RED processes unavailable. Optional cv2/NumPy ABI warning unchanged.

Independent read-only code and final result/document review GO,0 unresolved Critical/Important, after fixing2 code findings (global bounded capture/metadata; nonfinite Validation metrics) and the report JSON float/int identity issue. Final review verified777 raw results/22 exclusions, Python protocol roundtrip/9 source identities and all21 protected hashes. Runtime code unchanged after final tests. `TERMINAL_ACCOUNTING_FIXED=YES`, `FAILURE_CAPTURE_READY=YES`, `FORMAL_RETRAIN_AUTHORIZED=NO`. Raw regression/failure evidence local-only; original87 untracked retained. Next: stop; any fresh formal seed0 attempt requires separate authorization and a new explicit preflight identity, never warm-start continuation of L9.
