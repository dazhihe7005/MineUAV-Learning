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
