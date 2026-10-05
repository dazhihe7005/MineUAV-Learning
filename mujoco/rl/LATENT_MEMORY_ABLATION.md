# No-Memory GRU Ablation — seed 0

Question: Does the one-step prediction improvement require cross-time recurrent memory, rather than just the GRU/head architecture?

## Fixed setup and intervention

Branch `feat/latent-memory-ablation` starts from pushed one-step commit `ad2a7a21f203fea3bfe4a010d58b58e2f69cee4c`. All three old model files and the previous report stay immutable; their hashes and every Test metric are reproduced before the new run.

Exactly reuse `pi_hidden_state_seed0`, original target-disjoint splits and saved normalization statistics. Train/Val/Test: 588/126/126 episodes; 146631/30676/32296 transitions. No resampling; no terminal/cross-episode transition changes. Prediction remains raw `delta_o=o_next-o` in 7D.

Both History models have identical parameter names, shapes, seed0 initialization and 23815 parameters:

- GRU(input11, hidden64, one layer), input `[o_t7, previous executed normalized action4]`.
- Dynamics head: `[z_t64,current_action4]` → Linear68→64→64→7, Tanh hidden activations.
- Full History passes the prior hidden state through ordered timesteps.
- No-Memory reshapes B×T timesteps into B×T independent length-1 GRU lanes, each initialized with hidden zero. This is not hidden-state detachment: prior hidden *values* are never used. The streaming API also ignores supplied prior hidden state.

Shared trainer is unchanged: seed0, Adam lr=.001, 16 complete episodes per batch, identical `default_rng(0)` episode shuffle schedule, 60epochs, normalized masked MSE, validation-best selection. No PI input/auxiliary loss, no policy/controller changes.

No-Memory best epoch57, final epoch60. Best Train normalized MSE=.299982; best Val=.342463. Final online-average Train=.302711, Val=.344305. Total experiment elapsed40.74s. Only the new experimental model was trained; test fixtures use small synthetic data independently of the experimental artifacts.

## Shared Test comparison

| Model | Delta RMSE | MAE | R² | Normalized RMSE |
|---|---:|---:|---:|---:|
| Markov MLP | .01124036 | .00450828 | .668054 | .605748 |
| No-Memory GRU | .01109201 | .00445811 | .676758 | .598015 |
| Full-History GRU | .00564894 | .00111001 | .916162 | .247433 |
| PI-state augmented MLP | .00959443 | .00355193 | .758151 | .540332 |

Overall physical metrics mix m, m/s and rad; use per-dimension units or dimensionless normalized RMSE for scale interpretation.

No-Memory per-dimension results:

| Delta dimension | RMSE | MAE | R² |
|---|---:|---:|---:|
| error_x (m) | .00129436 | .00082029 | .993531 |
| error_y (m) | .00140348 | .00088727 | .992162 |
| error_z (m) | .00069659 | .00041461 | .992599 |
| vx (m/s) | .01902853 | .01166235 | .266471 |
| vy (m/s) | .01869009 | .01175149 | .264059 |
| vz (m/s) | .01154298 | .00354931 | .879200 |
| yaw_error (rad) | .00352902 | .00212149 | .277869 |

Critical-region delta RMSE (same thresholds and sample identities):

| Region | Markov | No-Memory | Full History | PI baseline |
|---|---:|---:|---:|---:|
| distance<.1m, n431 | .00400003 | .00409706 | .00042697 | .00152275 |
| .1–.2m, n1691 | .00489509 | .00498398 | .00054541 | .00220548 |
| .2–.5m, n7460 | .00483066 | .00485487 | .00094312 | .00294437 |
| ≥.5m, n22714 | .01303433 | .01284624 | .00671226 | .01129746 |
| Small PI, n10332 | .00189529 | .00208691 | .00069124 | .00127857 |
| Medium PI, n10921 | .00947851 | .00942339 | .00476318 | .00741824 |
| Large PI, n11043 | .01665221 | .01636837 | .00839287 | .01460355 |

PI norm tertiles remain q1=.0507170, q2=.2328452m/s², derived in the prior experiment from Train only. True PI is used for evaluation grouping, never for No-Memory input/loss.

## Interpretation and boundaries

Full History relative reduction versus No-Memory = `(RMSE_no_memory−RMSE_history)/RMSE_no_memory`: **49.07199% physical**, **58.62430% normalized**.

No-Memory relative reduction versus Markov = `(RMSE_markov−RMSE_no_memory)/RMSE_markov`: **1.31978% physical**, **1.27665% normalized**. Within .5m, No-Memory is actually slightly worse than Markov.

The same GRU/head without recurrent history is close to Markov and clearly below Full History. This supports cross-time recurrent memory as an important source of the one-step gain in this fixed-seed setting, rather than architecture/parameter count alone.

Limits: only one training seed and nominal one-step 25Hz prediction. Identical nominal parameter tensors do not imply identical active temporal paths: zero hidden naturally disables those paths and recurrent-weight gradients. No-Memory still retains explicit previous-action information. No-Memory versus Markov additionally differs in inputs/capacity; the controlled memory comparison is Full History versus No-Memory. PI-state augmented MLP is not a complete MuJoCo-state lower bound. Test near-target coverage is only431 transitions. No statistical multi-seed/generalization or long-horizon claim is made.

It is reasonable to proceed next to **frozen multi-step rollout evaluation**, with separately specified horizon/teacher-forcing/history handling and no model retraining. This experiment stops here; it does not perform any multi-step evaluation.

## Verification and artifacts

```bash
cd ~/MineUAV-Learning
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*latent_memory*.py' -v
.venv/bin/python mujoco/rl/run_latent_memory_ablation.py --verify
```

Initial training command: `.venv/bin/python mujoco/rl/run_latent_memory_ablation.py`. Existing outputs are protected against overwrite. `--verify` only reads existing data/models and reconstructs all four models' physical, normalized, per-dimension and region metrics. It also connects the validation-best assertion to checkpoint metadata/config and recomputes the actual saved model's validation loss.

Verification: 8 focused ablation tests pass, including tampered-report rejection, identical initial parameters, zero temporal input gradients, streaming memory rejection and model reload. The tracked repository regression suite plus these tests runs 260 tests with no failures and one optional GUI skip. Independent read-only review found no blocking issue; its validation-checkpoint verification suggestion was implemented and regression-tested.

- Model: `mujoco/rl/models/history_latent_gru_no_memory.pt` (~99KB).
- Report: `mujoco/reports/latent_memory_ablation_seed0.json` (full metrics, epoch history, preserved statistics, dataset/split/model hashes and intervention config).
- Figures: `memory_ablation_model_comparison.png`, `memory_ablation_vs_target_distance.png`, `memory_ablation_vs_integral_magnitude.png` under `mujoco/reports/`.
- No new raw dataset, no latent-array dump, no intermediate checkpoint copies. Existing raw NPZ, historical artifacts and logs remain local/ignored.
