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
