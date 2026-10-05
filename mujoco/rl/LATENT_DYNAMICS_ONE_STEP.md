# One-Step Latent Dynamics — seed 0

This is offline supervised dynamics prediction, not policy training or a world-model framework. Reward V2, PI, actions, observations and simulator files are unchanged.

## Data and alignment

Reuse `datasets/pi_hidden_state_seed0/{train,val,test}.npz`, with original SHA256 checks. No dataset regeneration. Seven trajectory sources per target: scripted, oracle PPO, 7D PI PPO, zero, IID random, held random, alternating. Successful, timed-out and flight-area failure episodes are represented.

| Split | Targets | Episodes | Stored steps | Valid transitions |
|---|---:|---:|---:|---:|
| Train | 84 | 588 | 147219 | 146631 |
| Validation | 18 | 126 | 30802 | 30676 |
| Test | 18 | 126 | 32422 | 32296 |

All seven trajectories of a target stay in one split. Original terminal next observations were not saved: exclude exactly one final command per episode (840 total), rather than inventing terminal data or crossing episode boundaries.

At timestep t: `o_t=inputs[t,:7]`, **previous** executed normalized action `a_(t-1)=inputs[t,7:11]`, **current** action `a_t=actions[t]`, next observation `inputs[t+1,:7]`. Target is the raw subtraction `delta_o=o_(t+1)-o_t`, including yaw (zero wrap jumps were found). Physical action scaling is unchanged: [1.5,1.5,1,1].

The original generation manifest and derived split/transition manifest are versioned; the NPZ bodies and derived latent arrays are local-only. Reproduction requires these exact existing NPZ files, not newly sampled replacements.

## Models and training

| Model | Inputs | Architecture | Parameters | Best epoch |
|---|---|---|---:|---:|
| Markov | o_t7 + a_t4 | MLP11→64→64→7 | 5383 | 57 |
| History latent | [o_t7,a_(t-1)4] sequence, then a_t4 | GRU11→64, one layer; head68→64→64→7 | 23815 | 60 |
| PI-state Oracle | o_t7 + true PI3 + a_t4 | MLP14→64→64→7 | 5575 | 60 |

All MLP hidden activations are Tanh; output linear. Each model: seed0, CPU, Adam lr=0.001, 60 complete epochs, 16 whole episodes per batch, identical episode-order schedule. Whole ordered sequences reset hidden state; right padding is excluded from loss. No early stopping or Test-based selection.

Train-only population mean/std are fitted separately for observation, executed action, delta, and Oracle PI. Std floor 1e-6; previous actions reuse current-action statistics. Shared loss: mean normalized delta squared error over valid timesteps and all 7 dimensions. History does not read PI, including through an auxiliary loss.

| Model | Best checkpoint Train MSE | Best Val MSE | Epoch60 online Train MSE | Epoch60 Val MSE |
|---|---:|---:|---:|---:|
| Markov | .308411 | .352604 | .310197 | .354086 |
| History | .021732 | .058853 | .022333 | .058853 |
| Oracle | .218457 | .264425 | .222023 | .264425 |

Total run elapsed 131.81s; models 12.75 / 101.53 / 12.30s respectively. Same epoch/sample budget, not identical wall-clock cost.

## Shared Test results

| Model | Physical delta RMSE | MAE | R² | Standardized RMSE |
|---|---:|---:|---:|---:|
| Markov | .0112404 | .00450828 | .668054 | .605748 |
| History | .00564894 | .00111001 | .916162 | .247433 |
| Oracle | .00959443 | .00355193 | .758151 | .540332 |

Physical overall metrics mix m, m/s and rad; do not assign a single physical unit. Standardized metrics are dimensionless. Overall R² is 1−sum(SSE_dimension)/sum(SST_dimension), centering each dimension independently. Since the same measured o_t is added back, next-observation reconstruction error equals delta error exactly.

Each cell below is **RMSE / MAE / R²**:

| Delta dimension | Markov | History | Oracle |
|---|---|---|---|
| error_x (m) | .00147815 / .00087060 / .99156 | .00069911 / .00039243 / .99811 | .00175221 / .00100524 / .98814 |
| error_y (m) | .00143111 / .00084631 / .99185 | .00066687 / .00040245 / .99823 | .00185273 / .00104147 / .98634 |
| error_z (m) | .00078896 / .00044240 / .99051 | .00050638 / .00027554 / .99609 | .00095010 / .00057614 / .98623 |
| vx (m/s) | .01941734 / .01195404 / .23619 | .00731253 / .00176649 / .89167 | .01600713 / .00861162 / .48092 |
| vy (m/s) | .01901595 / .01188990 / .23817 | .00709778 / .00192853 / .89386 | .01564175 / .00830037 / .48455 |
| vz (m/s) | .01133164 / .00340155 / .88358 | .01081321 / .00259610 / .89399 | .01111629 / .00320458 / .88797 |
| yaw_error (rad) | .00353823 / .00215314 / .27410 | .00118615 / .00040855 / .91842 | .00353581 / .00212413 / .27509 |

Distance-region physical delta RMSE:

| Distance | Test transitions | Markov | History | Oracle |
|---|---:|---:|---:|---:|
| <0.1m | 431 | .00400003 | .00042697 | .00152275 |
| 0.1–0.2m | 1691 | .00489509 | .00054541 | .00220548 |
| 0.2–0.5m | 7460 | .00483066 | .00094312 | .00294437 |
| ≥0.5m | 22714 | .01303433 | .00671226 | .01129746 |

PI norm bins use Train tertiles q1=.0507170, q2=.2328452 m/s². True PI is used for grouping only, never as History input.

| PI magnitude | Test transitions | Markov RMSE | History RMSE | Oracle RMSE |
|---|---:|---:|---:|---:|
| Small, <q1 | 10332 | .00189529 | .00069124 | .00127857 |
| Medium, q1≤norm<q2 | 10921 | .00947851 | .00476318 | .00741824 |
| Large, ≥q2 | 11043 | .01665221 | .00839287 | .01460355 |

RMSE gap closure `(Markov−History)/(Markov−Oracle)` = 3.397 physical, 5.477 standardized. These values exceed 1 because History beats this **PI-state-only** Oracle. They are not literal percentages of a full-state information ceiling. Oracle omits attitude, angular rates and other controller memory. History has more parameters and temporal information, so this comparison does not isolate PI recovery causally.

## Latent and post-hoc probe

Test z norm mean/std/max = 1.66958 / .47515 / 4.03256; finite, no NaN/explosion. Same-instance A→B→A reset is exact. Streamed versus full-sequence predictions differ at most 4.84e-7 in normalized delta units.

Frozen encoder OLS, fitted on Train only, predicts PI with Test overall RMSE=.140976m/s², MAE=.0737793m/s², R²=.525441. x/y/z RMSE=.142168/.169939/.102622m/s², MAE=.080743/.084707/.055889, R²=.607906/.466303/.472841. Encoder model hash is unchanged. PI is partially linearly decodable, not precisely reconstructed; no probe loss enters dynamics training.

## Conclusion and limits

Under this fixed seed/split, history-derived latent state materially improves one-step dynamics prediction over current-observation-only Markov prediction, including near-target and large-PI regions. This supports useful learned predictive state without PI supervision. It does not establish a minimal physical state, exact PI identification, causal latent semantics, or robust long-horizon dynamics.

Only nominal simulation, one seed and one-step 25Hz predictions were tested; only 431 Test transitions lie below 0.1m. Vertical-velocity improvement is much smaller than horizontal improvement. All models received the same epoch/sample budget, but recurrent parameter capacity is larger. Multi-step prediction is a reasonable next experiment **only after a separate instruction**; nothing here executes it.

## Commands and artifacts

```bash
cd ~/MineUAV-Learning
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test*latent_dynamics*.py' -v
.venv/bin/python mujoco/rl/verify_latent_dynamics_one_step.py
```

Initial training command: `.venv/bin/python mujoco/rl/run_latent_dynamics_one_step.py`. The trainer refuses to overwrite existing representative models/reports. Dataset NPZs must be present with the recorded hashes; the verifier only reads existing data/models and never trains.

- Report: `mujoco/reports/latent_dynamics_one_step_seed0.json` (full normalization, epoch curves, metrics and hashes).
- Models: `mujoco/rl/models/{markov_dynamics_mlp,history_latent_gru,oracle_dynamics_mlp,latent_pi_linear_probe}.pt`.
- Figures: five specified PNGs in `mujoco/reports/`.
- Manifests: original `mujoco/rl/datasets/pi_hidden_state_seed0/manifest.json`, derived `mujoco/rl/datasets/latent_dynamics_one_step_seed0/manifest.json`.
- Local-only: original raw NPZ (~11MB), derived z arrays (~43MB), temporary tests/logs and all historical diagnostics.
