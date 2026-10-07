# Joint v1 vs Autonomous Consistency v3 — paired training-seed replication

Branch `feat/joint-v1-v3-multiseed`, from
`feat/joint-autonomous-consistency-v3` at
`fa357af3c1a9fc0f24e635e1b073b180bdca4b64`.
No main merge/rebase, new objective, upstream retraining or follow-on experiment.

## Research question and scope

Does the long-horizon benefit of autonomous latent-consistency training replicate
across five paired **joint-training seeds**, with one fixed upstream pretrained
initialization? This is **not** random-initialization or full end-to-end pipeline
replication. Seed0 is reused from immutable existing v1/v3 checkpoints and reports;
only seeds1–4 are newly trained, once per condition (eight runs).

All runs begin from exactly the same pre-joint checkpoints:
`history_latent_gru_multistep10.pt`, `latent_transition_multistep10_mlp.pt`, and
`latent_observation_decoder.pt`. The existing initializer folds their affine
latent affine transformations into T/D weights exactly as in original Joint v1.
Observation, previous-action and current-action normalization remain runtime operations.
Initial component parameter hashes, after folding, are:

```
E 25a95644f2a2a7b1fc69cd336bd13cfe771f9dca65ee2c9686f64106ccf71e75
T 233b0c6d8eac9cbf33b2e3b131784d091a26eca524bd91f54e842f62d924406f
D 18b8e42ec7e78a83149fdc3b6ae1aaab8b26ad40a59b194a2bb443ec5ab2abfd
```

Architecture remains74,119parameters: GRU11→64(one layer), residual Tanh
MLP68→128→128→64, and Tanh decoder64→128→128→7. All E/T/D update.
E reads observation7D and **previous** executed action4D; T receives **current**
recorded action. PI state is never a model input or auxiliary target.

## Pairing, fixed budget and objectives

Python/NumPy/PyTorch seed s∈{0,1,2,3,4}, CPU one thread, deterministic algorithms.
Adam lr=.0003, default betas=.9/.999, eps=1e-8;16complete episodes/batch;
K10(0.4s),60epochs, no new clipping or schedule. At each epoch, both conditions
use the identical `default_rng(s).permutation(Train episode count)` realization.
Every legal K10 start within selected complete episodes participates. For new runs,
SHA256 of each realized little-endian int64 episode permutation is recorded **during
training**, checked against the prescribed rule and compared within each pair.
Seed0 order hashes are reconstructed from its archived rule, not retroactively
measured. Seeds change ordering/optimizer trajectory, not pretrained parameters.

From a causal true-history prefix, E produces z[t] with gradient retained. Both
conditions perform the identical10-step autonomous T composition, retaining full
BPTT; decoded observations never feed E or T. Future observations/latents are not
state inputs. Let Dnorm be the normalized decoder readout:

```
Lobs = mean_over_windows,k=0..10,7dims
       (Dnorm(zpred[t+k]) - (o[t+k] - Train_obs_mean)/Train_obs_std)^2

v1: Ltotal = Lobs

v3: zref[t+k] = E(real history through t+k)        # detached target only
    Lauto = mean_over_windows,k=1..10,64dims
            ((zpred[t+k] - stopgrad(zref[t+k]))/sigma_initial_Train)^2
    Ltotal = Lobs + 0.1 * Lauto
```

No local consistency loss. V3 source E and all composed T steps receive the
auxiliary gradient; reference E receives none; D receives only Lobs gradients.
The original Train-only initial latent std is fixed, floor1e-6. Both conditions
select checkpoint **only by Validation Lobs**, not Test, Ltotal or Lauto. Complete
all60epochs regardless of selected best epoch. Delivered checkpoint hashes refer
to the Validation-best model, not an unsaved final-epoch model when best is earlier.
Best-at-cap does not establish convergence or authorize increasing the budget.

## Data, normalization and manifest integrity

Reuse `pi_hidden_state_seed0` without trajectory regeneration: target-disjoint
84/18/18Train/Val/Test groups;588/126/126episodes. Reuse original Train-only
observation/action statistics and K10Train/Val manifests141339/29542windows.
Original Test start-point manifest SHA256:
`57ba97fd3f0c8e7befcbca5d76da73179e8880c789c5072a4651ef2e6566d6e9`.
All legal starts at H1/5/10/25/50 give32296/31792/31162/29272/26122windows.
Normalization semantic hash:
`cf53ef60049cbd55ad828f112644f3e81bbf0bc94ab236ddbba332b6744e1be2`.

Only Train/Val timestep values are opened before checkpoint selection. Test files
may be hashed for provenance but their values are first loaded after selection.
Dataset, source reports/checkpoints, normalization and manifests are verified
against original hashes, with source immutability checked again after evaluation.
No duplicate dataset is generated. Latent arrays used in memory for evaluation
and diagnostics are not persisted or committed.

## Frozen evaluation and minimum diagnostics

Encode real history through start t once; advance only predicted latent and saved
actions. Evaluate endpoint normalized7D observation RMSE and physical per-dimension
RMSE/MAE at1/5/10/25/50steps, plus horizontal velocity, yaw, current reconstruction,
distance/PI groups, and each model's own-Train-scale latent consistency/cosine.
H25/H50 exceed the K10 training horizon. Corrupting/deleting future observations
must leave selected-window latents and predictions bit-identical.

All conditions evaluate C1/5/10/25/never corrections on the same26122H50windows.
Corrections replace the current state **before** the next transition, never after
the evaluated endpoint; C50≡never. Corrections use real future histories only as
a diagnostic, not deployable inference. Report gaps never−C1 and never−C10.

Retain Train/Test latent norm, per-dimension std, effective rank, nonfinite flags
and own-model H50 consistency. A descriptive potential-collapse warning is
≥32/64dimensions with std<1e-6 or total variance<1e-8; it is not a mathematical
sufficiency test. Do not discard anomalous seeds; full heavy audit is reserved for
an actually anomalous seed. No default covariance/PCA/sensitivity analyses.

## Statistics and failure accounting

For each horizon list every seed, paired delta=v1−v3 and
improvement%=100(v1−v3)/v1 (positive is better). Report mean, **sample SD ddof1**,
median/min/max and wins/losses/ties. No significance claim or p-value. Exactly
five requested identities remain, including failures. Failed metrics are null;
valid-pair count and missing-pair count are explicit, never silently pooled away.
Nonfinite returned diagnostics persist as failed records with offending field paths.
Available surviving plot points remain visible and missing counterparts annotated.

For completed new branches, the verifier reopens selected models and reproduces
saved-best Train/Val losses, Test metrics, orders and source hashes. Seed0 reuse
reproduces archived Test/reconstruction/correction metrics and verifies immutable
sources; it does not recompute selected seed0 Train/Val losses. All five pairs enter
the recomputed statistics. For completed new branches it checks logged history
arithmetic/nonnegative finite losses/gradients and final summaries, but
does **not** replay historical optimization or claim historical gradients reproduced.
Failed rows still must satisfy initialization/configuration/normalization invariants;
unavailable trajectory checks are qualified as available/completed-branch checks.

## Reproduction and artifact policy

```
# Each new branch once; seed0 has no training option.
.venv/bin/python mujoco/rl/run_joint_v1_v3_multiseed.py --train v1 --seed 1
.venv/bin/python mujoco/rl/run_joint_v1_v3_multiseed.py --train v3 --seed 1
# Repeat --seed 2,3,4 for each condition without changing any other setting.
.venv/bin/python mujoco/rl/run_joint_v1_v3_multiseed.py --assemble
.venv/bin/python mujoco/rl/run_joint_v1_v3_multiseed.py --verify
.venv/bin/python -m unittest discover -s mujoco/rl -p 'test_joint_multiseed*.py' -v
```

Existing models/report fragments are never overwritten or automatically retrained.
Commit experiment code/tests, eight small selected models, one compiled report,
seven requested figures, this methods file and EXPERIMENT_LOG. Existing manifests
are referenced by hash; no duplicate manifests/raw trajectories/traces/intermediate
models are committed. Per-branch JSON fragments, execution logs and cache are
ignored/local-only. Unrelated historical untracked experiments remain untouched.

The local full-suite harness also exercises historical untracked diagnostics.
Missing legacy alignment helpers/report are loaded read-only from fixed Git commit
`8879754a43302e9c55b27de4ce3c629c7e4e1a53`, not merged or restored into this branch.
Explicit desktop/absent historical-source skips are reported separately; new
algorithm tests must all pass. This harness stays ignored/local-only.

## Interpretation boundary and stopping rule

Five seeds share upstream training, nominal simulation, one dataset and overlapping
windows; future actions are recorded, not policy-generated. Native latent coordinates
can differ and are not physical cross-model quantities. Fixedλ.1/K10/60epochs,
moving online targets and Validation selection remain limitations. This tests
robustness to **joint-training stochasticity**, not broad simulator or initialization
robustness. No policy, reward, planning, model-based control or complete agent.

After evaluation/report/tests/commit/push, stop the deterministic consistency
objective family. Baseline promotion depends on paired results and reconstruction/
variance checks; it does not authorize further lambda/K/architecture/EMA experiments.
Only one suggested next experiment is documented, never executed here.

## Measured results — paired seeds0–4

All eight new runs completed60epochs, without failed/replaced seeds. Seed0 model
files stayed byte-identical and archived metrics were reproduced read-only.
Initial E/T/D hashes match for every run. All60recorded epoch orders agree within
every new pair; seed0 pairing is reconstructed. Validation-best epochs coincide
within pairs:60/59/58/51/60 for seeds0/1/2/3/4. All selected E/T/D hashes changed
from initialization; individual full hashes/loss histories remain in JSON.

Normalized observation endpoint RMSE on the same Test manifests:

| Seed | Model | Best epoch | H1 | H5 | H10 | H25 | H50 |
|---|---|---:|---:|---:|---:|---:|---:|
|0|v1 reused|60|.043320|.070828|.102727|.230850|.510667|
|0|v3 reused|60|.044057|.071549|.104600|.201413|.308050|
|1|v1|59|.042739|.069799|.101778|.230792|.505980|
|1|v3|59|.043286|.070445|.102845|.197542|.297185|
|2|v1|58|.043098|.070157|.102542|.235733|.567516|
|2|v3|58|.043637|.070700|.103619|.202670|.313745|
|3|v1|51|.044189|.071383|.103589|.236094|.527100|
|3|v3|51|.044720|.072007|.105102|.203781|.308057|
|4|v1|60|.043393|.071001|.103645|.242633|.544148|
|4|v3|60|.043971|.071219|.103923|.199249|.301739|

| Seed | H25 delta v1−v3 | H25 improvement | H50 delta v1−v3 | H50 improvement |
|---|---:|---:|---:|---:|
|0|.029438|12.75%|.202618|39.68%|
|1|.033250|14.41%|.208794|41.27%|
|2|.033062|14.03%|.253771|44.72%|
|3|.032314|13.69%|.219043|41.56%|
|4|.043384|17.88%|.242409|44.55%|

H25 v1 mean±sampleSD=.235221±.004865, v3=.200931±.002534;
paired delta=.034290±.005309, median=.033062, range=.029438–.043384.
H50 v1=.531082±.025303 (median.527100, range.505980–.567516),
v3=.305755±.006403 (median.308050, range.297185–.313745);
paired delta=.225327±.021962, median=.219043, range=.202618–.253771.
V3 wins **5/5 at both H25 and H50**. Mean per-pair percentage improvements are
14.55±1.96% and42.35±2.20%, respectively. The improvement computed from H50
group means is42.43%; this is a different aggregation from mean paired percentage.
No extreme seed reverses the result; no formal significance test is claimed.

### Short-horizon and representation trade-off

V3 loses5/5pairs at H1/H5/H10, by1.20–1.70%/.31–1.02%/.27–1.82%, respectively.
Group means v1→v3 are .043348→.043934, .070634→.071184,
.102856→.104018. The small short-term regression is stable, not absent.
Current reconstruction instead improves slightly5/5:
v1 .038386±.000491 → v3 .038167±.000506 (mean paired improvement.57%).
There is no systematic severe reconstruction sacrifice.

| Seed | Reconstruction v1/v3 | H50 horizontal v1/v3 (m/s) | H50 yaw v1/v3 (rad) |
|---|---|---|---|
|0|.038286/.038199|.318011/.187326|.156039/.070760|
|1|.037817/.037512|.315392/.184477|.145435/.055118|
|2|.038312/.038003|.361055/.194940|.181271/.060909|
|3|.039176/.038920|.323126/.192118|.182143/.056367|
|4|.038338/.038202|.348456/.183870|.140151/.064629|

Horizontal mean±SD .333208±.020359 → .188546±.004836m/s;
paired reductions[.130685,.130915,.166116,.131008,.164587], mean.144662±.018895.
Yaw .161008±.019745 → .061557±.006383rad;
paired reductions[.085279,.090317,.120362,.125776,.075522], mean.099451±.022289.
Both dimensions improve5/5; full physical per-dimension metrics remain in JSON.

### Correction and latent-health checks

Mean H50 correction curves on the identical cohort:

| Model | C1 | C5 | C10 | C25 | Never |
|---|---:|---:|---:|---:|---:|
|v1|.046113|.075876|.109703|.242363|.531082|
|v3|.046694|.076319|.110730|.208816|.305755|

Every seed has a monotonic correction curve. Never−C1 shrinks5/5:
.484969±.025287→.259062±.006228 (mean paired gap reduction46.50%).
Never−C10 similarly .421379±.025056→.195025±.005994 (53.63%).
V3's C1/C10 errors are slightly worse, so the gap reduction is not an across-the-board
local prediction gain. Neither condition eliminates the diagnostic correction benefit.

H50 own-model Train-scale discrepancy v1/v3 by seed:
[.749678,.739651,.777577,.728734,.765066] /
[.356615,.350178,.367172,.362707,.354282];
cosines [.878943,.877490,.876302,.890122,.882994] /
[.976299,.976999,.975808,.975763,.976650]. These are internal-coordinate diagnostics,
not coordinate-invariant physical errors.

Every Train/Test latent has0near-zero-std dimensions. Effective rank ranges:
v1Train12.27–12.42/Test12.05–12.19; v3Train11.96–12.12/Test11.73–11.90.
All64dimensions retain variance (minimum std>.107). Max evaluated latent norm is
≤5.219 for v1 and≤4.254 for v3. No NaN/Inf, numerical explosion, collapse warning
or abnormal seed occurred; therefore no additional heavy audit was triggered.

## Conclusion and one recommended next experiment

Autonomous Consistency v3 receives **preliminary five-seed support for long-horizon
prediction under fixed upstream initialization**, with a consistent small short-term
trade-off and no reconstruction/collapse failure. Promote the **v3 formulation** to
the current deterministic latent dynamics baseline; retain existing seed0 as the
reference checkpoint rather than selecting the Test-best seed. This is not overall
dominance at all horizons, nor proof of unique mechanism or end-to-end robustness.

The deterministic consistency-objective family is formally stopped: no lambda/K
sweep, mixed/local auxiliary objective, EMA or added network is executed.
One next suggestion only: paired end-to-end replication across **independently
pretrained upstream initializations** to test the robustness not measured here.

## Verification evidence

Related TDD suite:13passed/0failed/0skipped, including paired ordering,
seed0 archived-trainer bitwise equivalence, reload/selection, failure retention,
nonfinite serialization and report-tampering rejection. Final local full discovery
with the read-only legacy Git fixture:434tests,430passed/0failed/4skipped,
116.74s. Skips are one optional desktop-X11 viewer test and three historical
Recurrent-source-worktree tests; no new algorithm test is skipped.
All seven final figures were visually checked; legends are outside paired/bar
axes so they do not cover seed results. Baseline source artifacts remain immutable.
The independent read-only saved-model verifier passes: original seed0 metrics,
all new selected Train/Val losses, complete histories/order hashes, frozen Test
metrics, source/model hashes and five-pair statistics reproduce. No optimizer is
created during report assembly or verification.
