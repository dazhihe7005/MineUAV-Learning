# MineUAV RL Baseline v1 Robustness Audit — Design

## Purpose and frozen boundary

Evaluate five saved Reward V2, low-std 100k PPO policies against controlled simulation mismatch before any Sim2Real training. No PPO learning, policy selection, reward/interface/timing/controller change, or nominal MJCF rewrite is permitted. The existing 100 holdout waypoint seeds 20271001–20271100 are used in the same order for every policy and condition, with deterministic actions.

## Injection design

Construct one headless `MineUAVEnv(reward_version="v2", target_distribution="full")` per independent evaluation. Each instance loads `mine_uav_dynamics_v2.xml` into its own `MjModel`. A narrow subclass of the environment replaces only the physical control application and per-physics-step hooks; the baseline path has identical default behavior. The compiled nominal model and current nominal controller/allocator are kept separate: a model-only mass, inertia or thrust-coefficient override does not retune control gains or the allocation matrix.

- Mass factor changes `body_mass[mine_uav]` alone; `body_inertia`, COM and actuator gear remain nominal. Call `mujoco.mj_setConst` after mass/inertia overrides.
- Inertia factor multiplies the three compiled principal inertias together; their inertial orientation is unchanged, so the full 3×3 tensor scales uniformly. Mass and COM remain nominal.
- Thrust coefficient factor multiplies only the translational +Z component of each of four `actuator_gear` vectors. Their yaw-torque gear entries, motor positions, allocator, and controller remain nominal.
- Motor lag uses a separate four-element actual motor-speed state `omega_actual`, initially zero like nominal reset's `ctrl=0`. On each 100 Hz controller update, `omega_cmd=sqrt(clip(u_cmd,0,u_max))` is held. At every 500 Hz physics step, advance the exact zero-order-hold first-order response `omega_actual=omega_cmd+(omega_actual-omega_cmd)*exp(-dt/tau)` and set `data.ctrl=omega_actual**2`; `tau=0` is bit-for-bit nominal instantaneous actuation. The same actual `u` generates thrust and yaw torque through the original four 6D site gears. No controller/allocator change.
- Constant ±X world force is applied at the UAV body COM through `data.xfrc_applied[body_id,:3]`, with zero added torque, before each physics step. Reset clears prior force and restores the selected constant value.

Each factor is swept alone against nominal: mass/inertia/k_f 0.8,0.9,0.95,1,1.05,1.1,1.2; motor lag 0,20,40,60,80,100 ms; X force -20,-15,-10,-5,0,+5,+10,+15,+20 N. Nominal is evaluated once and reused as the zero/factor-1 row. No motor-to-motor mismatch or random disturbance is introduced.

## Measurement and outputs

Reuse `diagnose_ppo_braking.evaluate_episode`, `summarize_policy`, `reward_v2_diagnostics.summarize_episodes`, and `reward_v4_diagnostics.crossing_counts` so success, distance, near-target speed, signed target-plane crossing and action saturation retain baseline definitions. Record each episode's terminal reason and metrics, then each scenario/policy's success, mean final distance, successful completion time, mean episode length, crossing fraction, mean speed for distance <0.1 m, action saturation, failure and timeout counts. Undefined successful completion or near-target means are JSON null, not zero.

The runner writes isolated per-factor JSON parts and finally `mujoco/reports/ppo_baseline_v1_robustness.json`. Group statistics across exactly five policies use mean, sample standard deviation, min and max. Sensitivity figures show group mean success versus disturbance with spread, one uncluttered figure per factor. The report labels ranges by observed group mean >=90%, 70–<90%, <70%, and also reports worst-policy success so group mean does not conceal failures. The factor whose tested levels yield the largest drop versus nominal is identified empirically, not assumed.

## Verification and caveats

Unit tests first pin single-factor isolation, nominal equivalence, rotor speed lag math/reset, force frame/sign, waypoint identity, metric aggregation, null handling, and report preservation. A small physical probe checks mass-only acceleration under known force, inertia-only angular response, k_f-only actuator wrench and ±world-X external force. Re-run the existing RL test suite. Verify a fresh nominal audit matches the prior holdout evaluation for the same five checkpoints within numerical tolerance before running sweeps.

The 7 kg mass is confirmed, while COM/inertia, thrust mapping, rotor signs and controller are baseline estimates. This audit measures sensitivity of that simulator and policy; it does not certify real-flight robustness.
