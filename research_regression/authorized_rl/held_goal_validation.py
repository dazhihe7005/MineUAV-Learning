"""Separate R28 evaluator: copied frozen L6 bookkeeping with explicit task wrapper.
No change to historical L6 evaluator, physics, controller, rewards or checkpoints.
"""
import copy,time,math
from pathlib import Path
import numpy as np
import torch
from research_ops.storage import digest
from lidar_integration.ppo_formal_validation import LidarPPOValidationEvaluator,_optimizer_states_equal,TERMINAL_OUTCOMES
from lidar_integration.ppo_formal_scheduler import require_memory
from lidar_integration.navigation_env import LidarNavigationEnv
from lidar_integration.ppo_preflight import HeadingObservation
from lidar_integration.ppo_scenarios import build_ppo_scene
from lidar_integration.ppo_training_runner import _stats,_weights_sha
from lidar_integration.terminal_accounting import classify_terminal
from lidar_integration.failure_evidence import NOT_AVAILABLE,transition_failure_context
from .held_goal_profile import held_goal_manifest
from .goal_heading_diagnostic import heading_task
from .goal_heading_trace import HoldInitialHeading

class HeldGoalValidationEvaluator(LidarPPOValidationEvaluator):
    def __init__(self,prereg,upstream_root,*,distance_stage,failure_recorder=None):
        if distance_stage not in ('SHORT','ORIGINAL'):raise ValueError('R28 unknown Validation distance stage')
        super().__init__(prereg,upstream_root,failure_recorder=failure_recorder)
        manifest=held_goal_manifest(prereg);self.distance_stage=distance_stage
        self.rows=tuple(copy.deepcopy(v['scenario']) for v in manifest['validation'][distance_stage])
        self._rows_sha256=digest(self.rows);self._manifest_sha=digest(manifest)

    def _check_observation(self,env,obs,info,task):
        plant=env.unwrapped
        if (not np.isfinite(plant.data.qpos).all() or not np.isfinite(plant.data.qvel).all()
                or not np.isfinite(plant.data.ctrl).all() or abs(plant.target_yaw-task['goal_yaw_rad'])>1e-12
                or info['goal_yaw_mode']!='HOLD_INITIAL_ENU_YAW' or abs(info['target_yaw_rad']-task['goal_yaw_rad'])>1e-12):
            raise RuntimeError('R28 Validation physical task/finite state changed')
        w,x,y,z=plant.data.qpos[3:7];yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        expected=math.atan2(math.sin(task['goal_yaw_rad']-yaw),math.cos(task['goal_yaw_rad']-yaw))
        if (np.asarray(obs['state']).shape!=(7,) or not np.isfinite(obs['state']).all()
                or abs(float(obs['state'][6])-expected)>1e-6):
            raise RuntimeError('R28 Validation observation has wrong declared task heading')

    def evaluate(self, model, *, max_policy_steps: int = 375) -> dict:
        if (isinstance(max_policy_steps, bool) or not isinstance(max_policy_steps, int)
                or not 1 <= max_policy_steps <= 375):
            raise ValueError("Validation probe budget must be 1..375 policy steps")
        if digest(self.rows)!=self._rows_sha256:raise ValueError('R28 frozen Validation rows changed')
        require_memory()
        policy_before = _weights_sha(model.policy)
        optimizer_before = copy.deepcopy(model.policy.optimizer.state_dict())
        training_steps_before = model.num_timesteps
        rows = []
        whole_started = time.perf_counter()
        for row in self.rows:
            task=heading_task(row)
            require_memory()
            scene = build_ppo_scene(row)
            if scene.geometry_hash != row["obstacle_configuration_hash"]:
                raise ValueError("Validation compiled geometry identity mismatch")
            env = HoldInitialHeading(HeadingObservation(LidarNavigationEnv(
                row["layout"], row["scene_seed"], self.upstream_root, scene=scene)))
            returned, classification, action = None, None, NOT_AVAILABLE
            completed_before, known_reward = 0, NOT_AVAILABLE
            policy_step = NOT_AVAILABLE
            try:
                obs, info = env.reset(seed=row["scene_seed"], options={
                    "start_seed": row["start_seed"], "target_seed": row["target_seed"],
                    "start_position": row["start_xyz"], "target_position": row["target_xyz"],
                    "start_yaw_rad": row["start_yaw_rad"]})
                if info["obstacle_configuration_hash"] != row["obstacle_configuration_hash"]:
                    raise RuntimeError("Validation reset scene identity changed")
                self._check_observation(env,obs,info,task)
                previous_position = env.unwrapped.data.qpos[:3].copy()
                path_length = 0.
                path_available = bool(np.isfinite(previous_position).all())
                rewards = 0.
                minimum_range = None
                ages, inference_ms = [], []
                final_info = None
                outcome = "probe_cutoff"
                for policy_step in range(1, max_policy_steps + 1):
                    returned, classification = None, None
                    completed_before = policy_step - 1
                    require_memory()
                    started = time.perf_counter()
                    with torch.inference_mode():
                        action, _ = model.predict(obs, deterministic=True)
                    inference_ms.append((time.perf_counter() - started) * 1000.)
                    if np.asarray(action).shape != (4,) or not np.isfinite(action).all():
                        raise RuntimeError("Validation model emitted invalid action")
                    returned = env.step(action)
                    obs, reward, terminated, truncated, final_info = returned
                    classification = classify_terminal(final_info, terminated, truncated)
                    self._check_observation(env,obs,final_info,task)
                    if not np.isfinite(reward):
                        raise RuntimeError("Validation returned nonfinite reward")
                    rewards += float(reward)
                    position = env.unwrapped.data.qpos[:3].copy()
                    path_available = path_available and bool(np.isfinite(position).all())
                    if path_available:
                        path_length += float(np.linalg.norm(position - previous_position))
                    previous_position = position
                    if final_info["lidar_age_s"] is not None:
                        ages.append(float(final_info["lidar_age_s"]))
                    frame = env.unwrapped.last_frame
                    if frame is not None and np.any(frame.valid_mask):
                        sample = float(np.min(frame.ranges[frame.valid_mask]))
                        minimum_range = sample if minimum_range is None else min(minimum_range, sample)
                    if terminated or truncated:
                        outcome = classification.terminal_class
                        break
                    known_reward = float(reward)
                if final_info is None:
                    raise RuntimeError("Validation did not execute a policy step")
                rows.append({
                    "scenario_identity_sha256": task["identity_sha256"],
                    "base_scenario_identity_sha256":row.get("base_identity_sha256",row["identity_sha256"]),
                    "distance_task_identity_sha256":row["identity_sha256"],
                    "task_goal_yaw_rad":task["goal_yaw_rad"],"goal_yaw_mode":"HOLD_INITIAL_ENU_YAW",
                    "layout": row["layout"], "outcome": outcome,
                    "terminal_class": classification.terminal_class,
                    "raw_termination_reason": classification.raw_termination_reason,
                    "success": outcome == "success", "collision": outcome == "collision",
                    "timeout": outcome == "time_limit",
                    "physical_failure": outcome == "physical_failure",
                    "sensor_failure": outcome == "sensor_failure",
                    "policy_steps": policy_step, "episode_return": rewards,
                    "trajectory_length_m": path_length if path_available else NOT_AVAILABLE,
                    "completion_time_s": float(env.unwrapped.data.time) if outcome == "success" else None,
                    "final_target_error_m": float(np.linalg.norm(
                        env.unwrapped.target_position - position))
                        if np.isfinite(position).all() else NOT_AVAILABLE,
                    "minimum_valid_lidar_range_m": minimum_range,
                    "mean_lidar_age_s": float(np.mean(ages)) if ages else None,
                    "inference_ms": _stats(inference_ms),
                    "collision_contact_pairs": final_info.get("contact_pairs", []),
                    "last_sim_time_s": float(env.unwrapped.data.time),
                })
            except BaseException as error:
                if self.failure_recorder is not None:
                    context = transition_failure_context(env, completed_before=completed_before,
                        action=action, known_reward=known_reward, returned=returned,
                        outcome=classification, scenario_identity=task["identity_sha256"], phase="VALIDATION")
                    context["policy_step_attempt"] = policy_step
                    try:
                        self.failure_recorder.capture(context, error)
                    except BaseException as capture_error:
                        raise RuntimeError(f"Validation failure evidence persistence failed: {capture_error}") from error
                raise
            finally:
                env.close()
        if _weights_sha(model.policy) != policy_before or model.num_timesteps != training_steps_before:
            raise RuntimeError("Validation mutated training model")
        # The optimizer is not invoked; compare serialized state-dict values,
        # including any tensor-valued moments, without consuming its RNG.
        optimizer_after = model.policy.optimizer.state_dict()
        if not _optimizer_states_equal(optimizer_before, optimizer_after):
            raise RuntimeError("Validation mutated PPO optimizer state")
        counts = {key: sum(row["outcome"] == key for row in rows)
                  for key in (*TERMINAL_OUTCOMES, "probe_cutoff")}
        complete = counts["probe_cutoff"] == 0
        rates = {
            "navigation_success_rate": counts["success"] / len(rows) if complete else None,
            "collision_rate": counts["collision"] / len(rows) if complete else None,
            "timeout_rate": counts["time_limit"] / len(rows) if complete else None,
            "physical_failure_rate": counts["physical_failure"] / len(rows) if complete else None,
            "sensor_failure_rate": counts["sensor_failure"] / len(rows) if complete else None,
        }
        times_to_goal = [row["completion_time_s"] for row in rows if row["success"]]
        available_paths = [row["trajectory_length_m"] for row in rows
                           if row["trajectory_length_m"] != NOT_AVAILABLE]
        return {
            "split": "validation", "goal_yaw_mode":"HOLD_INITIAL_ENU_YAW",
            "distance_stage":self.distance_stage,"held_goal_manifest_sha256":self._manifest_sha,
            "claim":"new heldheading tasks;not originalYaw0 success", "model_status": "UNTRAINED_FROZEN_INITIAL_POLICY"
            if training_steps_before == 0 else "FROZEN_MODEL_DIAGNOSTIC",
            "probe_only": not complete, "episodes": rows,
            "overall": {**counts, **rates, "total": len(rows),
                        "mean_return": float(np.mean([row["episode_return"] for row in rows])),
                        "mean_trajectory_length_m": float(np.mean(available_paths))
                            if available_paths else NOT_AVAILABLE,
                        "trajectory_length_available_count": len(available_paths),
                        "mean_time_to_goal_s": float(np.mean(times_to_goal)) if times_to_goal else None},
            "per_scene": {row["scenario_identity_sha256"]: dict(row) for row in rows},
            "policy_sha256_before_after": policy_before,
            "optimizer_unchanged": True,
            "training_timesteps_before_after": training_steps_before,
            "wall_seconds": time.perf_counter() - whole_started,
        }

def evaluate_held_validation(model,prereg,upstream_root,*,failure_recorder=None,max_policy_steps=375):
    from .centered_noise_training import annotate_noise_validation
    return {stage:annotate_noise_validation(HeldGoalValidationEvaluator(prereg,upstream_root,
        distance_stage=stage,failure_recorder=failure_recorder).evaluate(model,max_policy_steps=max_policy_steps))
        for stage in ('ORIGINAL','SHORT')}
