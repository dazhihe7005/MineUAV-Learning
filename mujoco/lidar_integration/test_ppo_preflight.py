"""Offline RED/GREEN checks for the LiDAR PPO training interface."""

import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from .ppo_preflight import HeadingObservation, LidarStateExtractor, make_ppo
from .ppo_scenarios import build_ppo_scene, make_scenario_manifest, path_exists
from .ppo_training_runner import (SMOKE_PROTOCOL, _StepTiming, protocol_hash,
                                  validate_smoke_budget, validated_training_diagnostics)
from .ppo_reward_audit import FIXED_FRONT_BOX_WAYPOINTS, _audit_action, audit_reward
from .ppo_preflight_analysis import build_report


class _DummyNav(gym.Env):
    action_space = spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
    observation_space = spaces.Dict({
        "lidar": spaces.Box(0, 1, shape=(3, 12, 24), dtype=np.float32),
        "state": spaces.Box(-100, 100, shape=(7,), dtype=np.float32),
        "lidar_age": spaces.Box(0, 100, shape=(1,), dtype=np.float32),
    })

    def __init__(self):
        self.yaw = 0.
        self.data = type("Data", (), {})()
        self.data.qpos = np.array([0., 0., 1., 1., 0., 0., 0.])

    def reset(self, *, seed=None, options=None):
        return self._obs(), {"state_source": "SIM_GROUND_TRUTH_STATE"}

    def step(self, action):
        return self._obs(), 0., False, False, {}

    def _obs(self):
        return {"lidar": np.ones((3, 12, 24), np.float32),
                "state": np.zeros(7, np.float32), "lidar_age": np.zeros(1, np.float32)}


@pytest.mark.parametrize("degrees,expected", [(0, (0, 1)), (90, (1, 0)),
                                               (-90, (-1, 0)), (180, (0, -1)),
                                               (-180, (0, -1))])
def test_heading_from_current_physical_quaternion(degrees, expected):
    source = _DummyNav()
    angle = math.radians(degrees)
    source.data.qpos[3:7] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
    wrapped = HeadingObservation(source)
    obs, info = wrapped.reset()
    np.testing.assert_allclose(obs["heading"], expected, atol=1e-6)
    assert info["state_source"] == "SIM_GROUND_TRUTH_STATE"


def test_cnn_forward_backward_and_ppo_distribution():
    env = HeadingObservation(_DummyNav())
    extractor = LidarStateExtractor(env.observation_space)
    sample = {key: torch.as_tensor(value[None, ...])
              for key, value in env.reset()[0].items()}
    features = extractor(sample)
    assert features.shape == (1, 74)
    features.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in extractor.parameters())
    model = make_ppo(env, n_steps=32, seed=0)
    action, _ = model.predict(env.reset()[0], deterministic=True)
    assert action.shape == (4,) and np.isfinite(action).all()
    assert np.max(np.abs(action)) <= 1
    obs_tensor, _ = model.policy.obs_to_tensor(env.reset()[0])
    values, log_prob, entropy = model.policy.evaluate_actions(
        obs_tensor, torch.zeros((1, 4)))
    assert values.shape == (1, 1)
    assert log_prob.shape == entropy.shape == (1,)
    assert all(torch.isfinite(x).all() for x in (values, log_prob, entropy))
    batch = {key: tensor.repeat(4, *([1] * (tensor.ndim - 1)))
             for key, tensor in sample.items()}
    assert extractor(batch).shape == (4, 74)
    batch_value, batch_log_prob, batch_entropy = model.policy.evaluate_actions(
        batch, torch.zeros((4, 4)))
    assert batch_value.shape == (4, 1)
    assert batch_log_prob.shape == batch_entropy.shape == (4,)
    assert all(torch.isfinite(x).all() for x in (batch_value, batch_log_prob, batch_entropy))


def test_diverse_scenario_split_is_feasible_and_sealed():
    first = make_scenario_manifest()
    assert first == make_scenario_manifest()
    assert {key: len(rows) for key, rows in first.items()} == {
        "train": 12, "validation": 6, "test": 6}
    identities = [row["identity_sha256"] for rows in first.values() for row in rows]
    assert len(set(identities)) == 24
    assert len({row["scene_seed"] for rows in first.values() for row in rows}) == 24
    train_positions = {tuple(row["start_xyz"]) for row in first["train"]}
    assert len(train_positions) > 1
    assert len({tuple(row["obstacle_variation"][0]) for row in first["train"]
                if row["obstacle_variation"]}) > 1
    for row in first["train"]:
        scene = build_ppo_scene(row)
        assert path_exists(scene, row["start_xyz"], row["target_xyz"])
        assert not scene.overlaps_obstacle(row["start_xyz"], clearance_m=.20)
        assert not scene.overlaps_obstacle(row["target_xyz"], clearance_m=.20)


def test_scenario_uses_actual_reset_pose_not_legacy_endpoint_sampler():
    from .navigation_env import LidarNavigationEnv
    row = make_scenario_manifest()["train"][0]
    env = LidarNavigationEnv(row["layout"], row["scene_seed"],
                             "/tmp/mineuav-mujoco-lidar-audit", scene=build_ppo_scene(row))
    try:
        _, info = env.reset(seed=0, options={"start_position": row["start_xyz"],
                                            "target_position": row["target_xyz"],
                                            "start_yaw_rad": row["start_yaw_rad"]})
        np.testing.assert_allclose(env.data.qpos[:3], row["start_xyz"])
        assert info["obstacle_configuration_hash"] == row["obstacle_configuration_hash"]
    finally:
        env.close()


def test_smoke_budget_and_config_are_frozen():
    assert SMOKE_PROTOCOL["seed"] == 0
    assert SMOKE_PROTOCOL["workers"] == 1
    assert SMOKE_PROTOCOL["total_timesteps"] == 512
    assert SMOKE_PROTOCOL["n_steps"] == 128
    assert len(protocol_hash(SMOKE_PROTOCOL)) == 64
    validate_smoke_budget(SMOKE_PROTOCOL)
    with pytest.raises(ValueError):
        validate_smoke_budget({**SMOKE_PROTOCOL, "total_timesteps": 513})
    with pytest.raises(ValueError):
        validate_smoke_budget({**SMOKE_PROTOCOL, "workers": 2})
    assert validated_training_diagnostics({"train/value_loss": 1.}) == {
        "train/value_loss": 1.}
    with pytest.raises(ValueError, match="nonfinite training diagnostic"):
        validated_training_diagnostics({"train/value_loss": float("nan")})


def test_actor_critic_optimizer_and_checkpoint_roundtrip(tmp_path):
    env = HeadingObservation(_DummyNav())
    model = make_ppo(env, n_steps=32, seed=0)
    obs = env.reset()[0]
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    before_actor = model.policy.action_net.weight.detach().clone()
    before_critic = model.policy.value_net.weight.detach().clone()
    values, log_prob, entropy = model.policy.evaluate_actions(
        obs_tensor, torch.zeros((1, 4)))
    loss = values.square().mean() - log_prob.mean() - .01 * entropy.mean()
    model.policy.optimizer.zero_grad()
    loss.backward()
    assert torch.isfinite(model.policy.action_net.weight.grad).all()
    assert torch.isfinite(model.policy.value_net.weight.grad).all()
    model.policy.optimizer.step()
    assert not torch.equal(before_actor, model.policy.action_net.weight)
    assert not torch.equal(before_critic, model.policy.value_net.weight)
    path = tmp_path / "SMOKE_ONLY_roundtrip"
    model.save(path)
    restored = type(model).load(path.with_suffix(".zip"), env=env, device="cpu")
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                  restored.predict(obs, deterministic=True)[0])


def test_reward_evidence_hash_and_component_validation(tmp_path):
    import json
    raw = tmp_path / "front_box.npz"
    np.savez(raw, reward=np.array([-29.]),
             reward_component_progress=np.array([1.]),
             reward_component_action=np.array([0.]),
             reward_component_brake=np.array([0.]),
             reward_component_success=np.array([0.]),
             reward_component_collision=np.array([-30.]),
             reward_component_timeout_or_time_cost=np.array([0.]),
             reward_component_physical_failure=np.array([0.]),
             reward_component_sensor_failure=np.array([0.]))
    # A tampered historical trace must be rejected before any interpretation.
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"reward": {"version": "NAV_V2_COLLISION_PREFLIGHT_V1"},
                                  "raw_local_only": [{"local_only_path": raw.name,
                                                      "sha256": "0" * 64}],
                                  "episodes": []}))
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_reward(report)


def test_historical_reward_evidence_is_hash_verified_and_limited():
    report = Path(__file__).resolve().parents[1] / "reports" / "mujoco_lidar_navigation_env.json"
    if not (report.parent / "mujoco_lidar_navigation_env_parts/preflight_run_01/front_box.npz").exists():
        pytest.skip("Phase L5 raw trajectories are local-only and not in a clean clone")
    result = audit_reward(report)
    assert result["reward_version"] == "NAV_V2_COLLISION_PREFLIGHT_V1"
    assert result["fixed_small_scene_ordering"]
    assert result["same_scene_three_policy_comparison"] == "NOT TESTED"
    assert result["A_ideal_hover_math_not_executed"]["return"] == -5.75


def test_smoke_monitor_counts_physics_across_episode_reset_and_fails_on_low_memory(monkeypatch):
    import lidar_integration.ppo_training_runner as runner

    class Fake(gym.Env):
        action_space = _DummyNav.action_space
        observation_space = _DummyNav.observation_space

        def __init__(self):
            self.lidar_frame_count = 0
            self.last_scan_ms = 0.
            self.last_encode_ms = 0.

        def reset(self, *, seed=None, options=None):
            return _DummyNav()._obs(), {}

        def step(self, action):
            return _DummyNav()._obs(), 0., True, False, {
                "physics_steps": 20, "termination_reason": "collision"}

    monkeypatch.setattr(runner, "_available_mib", lambda: 2048.)
    env = _StepTiming(Fake())
    env.reset()
    env.step(np.zeros(4))
    env.reset()
    env.step(np.zeros(4))
    assert env.physics_ticks_total == 40
    monkeypatch.setattr(runner, "_available_mib", lambda: 1024.)
    with pytest.raises(MemoryError):
        env.step(np.zeros(4))


def test_smoke_report_rechecks_prereg_and_checkpoint_hashes():
    root = Path(__file__).resolve().parents[1] / "reports"
    parts = root / "mujoco_lidar_ppo_training_preflight_parts" / "smoke_run_01"
    smoke = parts / "smoke_summary.json"
    prereg = root / "mujoco_lidar_ppo_training_prereg_seed0.json"
    checkpoint = parts / "SMOKE_ONLY_lidar_ppo_seed0.zip"
    l5 = root / "mujoco_lidar_navigation_env.json"
    audit = root / "mujoco_lidar_ppo_training_preflight_parts" / "reward_same_scene_01" / "audit_summary.json"
    if not all(item.exists() for item in (smoke, prereg, checkpoint)):
        pytest.skip("one-shot smoke evidence is local-only in a clean clone")
    result = build_report(smoke, prereg, checkpoint, l5, audit if audit.exists() else None)
    assert result["status"]["bounded_training_smoke"] == "PASS"
    assert result["status"]["obstacle_avoidance_learned"] == "NOT TESTED"
    assert result["smoke"]["physics_ticks_total"] == 10240
    if audit.exists():
        assert result["status"]["same_scene_ABC_reward_trajectory_comparison"] == "PASS"
        assert result["same_scene_reward_audit"]["observed_ordering_C_gt_A_gt_B"]


def test_reward_audit_actions_are_fixed_and_nonlearning():
    assert FIXED_FRONT_BOX_WAYPOINTS == ((.45, .85, 1.), (1.45, .85, 1.))
    state = np.array([1.8, 0., 0., 0., 0., 0., 0.], dtype=np.float32)
    position = np.array([0., 0., 1.])
    zero, stage = _audit_action("A_zero_command", state, position, 0)
    np.testing.assert_array_equal(zero, np.zeros(4))
    assert stage == 0
    direct, _ = _audit_action("B_obstacle_blind_direct", state, position, 0)
    detour, _ = _audit_action("C_privileged_fixed_waypoint_detour", state, position, 0)
    assert direct[0] > 0 and abs(direct[1]) < 1e-6
    assert detour[0] > 0 and detour[1] > 0
