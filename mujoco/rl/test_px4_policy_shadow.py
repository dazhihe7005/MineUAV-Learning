"""Offline contracts for frozen PI-policy inference and PX4-format commands."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from px4_policy_shadow import (
    ActionAdapter,
    ObservationGate,
    FrozenPolicy,
    build_observation,
    shadow_step,
)
from px4_policy_shadow_analysis import read_cohort, overall_status, require_deterministic


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "rl" / "models"


def test_build_observation_world_error_and_wrapped_target_minus_yaw():
    obs = build_observation([2., -1., 3.], [0.2, -0.3, 0.4],
                            math.radians(-179), [3., 1., 2.], math.radians(179))
    np.testing.assert_allclose(obs[:6], [1., 2., -1., 0.2, -0.3, 0.4], atol=1e-7)
    assert obs[6] == pytest.approx(math.radians(-2), abs=1e-7)


def test_observation_gate_rejects_stale_repeated_and_impossible_jump():
    gate = ObservationGate(max_age_s=.12, max_yaw_rate_rad_s=4.)
    obs = np.zeros(7)
    np.testing.assert_array_equal(gate.accept(obs, 1_000_000, 1_080_000), obs)
    with pytest.raises(ValueError, match="timestamp"):
        gate.accept(obs, 1_000_000, 1_080_000)
    with pytest.raises(ValueError, match="stale"):
        gate.accept(obs, 1_040_000, 1_200_001)
    jump = obs.copy(); jump[6] = 1.
    with pytest.raises(ValueError, match="yaw"):
        gate.accept(jump, 1_040_000, 1_050_000)
    np.testing.assert_array_equal(gate.accept(obs, 1_040_000, 1_050_000), obs)


def test_observation_gate_handles_pi_wrap_without_false_jump():
    gate = ObservationGate(max_age_s=.12, max_yaw_rate_rad_s=4.)
    left = np.zeros(7); left[6] = math.pi - .03
    right = np.zeros(7); right[6] = -math.pi + .03
    gate.accept(left, 1_000_000, 1_000_000)
    gate.accept(right, 1_040_000, 1_040_000)


@pytest.mark.parametrize("bad", [np.zeros(6), np.full(7, np.nan), np.r_[np.zeros(6), 4.]])
def test_observation_gate_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        ObservationGate().accept(bad, 1, 1)


def test_action_adapter_scales_then_rotates_enu_to_ned_and_yaw_sign():
    adapter = ActionAdapter()
    command = adapter.convert([1., -.5, .25, -.75], 1_000_000, 1_000_000)
    np.testing.assert_allclose(command.velocity_ned_m_s, [-.75, 1.5, -.25])
    assert command.yaw_rate_ned_rad_s == pytest.approx(.75)
    np.testing.assert_array_equal(adapter.convert([0., 0., 0., 0.], 1, 1).velocity_ned_m_s, [0., 0., 0.])


@pytest.mark.parametrize("action", [[1.01, 0, 0, 0], [0, -1.01, 0, 0], [0, 0, 2, 0], [0, 0, 0, math.nan], [0, 0, 0]])
def test_action_adapter_rejects_unsafe_or_malformed_action(action):
    with pytest.raises(ValueError):
        ActionAdapter().convert(action, 1, 1)


def test_action_adapter_rejects_stale_state():
    with pytest.raises(ValueError, match="stale"):
        ActionAdapter().convert([0, 0, 0, 0], 1_000_000, 1_120_001)


@pytest.mark.parametrize("name,kind,sha", [
    ("uav_bc_mlp_seed0.pt", "bc", "7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c"),
    ("uav_bc_yaw_augmented_seed0.pt", "bc", "d29b93555ef46f231228c23345761f76b5758f6451648375fcc2b7e4501f03ce"),
    ("uav_ppo_fixed_protocol_seed0.zip", "ppo", "9a1e211611436d058a9fafe218c8f0e965572dfe6389c4dff869a9dcedcfd49b"),
    ("uav_ppo_fixed_protocol_seed1.zip", "ppo", "96d66efd3d1a943d6320d1013abb20cb429cffb1f992ecbb24e94775e91a9a33"),
    ("uav_ppo_fixed_protocol_seed2.zip", "ppo", "61da0c0f000f544f6963254a1f4c3aa086161592988900ff0d4bfd0a3d2e737d"),
    ("uav_ppo_fixed_protocol_seed3.zip", "ppo", "0f0bddee3c711b925b7a9ac7336d23ca47afb250196773a3bf6cda40851c8cd6"),
    ("uav_ppo_fixed_protocol_seed4.zip", "ppo", "80224e40f26faf1802388d9f9f264e5ec120bb40cfdc9efe91c77f921d6a5125"),
])
def test_frozen_checkpoint_identity_determinism_and_no_parameter_mutation(name, kind, sha):
    path = MODELS / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    policy = FrozenPolicy.load(path, kind=kind, expected_sha256=sha)
    before = policy.parameter_hash()
    obs = np.array([.2, -.4, .1, .3, -.2, .1, .01], dtype=np.float32)
    first = policy.predict(obs)
    second = policy.predict(obs)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (4,) and np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1
    assert policy.parameter_hash() == before
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha


def test_bad_checkpoint_hash_fails_before_deserialization():
    with pytest.raises(ValueError, match="SHA256"):
        FrozenPolicy.load(MODELS / "uav_bc_mlp_seed0.pt", kind="bc", expected_sha256="0" * 64)


def test_bc_uses_checkpoint_stats_while_ppo_receives_raw_7d():
    bc = FrozenPolicy.load(MODELS / "uav_bc_mlp_seed0.pt", kind="bc",
                           expected_sha256="7d7cf248c1fc672ec859bb906f4e8e0f442fbee9c6daea8fec54d4eee64c772c")
    ppo = FrozenPolicy.load(MODELS / "uav_ppo_fixed_protocol_seed0.zip", kind="ppo",
                            expected_sha256="9a1e211611436d058a9fafe218c8f0e965572dfe6389c4dff869a9dcedcfd49b")
    obs = np.array([.1, -.2, .05, .3, 0., -.1, .2], dtype=np.float32)
    np.testing.assert_array_equal(bc.predict(obs), bc.model.predict(obs))
    np.testing.assert_array_equal(ppo.predict(obs), ppo.model.predict(obs, deterministic=True)[0])
    assert bc.model.stats["provenance"] == "train only"
    assert not hasattr(ppo.model, "obs_rms")


def test_shadow_step_returns_commands_without_any_sender():
    class FixedPolicy:
        def predict(self, obs):
            assert obs.shape == (7,)
            return np.array([.5, 0, -.25, .1])
    result = shadow_step({"fixed": FixedPolicy()}, np.zeros(7), 1_000_000, 1_000_000,
                         gate=ObservationGate())
    np.testing.assert_allclose(result["fixed"].velocity_ned_m_s, [0., .75, .25])
    assert result["fixed"].yaw_rate_ned_rad_s == pytest.approx(-.1)


def test_shadow_step_enforces_causal_gate_on_each_call():
    class FixedPolicy:
        def predict(self, obs):
            return np.zeros(4)
    gate = ObservationGate()
    shadow_step({"fixed": FixedPolicy()}, np.zeros(7), 1_000_000, 1_000_000, gate=gate)
    with pytest.raises(ValueError, match="timestamp"):
        shadow_step({"fixed": FixedPolicy()}, np.zeros(7), 1_000_000, 1_010_000, gate=gate)
    impossible = np.zeros(7); impossible[6] = 1.
    with pytest.raises(ValueError, match="yaw"):
        shadow_step({"fixed": FixedPolicy()}, impossible, 1_040_000, 1_040_000, gate=gate)


def test_replay_cohort_checks_trace_sha_and_observation_action_alignment(tmp_path):
    path = tmp_path / "fresh_000_scripted.npz"
    np.savez(path, observations=np.array([[.2, 0, 0, 0, 0, 0, 0],
                                          [.1, 0, 0, 0, 0, 0, 0]], dtype=np.float32),
             actions=np.array([[.1, 0, 0, 0]], dtype=np.float32),
             physical_state_valid=np.array([True, True]))
    row = {"trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "controller": "scripted", "split": "fresh", "key": "fresh/000",
           "target_id": "ppo-bc-nominal-2026101501-000"}
    row["record_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()
    (tmp_path / "fresh_000_scripted.json").write_text(json.dumps(row))
    obs, actions, episode, sample_time_us, counts = read_cohort(1, tmp_path)
    assert counts["accepted"] == 1
    np.testing.assert_array_equal(obs[0], np.array([.2, 0, 0, 0, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(actions[0], np.array([.1, 0, 0, 0], dtype=np.float32))
    np.testing.assert_array_equal(episode, [0])
    np.testing.assert_array_equal(sample_time_us, [0])
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="identity"):
        read_cohort(1, tmp_path)


def test_replay_cohort_rejects_mutated_record_even_when_trace_is_unchanged(tmp_path):
    path = tmp_path / "fresh_000_scripted.npz"
    np.savez(path, observations=np.zeros((2, 7), np.float32),
             actions=np.zeros((1, 4), np.float32), physical_state_valid=np.ones(2, bool))
    row = {"trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "controller": "scripted", "split": "fresh", "key": "fresh/000",
           "target_id": "ppo-bc-nominal-2026101501-000"}
    row["record_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()
    row["target_id"] = "unrelated-target"
    (tmp_path / "fresh_000_scripted.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="identity"):
        read_cohort(1, tmp_path)


def test_replay_cohort_rejects_unpinned_trace_set_even_with_valid_local_hashes(tmp_path):
    path = tmp_path / "fresh_000_scripted.npz"
    np.savez(path, observations=np.zeros((2, 7), np.float32),
             actions=np.zeros((1, 4), np.float32), physical_state_valid=np.ones(2, bool))
    row = {"trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "controller": "scripted", "split": "fresh", "key": "fresh/000",
           "target_id": "ppo-bc-nominal-2026101501-000"}
    row["record_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()
    (tmp_path / "fresh_000_scripted.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="pinned cohort"):
        read_cohort(1, tmp_path, expected_digest="0" * 64)


def test_replay_cohort_fails_closed_on_yaw_discontinuity(tmp_path):
    path = tmp_path / "fresh_000_scripted.npz"
    obs = np.zeros((3, 7), np.float32)
    obs[1, 6] = 1.
    np.savez(path, observations=obs, actions=np.zeros((2, 4), np.float32),
             physical_state_valid=np.ones(3, bool))
    row = {"trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "controller": "scripted", "split": "fresh", "key": "fresh/000",
           "target_id": "ppo-bc-nominal-2026101501-000"}
    row["record_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()
    (tmp_path / "fresh_000_scripted.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="validity gate rejection"):
        read_cohort(1, tmp_path)


def test_overall_preflight_cannot_pass_when_required_cpu_timing_fails():
    checks = {"model_load_identity": "PASS", "observation_action_mapping": "PASS",
              "shadow_inference_no_transmission": "PASS", "cpu_25hz_single_inference": "FAIL",
              "px4_offboard_control": "NOT TESTED"}
    assert overall_status(checks) == "FAIL"
    checks["cpu_25hz_single_inference"] = "PASS"
    assert overall_status(checks) == "PASS"


def test_shadow_report_rejects_nondeterministic_policy():
    class AlternatingPolicy:
        def __init__(self):
            self.calls = 0
        def predict(self, obs):
            self.calls += 1
            return np.array([self.calls % 2, 0, 0, 0], dtype=np.float32)
    with pytest.raises(ValueError, match="nondeterministic"):
        require_deterministic(AlternatingPolicy(), np.zeros(7))
