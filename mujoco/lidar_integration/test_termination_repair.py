"""L9R: controlled MuJoCo terminal/failure checks; no PPO optimizer updates."""

from research_regression.upstream import resolve_upstream
from pathlib import Path

import mujoco
import numpy as np
import pytest

from .ppo_formal_scheduler import MultiScenarioTrainingEnv, load_l6_prereg
from .ppo_training_orchestration import TrainingMetricsEnv


UPSTREAM = resolve_upstream()
PREREG = Path(__file__).resolve().parents[1] / "reports/mujoco_lidar_ppo_training_prereg_seed0.json"


def test_forced_mujoco_boundary_failure_is_counted_without_l9_exception():
    """FORCED_TEST_STATE catches rejection of a valid physical-failure terminal."""
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler)
    try:
        env.reset()
        scheduler.active.unwrapped.data.qpos[2] = 100.
        mujoco.mj_forward(scheduler.active.unwrapped.model, scheduler.active.unwrapped.data)
        _, reward, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert terminated and not truncated
        assert info["terminal_class"] == "physical_failure"
        assert info["raw_termination_reason"] == "outside_flight_area"
        assert info["termination_reason"] == "outside_flight_area"
        assert env.terminal_counts["physical_failure"] == 1
        assert len(env.episode_returns) == 1 and env.episode_returns[0] == reward
    finally:
        env.close()


def _info(reason, flag):
    return {"termination_reason": reason,
            "success": flag == "success", "collision": flag == "collision",
            "timeout": flag == "time_limit", "physical_failure": flag == "physical_failure",
            "sensor_failure": flag == "sensor_failure"}


@pytest.mark.parametrize("raw, want, terminated, truncated", [
    ("success", "success", True, False),
    ("collision", "collision", True, False),
    ("time_limit", "time_limit", False, True),
    ("sensor_failure", "sensor_failure", True, False),
    ("physical_failure", "physical_failure", True, False),
    ("z_below_zero", "physical_failure", True, False),
    ("outside_flight_area", "physical_failure", True, False),
    ("excessive_tilt", "physical_failure", True, False),
    ("nonfinite_state", "physical_failure", True, False),
    (None, None, False, False),
])
def test_shared_contract_preserves_known_raw_reason(raw, want, terminated, truncated):
    from .terminal_accounting import classify_terminal
    info = _info(raw, want)
    before = dict(info)
    outcome = classify_terminal(info, terminated, truncated)
    assert outcome.terminal_class == want
    assert outcome.raw_termination_reason == raw
    assert info == before


@pytest.mark.parametrize("reason, flag, terminated, truncated", [
    ("unknown_motor_issue", "physical_failure", True, False),
    ("outside_flight_area", "collision", True, False),
    ("success", "success", False, False),
    ("success", "success", True, True),
    ("time_limit", "time_limit", True, False),
    ("collision", "collision", False, True),
    ("outside_flight_area", "physical_failure", False, True),
    ("sensor_failure", "sensor_failure", False, True),
    (None, None, True, False),
])
def test_shared_contract_rejects_unknown_or_inconsistent_terminal(reason, flag,
                                                                terminated, truncated):
    from .terminal_accounting import classify_terminal
    with pytest.raises(RuntimeError):
        classify_terminal(_info(reason, flag), terminated, truncated)


def test_shared_contract_rejects_multiple_or_missing_flags_and_spoofed_class():
    from .terminal_accounting import classify_terminal
    for changes in ({"collision": True}, {"success": 1},
                    {"terminal_class": "physical_failure"},
                    {"raw_termination_reason": "collision"}):
        info = {**_info("success", "success"), **changes}
        with pytest.raises(RuntimeError):
            classify_terminal(info, True, False)
    incomplete = _info("success", "success")
    incomplete.pop("sensor_failure")
    with pytest.raises(RuntimeError):
        classify_terminal(incomplete, True, False)


def _force_state(plant, raw):
    """FORCED_TEST_STATE, not a natural training failure or a flight trajectory."""
    if raw == "outside_flight_area":
        plant.data.qpos[2] = 100.
        mujoco.mj_forward(plant.model, plant.data)
    elif raw == "z_below_zero":
        # Pre-integration state invariant injection; retain the contact-free
        # reset contact set. A separate refreshed-contact test checks priority.
        assert plant.data.ncon == 0
        plant.data.qpos[2] = -1.
    elif raw == "excessive_tilt":
        plant.data.qpos[3:7] = [np.sqrt(.5), np.sqrt(.5), 0., 0.]
        mujoco.mj_forward(plant.model, plant.data)
    elif raw == "nonfinite_state":
        plant.data.ctrl[0] = np.nan
    elif raw == "collision":
        plant.data.qpos[:3] = plant.data.geom_xpos[plant.model.geom("nav_wall_x_pos").id]
        mujoco.mj_forward(plant.model, plant.data)
        assert plant.data.ncon > 0
    elif raw == "sensor_failure":
        plant.lidar_buffer.reset(plant._experiment_id)
    elif raw == "time_limit":
        plant.episode_steps = plant.max_episode_steps - 1
    elif raw == "success":
        plant.data.qpos[:3] = plant.target_position
        plant.data.qvel[:] = 0.
        plant.previous_distance = 0.
        mujoco.mj_forward(plant.model, plant.data)
    else:
        raise AssertionError(raw)


def _forced_transition(raw, *, measured):
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler) if measured else scheduler
    try:
        env.reset()
        _force_state(scheduler.active.unwrapped, raw)
        for _ in range(5 if raw == "success" else 1):
            transition = env.step(np.zeros(4, dtype=np.float32))
        stats = dict(env.terminal_counts) if measured else None
        return transition, stats
    finally:
        env.close()


@pytest.mark.parametrize("raw, want, expected_terminal, expected_truncated", [
    ("success", "success", True, False), ("collision", "collision", True, False),
    ("time_limit", "time_limit", False, True), ("sensor_failure", "sensor_failure", True, False),
    ("z_below_zero", "physical_failure", True, False),
    ("outside_flight_area", "physical_failure", True, False),
    ("excessive_tilt", "physical_failure", True, False),
    ("nonfinite_state", "physical_failure", True, False),
])
def test_forced_mujoco_all_terminals_preserve_reward_and_raw_reason(raw, want,
                                                                  expected_terminal, expected_truncated):
    # Sequential environments keep a single active MuJoCo instance.
    original, _ = _forced_transition(raw, measured=False)
    repaired, counts = _forced_transition(raw, measured=True)
    assert repaired[1:4] == original[1:4]
    assert repaired[2] == expected_terminal and repaired[3] == expected_truncated
    assert repaired[4]["reward_breakdown"] == original[4]["reward_breakdown"]
    for key in original[0]:
        np.testing.assert_allclose(repaired[0][key], original[0][key], atol=0, rtol=0, equal_nan=True)
    assert repaired[4]["raw_termination_reason"] == raw
    assert repaired[4]["termination_reason"] == raw
    assert repaired[4]["terminal_class"] == want
    assert counts[want] == 1 and sum(counts.values()) == 1
    assert sum(bool(repaired[4][key]) for key in
               ("success", "collision", "timeout", "physical_failure", "sensor_failure")) == 1


def test_refreshed_floor_contact_keeps_collision_priority_over_negative_altitude():
    scheduler = MultiScenarioTrainingEnv(load_l6_prereg(PREREG), UPSTREAM, seed=0)
    env = TrainingMetricsEnv(scheduler)
    try:
        env.reset()
        plant = scheduler.active.unwrapped
        plant.data.qpos[2] = -.05
        mujoco.mj_forward(plant.model, plant.data)
        assert plant.data.ncon > 0
        _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
        assert terminated and not truncated
        assert info["terminal_class"] == "collision"
        assert info["raw_termination_reason"] == "collision"
        assert info["physical_failure"] is False
    finally:
        env.close()
