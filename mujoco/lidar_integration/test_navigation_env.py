"""Real MuJoCo obstacle-contact, timing, reward and missing-sensor regressions."""

from research_regression.upstream import resolve_upstream
import hashlib
import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import pytest

from lidar_integration.scene import BASE_MODEL, BASE_SHA256


UPSTREAM = resolve_upstream()


@pytest.fixture
def make_env():
    if not UPSTREAM.is_dir():
        pytest.skip("pinned upstream MuJoCo-LiDAR checkout unavailable")
    from lidar_integration.navigation_env import LidarNavigationEnv

    created = []

    def factory(layout="empty", seconds=15.):
        env = LidarNavigationEnv(layout=layout, scene_seed=1001,
                                 upstream_root=UPSTREAM,
                                 max_episode_seconds=seconds)
        created.append(env)
        return env

    yield factory
    for env in created:
        env.close()


def test_reset_and_step_use_old_controller_with_dict_observation_and_exact_rates(make_env):
    env = make_env()
    obs, info = env.reset(seed=0, options={"start_seed": 1101, "target_seed": 1201})
    assert isinstance(env, gym.Env)
    assert set(obs) == {"lidar", "state", "lidar_age"}
    assert env.observation_space.contains(obs)
    assert obs["lidar"].shape == (3, 12, 24)
    assert obs["state"].shape == (7,)
    assert obs["lidar_age"].shape == (1,)
    assert obs["lidar"].dtype == obs["state"].dtype == np.float32
    assert info["state_source"] == "SIM_GROUND_TRUTH_STATE"
    assert info["lidar_source_time_s"] == pytest.approx(0.)
    assert env.model.opt.timestep == pytest.approx(.002)
    assert env.control_dt == pytest.approx(.01)
    assert env.policy_dt == pytest.approx(.04)
    start = env.data.qpos.copy()
    obs, _, terminated, truncated, info = env.step(np.array([1., -1., .5, -.25]))
    assert not terminated and not truncated
    assert info["physics_steps"] == 20
    assert info["controller_updates"] == 4
    assert env.data.time == pytest.approx(.04)
    assert not np.array_equal(env.data.qpos, start)  # genuine mj_step, not pose replay
    np.testing.assert_allclose(info["velocity_command_m_s"], [1.5, -1.5, .5])
    assert info["yaw_rate_command_rad_s"] == pytest.approx(-.25)
    assert obs["lidar_age"][0] == pytest.approx(.04)
    assert info["lidar_source_time_s"] == pytest.approx(0.)
    assert env.observation_space.contains(obs)


def test_lidar_is_generated_at_physics_tick_50_and_never_from_future(make_env):
    env = make_env()
    env.reset(seed=0)
    assert env.last_frame.frame_timestamp == pytest.approx(0.)
    assert env.last_frame.ray_count == 24000
    rows = []
    for _ in range(5):
        obs, _, terminated, truncated, info = env.step(np.zeros(4))
        assert not terminated and not truncated
        rows.append((round(float(env.data.time), 2),
                     round(float(info["lidar_source_time_s"]), 2),
                     float(obs["lidar_age"][0])))
    assert [row[:2] for row in rows] == [(.04, 0.), (.08, 0.),
                                         (.12, .10), (.16, .10), (.20, .20)]
    assert all(0 <= age <= .15 for _, _, age in rows)
    assert env.lidar_frame_count == 3
    assert env.last_frame.frame_timestamp == pytest.approx(.2)


def test_real_contact_with_box_terminates_but_clearance_does_not(make_env):
    env = make_env("front_box")
    env.reset(seed=0)
    near_face = env.scene.obstacle_aabbs[0][0][0]
    env.data.qpos[:3] = [near_face - .18, 0., 1.]
    env.data.qvel[:] = 0
    mujoco.mj_forward(env.model, env.data)
    assert env._contact_kind() is None  # central half-width .12 + .06 clearance
    # Advance the original controller and full MuJoCo dynamics into the box.
    env.reset(seed=0)
    result = None
    for _ in range(180):
        result = env.step(np.array([1., 0., 0., 0.], dtype=np.float32))
        if result[2] or result[3]:
            break
    assert result is not None
    _, reward, terminated, truncated, info = result
    assert terminated and not truncated
    assert info["termination_reason"] == "collision"
    assert info["collision_kind"] == "obstacle"
    assert info["contact_pairs"]
    assert info["reward_breakdown"]["collision"] == -30.
    assert info["reward_breakdown"]["progress"] == 0.
    assert info["reward_breakdown"]["success"] == 0.
    assert reward < 0


def test_boundary_wall_contact_is_a_collision_not_a_free_flight_escape(make_env):
    env = make_env("empty")
    env.reset(seed=0)
    env.data.qpos[:3] = [env.scene.boundary_half_extent_m - .10, 0., 1.]
    env.data.qvel[:] = 0.
    mujoco.mj_forward(env.model, env.data)
    assert env._contact_kind() == "boundary"
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated
    assert info["collision"] and info["collision_kind"] == "boundary"
    assert info["reward_breakdown"]["progress"] == 0.


def test_real_contact_precedes_simultaneous_tilt_failure(make_env):
    env = make_env("empty")
    env.reset(seed=0)
    env.data.qpos[:3] = [env.scene.boundary_half_extent_m - .10, 0., 1.]
    env.data.qpos[3:7] = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0., 0.]
    mujoco.mj_forward(env.model, env.data)
    assert env._contact_kind() == "boundary"
    assert env._failure_reason() == "collision"


def test_reset_refuses_target_inside_a_physical_obstacle(make_env):
    env = make_env("front_box")
    center = [sum(bounds) / 2 for bounds in env.scene.obstacle_aabbs[0]]
    with pytest.raises(ValueError, match="target.*obstacle"):
        env.reset(seed=0, options={"target_position": center})


def test_success_timeout_and_physical_failure_remain_distinct(make_env):
    success_env = make_env("empty")
    success_env.reset(seed=0, options={"target_position": [0., 0., 1.]})
    for _ in range(4):
        _, _, terminated, truncated, _ = success_env.step(np.zeros(4))
        assert not terminated and not truncated
    _, _, terminated, truncated, info = success_env.step(np.zeros(4))
    assert terminated and not truncated
    assert info["termination_reason"] == "success"
    assert info["reward_breakdown"]["success"] == 10.

    timeout_env = make_env("empty", seconds=.08)
    timeout_env.reset(seed=0)
    timeout_env.step(np.zeros(4))
    _, _, terminated, truncated, info = timeout_env.step(np.zeros(4))
    assert not terminated and truncated
    assert info["termination_reason"] == "time_limit"
    assert info["reward_breakdown"]["timeout_or_time_cost"] == pytest.approx(-2.01)

    failure_env = make_env("empty")
    failure_env.reset(seed=0)
    failure_env.data.qpos[0] = 6.1
    _, _, terminated, truncated, info = failure_env.step(np.zeros(4))
    assert terminated and not truncated
    assert info["termination_reason"] == "outside_flight_area"
    assert info["reward_breakdown"]["physical_failure"] == -30.


def test_stale_scan_stops_before_another_action_and_reset_clears_cache_and_pi(make_env):
    env = make_env("empty")
    env.reset(seed=0)
    env.step(np.array([.3, 0., 0., 0.]))
    assert np.linalg.norm(env.velocity_controller.integral_error) > 0
    env.reset(seed=1)
    np.testing.assert_array_equal(env.velocity_controller.integral_error, [0., 0., 0.])
    assert env.lidar_frame_count == 1
    assert env.data.time == pytest.approx(0.)
    # Simulate an interrupted sensor producer; no old scan gets renewed.
    env._scan_due = lambda: None
    terminal = None
    for _ in range(6):
        terminal = env.step(np.zeros(4))
        if terminal[2] or terminal[3]:
            break
    assert terminal is not None
    obs, _, terminated, truncated, info = terminal
    assert terminated and not truncated
    assert info["termination_reason"] == "sensor_failure"
    assert info["lidar_status"] == "STALE"
    assert info["reward_breakdown"]["collision"] == 0.
    assert np.all(obs["lidar"][0] == 1.)
    assert np.all(obs["lidar"][1:] == 0.)
    with pytest.raises(RuntimeError):
        env.step(np.ones(4))


def test_physical_failure_is_not_relabelled_as_sensor_failure(make_env):
    from lidar_integration.async_observation import ObservationRead

    env = make_env("empty")
    env.reset(seed=0)
    fresh = env._read_lidar()
    stale = ObservationRead("STALE", None, .20, 0., 0., False)
    calls = iter((fresh, stale))
    env._read_lidar = lambda: next(calls)
    env.data.qpos[0] = 6.1
    _, _, terminated, truncated, info = env.step(np.zeros(4))
    assert terminated and not truncated
    assert info["termination_reason"] == "outside_flight_area"
    assert info["physical_failure"] and not info["sensor_failure"]
    assert info["reward_breakdown"]["physical_failure"] == -30.


def test_reward_v2_terms_and_terminal_exclusivity_are_numeric():
    from lidar_integration.navigation_env import navigation_reward

    reward, parts = navigation_reward(2., 1.9, 0., np.zeros(4), "running", 1)
    assert reward == pytest.approx(.99)
    assert parts["progress"] == pytest.approx(1.)
    assert parts["timeout_or_time_cost"] == pytest.approx(-.01)
    reward, parts = navigation_reward(2., 1., 0., np.zeros(4), "collision", 1)
    assert parts["progress"] == 0. and parts["success"] == 0.
    assert parts["collision"] == -30.
    assert reward == pytest.approx(-30.01)
    reward, parts = navigation_reward(.4, .3, 5., np.zeros(4), "running", 1)
    assert parts["brake"] == pytest.approx(-6.25)
    reward, parts = navigation_reward(.1, .05, 0., np.zeros(4), "success", 1)
    assert parts["success"] == 10.
    assert parts["collision"] == 0.
    assert reward == pytest.approx(10.49)
    assert all(np.isfinite(list(parts.values())))


def test_fixed_layout_collision_or_physical_failure_cannot_outscore_hover_timeout():
    from lidar_integration.navigation_env import navigation_reward

    # Across the fixed seed manifest d_initial<2m. Give the unsafe policy
    # the impossible best-case full 2m progress, zero braking/action cost,
    # and only one time-cost step. It must still score below 15s hovering.
    hover = -0.01 * 375 - 2.
    for outcome in ("collision", "physical_failure"):
        terminal, parts = navigation_reward(2., 0., 0., np.zeros(4), outcome, 1)
        assert parts["progress"] == 0.
        best_possible_unsafe_return = 20. + terminal
        assert best_possible_unsafe_return < hover


def test_old_physical_model_and_nominal_env_remain_unchanged(make_env):
    from mine_uav_env import MineUAVEnv

    assert hashlib.sha256(BASE_MODEL.read_bytes()).hexdigest() == BASE_SHA256
    old = MineUAVEnv(reward_version="v2")
    new = make_env("empty")
    try:
        old_obs, _ = old.reset(seed=43)
        new_obs, _ = new.reset(seed=43)
        assert old_obs.shape == (7,)
        assert new_obs["lidar"].shape == (3, 12, 24)
        assert old.model.ngeom < new.model.ngeom
        assert old.action_space.shape == new.action_space.shape == (4,)
        assert new.velocity_controller.ki_xy == .5
        assert new.velocity_controller.ki_z == .8
    finally:
        old.close()


def test_navigation_environment_import_does_not_load_ppo_trainer():
    command = [sys.executable, "-c",
               "import sys; import lidar_integration.navigation_env; "
               "assert 'stable_baselines3' not in sys.modules"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_gymnasium_environment_checker_accepts_deterministic_info(make_env):
    from gymnasium.utils.env_checker import check_env

    env = make_env("empty")
    check_env(env)
