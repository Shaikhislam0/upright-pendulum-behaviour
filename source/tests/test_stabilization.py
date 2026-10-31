import math

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from upright_pendulum import (
    CartPendulumEnv, StabilizationCriteria, StabilizationMonitor, StabilizationWrapper,
)
from upright_pendulum.physics import integrate

UP = [0, 0, 0, 0]
DOWN = [0, 0, math.pi, 0]


def feed(monitor, end_index, state_at_index):
    for index in range(end_index + 1):
        monitor.observe(index * 0.002, state_at_index(index))


def test_all_four_conditions_required_and_boundaries_inclusive():
    c = StabilizationCriteria()
    assert c.contains([c.position, -c.velocity, c.angle, -c.angular_velocity])
    limits = [c.position, c.velocity, c.angle, c.angular_velocity]
    for coordinate, limit in enumerate(limits):
        for sign in (-1, 1):
            state = np.zeros(4)
            state[coordinate] = sign * limit * 1.001
            assert not c.contains(state)
    assert not c.contains(DOWN)


def test_upright_after_multiple_rotations():
    c = StabilizationCriteria()
    for turns in (-10, -2, 0, 3, 10):
        assert c.contains([0, 0, turns * 2 * math.pi, 0])


def test_hold_requires_elapsed_intervals_and_first_time_is_sticky():
    monitor = StabilizationMonitor()
    feed(monitor, 999, lambda _: UP)
    assert monitor.success_time is None  # 1000 samples span only 1.998 seconds
    monitor.observe(2.0, UP)
    assert monitor.success_time == 2.0
    monitor.observe(2.002, UP)
    assert monitor.success_time == 2.0


def test_records_completion_not_entry():
    monitor = StabilizationMonitor()
    feed(monitor, 3150, lambda i: UP if i >= 2150 else DOWN)
    assert monitor.entry_time == pytest.approx(4.3)
    assert monitor.success_time == pytest.approx(6.3)


def test_single_substep_violation_resets_whole_hold():
    monitor = StabilizationMonitor()
    # A 2-ms violation between control observations at 1.000 and 1.020 seconds.
    feed(monitor, 1503, lambda i: DOWN if i == 503 else UP)
    assert monitor.entry_time == pytest.approx(1.008)
    assert monitor.success_time is None
    monitor.observe(3.008, UP)
    assert monitor.success_time == pytest.approx(3.008)


def test_leaving_after_success_preserves_event_but_records_loss():
    monitor = StabilizationMonitor()
    feed(monitor, 1100, lambda i: UP if i <= 1000 else DOWN)
    result = monitor.summary()
    assert result["success_time"] == 2.0
    assert result["first_post_success_exit_time"] == pytest.approx(2.002)
    assert not result["in_region"]
    assert not result["right_censored"]
    assert result["observed_time"] == 2.0
    assert result["followup_time"] == 2.2


def test_censoring_and_monitor_reset():
    monitor = StabilizationMonitor()
    feed(monitor, 10000, lambda _: DOWN)
    result = monitor.summary()
    assert result["right_censored"] and not result["event_observed"]
    assert result["success_time"] is None
    assert result["observed_time"] == result["followup_time"] == 20
    monitor.reset()
    assert monitor.success_time is None and monitor.last_time is None
    with pytest.raises(RuntimeError):
        monitor.summary()


def test_event_completed_exactly_at_horizon_counts():
    monitor = StabilizationMonitor()
    feed(monitor, 10000, lambda i: UP if i >= 9000 else DOWN)
    assert monitor.success_time == 20
    assert monitor.summary()["event_observed"]
    shorter = StabilizationMonitor()
    feed(shorter, 10000, lambda i: UP if i >= 9001 else DOWN)
    assert shorter.summary()["right_censored"]


def test_gaps_duplicate_times_and_invalid_samples_are_rejected():
    monitor = StabilizationMonitor()
    for time in (0.1, -1, math.nan):
        with pytest.raises(ValueError):
            monitor.observe(time, UP)
    monitor.observe(0, UP)
    for time in (0, -0.002, 0.004):
        with pytest.raises(ValueError):
            monitor.observe(time, UP)
    with pytest.raises(ValueError):
        monitor.observe(0.002, [0, 0, math.nan, 0])
    assert monitor.last_time == 0


@pytest.mark.parametrize("field", ["position", "velocity", "angle", "angular_velocity", "hold_time"])
def test_invalid_criteria(field):
    for value in (0, -1, math.inf, math.nan):
        with pytest.raises(ValueError):
            StabilizationCriteria(**{field: value})


def test_observer_does_not_change_physics_or_leak_state():
    initial = [0, 0, 2.4, 1.3]
    expected = integrate(initial, 1.7, 0.02)
    times = []

    def observer(time, state):
        times.append(time)
        state[:] = 999  # Mutating the observed copy cannot change integration.

    actual = integrate(initial, 1.7, 0.02, observer=observer)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(times, np.arange(1, 11) * 0.002, atol=1e-16)


def test_wrapper_preserves_trajectory_and_control_timing():
    base = CartPendulumEnv(max_episode_steps=10)
    observed = StabilizationWrapper(CartPendulumEnv(max_episode_steps=10))
    base.reset(seed=10)
    observed.reset(seed=10)
    for action in np.linspace(-10, 10, 10):
        ordinary = base.step([action])
        monitored = observed.step([action])
        np.testing.assert_array_equal(ordinary[0], monitored[0])
        assert ordinary[1:4] == monitored[1:4]
        assert monitored[4]["elapsed_time"] == ordinary[4]["elapsed_time"]
    assert monitored[4]["episode_measurement"]["followup_time"] == 0.2


def test_wrapper_success_does_not_end_episode_and_reset_clears_event():
    env = StabilizationWrapper(CartPendulumEnv(max_episode_steps=101))
    env.reset(options={"state": UP})
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step([0])
    assert info["stabilization"]["success_time"] == 2
    assert (reward, terminated, truncated) == (0.0, False, False)
    _, _, _, truncated, info = env.step([0])
    assert truncated and info["episode_measurement"]["event_observed"]
    _, info = env.reset()
    assert info["stabilization"]["success_time"] is None


def test_wrapped_environment_gymnasium_contract():
    # The measurement wrapper must retain the Gymnasium reset/step contract.
    env = StabilizationWrapper(CartPendulumEnv())
    with pytest.warns(UserWarning):
        check_env(env, skip_render_check=True)
    env.close()


def test_detection_resolution_agrees_when_substep_is_halved():
    records = []
    for step in (0.002, 0.001):
        env = StabilizationWrapper(CartPendulumEnv(integration_dt=step, max_episode_steps=200))
        env.reset(options={"state": [0, 0, 1e-6, 0]})
        for _ in range(200):
            _, _, _, _, info = env.step([0])
        records.append(info["episode_measurement"])
    assert all(r["success_time"] == 2.0 for r in records)
    assert all(r["first_post_success_exit_time"] is not None for r in records)
    assert abs(records[0]["first_post_success_exit_time"]
               - records[1]["first_post_success_exit_time"]) <= 0.002 + 1e-12
