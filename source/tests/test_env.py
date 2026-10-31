import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from upright_pendulum import CartPendulumEnv, ENV_ID


def test_gymnasium_contract():
    env = gym.make(ENV_ID, render_mode="ansi")
    check_env(env.unwrapped)
    obs, info = env.reset(seed=7)
    assert env.observation_space.contains(obs)
    assert isinstance(env.render(), str)
    env.close()


def test_default_initial_state_and_time_limit():
    env = CartPendulumEnv(max_episode_steps=2)
    obs, info = env.reset(seed=1)
    np.testing.assert_array_equal(obs, [0, 0, np.pi, 0])
    assert info["elapsed_time"] == 0
    assert env.step([0])[2:4] == (False, False)
    obs, reward, terminated, truncated, info = env.step([0])
    assert (reward, terminated, truncated) == (0.0, False, True)
    assert info["elapsed_time"] == 0.04
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0])


def test_rotations_are_unwrapped_and_never_angle_terminated():
    env = CartPendulumEnv(max_episode_steps=None)
    env.reset(options={"state": [0, 0, np.pi, 10]})
    angles = []
    for _ in range(100):
        obs, _, terminated, truncated, info = env.step([0])
        assert not terminated and not truncated
        assert -np.pi <= info["angle_error"] <= np.pi
        angles.append(obs[2])
    assert max(angles) > 2 * np.pi


def test_seeding_and_observation_ownership():
    env = CartPendulumEnv()
    options = {"noise_scale": [0.1, 0.2, 0.3, 0.4]}
    a, _ = env.reset(seed=42, options=options)
    b, _ = env.reset(seed=42, options=options)
    np.testing.assert_array_equal(a, b)
    c, _ = env.reset(seed=43, options=options)
    assert not np.array_equal(a, c)
    c[:] = 99
    assert not np.any(env.state == 99)
    state = env.state
    state[:] = 98
    assert not np.any(env.state == 98)


@pytest.mark.parametrize("action", [0, [0, 1], [np.nan], [np.inf], [10.01], [-10.01]])
def test_invalid_actions_are_rejected(action):
    env = CartPendulumEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step(action)


def test_force_endpoints_and_reset_needed():
    env = CartPendulumEnv()
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0])
    env.reset()
    assert env.step([10])[4]["applied_force"] == 10
    assert env.step([-10])[4]["applied_force"] == -10


@pytest.mark.parametrize("options", [
    {"state": [0, 0]}, {"state": [0, np.nan, 0, 0]},
    {"noise_scale": [-1, 0, 0, 0]}, {"noise_scale": 0.1}, {"typo": 1},
])
def test_invalid_reset_options(options):
    with pytest.raises(ValueError):
        CartPendulumEnv().reset(options=options)
