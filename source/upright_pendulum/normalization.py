"""Optional SB3 observation normalization, with explicit evaluation freezing."""


def make_observation_normalizer(env, *, gamma=0.99):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    return VecNormalize(
        DummyVecEnv([lambda: env]), norm_obs=True, norm_reward=False,
        clip_obs=10.0, gamma=gamma, epsilon=1e-8,
    )


def load_frozen_observation_normalizer(path, env):
    """Load a trusted local statistics file; never update it during evaluation.

    The caller may use normalize_obs on individual observations without using
    vector stepping, retaining the original physical-state evaluation loop.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    normalizer = VecNormalize.load(str(path), DummyVecEnv([lambda: env]))
    normalizer.training = False
    normalizer.norm_reward = False
    if not normalizer.norm_obs:
        normalizer.close()
        raise ValueError("the statistics file does not enable observation normalization")
    return normalizer
