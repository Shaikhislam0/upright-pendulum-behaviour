"""Policy-command adapter: normalized commands map to actual force."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class NormalizedForceWrapper(gym.ActionWrapper):
    """Map a in [-1, 1] to F = F_max * a without silently clipping inputs.

    SB3's PPO clips its sampled Gaussian actions to this declared space before
    stepping the environment. The physical environment still accepts newtons.
    """

    def __init__(self, env):
        super().__init__(env)
        original = env.action_space
        if (not isinstance(original, spaces.Box) or original.shape != (1,)
                or not np.all(np.isfinite(original.low))
                or not np.all(np.isfinite(original.high))
                or original.high[0] <= 0 or original.low[0] != -original.high[0]):
            raise ValueError("expected one finite, symmetric force action interval")
        self.force_limit = float(original.high[0])
        self.force_dtype = original.dtype
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def action(self, action):
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (1,) or not np.all(np.isfinite(command)) or np.any(np.abs(command) > 1):
            raise ValueError("normalized action must have shape (1,) and lie in [-1, 1]")
        return (command * self.force_limit).astype(self.force_dtype)
