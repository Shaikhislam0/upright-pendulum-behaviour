"""Fixed reward rescaling for optimization; physical rewards stay accessible."""

import math

import gymnasium as gym


class TrainingRewardScale(gym.Wrapper):
    """Multiply feedback to the learner by a positive constant.

    Put this outside Monitor so episode returns in Monitor/progress logs retain
    their original units. Never apply it to the evaluation environment.
    """

    def __init__(self, env, scale=1.0):
        if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                or not math.isfinite(scale) or scale <= 0):
            raise ValueError("training reward scale must be a finite positive number")
        super().__init__(env)
        self.scale = float(scale)

    def step(self, action):
        observation, raw_reward, terminated, truncated, info = self.env.step(action)
        reward = float(raw_reward) * self.scale
        if not math.isfinite(reward):
            raise FloatingPointError("scaled training reward is not finite")
        return observation, reward, terminated, truncated, {
            **info, "raw_reward": float(raw_reward), "training_reward": reward,
            "training_reward_scale": self.scale,
        }
