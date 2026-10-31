"""Alternative encodings of the same physical cart–pendulum state."""

from dataclasses import dataclass
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .physics import state_vector

ANGLE_REPRESENTATIONS = ("unwrapped", "wrapped", "sin_cos")


@dataclass(frozen=True)
class ObservationConfig:
    angle_representation: str = "sin_cos"
    position_scale: float = 1.0  # m
    cart_velocity_scale: float = 1.0  # m/s
    angular_velocity_scale: float = 5.0  # rad/s

    def __post_init__(self):
        if self.angle_representation not in ANGLE_REPRESENTATIONS:
            raise ValueError(f"angle_representation must be one of {ANGLE_REPRESENTATIONS}")
        for name in ("position_scale", "cart_velocity_scale", "angular_velocity_scale"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")


def encode_observation(state, config: ObservationConfig = ObservationConfig()):
    """Return five float32 inputs; raw physics always remains float64.

    Slots: scaled x, scaled v, angle feature A, angle feature B, scaled omega.
    Scalar-angle variants use a zero in slot B to match network dimensions.
    Scaling is fixed, with no clipping or running normalization.
    """
    x, v, theta, omega = state_vector(state)
    if config.angle_representation == "sin_cos":
        a, b = math.sin(theta), math.cos(theta)
    elif config.angle_representation == "wrapped":
        a, b = math.atan2(math.sin(theta), math.cos(theta)) / math.pi, 0.0
    else:
        a, b = theta / math.pi, 0.0
    with np.errstate(over="raise", invalid="raise"):
        observation = np.asarray([
            x / config.position_scale, v / config.cart_velocity_scale,
            a, b, omega / config.angular_velocity_scale,
        ], dtype=np.float32)
    if not np.all(np.isfinite(observation)):
        raise FloatingPointError("observation encoding produced a nonfinite value")
    return observation


class PolicyObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, config: ObservationConfig | None = None):
        if env.observation_space.shape != (4,):
            raise ValueError("expected raw four-state observations before encoding")
        super().__init__(env)
        self.config = config if config is not None else ObservationConfig()
        mode = self.config.angle_representation
        a_bound = np.inf if mode == "unwrapped" else 1.0
        b_bound = 1.0 if mode == "sin_cos" else 0.0
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -a_bound, -b_bound, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, a_bound, b_bound, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        return encode_observation(observation, self.config)
