"""Configurable dense reward, separate from physical dynamics and success."""

from dataclasses import dataclass, fields
import math

import gymnasium as gym

from .physics import state_vector


@dataclass(frozen=True)
class RewardConfig:
    """Provisional candidate weights, not empirically optimized coefficients.

    Scales carry SI units; penalties are dimensionless. Reward is evaluated
    at the resulting state with the force applied during that control step.
    """

    position_penalty: float = 0.10
    cart_velocity_penalty: float = 0.01
    angular_velocity_penalty: float = 0.01
    force_penalty: float = 0.001
    position_scale: float = 1.0  # m
    cart_velocity_scale: float = 1.0  # m/s
    angular_velocity_scale: float = 5.0  # rad/s
    force_scale: float = 10.0  # N
    form: str = "quadratic"  # Historical baseline, or the declared bounded candidate.

    def __post_init__(self):
        if self.form not in ("quadratic", "bounded"):
            raise ValueError("reward form must be quadratic or bounded")
        for field in fields(self):
            if field.name == "form":
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field.name} must be a finite number")
            if field.name.endswith("_scale"):
                if value <= 0:
                    raise ValueError(f"{field.name} must be positive")
            elif value < 0:
                raise ValueError(f"{field.name} must be nonnegative")


def reward_components(state, force: float, config: RewardConfig = RewardConfig()) -> dict[str, float]:
    """Positive upright benefit and positive penalty magnitudes; total subtracts costs."""
    x, v, theta, omega = state_vector(state)
    if not math.isfinite(force):
        raise ValueError("force must be finite")
    if config.form == "bounded":
        def gate(value, scale):
            # Algebraically 1/(1+(value/scale)^2), stable at extreme values.
            inverse = 1.0 / math.hypot(1.0, float(value) / scale)
            return inverse * inverse

        centered = gate(x, config.position_scale)
        slow_cart = gate(v, config.cart_velocity_scale)
        slow_pole = gate(omega, config.angular_velocity_scale)
        gentle_force = gate(force, config.force_scale)
        components = {
            "upright": (1 + math.cos(theta)) / 2 * centered * (1 + slow_cart) / 2 * (1 + slow_pole) / 2,
            "position_cost": config.position_penalty * (1 - centered),
            "cart_velocity_cost": config.cart_velocity_penalty * (1 - slow_cart),
            "angular_velocity_cost": config.angular_velocity_penalty * (1 - slow_pole),
            "force_cost": config.force_penalty * (1 - gentle_force),
        }
        components["total"] = components["upright"] - sum(value for key, value in components.items() if key.endswith("_cost"))
        if not all(math.isfinite(value) for value in components.values()):
            raise FloatingPointError("reward calculation produced a nonfinite value")
        return components
    components = {
        "upright": (1.0 + math.cos(theta)) / 2.0,
        "position_cost": config.position_penalty * (x / config.position_scale)**2,
        "cart_velocity_cost": config.cart_velocity_penalty * (v / config.cart_velocity_scale)**2,
        "angular_velocity_cost": config.angular_velocity_penalty * (omega / config.angular_velocity_scale)**2,
        "force_cost": config.force_penalty * (force / config.force_scale)**2,
    }
    components["total"] = components["upright"] - sum(
        value for name, value in components.items() if name.endswith("_cost"))
    if not all(math.isfinite(value) for value in components.values()):
        raise FloatingPointError("reward calculation produced a nonfinite value")
    return {name: float(value) for name, value in components.items()}


class DenseRewardWrapper(gym.Wrapper):
    """Apply before observation encoding; expects the raw four-state observation."""

    def __init__(self, env, config: RewardConfig | None = None):
        if env.observation_space.shape != (4,):
            raise ValueError("apply the reward wrapper before observation encoding")
        super().__init__(env)
        self.config = config if config is not None else RewardConfig()

    def step(self, action):
        state, _, terminated, truncated, info = self.env.step(action)
        components = reward_components(state, info["applied_force"], self.config)
        return state, components["total"], terminated, truncated, {
            **info, "reward_components": components,
        }
