"""Sampled sustained-stabilization measurement, independent of reward.

Success is the first completed holding interval, not its entry time. The
continuous-time definition is approximated at physics substep endpoints.
"""

from dataclasses import dataclass
import math

import gymnasium as gym

from .env import CartPendulumEnv
from .physics import state_vector


@dataclass(frozen=True)
class StabilizationCriteria:
    position: float = 0.10  # m
    velocity: float = 0.10  # m/s
    angle: float = math.radians(5)  # rad, wrapped about upright
    angular_velocity: float = 0.20  # rad/s
    hold_time: float = 2.0  # s

    def __post_init__(self):
        for name in ("position", "velocity", "angle", "angular_velocity", "hold_time"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.angle > math.pi:
            raise ValueError("angle tolerance must be at most pi radians")

    def contains(self, state) -> bool:
        x, v, theta, omega = state_vector(state)
        angle_error = math.atan2(math.sin(theta), math.cos(theta))
        return bool(abs(x) <= self.position and abs(v) <= self.velocity
                    and abs(angle_error) <= self.angle and abs(omega) <= self.angular_velocity)


class StabilizationMonitor:
    """Track chronological samples with an explicit maximum allowed gap.

    Gaps larger than max_sample_gap are rejected rather than treated as an
    observed holding interval. A sampled pass still cannot rule out violations
    between samples. No threshold-crossing interpolation is performed.
    """

    def __init__(self, criteria=None, *, max_sample_gap=0.002):
        self.criteria = criteria if criteria is not None else StabilizationCriteria()
        if not math.isfinite(max_sample_gap) or max_sample_gap <= 0:
            raise ValueError("max_sample_gap must be finite and positive")
        self.max_sample_gap = float(max_sample_gap)
        self.reset()

    def reset(self):
        self.last_time = None
        self.entry_time = None
        self.success_time = None
        self.first_post_success_exit_time = None
        self.in_region = False

    def observe(self, time, state):
        if not math.isfinite(time) or time < 0:
            raise ValueError("sample time must be finite and nonnegative")
        if self.last_time is None:
            if time != 0:
                raise ValueError("the first sample must be at episode time zero")
        else:
            if time <= self.last_time:
                raise ValueError("sample times must increase strictly")
            if time - self.last_time > self.max_sample_gap + 1e-12:
                raise ValueError("sample gap exceeds the declared monitoring resolution")
        inside = self.criteria.contains(state)
        if inside:
            if self.entry_time is None:
                self.entry_time = float(time)
            elapsed = time - self.entry_time
            if self.success_time is None and (
                elapsed >= self.criteria.hold_time
                or math.isclose(elapsed, self.criteria.hold_time, rel_tol=0, abs_tol=1e-12)
            ):
                self.success_time = float(time)
        else:
            self.entry_time = None
            if self.success_time is not None and self.first_post_success_exit_time is None:
                self.first_post_success_exit_time = float(time)
        self.last_time = float(time)
        self.in_region = inside

    def status(self):
        held = 0.0 if self.entry_time is None else self.last_time - self.entry_time
        return {
            "in_region": self.in_region,
            "continuous_hold_time": held,
            "has_succeeded": self.success_time is not None,
            "success_time": self.success_time,
            "first_post_success_exit_time": self.first_post_success_exit_time,
        }

    def summary(self):
        """Episode record: no-event observations are right-censored at follow-up.

        Call at the end of evaluation. Calling earlier gives a partial record,
        not evidence of failure over the intended evaluation horizon.
        """
        if self.last_time is None:
            raise RuntimeError("cannot summarize without an initial observation")
        event = self.success_time is not None
        return {
            **self.status(),
            "event_observed": event,
            "right_censored": not event,
            "observed_time": self.success_time if event else self.last_time,
            "followup_time": self.last_time,
        }


class StabilizationWrapper(gym.Wrapper):
    """Add success diagnostics to a direct CartPendulumEnv instance.

    Observe every internal physics substep (2 ms by default), while retaining
    the original 20 ms control interval, rewards, and termination behavior.
    Apply other Gymnasium wrappers outside this one.
    """

    def __init__(self, env: CartPendulumEnv, criteria=None):
        if not isinstance(env, CartPendulumEnv):
            raise TypeError("wrap a direct CartPendulumEnv before adding other wrappers")
        super().__init__(env)
        self.monitor = StabilizationMonitor(
            criteria, max_sample_gap=min(env.integration_dt, env.control_dt))

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.monitor.reset()
        self.monitor.observe(0.0, observation)
        return observation, {**info, "stabilization": self.monitor.status()}

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step_observed(
            action, observer=self.monitor.observe)
        info = {**info, "stabilization": self.monitor.status()}
        if terminated or truncated:
            info["episode_measurement"] = self.monitor.summary()
        return observation, reward, terminated, truncated, info
