"""Training-only local balancing task; the swing-up environment is unchanged."""

from dataclasses import asdict, dataclass
import math

import gymnasium as gym


@dataclass(frozen=True)
class LocalBalanceTask:
    """Fixed diagnostic definition, not a reward/hyperparameter search.

    Initial coordinates are sampled independently and uniformly within these
    half-widths about [0,0,0,0]. Failure is checked at 20 ms control endpoints.
    """

    initial_position_m: float = 0.25
    initial_velocity_m_s: float = 0.25
    initial_angle_rad: float = 0.15
    initial_angular_velocity_rad_s: float = 0.30
    failure_position_m: float = 1.0
    failure_angle_rad: float = math.pi / 6

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.initial_position_m < self.failure_position_m:
            raise ValueError("initial position range must lie inside the failure boundary")
        if not self.initial_angle_rad < self.failure_angle_rad < math.pi:
            raise ValueError("initial angle range must lie inside a failure boundary below pi")

    def initial_half_widths(self):
        return [self.initial_position_m, self.initial_velocity_m_s,
                self.initial_angle_rad, self.initial_angular_velocity_rad_s]


def local_evaluation_cases():
    """Fixed diagnostic probes; not samples for a population success rate."""
    return {
        "displaced_tilted": [0.2, 0.0, 0.1, 0.0],
        "mirrored": [-0.2, 0.0, -0.1, 0.0],
        "moving": [0.0, 0.25, 0.0, 0.3],
        "equivalent_after_rotation": [0.2, 0.0, 2 * math.pi + 0.1, 0.0],
    }


class LocalBalanceTrainingWrapper(gym.Wrapper):
    """Random local starts and true failure terminations, only during training.

    No reward modification, no success termination, no time-limit replacement.
    Apply inside Monitor so each local training episode is recorded correctly.
    Evaluation must use the ordinary environment without this wrapper.
    """

    def __init__(self, env, task=None):
        super().__init__(env)
        self.task = LocalBalanceTask() if task is None else task
        if not isinstance(self.task, LocalBalanceTask):
            raise TypeError("task must be LocalBalanceTask")
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("local training reset states are defined by LocalBalanceTask")
        observation, info = self.env.reset(seed=seed, options={
            "state": [0.0, 0.0, 0.0, 0.0], "noise_scale": self.task.initial_half_widths(),
        })
        self._needs_reset = False
        return observation, info

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("reset the local training task before stepping")
        observation, reward, terminated, truncated, info = self.env.step(action)
        x, _, theta, _ = self.unwrapped.state
        angle = math.atan2(math.sin(theta), math.cos(theta))
        failed_position = abs(x) > self.task.failure_position_m
        failed_angle = abs(angle) > self.task.failure_angle_rad
        failed = failed_position or failed_angle
        terminated = bool(terminated or failed)
        self._needs_reset = terminated or truncated
        info = {**info, "local_balance_failure": failed,
                "local_balance_position_failure": failed_position,
                "local_balance_angle_failure": failed_angle}
        if failed:
            # This is a training diagnostic at early termination, not a
            # right-censored 20-second evaluation for the swing-up estimand.
            info["episode_measurement"] = self.env.get_wrapper_attr("monitor").summary()
        return observation, reward, terminated, truncated, info
