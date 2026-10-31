"""Training-only mixed-start curriculum; evaluation construction never uses it."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import gymnasium as gym
import numpy as np


@dataclass(frozen=True)
class CurriculumConfig:
    total_generations: int = 500
    position_half_width: float = .1
    velocity_half_width: float = .1
    near_upright_angle_half_width: float = .15
    angular_velocity_half_width: float = .2

    def __post_init__(self):
        if type(self.total_generations) is not int or self.total_generations < 1:
            raise ValueError("total_generations must be a positive integer")
        for name in ("position_half_width", "velocity_half_width", "near_upright_angle_half_width", "angular_velocity_half_width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class TrainingStartCurriculum(gym.Wrapper):
    """Sample starts on natural resets, using an independent seeded generator.

    Five fixed training phases. In the first four, 20% exact downward, 20%
    near upright and 60% uniform angle in an expanding range. In the fifth,
    80% exact downward and 20% near upright. No evaluation feedback is used.
    Ongoing episodes are never reset merely because the generation changes.
    """

    def __init__(self, env, config=None, *, seed=0):
        super().__init__(env)
        self.config = config or CurriculumConfig()
        self.generation = 0
        self.rng = np.random.default_rng(seed)
        self.start_counts = {"exact_downward": 0, "near_upright": 0, "expanding_angle": 0, "explicit": 0}
        self.last_start = None

    @property
    def phase(self):
        return min(4, (max(1, self.generation) - 1) * 5 // self.config.total_generations)

    def set_generation(self, generation):
        if type(generation) is not int or not 0 <= generation <= self.config.total_generations:
            raise ValueError("generation is outside the declared curriculum")
        self.generation = generation

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        options = {} if options is None else dict(options)
        if "state" in options:
            kind, angle_range = "explicit", None
        else:
            choice = self.rng.random()
            if choice < (.8 if self.phase == 4 else .2):
                kind, angle_range = "exact_downward", 0.
                state = np.array([0., 0., math.pi, 0.])
            else:
                kind = "near_upright" if self.phase == 4 or choice < .4 else "expanding_angle"
                angle_range = self.config.near_upright_angle_half_width if kind == "near_upright" else (
                    self.config.near_upright_angle_half_width, math.pi / 3, 2 * math.pi / 3, math.pi)[self.phase]
                widths = np.array([self.config.position_half_width, self.config.velocity_half_width,
                                   angle_range, self.config.angular_velocity_half_width])
                state = self.rng.uniform(-widths, widths)
            options["state"] = state
        observation, info = self.env.reset(seed=seed, options=options)
        self.start_counts[kind] += 1
        self.last_start = {"kind": kind, "phase": self.phase + 1, "generation": self.generation,
                           "angle_half_width_rad": angle_range, "state": self.unwrapped.state.tolist()}
        return observation, {**info, "training_start": deepcopy(self.last_start)}

    def snapshot(self):
        return {"generation": self.generation, "rng_state": deepcopy(self.rng.bit_generator.state),
                "start_counts": deepcopy(self.start_counts), "last_start": deepcopy(self.last_start)}

    def restore(self, state):
        self.set_generation(state["generation"])
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
        self.start_counts = deepcopy(state["start_counts"])
        self.last_start = deepcopy(state["last_start"])


def attach_curriculum(member, config):
    """Attach inside Monitor, before the first training reset or policy update."""
    from stable_baselines3.common.monitor import Monitor
    if member.model.num_timesteps or member.model._last_obs is not None:
        raise ValueError("attach a curriculum only to a fresh member")
    monitor = member.env.venv.envs[0]
    if not isinstance(monitor, Monitor) or isinstance(monitor.env, TrainingStartCurriculum):
        raise ValueError("expected a Monitor without an existing curriculum")
    monitor.env = TrainingStartCurriculum(monitor.env, config, seed=member.settings.seed)
    member.model.pendulum_curriculum = asdict(config)
    return monitor.env


def curriculum_layer(member):
    env = member.env.venv.envs[0]
    while isinstance(env, gym.Wrapper):
        if isinstance(env, TrainingStartCurriculum):
            return env
        env = env.env
    raise ValueError("training environment has no start curriculum")
