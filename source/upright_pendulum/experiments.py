"""Load reviewable experiment choices; construct environments without training."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from .env import CartPendulumEnv
from .observations import ObservationConfig, PolicyObservationWrapper
from .rewards import DenseRewardWrapper, RewardConfig
from .stabilization import StabilizationWrapper


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "sin_cos_small_penalty"
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    schema_version: int = 1

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a nonempty string")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported schema_version; expected 1")
        if not isinstance(self.reward, RewardConfig) or not isinstance(self.observation, ObservationConfig):
            raise ValueError("use RewardConfig and ObservationConfig settings")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("experiment configuration must be a JSON object")
        unknown = set(data) - {"name", "reward", "observation", "schema_version"}
        if unknown:
            raise ValueError(f"unknown experiment settings: {sorted(unknown)}")
        try:
            return cls(
                name=data.get("name", "sin_cos_small_penalty"),
                reward=RewardConfig(**data.get("reward", {})),
                observation=ObservationConfig(**data.get("observation", {})),
                schema_version=data.get("schema_version", 1),
            )
        except TypeError as error:
            raise ValueError(f"invalid reward or observation settings: {error}") from error

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def make_experiment_env(config: ExperimentConfig | None = None, *, max_episode_steps=1000):
    """Fixed physical baseline and success criterion; selectable reward/input.

    Order matters: physical state -> success measurement -> reward -> encoding.
    Actions remain actual forces in [-10, 10] N at 50 Hz. Default horizon: 20 s.
    """
    config = config if config is not None else ExperimentConfig()
    env = StabilizationWrapper(CartPendulumEnv(max_episode_steps=max_episode_steps))
    env = DenseRewardWrapper(env, config.reward)
    return PolicyObservationWrapper(env, config.observation)
