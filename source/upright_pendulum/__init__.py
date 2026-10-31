"""Nonlinear cart–rigid-pendulum physics. No learning code."""

from gymnasium.envs.registration import register, registry

from .physics import Parameters, dynamics, energy, horizontal_momentum, integrate
from .env import CartPendulumEnv
from .stabilization import StabilizationCriteria, StabilizationMonitor, StabilizationWrapper
from .rewards import RewardConfig, reward_components, DenseRewardWrapper
from .observations import ObservationConfig, encode_observation, PolicyObservationWrapper
from .experiments import ExperimentConfig, make_experiment_env

ENV_ID = "UprightPendulum/CartPendulum-v0"
if ENV_ID not in registry:
    register(id=ENV_ID, entry_point="upright_pendulum.env:CartPendulumEnv")

__all__ = [
    "CartPendulumEnv", "Parameters", "dynamics", "energy",
    "horizontal_momentum", "integrate", "ENV_ID",
    "StabilizationCriteria", "StabilizationMonitor", "StabilizationWrapper",
    "RewardConfig", "reward_components", "DenseRewardWrapper",
    "ObservationConfig", "encode_observation", "PolicyObservationWrapper",
    "ExperimentConfig", "make_experiment_env",
]
