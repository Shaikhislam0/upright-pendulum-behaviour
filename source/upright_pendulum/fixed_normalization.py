"""Freeze the observation coordinates while leaving PPO learning enabled."""

import hashlib
import json


def observation_statistics(member):
    normalizer = member.env
    if not normalizer.norm_obs or normalizer.norm_reward:
        raise ValueError("this intervention requires observation-only normalization")
    return {"mean": normalizer.obs_rms.mean.tolist(),
            "variance": normalizer.obs_rms.var.tolist(),
            "count": float(normalizer.obs_rms.count),
            "clip_obs": float(normalizer.clip_obs), "epsilon": float(normalizer.epsilon)}


def freeze_observation_statistics(member, generation):
    if member.model.num_timesteps != generation * member.settings.rollout_steps:
        raise ValueError("freeze generation does not match training progress")
    snapshot = observation_statistics(member)
    member.env.training = False
    member.model.pendulum_fixed_observation_statistics = True
    member.model.pendulum_observation_freeze_generation = generation
    member.model.pendulum_observation_freeze_sha256 = statistics_hash(snapshot)
    return snapshot


def statistics_hash(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_fixed_statistics(member, expected):
    if (member.env.training or
            not getattr(member.model, "pendulum_fixed_observation_statistics", False) or
            observation_statistics(member) != expected or
            member.model.pendulum_observation_freeze_sha256 != statistics_hash(expected)):
        raise RuntimeError("the fixed observation transform changed")
