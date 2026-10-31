"""Trusted-local recovery state, committed after policy/statistics checkpoints.

Legacy reconstruction is deliberately restricted to this study's deterministic
episode boundary and the validated SB3 Gaussian/NumPy sampling schedule.
"""

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import zipfile

import gymnasium as gym
import numpy as np


def _layers(member):
    layers = []
    env = member.env.venv.envs[0]
    while isinstance(env, gym.Wrapper):
        layers.append(env)
        env = env.env
    return layers, env


def save_recovery_state(member, generation, checkpoint, statistics):
    layers, base = _layers(member)
    wrappers = []
    for layer in layers:
        name = type(layer).__name__
        state = {}
        if name == "Monitor":
            for field in ("rewards", "needs_reset", "total_steps", "episode_returns", "episode_lengths",
                          "episode_times", "current_reset_info", "t_start"):
                state[field] = copy.deepcopy(getattr(layer, field))
        elif name == "StabilizationWrapper":
            state = copy.deepcopy(vars(layer.monitor))
        elif name == "TrainingBoundary":
            state = {"needs_reset": layer.needs_reset}
        elif name == "TrainingStartCurriculum":
            state = layer.snapshot()
        wrappers.append((name, state))
    venv = member.env.venv
    payload = {"schema": 1, "generation": generation, "member_id": member.member_id,
        "settings": asdict(member.settings),
        "sha256": {"checkpoint": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                   "statistics": hashlib.sha256(Path(statistics).read_bytes()).hexdigest()},
        "random_stream": copy.deepcopy(vars(member.random_stream)),
        "base_env": copy.deepcopy(vars(base)), "wrappers": wrappers,
        "vector_env": {field: copy.deepcopy(getattr(venv, field)) for field in
            ("buf_obs", "buf_dones", "buf_rews", "buf_infos", "reset_infos", "_seeds", "_options")},
        "normalizer_returns": member.env.returns.copy()}
    target = Path(checkpoint).with_suffix(".resume.pkl")
    temporary = target.with_suffix(".pkl.tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, target)


def restore_member(member_id, config, settings, output, source_member, generation, *, legacy=False):
    import torch
    from stable_baselines3.common.vec_env import VecNormalize
    from .ppo_variants import load_training_model
    from .train_generations import TrainingMember
    from .train_pilot import preserve_random_states

    source_member = Path(source_member)
    checkpoint = source_member / f"generation_{generation:06d}.zip"
    statistics = checkpoint.with_suffix(".normalization.pkl")
    sidecar = checkpoint.with_suffix(".resume.pkl")
    with zipfile.ZipFile(checkpoint) as archive:
        model_data = json.loads(archive.read("data"))
    member = TrainingMember(member_id, config, settings, output,
        training_boundary_m=model_data.get("pendulum_training_boundary_m"),
        boundary_failure_cost=model_data.get("pendulum_boundary_failure_cost", 10.0))
    try:
        if model_data.get("pendulum_curriculum") is not None:
            from .curriculum import CurriculumConfig, attach_curriculum
            attach_curriculum(member, CurriculumConfig(**model_data["pendulum_curriculum"]))
        payload = None
        if sidecar.exists() and not legacy:
            with sidecar.open("rb") as file:
                payload = pickle.load(file)  # Only trusted local checkpoint files.
            if (payload["schema"] != 1 or payload["generation"] != generation or
                    payload["member_id"] != member_id or payload["settings"] != asdict(settings)):
                raise ValueError("recovery state does not match the declared member/settings")
            for field, path in (("checkpoint", checkpoint), ("statistics", statistics)):
                if hashlib.sha256(path.read_bytes()).hexdigest() != payload["sha256"][field]:
                    raise ValueError("checkpoint/statistics pair differs from its committed recovery state")
        elif not legacy:
            raise ValueError("full recovery state is absent; legacy reconstruction must be explicit")

        logger = member.model.logger
        with preserve_random_states():
            member.env = VecNormalize.load(str(statistics), member.env.venv)
            member.model = load_training_model(checkpoint, env=member.env, device="cpu", force_reset=False)
        # Freezing the observation transform must survive a restart without
        # disabling PPO optimization. Older checkpoints keep their old behavior.
        fixed = getattr(member.model, "pendulum_fixed_observation_statistics", False)
        if type(fixed) is not bool:
            raise ValueError("invalid fixed-observation-statistics checkpoint marker")
        if fixed and (not member.env.norm_obs or member.env.norm_reward):
            raise ValueError("fixed observation statistics require observation-only normalization")
        member.env.training = not fixed
        member.model.set_logger(logger)
        if (member.model.num_timesteps != generation * settings.rollout_steps or
                member.model.pendulum_generation != generation or member.model.seed != settings.seed):
            raise ValueError("saved model generation, step count or seed disagrees")
        layers, base = _layers(member)
        if payload is not None:
            if [name for name, _ in payload["wrappers"]] != [type(layer).__name__ for layer in layers]:
                raise ValueError("environment wrapper chain changed")
            vars(base).clear()
            vars(base).update(payload["base_env"])
            for layer, (name, state) in zip(layers, payload["wrappers"]):
                if name == "StabilizationWrapper":
                    vars(layer.monitor).clear()
                    vars(layer.monitor).update(state)
                elif name == "TrainingStartCurriculum":
                    layer.restore(state)
                else:
                    vars(layer).update(state)
            for field, value in payload["vector_env"].items():
                setattr(member.env.venv, field, value)
            member.env.returns = payload["normalizer_returns"]
            vars(member.random_stream).update(payload["random_stream"])
            member.recovery_method = "full_saved_state"
        else:
            # This legacy run has exactly one deterministic 1,000-step episode
            # per generation. Each Gaussian action consumes one (1,1) float32
            # standard-normal draw; each optimizer epoch consumes one permutation.
            if (settings.rollout_steps != 1000 or base.max_episode_steps != 1000 or
                    getattr(member.model, "pendulum_training_boundary_m", None) is not None or
                    getattr(member.model, "pendulum_curriculum", None) is not None or
                    member.model.use_sde or member.model.target_kl is not None or
                    member.model._n_updates != generation * settings.epochs or
                    member.model.policy.squash_output or member.env.norm_reward or
                    not np.all(member.model._last_episode_starts)):
                raise ValueError("legacy recovery requires the validated complete-episode Gaussian PPO protocol")
            with member.random_stream.active():
                with torch.no_grad():
                    for _ in range(generation):
                        for _ in range(settings.rollout_steps):
                            torch.empty((1, 1), dtype=torch.float32, device="cpu").normal_()
                        for _ in range(settings.epochs):
                            np.random.permutation(settings.rollout_steps)
            # Reset the raw vector env only: VecNormalize.reset would add an
            # extra statistics sample. The saved episode already auto-reset.
            raw = member.env.venv.reset()
            np.testing.assert_array_equal(member.env.normalize_obs(raw), member.model._last_obs)
            for _ in range(generation):
                base.np_random.uniform(np.zeros(4), np.zeros(4))
            member.env.venv.buf_dones[:] = True
            member.env.returns[:] = 0
            for layer in layers:
                if type(layer).__name__ == "Monitor":
                    layer.total_steps = generation * settings.rollout_steps
            member.recovery_method = "legacy_sampling_schedule_reconstructed_at_episode_boundary"
        return member
    except BaseException:
        member.close()
        raise
