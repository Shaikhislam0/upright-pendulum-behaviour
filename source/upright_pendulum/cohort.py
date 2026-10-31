"""Synchronized evaluation of a fixed cohort; no training takes place here."""

from dataclasses import dataclass
from functools import partial
import math

import gymnasium as gym
import numpy as np
from gymnasium.wrappers.vector import DictInfoToList

from .experiments import ExperimentConfig
from .train_pilot import make_pilot_env


@dataclass(frozen=True)
class EvaluationSettings:
    horizon_s: float = 20.0
    sample_interval_s: float = 0.1
    parallel: bool = False

    def __post_init__(self):
        for name in ("horizon_s", "sample_interval_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isclose(self.horizon_s / .02, round(self.horizon_s / .02), abs_tol=1e-10, rel_tol=0):
            raise ValueError("horizon_s must be a multiple of the 0.02 s control interval")
        if self.horizon_s < .02 or self.sample_interval_s < .002:
            raise ValueError("horizon must be >=0.02 s; sampling interval must be >=0.002 s")
        if type(self.parallel) is not bool:
            raise ValueError("parallel must be boolean")

    @property
    def steps(self):
        return round(self.horizon_s / .02)

    def sample_times(self):
        count = math.floor(self.horizon_s / self.sample_interval_s + 1e-12)
        times = [i * self.sample_interval_s for i in range(count + 1)]
        if math.isclose(times[-1], self.horizon_s, rel_tol=0, abs_tol=1e-12):
            times[-1] = self.horizon_s
        else:
            times.append(self.horizon_s)
        return np.array(times, dtype=np.float64)


def cumulative_counts(success_times, sample_times):
    """Completed holds by each reporting time; failures stay in the denominator.

    None denotes no observed event. Counts remain sticky after a later exit.
    This is different from counting the simulations currently in the region.
    """
    if len(success_times) == 0:
        raise ValueError("a cohort must have at least one member")
    times = np.asarray(sample_times, dtype=np.float64)
    if times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("sample_times must be finite, nonnegative and strictly increasing")
    events = []
    for event in success_times:
        if event is None:
            continue
        if isinstance(event, (bool, np.bool_)) or not math.isfinite(event) or event < 0:
            raise ValueError("success times must be None or finite nonnegative times")
        events.append(float(event))
    return np.searchsorted(np.sort(events), times + 1e-12, side="right")


class PhysicalStateInfo(gym.Wrapper):
    """Carry states and optional event times safely through batched info.

    Gymnasium infers an info array's dtype from the first member. A mixture of
    float event times and None can become NaN depending on cohort order. Use a
    numeric transport sentinel, decoded immediately after vector unbatching.
    """

    def pack(self, info):
        info = {**info, "physical_state": self.unwrapped.state}
        for key in ("stabilization", "episode_measurement"):
            if key in info:
                info[key] = dict(info[key])
                for field in ("success_time", "first_post_success_exit_time"):
                    if info[key][field] is None:
                        info[key][field] = -1.0
        return info

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, self.pack(info)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, self.pack(info)


def _decode_infos(infos):
    for info in infos:
        for key in ("stabilization", "episode_measurement"):
            if key in info:
                for field in ("success_time", "first_post_success_exit_time"):
                    value = info[key][field]
                    info[key][field] = None if value == -1 else float(value)
    return infos


def _evaluation_env(config, steps, initial_state):
    env = PhysicalStateInfo(make_pilot_env(config, max_episode_steps=steps))
    return InitialState(env, initial_state)


class InitialState(gym.Wrapper):
    def __init__(self, env, state):
        super().__init__(env)
        self.initial_state = list(state)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("cohort initial states are fixed before evaluation")
        return self.env.reset(seed=seed, options={"state": self.initial_state})


def evaluate_cohort(policies, *, config=None, settings=None, member_ids=None,
                    initial_states=None, trajectory_callback=None, sample_callback=None):
    """Advance every member on the same physical clock, with no autoresets.

    Each policy is a deterministic callable: encoded observation -> normalized
    action. Standard use supplies one independently trained policy per member.
    Optional initial_states exists for measurement tests; the study uses the
    exact downward start for every member. Parallel mode uses worker processes;
    synchronous mode has the same shared simulation clock on one CPU process.
    """
    config = ExperimentConfig() if config is None else config
    settings = EvaluationSettings() if settings is None else settings
    n = len(policies)
    if n == 0:
        raise ValueError("a cohort must contain at least one policy")
    member_ids = list(range(n)) if member_ids is None else list(member_ids)
    if len(member_ids) != n or len(set(member_ids)) != n:
        raise ValueError("member IDs must be unique and match the cohort size")
    initial_states = [[0, 0, math.pi, 0] for _ in range(n)] if initial_states is None else initial_states
    if len(initial_states) != n:
        raise ValueError("supply exactly one initial state per member")
    factories = [partial(_evaluation_env, config, settings.steps, state) for state in initial_states]
    vector_type = gym.vector.AsyncVectorEnv if settings.parallel else gym.vector.SyncVectorEnv
    env = DictInfoToList(vector_type(factories, autoreset_mode=gym.vector.AutoresetMode.DISABLED))
    report_times = settings.sample_times()
    samples = []
    next_sample = 0
    latest_success = [None] * n

    def record_samples(elapsed):
        nonlocal next_sample
        while next_sample < len(report_times) and report_times[next_sample] <= elapsed + 1e-12:
            time = float(report_times[next_sample])
            count = int(cumulative_counts(latest_success, [time])[0])
            row = {"evaluation_time_s": time, "achieved_balance": count,
                   "n_simulations": n, "percentage": 100.0 * count / n}
            samples.append(row)
            if sample_callback is not None:
                sample_callback(row)
            next_sample += 1

    def record_states(infos, actions=None, rewards=None):
        if trajectory_callback is None:
            return
        for index, info in enumerate(infos):
            state = info["physical_state"]
            status = info["stabilization"]
            trajectory_callback({"member_id": member_ids[index], "time_s": float(info["elapsed_time"]),
                **dict(zip(("x_m", "v_m_s", "theta_rad_unwrapped", "omega_rad_s"), map(float, state))),
                "normalized_action": None if actions is None else float(actions[index, 0]),
                "force_N": None if actions is None else float(info["applied_force"]),
                "reward": None if rewards is None else float(rewards[index]),
                "in_region": bool(status["in_region"]), "has_achieved_balance": bool(status["has_succeeded"]),
                "first_success_time_s": status["success_time"]})

    try:
        observations, infos = env.reset(seed=0)
        infos = _decode_infos(infos)
        record_states(infos)
        record_samples(0)
        for step in range(1, settings.steps + 1):
            actions = np.array([policy(observation.copy()) for policy, observation in zip(policies, observations)], dtype=np.float32)
            if actions.shape != (n, 1) or not np.all(np.isfinite(actions)) or np.any(np.abs(actions) > 1):
                raise ValueError("each policy must return one finite normalized action in [-1,1]")
            observations, rewards, terminated, truncated, infos = env.step(actions)
            infos = _decode_infos(infos)
            elapsed = step * .02
            if any(not math.isclose(float(info["elapsed_time"]), elapsed, abs_tol=1e-12, rel_tol=0) for info in infos):
                raise RuntimeError("cohort members lost their common evaluation clock")
            if np.any(terminated) or (step < settings.steps and np.any(truncated)):
                raise RuntimeError("cohort evaluation ended early; denominator must not change")
            latest_success = [info["stabilization"]["success_time"] for info in infos]
            record_states(infos, actions, rewards)
            record_samples(elapsed)
        if not np.all(truncated) or next_sample != len(report_times):
            raise RuntimeError("cohort did not finish the complete evaluation horizon")
        members = [{"member_id": member_ids[i], **{
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in info["episode_measurement"].items()
        }} for i, info in enumerate(infos)]
        return {"n_simulations": n, "horizon_s": settings.horizon_s, "samples": samples, "members": members}
    finally:
        env.close()
