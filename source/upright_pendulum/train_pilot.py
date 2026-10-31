"""One CPU PPO pilot, with saved and independently reloaded evaluations.

Run: python -m upright_pendulum.train_pilot
Short implementation check: python -m upright_pendulum.train_pilot --smoke
No multi-seed experiment, automatic reward search, or H(e,t) estimate is run.
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import time

import numpy as np

from .actions import NormalizedForceWrapper
from .experiments import ExperimentConfig, make_experiment_env
from .physics import Parameters, state_vector
from .stabilization import StabilizationCriteria
from .training_rewards import TrainingRewardScale
from .normalization import load_frozen_observation_normalizer, make_observation_normalizer


@dataclass(frozen=True)
class PilotSettings:
    total_steps: int = 100_000
    checkpoint_every: int = 20_000
    rollout_steps: int = 1000
    batch_size: int = 250
    epochs: int = 10
    seed: int = 0
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.0
    training_reward_scale: float = 1.0
    normalize_observations: bool = False
    initial_action_std: float = 1.0  # normalized command units; remains learnable

    def __post_init__(self):
        if type(self.normalize_observations) is not bool:
            raise ValueError("normalize_observations must be a boolean")
        for name in ("total_steps", "checkpoint_every", "rollout_steps", "batch_size", "epochs"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.rollout_steps < 2 or self.batch_size < 2:
            raise ValueError("rollout_steps and batch_size must be at least 2")
        if self.total_steps % self.rollout_steps or self.checkpoint_every % self.rollout_steps:
            raise ValueError("total_steps and checkpoint_every must be multiples of rollout_steps")
        if self.rollout_steps % self.batch_size:
            raise ValueError("batch_size must divide rollout_steps for this single-environment pilot")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for name in ("learning_rate", "gamma", "gae_lambda", "clip_range", "entropy_coefficient", "training_reward_scale", "initial_action_std"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or not 0 < self.clip_range < 1:
            raise ValueError("learning_rate must be positive and clip_range must lie in (0, 1)")
        if not 0 < self.gamma < 1 or not 0 <= self.gae_lambda <= 1 or self.entropy_coefficient < 0:
            raise ValueError("invalid discount, GAE, or entropy coefficient")
        if self.training_reward_scale <= 0:
            raise ValueError("training_reward_scale must be positive")
        if self.initial_action_std <= 0:
            raise ValueError("initial_action_std must be positive")

    def checkpoints(self):
        return [0, *range(self.checkpoint_every, self.total_steps, self.checkpoint_every), self.total_steps]


def make_pilot_env(config, *, max_episode_steps=1000):
    return NormalizedForceWrapper(make_experiment_env(config, max_episode_steps=max_episode_steps))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


@contextmanager
def preserve_random_states():
    """Loading an SB3 model may reseed global RNGs; keep training unaffected."""
    import torch
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


def evaluate_saved_checkpoint(checkpoint, config, output_csv, *, expected_steps, normalization_path=None,
                              initial_state=None):
    """One deterministic episode, always 20 s; default start [0, 0, pi, 0].

    This produces one event record, not a probability estimate. States in the
    CSV are at control endpoints; success is checked at 2 ms substeps.
    """
    from stable_baselines3 import PPO

    initial_state = state_vector([0, 0, np.pi, 0] if initial_state is None else initial_state)
    with preserve_random_states():
        env = make_pilot_env(config)
        normalizer = None
        try:
            if normalization_path is not None:
                normalizer = load_frozen_observation_normalizer(normalization_path, env)
            model = PPO.load(checkpoint, env=normalizer if normalizer is not None else env, device="cpu")
            requires_normalization = getattr(model, "pendulum_normalize_observations", False)
            if requires_normalization != (normalizer is not None):
                raise ValueError("checkpoint requires matching observation-normalization statistics")
            if model.num_timesteps != expected_steps:
                raise RuntimeError("loaded checkpoint has an unexpected training-step count")
            observation, info = env.reset(seed=0, options={"state": initial_state})
            initial = env.unwrapped.state
            total_reward = 0.0
            max_cart_displacement = abs(float(initial[0]))
            effort = 0.0
            saturation_steps = 0
            fieldnames = [
                "time_s", "x_m", "v_m_s", "theta_rad_unwrapped", "omega_rad_s",
                "normalized_action", "force_N", "reward", "upright", "position_cost",
                "cart_velocity_cost", "angular_velocity_cost", "force_cost",
                "in_region", "continuous_hold_time_s", "first_success_time_s",
            ]
            with Path(output_csv).open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({"time_s": 0.0, "x_m": initial[0], "v_m_s": initial[1],
                                 "theta_rad_unwrapped": initial[2], "omega_rad_s": initial[3],
                                 "in_region": info["stabilization"]["in_region"],
                                 "continuous_hold_time_s": 0.0})
                for index in range(1000):
                    policy_input = observation if normalizer is None else normalizer.normalize_obs(observation)
                    action, _ = model.predict(policy_input, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    state = env.unwrapped.state
                    force = info["applied_force"]
                    measurement = info["stabilization"]
                    components = info["reward_components"]
                    total_reward += reward
                    max_cart_displacement = max(max_cart_displacement, abs(float(state[0])))
                    effort += force**2 * env.unwrapped.control_dt
                    saturation_steps += abs(float(action[0])) >= 1 - 1e-7
                    writer.writerow({
                        "time_s": info["elapsed_time"], "x_m": state[0], "v_m_s": state[1],
                        "theta_rad_unwrapped": state[2], "omega_rad_s": state[3],
                        "normalized_action": float(action[0]), "force_N": force, "reward": reward,
                        **{key: value for key, value in components.items() if key != "total"},
                        "in_region": measurement["in_region"],
                        "continuous_hold_time_s": measurement["continuous_hold_time"],
                        "first_success_time_s": measurement["success_time"],
                    })
                    if terminated or truncated:
                        break
            if index != 999 or terminated or not truncated:
                raise RuntimeError("evaluation did not complete the fixed 20-second horizon")
            if not np.isfinite(total_reward):
                raise FloatingPointError("evaluation reward is not finite")
            return {
                "environment_steps": expected_steps,
                **info["episode_measurement"],
                "episode_return": float(total_reward),
                "max_abs_cart_position_m": max_cart_displacement,
                "force_squared_integral_N2_s": float(effort),
                "saturated_action_fraction": saturation_steps / 1000,
                "evaluation_steps": 1000,
                "trajectory_file": Path(output_csv).name,
                "checkpoint_file": Path(checkpoint).name,
                "normalization_file": None if normalization_path is None else Path(normalization_path).name,
            }
        finally:
            if normalizer is not None:
                normalizer.close()
            else:
                env.close()


def run_pilot(config, settings, output, *, smoke=False, local_balance=False):
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.monitor import Monitor
    except ImportError as error:
        raise RuntimeError(
            'Training dependencies are missing. Install the project training extra with '
            'python -m pip install -e ".[training]" in the project virtual environment.'
        ) from error

    if type(local_balance) is not bool:
        raise ValueError("local_balance must be a boolean")
    from .local_balance import LocalBalanceTask, LocalBalanceTrainingWrapper, local_evaluation_cases

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing run.
    checkpoints_dir = output / "checkpoints"
    trajectories_dir = output / "trajectories"
    checkpoints_dir.mkdir()
    trajectories_dir.mkdir()
    config.save(output / "experiment.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    network = {"pi": [64, 64], "vf": [64, 64]}
    manifest = {
        "status": "running", "kind": "technical_smoke" if smoke else "single_seed_pilot",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config.to_dict(), "pilot_settings": asdict(settings),
        "algorithm": "Stable-Baselines3 PPO", "device": "cpu", "torch_threads": 1,
        "deterministic_algorithms": True, "network": network, "activation": "Tanh",
        "fixed_ppo_settings": {"vf_coef": 0.5, "max_grad_norm": 0.5,
                               "normalize_advantage": True, "target_kl": None, "use_sde": False},
        "physical_parameters": asdict(Parameters()), "control_dt_s": 0.02,
        "integration_dt_s": 0.002, "episode_horizon_s": 20.0,
        "stabilization_criteria": asdict(StabilizationCriteria()),
        "policy_action_mapping": "F_N = 10 * clip(Gaussian policy action, -1, 1)",
        "action_exploration": {
            "distribution": "diagonal Gaussian before clipping",
            "initial_action_std_normalized": settings.initial_action_std,
            "initial_unclipped_force_std_N": 10 * settings.initial_action_std,
            "standard_deviation_remains_learnable": True,
            "evaluation": "deterministic clipped Gaussian mean; no sampled action noise",
        },
        "evaluation": {"deterministic_policy": True, "episodes_per_checkpoint": 1,
                       "initial_state": [0, 0, float(np.pi), 0],
                       "interpretation": "one event record per checkpoint; not an H(e,t) estimate"},
        "reward_reporting": {
            "ppo_feedback": "raw reward multiplied by pilot_settings.training_reward_scale",
            "training_monitor_and_progress_episode_returns": "raw reward units",
            "evaluation_returns_and_trajectory_rewards": "raw reward units",
            "value_loss": "squared scaled-return units; do not compare its magnitude across reward scales",
        },
        "observation_normalization": {
            "enabled": settings.normalize_observations,
            "method": "SB3 VecNormalize running mean/variance, training data only",
            "clip_obs": 10.0, "epsilon": 1e-8, "norm_reward": False,
            "evaluation_statistics_frozen": True,
            "note": "when enabled, each model requires its matching statistics pickle",
        },
        "checkpoints_after_completed_updates": settings.checkpoints(),
        "versions": {"python": platform.python_version(),
                     **{name: metadata.version(name) for name in ("numpy", "gymnasium", "torch", "stable-baselines3")}},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
    }
    if local_balance:
        manifest.update({
            "kind": "local_balance_technical_smoke" if smoke else "local_balance_learning_diagnostic",
            "training_task": {
                "name": "local_balance", "settings": asdict(LocalBalanceTask()),
                "initial_distribution": "independent uniform coordinates about upright",
                "early_failure_is_terminated_not_truncated": True,
                "failure_check_interval_s": 0.02,
                "reward_formula_unchanged": True,
                "interpretation": "separate simplified training task; not a controlled swing-up comparison",
            },
            "evaluation": {
                "deterministic_policy": True, "episodes_per_checkpoint": 4,
                "initial_states": local_evaluation_cases(), "local_failure_termination": False,
                "interpretation": "fixed local recovery probes, not independent trials or an H(e,t) estimate",
            },
        })
    write_json(output / "manifest.json", manifest)
    packages = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    (output / "installed_packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    start = time.perf_counter()
    train_env = None
    logger = None
    model = None
    records = []
    try:
        class PilotProgress(BaseCallback):
            def _on_step(self):
                if self.num_timesteps % 5000 == 0:
                    print(f"Collected {self.num_timesteps:,}/{settings.total_steps:,} training steps...", flush=True)
                return True

        progress = PilotProgress()
        training_task_env = make_pilot_env(config)
        if local_balance:
            training_task_env = LocalBalanceTrainingWrapper(training_task_env)
        monitored = Monitor(training_task_env, filename=str(output / "training.monitor.csv"))
        train_env = TrainingRewardScale(monitored, settings.training_reward_scale)
        if settings.normalize_observations:
            train_env = make_observation_normalizer(train_env, gamma=settings.gamma)
        model = PPO(
            "MlpPolicy", train_env, seed=settings.seed, device="cpu", verbose=0,
            n_steps=settings.rollout_steps, batch_size=settings.batch_size, n_epochs=settings.epochs,
            learning_rate=settings.learning_rate, gamma=settings.gamma, gae_lambda=settings.gae_lambda,
            clip_range=settings.clip_range, ent_coef=settings.entropy_coefficient,
            vf_coef=0.5, max_grad_norm=0.5, normalize_advantage=True, target_kl=None, use_sde=False,
            policy_kwargs={"net_arch": network, "activation_fn": torch.nn.Tanh,
                           "log_std_init": float(np.log(settings.initial_action_std))},
        )
        logger = configure(str(output), ["csv"])
        model.set_logger(logger)
        # Custom SB3 save data prevents silent raw-input evaluation of a model
        # that was trained with running observation statistics.
        model.pendulum_normalize_observations = settings.normalize_observations
        model.pendulum_task = "local_balance" if local_balance else "swing_up"
        initial_parameters = {name: value.detach().clone() for name, value in model.policy.state_dict().items()}
        for target in settings.checkpoints():
            if target > model.num_timesteps:
                # Each chunk is a whole number of rollouts. learn() returns only
                # after the gradient updates, unlike a per-step checkpoint callback.
                model.learn(total_timesteps=target - model.num_timesteps, reset_num_timesteps=False,
                            callback=progress)
            if model.num_timesteps != target:
                raise RuntimeError("PPO overshot a checkpoint; check rollout divisibility")
            path = checkpoints_dir / f"ppo_{target:09d}.zip"
            model.save(path)
            normalization_path = None
            if settings.normalize_observations:
                normalization_path = checkpoints_dir / f"obs_normalization_{target:09d}.pkl"
                train_env.save(str(normalization_path))
            cases = local_evaluation_cases() if local_balance else {None: None}
            checkpoint_records = []
            for case_name, initial in cases.items():
                suffix = "" if case_name is None else f"_{case_name}"
                record = evaluate_saved_checkpoint(
                    path, config, trajectories_dir / f"evaluation_{target:09d}{suffix}.csv", expected_steps=target,
                    normalization_path=normalization_path, initial_state=initial)
                if local_balance:
                    record.update({"evaluation_case": case_name, "initial_state": initial})
                checkpoint_records.append(record)
                records.append(record)
            write_json(output / "evaluations.json", records)
            with (output / "evaluations.csv").open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
            if local_balance:
                successes = sum(r["event_observed"] for r in checkpoint_records)
                retained = sum(r["event_observed"] and r["first_post_success_exit_time"] is None for r in checkpoint_records)
                mean_return = np.mean([r["episode_return"] for r in checkpoint_records])
                print(f"Checkpoint {target:,}/{settings.total_steps:,}: local recovery {successes}/4; "
                      f"no exit after success {retained}/4; mean raw return={mean_return:.3f}", flush=True)
                write_json(output / "local_balance_summary.json", {
                    "purpose": "Fixed local recovery probes; not a success-probability estimate",
                    "technical_smoke": smoke,
                    "evaluated_environment_steps": target,
                    "cases_with_completed_hold": successes, "cases_without_post_success_exit": retained,
                    "number_of_fixed_cases": 4,
                    "all_fixed_cases_retained_stabilization": retained == 4,
                    "limitations": ["One training seed; fixed cases include a periodic equivalent, not independent trials.",
                                    "A short smoke run only checks implementation.",
                                    "Local balancing does not demonstrate swing-up."],
                })
            else:
                success = record["success_time"]
                outcome = f"success at {success:.3f} s" if success is not None else "no success within 20 s"
                print(f"Checkpoint {target:,}/{settings.total_steps:,}: {outcome}; return={record['episode_return']:.3f}", flush=True)
        parameters_changed = any(not torch.equal(initial_parameters[name], value)
                                 for name, value in model.policy.state_dict().items())
        if not parameters_changed:
            raise RuntimeError("no policy/value parameters changed during training")
        manifest.update({"status": "complete", "actual_training_steps": model.num_timesteps,
                         "network_parameters_changed": True,
                         "finished_utc": datetime.now(timezone.utc).isoformat(),
                         "wall_time_seconds": time.perf_counter() - start})
        write_json(output / "manifest.json", manifest)
        print(f"Completed. Results: {output.resolve()}", flush=True)
        return {"output": str(output.resolve()), "manifest": manifest, "evaluations": records}
    except BaseException as error:
        manifest.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                         "actual_training_steps": 0 if model is None else model.num_timesteps,
                         "error": f"{type(error).__name__}: {error}",
                         "finished_utc": datetime.now(timezone.utc).isoformat(),
                         "wall_time_seconds": time.perf_counter() - start})
        write_json(output / "manifest.json", manifest)
        raise
    finally:
        if train_env is not None:
            train_env.close()
        if logger is not None:
            logger.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sin_cos_small_penalty.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="multiply only training rewards by this constant; evaluation stays unscaled")
    parser.add_argument("--normalize-observations", action="store_true",
                        help="learn input normalization from training data; save and freeze statistics for evaluation")
    parser.add_argument("--steps", type=int, help="default 100000; must be a multiple of 1000")
    parser.add_argument("--checkpoint-every", type=int, help="default 20000; multiple of 1000")
    parser.add_argument("--output", type=Path, help="new directory; existing directories are refused")
    parser.add_argument("--smoke", action="store_true", help="256 training steps, 64-step rollouts, two epochs")
    args = parser.parse_args()
    if args.smoke and (args.steps is not None or args.checkpoint_every is not None):
        parser.error("--smoke sets its own short budget; omit --steps and --checkpoint-every")
    if args.smoke:
        settings = PilotSettings(total_steps=256, checkpoint_every=128, rollout_steps=64,
                                 batch_size=64, epochs=2, seed=args.seed,
                                 training_reward_scale=args.reward_scale,
                                 normalize_observations=args.normalize_observations)
    else:
        settings = replace(PilotSettings(), seed=args.seed, training_reward_scale=args.reward_scale,
                           normalize_observations=args.normalize_observations,
                           total_steps=100000 if args.steps is None else args.steps,
                           checkpoint_every=20000 if args.checkpoint_every is None else args.checkpoint_every)
    config = ExperimentConfig.load(args.config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    kind = "smoke" if args.smoke else "pilot"
    output = args.output if args.output else Path("runs") / f"{kind}_{stamp}_seed{settings.seed}"
    print(f"Starting {kind}: {settings.total_steps:,} training steps, config={config.name}, "
          f"seed={settings.seed}, training reward scale={settings.training_reward_scale:g}, "
          f"normalize observations={settings.normalize_observations}", flush=True)
    run_pilot(config, settings, output, smoke=args.smoke)


if __name__ == "__main__":
    main()
