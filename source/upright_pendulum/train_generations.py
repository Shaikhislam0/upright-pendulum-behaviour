"""Train independent PPO models and evaluate the cohort after every generation.

A generation is one complete rollout/update cycle per model. PPO's optimization
epochs and minibatches are inside that generation. No experience is shared.
"""

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import heapq
import json
from importlib import metadata
import os
from pathlib import Path
import platform
import random
import shutil
import time

import numpy as np

from .cohort import EvaluationSettings, evaluate_cohort
from .experiments import ExperimentConfig
from .normalization import load_frozen_observation_normalizer, make_observation_normalizer
from .physics import Parameters
from .stabilization import StabilizationCriteria
from .train_pilot import PilotSettings, make_pilot_env, preserve_random_states, write_json


@dataclass(frozen=True)
class GenerationSettings:
    models: int
    generations: int
    batch_steps: int = 1000
    minibatch_size: int = 250
    epochs: int = 10
    seed_base: int = 100
    evaluation_every: int = 1

    def __post_init__(self):
        for name in ("models", "generations", "evaluation_every"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        # Reuse the existing validated PPO scalar settings and divisibility rules.
        self.pilot(self.seed_base)
        if self.seed_base + self.models > 2**32:
            raise ValueError("member seeds must be distinct uint32 values")

    def pilot(self, seed):
        return PilotSettings(total_steps=self.generations * self.batch_steps,
            checkpoint_every=self.batch_steps * self.evaluation_every, rollout_steps=self.batch_steps,
            batch_size=self.minibatch_size, epochs=self.epochs, seed=seed,
            normalize_observations=True)

    def evaluation_generations(self):
        return sorted({0, *range(self.evaluation_every, self.generations + 1, self.evaluation_every), self.generations})


class RandomStream:
    """Isolate SB3's global CPU RNGs for each independently trained model.

    Gymnasium environments also own separate seeded generators. Saving global
    streams prevents cohort order, model count or evaluation from changing the
    samples used to train another member.
    """

    def __init__(self):
        self.capture()

    def capture(self):
        import torch
        self.python = random.getstate()
        self.numpy = np.random.get_state()
        self.torch = torch.random.get_rng_state().clone()

    @contextmanager
    def active(self):
        import torch
        with preserve_random_states():
            random.setstate(self.python)
            np.random.set_state(self.numpy)
            torch.random.set_rng_state(self.torch)
            try:
                yield
            finally:
                self.capture()


class TrainingMember:
    def __init__(self, member_id, config, settings, output, *, gradient_clipping="global",
                 training_boundary_m=None, boundary_failure_cost=10.0):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.monitor import Monitor

        if gradient_clipping not in ("global", "independent"):
            raise ValueError("gradient_clipping must be global or independent")
        algorithm = PPO
        if gradient_clipping == "independent":
            from .ppo_variants import IndependentGradientPPO
            algorithm = IndependentGradientPPO

        self.member_id = member_id
        self.settings = settings
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        training_env = make_pilot_env(config)
        monitor_options = {}
        if training_boundary_m is not None:
            from .training_boundary import TrainingBoundary
            training_env = TrainingBoundary(training_env, limit_m=training_boundary_m,
                                            failure_cost=boundary_failure_cost)
            monitor_options["info_keywords"] = ("training_boundary_hit",)
        self.env = make_observation_normalizer(Monitor(training_env,
            filename=str(self.output / "training.monitor.csv"), **monitor_options), gamma=settings.gamma)
        try:
            with preserve_random_states():
                self.model = algorithm("MlpPolicy", self.env, seed=settings.seed, device="cpu", verbose=0,
                    n_steps=settings.rollout_steps, batch_size=settings.batch_size,
                    n_epochs=settings.epochs, learning_rate=settings.learning_rate,
                    gamma=settings.gamma, gae_lambda=settings.gae_lambda,
                    clip_range=settings.clip_range, ent_coef=settings.entropy_coefficient,
                    vf_coef=.5, max_grad_norm=.5, normalize_advantage=True,
                    policy_kwargs={"activation_fn": torch.nn.Tanh,
                        "net_arch": {"pi": [64, 64], "vf": [64, 64]}, "log_std_init": 0.0})
                self.random_stream = RandomStream()
            self.model.pendulum_normalize_observations = True
            self.model.pendulum_gradient_clipping = gradient_clipping
            self.model.pendulum_training_boundary_m = training_boundary_m
            self.model.pendulum_boundary_failure_cost = boundary_failure_cost
            self.model.pendulum_task = "full_swing_up"
            self.model.set_logger(configure(str(self.output / "ppo_log"), ["csv"]))
        except BaseException:
            self.env.close()
            raise

    def update(self, generation):
        expected_before = (generation - 1) * self.settings.rollout_steps
        if self.model.num_timesteps != expected_before:
            raise RuntimeError("training member is not at the previous generation")
        update_started = time.perf_counter()
        with self.random_stream.active():
            self.model.learn(total_timesteps=self.settings.rollout_steps, reset_num_timesteps=False,
                             progress_bar=False, log_interval=None)
        if self.model.num_timesteps != generation * self.settings.rollout_steps:
            raise RuntimeError("one generation must collect exactly one fixed batch per model")
        # One row after optimization: old rows mixed current rollout metrics
        # with the preceding generation's training metrics.
        logger = self.model.logger
        logger.record("time/total_timesteps", self.model.num_timesteps)
        logger.record("time/generation", generation)
        logger.record("time/update_wall_s", time.perf_counter() - update_started)
        if self.model.ep_info_buffer:
            logger.record("rollout/ep_rew_mean", float(np.mean([episode["r"] for episode in self.model.ep_info_buffer])))
            logger.record("rollout/ep_len_mean", float(np.mean([episode["l"] for episode in self.model.ep_info_buffer])))
            if getattr(self.model, "pendulum_training_boundary_m", None) is not None:
                logger.record("rollout/boundary_failure_fraction", float(np.mean([
                    episode["training_boundary_hit"] for episode in self.model.ep_info_buffer])))
        logger.dump(step=self.model.num_timesteps)

    def save(self, generation):
        stem = self.output / f"generation_{generation:06d}"
        self.model.pendulum_generation = generation
        self.model.save(str(stem) + ".zip")
        self.env.save(str(stem) + ".normalization.pkl")
        from .checkpoint_state import save_recovery_state
        save_recovery_state(self, generation, Path(str(stem) + ".zip"),
                            Path(str(stem) + ".normalization.pkl"))
        return Path(str(stem) + ".zip"), Path(str(stem) + ".normalization.pkl")

    def close(self):
        self.env.close()


class FrozenPolicy:
    """Reload a saved policy/statistics pair; deterministic evaluation only."""

    def __init__(self, checkpoint, statistics, config, *, generation, expected_steps):
        from stable_baselines3 import PPO
        self.normalizer = load_frozen_observation_normalizer(statistics, make_pilot_env(config))
        try:
            self.model = PPO.load(checkpoint, device="cpu")
            if (self.model.num_timesteps != expected_steps or
                    getattr(self.model, "pendulum_generation", None) != generation or
                    not getattr(self.model, "pendulum_normalize_observations", False)):
                raise RuntimeError("checkpoint generation, steps or normalization do not match")
        except BaseException:
            self.close()
            raise

    def __call__(self, observation):
        normalized = self.normalizer.normalize_obs(observation)
        action, _ = self.model.predict(normalized, deterministic=True)
        return action

    def close(self):
        self.normalizer.close()


def evaluate_members(members, config, settings, evaluation, generation, output, trajectory_path):
    """Save/reload one disjoint group; same path for serial and worker execution."""
    paths = [member.save(generation) for member in members]
    hashes = {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for pair in paths for path in pair}
    policies = []
    try:
        with preserve_random_states():
            for checkpoint, statistics in paths:
                policies.append(FrozenPolicy(checkpoint, statistics, config, generation=generation,
                    expected_steps=generation * settings.batch_steps))
            with Path(trajectory_path).open("w", newline="", encoding="utf-8") as file:
                writer = None

                def trajectory(row):
                    nonlocal writer
                    if writer is None:
                        writer = csv.DictWriter(file, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)

                record = evaluate_cohort(policies, config=config, settings=evaluation,
                    member_ids=[member.member_id for member in members], trajectory_callback=trajectory)
        return record, hashes
    finally:
        for policy in policies:
            policy.close()


def merge_trajectories(parts, target):
    """Stream worker rows into the serial time/member order, retaining all data."""
    from contextlib import ExitStack
    with ExitStack() as stack:
        readers = [csv.DictReader(stack.enter_context(Path(path).open(newline="", encoding="utf-8")))
                   for path in parts]
        if not readers or any(reader.fieldnames != readers[0].fieldnames for reader in readers):
            raise RuntimeError("worker trajectory columns disagree")
        with Path(target).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=readers[0].fieldnames)
            writer.writeheader()
            writer.writerows(heapq.merge(*readers, key=lambda row: (float(row["time_s"]), row["member_id"])))
    # Only redundant worker parts are removed, after the complete merge closes.
    for path in parts:
        Path(path).unlink()


def run_generations(config, settings, output, *, evaluation=None, technical=False, workers=1, recipe=None,
                    confidence_level=None, min_free_gib=2.):
    """Record generations 0..G with fixed n and full downward-start evaluations."""
    import torch
    from .training_recipe import TrainingRecipe, make_member
    from .cohort_execution import atomic_json, collect_generations, validate_reserve
    from .cohort_results import validate_confidence
    validate_reserve(min_free_gib)
    validate_confidence(confidence_level)
    recipe = recipe if recipe is not None else TrainingRecipe()
    if not isinstance(recipe, TrainingRecipe):
        raise ValueError("recipe must be a TrainingRecipe")
    evaluation = evaluation or EvaluationSettings()
    if type(workers) is not int or not 1 <= workers <= settings.models:
        raise ValueError("workers must lie between 1 and the number of models")
    if workers > 1 and evaluation.parallel:
        raise ValueError("use the worker pool without --parallel-evaluation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    # Configure before SB3 imports matplotlib; keep font-cache writes in the run.
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))
    config.save(output / "experiment.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(Path(__file__).parent.glob("*.py"))}
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    manifest = {
        "status": "running", "kind": "technical_smoke" if technical else "exploratory_generation_cohort",
        "process_id": os.getpid(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "generation_definition": "one fixed rollout followed by one PPO train call per independently trained model",
        "optimization_epochs_per_generation": settings.epochs,
        "axes": {"horizontal": "generation", "vertical": "elapsed evaluation time (s)",
                 "color": "100 * fraction with first completed stabilization hold by time t"},
        "settings": asdict(settings), "ppo_per_member": asdict(settings.pilot(settings.seed_base)),
        "training_recipe": asdict(recipe),
        "evaluation_generations": settings.evaluation_generations(),
        "inference": {"confidence_level": confidence_level, "method": "Clopper-Pearson exact binomial",
                      "scope": "pointwise across independent training seeds; not simultaneous bands"},
        "resource_policy": {"min_free_gib": min_free_gib,
                            "recovery": "full cohort at evaluation checkpoints and controlled stop; abrupt loss may require repeating unsaved generations"},
        "evaluation": asdict(evaluation), "initial_state": [0, 0, np.pi, 0],
        "physical_parameters": asdict(Parameters()), "stabilization": asdict(StabilizationCriteria()),
        "control_dt_s": .02, "integration_dt_s": .002,
        "policy": {"actor": [64, 64], "critic": [64, 64], "activation": "Tanh",
                   "observation_normalization": "separate running statistics per member, frozen for evaluation",
                   "training_reward_scale": 1, "initial_action_std": 1,
                   "vf_coef": .5, "max_grad_norm": .5, "deterministic_evaluation": True},
        "member_seeds": {f"member_{i:04d}": settings.seed_base + i for i in range(settings.models)},
        "versions": {"python": platform.python_version(), **{name: metadata.version(name)
            for name in ("numpy", "gymnasium", "stable-baselines3", "torch", "matplotlib")}},
        "source_sha256": sources, "completed_generations": [], "checkpoint_sha256": {},
        "execution": {"workers": workers, "torch_threads_per_worker": 1,
                      "generation_barriers": "all training completes before evaluation; all evaluation before next training"},
        "timings": [],
    }
    if confidence_level is not None:
        manifest["versions"]["scipy"] = metadata.version("scipy")
    atomic_json(output / "manifest.json", manifest)
    members = []
    pool = None
    records = []
    started = time.perf_counter()
    try:
        if workers > 1:
            from .workers import WorkerPool
            pool = WorkerPool(workers, config, settings, evaluation, output, recipe=recipe)
        else:
            for i in range(settings.models):
                member_id = f"member_{i:04d}"
                members.append(make_member(member_id, config, settings.pilot(settings.seed_base + i),
                                           output / "models" / member_id, recipe))
        manifest["initialization_s"] = time.perf_counter() - started
        return collect_generations(members, pool, config, settings, evaluation, output, manifest, records,
            confidence_level=confidence_level, min_free_gib=min_free_gib)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_wall_s"] = time.perf_counter() - started
        manifest["environment_steps_per_model_completed"] = manifest.get("last_training_generation", 0) * settings.batch_steps
        manifest["cohort_training_steps_completed"] = manifest["environment_steps_per_model_completed"] * settings.models
        atomic_json(output / "manifest.json", manifest)
        for member in members:
            member.close()
        if pool is not None:
            pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=int, help="Number n of independently trained PPO models")
    parser.add_argument("--generations", type=int, help="Number of rollout/update cycles per model")
    parser.add_argument("--batch-steps", type=int, default=1000, help="Experience collected per model per generation")
    parser.add_argument("--minibatch-size", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=10, help="Optimization epochs inside each generation")
    parser.add_argument("--evaluation-every", type=int, default=1, help="Evaluate/retain every K generations, plus 0 and final")
    parser.add_argument("--confidence-level", type=float, help="Optional pointwise exact binomial intervals, e.g. 0.95")
    parser.add_argument("--min-free-gib", type=float, default=2., help="Pause before disk space drops below this reserve")
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--config", type=Path, help="Reward/input JSON; default sin_cos_small_penalty")
    parser.add_argument("--sample-interval", type=float, default=.1, help="Seconds between fraction reports")
    parser.add_argument("--parallel-evaluation", action="store_true", help="Run physics in separate worker processes")
    parser.add_argument("--workers", type=int, default=1, help="Persistent workers for disjoint independent model groups")
    parser.add_argument("--gradient-clipping", choices=("global", "independent"), default="global")
    parser.add_argument("--training-boundary", type=float, help="Optional training-only terminal cart-position boundary in metres")
    parser.add_argument("--boundary-failure-cost", type=float, default=10.)
    parser.add_argument("--smoke", action="store_true", help="Only test wiring: 2 models, 2 generations, 64 steps per batch")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.smoke:
        settings = GenerationSettings(models=2, generations=2, batch_steps=64, minibatch_size=64,
                                      epochs=2, seed_base=args.seed_base, evaluation_every=args.evaluation_every)
    else:
        if args.models is None or args.generations is None:
            parser.error("specify --models and --generations, or use --smoke for the technical check")
        settings = GenerationSettings(args.models, args.generations, args.batch_steps,
                                      args.minibatch_size, args.epochs, args.seed_base, args.evaluation_every)
    evaluation = EvaluationSettings(sample_interval_s=args.sample_interval, parallel=args.parallel_evaluation)
    config = ExperimentConfig() if args.config is None else ExperimentConfig.load(args.config)
    from .training_recipe import TrainingRecipe
    recipe = TrainingRecipe(args.gradient_clipping, args.training_boundary, args.boundary_failure_cost)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"{'generation_smoke' if args.smoke else 'generations'}_{timestamp}"
    print(f"{'Technical smoke' if args.smoke else 'Exploratory cohort'}: {settings.models} independent models, "
          f"{settings.generations} generations, {settings.batch_steps} training steps/model/generation.", flush=True)
    run_generations(config, settings, output, evaluation=evaluation, technical=args.smoke, workers=args.workers, recipe=recipe,
                    confidence_level=args.confidence_level, min_free_gib=args.min_free_gib)
    status = json.loads((output / "manifest.json").read_text())["status"]
    print(f"Run {status}. Results: {output.resolve()}")


if __name__ == "__main__":
    main()
