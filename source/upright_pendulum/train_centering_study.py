"""Paired reward fine-tuning from fixed G=500 parents, with an equal-budget control."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from numbers import Real
import os
from pathlib import Path
import shutil
import multiprocessing as mp

import gymnasium as gym
import numpy as np

from .calibrate import evaluate_development
from .checkpoint_state import restore_member
from .experiments import ExperimentConfig
from .resume_calibration import seed_log_from_boundary, verified_checkpoint
from .rewards import DenseRewardWrapper
from .train_generations import GenerationSettings
from .train_pilot import write_json


CONDITIONS = {"unchanged_reward": 1.0, "sharper_centering": .3}
CONTROL = "training_failure_at_3m"
CORE_FILES = ("physics.py", "env.py", "stabilization.py", "observations.py", "rewards.py",
              "actions.py", "experiments.py", "normalization.py", "train_pilot.py",
              "train_generations.py", "ppo_variants.py", "cohort.py", "calibrate.py", "training_boundary.py")


def extend_member(source, output, old_config, new_config, old_settings, new_settings, generation):
    """Restore every training state, then change only budget and declared reward."""
    if new_settings.total_steps <= old_settings.total_steps:
        raise ValueError("the declared new budget must extend the original budget")
    if {k: v for k, v in asdict(old_settings).items() if k != "total_steps"} != {
            k: v for k, v in asdict(new_settings).items() if k != "total_steps"}:
        raise ValueError("only total_steps may change in the PPO settings")
    if old_config.observation != new_config.observation:
        raise ValueError("reward fine-tuning must preserve observation encoding")
    member = restore_member("member_0000", old_config, old_settings, output, source, generation)
    try:
        if not isinstance(member.model.learning_rate, Real):
            raise ValueError("budget extension requires the original constant learning rate")
        if getattr(member.model, "pendulum_curriculum", None) is not None:
            raise ValueError("this comparison uses the fixed-downward boundary parents only")
        env = member.env.venv.envs[0]
        while isinstance(env, gym.Wrapper):
            if isinstance(env, DenseRewardWrapper):
                env.config = new_config.reward
                break
            env = env.env
        else:
            raise ValueError("training reward wrapper was not found")
        member.settings = new_settings
        member.model.pendulum_reward_config = asdict(new_config.reward)
        member.model.pendulum_finetune_parent = {"source": str(Path(source).resolve()), "generation": generation}
        return member
    except BaseException:
        member.close()
        raise


def run_trial(condition, seed, source_root, plan, root):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root)
    source = Path(source_root) / CONTROL / f"seed_{seed}"
    output = root / condition / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    old_config = ExperimentConfig.load(source / "experiment.json")
    new_config = replace(old_config, reward=replace(old_config.reward, position_scale=CONDITIONS[condition]))
    new_config.save(output / "experiment.json")
    old_config.save(output / "evaluation_experiment.json")
    old = GenerationSettings(1, plan["parent_generation"], seed_base=seed,
        batch_steps=plan["batch_steps"], minibatch_size=plan["minibatch_size"], epochs=plan["epochs"])
    new = replace(old, generations=plan["final_generation"])
    old_history = json.loads((source / "evaluations.json").read_text())
    evaluations = [row for row in old_history if row["generation"] < plan["parent_generation"]]
    for row in evaluations:
        name = f"generation_{row['generation']:06d}.csv"
        shutil.copy2(source / "trajectories" / name, output / "trajectories" / name)
    progress = {"status": "running", "condition": condition, "seed": seed,
                "last_training_generation": plan["parent_generation"], "last_evaluation_generation": None}
    member = None
    try:
        if (root / "stop.request").exists():
            progress["status"] = "interrupted"
            return {**progress, "evaluations": evaluations}
        member = extend_member(source / "models/member_0000", output / "models/member_0000",
            old_config, new_config, old.pilot(seed), new.pilot(seed), plan["parent_generation"])
        if (member.model.pendulum_gradient_clipping != "independent" or
                member.model.pendulum_training_boundary_m != 3. or
                member.model.pendulum_boundary_failure_cost != 10.):
            raise ValueError("parent optimizer or training boundary differs")
        seed_log_from_boundary(member, source, plan["parent_generation"])
        for generation in range(plan["parent_generation"], plan["final_generation"] + 1):
            if generation > plan["parent_generation"]:
                member.update(generation)
            if generation in plan["evaluation_generations"]:
                # Both branches report the same evaluation reward as well as
                # identical physical success criteria. Training reward differs.
                row = evaluate_development(member, old_config, new, generation, output)
                if not all(np.isfinite(v) for v in row["metrics"].values()):
                    raise FloatingPointError("nonfinite training diagnostics")
                if generation == plan["parent_generation"]:
                    prior = old_history[-1]
                    for field in ("event", "max_abs_x_m", "min_abs_angle_deg", "episode_return"):
                        if row[field] != prior[field]:
                            raise ValueError(f"restored parent evaluation changed: {field}")
                evaluations.append(row)
                write_json(output / "evaluations.json", evaluations)
                progress["last_evaluation_generation"] = generation
                print(f"{condition}, seed={seed}, generation={generation}/{plan['final_generation']}: "
                      f"success={row['event']['success_time']}, max |x|={row['max_abs_x_m']:.3f} m", flush=True)
            else:
                member.save(generation)
            progress.update(last_training_generation=generation, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "progress.json", progress)
            if (root / "stop.request").exists() and generation < plan["final_generation"]:
                progress["status"] = "interrupted"
                break
        else:
            progress["status"] = "complete"
        return {**progress, "evaluations": evaluations}
    except BaseException as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "progress.json", progress)
        if member is not None:
            member.close()


def run_study(source, output, *, smoke=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / "plan.json").read_text())
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("parent study must be complete")
    parent, final = (2, 4) if smoke else (500, 1000)
    if (original["generations"] != parent or original["conditions_training_boundary_m"][CONTROL] != 3. or
            original["boundary_failure_cost"] != 10.):
        raise ValueError("parent budget or condition differs")
    for name in CORE_FILES:
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != original["source_sha256"][name]:
            raise ValueError(f"parent training/evaluation implementation changed: {name}")
    for name, version in original["versions"].items():
        if metadata.version(name) != version:
            raise ValueError(f"parent library changed: {name}")
    if original["gradient_clipping"] != "independent" or original["reward_form"] != "bounded":
        raise ValueError("parent algorithm/reward differs")
    b = original["experience_steps_per_generation"]
    audit = []
    for seed in original["seeds"]:
        directory = source / CONTROL / f"seed_{seed}"
        settings = GenerationSettings(1, parent, seed_base=seed, batch_steps=b,
            minibatch_size=original["ppo_settings"]["batch_size"], epochs=original["ppo_settings"]["epochs"])
        if asdict(settings.pilot(seed)) != {**original["ppo_settings"], "seed": seed}:
            raise ValueError("parent scalar PPO settings differ")
        generation, rejected = verified_checkpoint(directory / "models/member_0000", settings.pilot(seed), parent)
        if generation != parent or rejected:
            raise ValueError("parent checkpoint is incomplete or altered")
        if ExperimentConfig.load(directory / "experiment.json").reward.position_scale != 1.:
            raise ValueError("parent position scale must be 1m")
        filenames = ["evaluations.json", "experiment.json", *[
            f"models/member_0000/generation_{parent:06d}{suffix}" for suffix in (".zip", ".normalization.pkl", ".resume.pkl")]]
        audit.append({"seed": seed, "source": str(directory), "sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames}})
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    grid = [parent, final] if smoke else [500, 600, 750, 1000]
    plan = {"kind": "technical smoke" if smoke else "paired centering-reward fine-tuning development study",
        "source": str(source), "parent_generation": parent, "final_generation": final, "conditions": CONDITIONS,
        "seeds": original["seeds"], "batch_steps": b, "minibatch_size": original["ppo_settings"]["batch_size"],
        "epochs": original["ppo_settings"]["epochs"], "evaluation_generations": grid,
        "additional_training_steps": len(CONDITIONS) * len(original["seeds"]) * (final - parent) * b,
        "prefix": "Both branches of a seed share all parent experience; branches are paired, not six independent initializations",
        "evaluation": "Original unbounded plant, exact downward start, full20s, original event at2ms; common parent evaluation reward for both branches",
        "selection_gate": "At least 2/3 seeds succeed without later exit at final predeclared G=1000 with finite diagnostics and original physical checks. No best-checkpoint selection; no automatic large run.",
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
        "versions": original["versions"]}
    write_json(output / "plan.json", plan)
    write_json(output / "parent_audit.json", audit)
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    status = {"status": "running", "process_id": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "completed_tasks": 0, "total_tasks": len(CONDITIONS) * len(original["seeds"])}
    write_json(output / "status.json", status)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(run_trial, condition, seed, str(source), plan, str(output))
                       for seed in original["seeds"] for condition in CONDITIONS]
            for future in as_completed(futures):
                results.append(future.result())
                results.sort(key=lambda row: (row["condition"], row["seed"]))
                write_json(output / "results.json", results)
                status["completed_tasks"] = sum(row["status"] == "complete" for row in results)
                write_json(output / "status.json", status)
        status["status"] = "complete" if all(r["status"] == "complete" for r in results) else "interrupted"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_or_stopped_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"centering_{'smoke' if args.smoke else 'development'}_{stamp}"
    run_study(args.source, output, smoke=args.smoke)
    print(f"Centering study finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
