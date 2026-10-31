"""Continue fixed G=600 parents with fixed input statistics and paired saved controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil

import numpy as np

from .calibrate import evaluate_development, latest_metrics
from .checkpoint_state import restore_member
from .experiments import ExperimentConfig
from .fixed_normalization import freeze_observation_statistics, check_fixed_statistics
from .resume_calibration import seed_log_from_boundary, verified_checkpoint
from .train_centering_study import CORE_FILES
from .train_generations import GenerationSettings
from .train_pilot import write_json


CONTROL_SOURCE = "unchanged_reward"
CONDITION = "fixed_statistics"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generation_settings(plan, seed):
    return GenerationSettings(1, plan["final_generation"], seed_base=seed,
        batch_steps=plan["batch_steps"], minibatch_size=plan["minibatch_size"], epochs=plan["epochs"])


def run_trial(seed, source_root, plan, root, resume=None):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root)
    source = (Path(resume["source_trial"]) if resume is not None else
              Path(source_root) / CONTROL_SOURCE / f"seed_{seed}")
    output = root / CONDITION / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    config = ExperimentConfig.load(source / "experiment.json")
    config.save(output / "experiment.json")
    config.save(output / "evaluation_experiment.json")
    settings = generation_settings(plan, seed)
    parent = resume["generation"] if resume is not None else plan["parent_generation"]
    history = json.loads((source / "evaluations.json").read_text())
    evaluations = [row for row in history if row["generation"] < parent]
    prior = next((row for row in history if row["generation"] == parent), None)
    for row in evaluations:
        name = f"generation_{row['generation']:06d}.csv"
        shutil.copy2(source / "trajectories" / name, output / "trajectories" / name)
    # A resumed member may save many generations before its next evaluation.
    # Commit inherited measurements immediately so another stop is recoverable.
    write_json(output / "evaluations.json", evaluations)
    progress = {"status": "running", "condition": CONDITION, "seed": seed,
                "last_training_generation": parent,
                "last_evaluation_generation": max((row["generation"] for row in evaluations), default=None)}
    member = None
    try:
        if (root / "stop.request").exists():
            progress["status"] = "interrupted"
            return {**progress, "evaluations": evaluations}
        member = restore_member("member_0000", config, settings.pilot(seed),
            output / "models/member_0000", source / "models/member_0000", parent)
        if (member.model.pendulum_gradient_clipping != "independent" or
                member.model.pendulum_training_boundary_m != 3. or
                member.model.pendulum_boundary_failure_cost != 10. or
                getattr(member.model, "pendulum_curriculum", None) is not None):
            raise ValueError("parent algorithm, boundary or curriculum differs")
        seed_log_from_boundary(member, source, parent)
        if resume is not None:
            frozen = json.loads((source / "frozen_observation_statistics.json").read_text())
            check_fixed_statistics(member, frozen)
            if member.model.pendulum_observation_freeze_generation != plan["parent_generation"]:
                raise ValueError("original observation-freeze anchor changed")
        else:
            if getattr(member.model, "pendulum_fixed_observation_statistics", False):
                raise ValueError("new intervention requires a rolling-statistics parent")
            frozen = freeze_observation_statistics(member, parent)
        write_json(output / "frozen_observation_statistics.json", frozen)
        for generation in range(parent, plan["final_generation"] + 1):
            if generation > parent:
                member.update(generation)
            check_fixed_statistics(member, frozen)
            metrics = latest_metrics(output / "models/member_0000/ppo_log/progress.csv")
            if not metrics or not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("missing or nonfinite training diagnostics")
            if generation in plan["evaluation_generations"]:
                row = evaluate_development(member, config, settings, generation, output)
                if generation == parent:
                    if prior is None:
                        raise ValueError("saved boundary evaluation is missing")
                    for field in ("event", "max_abs_x_m", "min_abs_angle_deg", "episode_return"):
                        if row[field] != prior[field]:
                            raise ValueError(f"restored parent evaluation changed: {field}")
                evaluations.append(row)
                write_json(output / "evaluations.json", evaluations)
                progress["last_evaluation_generation"] = generation
                print(f"{CONDITION}, seed={seed}, generation={generation}/{plan['final_generation']}: "
                      f"success={row['event']['success_time']}, max |x|={row['max_abs_x_m']:.3f} m", flush=True)
            else:
                member.save(generation)
            check_fixed_statistics(member, frozen)
            progress.update(last_training_generation=generation, fixed_statistics_verified=True,
                            updated_utc=datetime.now(timezone.utc).isoformat())
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
    parent, final = (2, 4) if smoke else (600, 1000)
    if json.loads((source / "status.json").read_text())["status"] != "complete":
        raise ValueError("matched control study must be complete")
    if (original["final_generation"] != final or original["conditions"][CONTROL_SOURCE] != 1. or
            parent not in original["evaluation_generations"]):
        raise ValueError("source budget, condition or parent evaluation differs")
    if not smoke and (original["seeds"] != [5000, 5001, 5002] or original["batch_steps"] != 1000):
        raise ValueError("use all three predeclared G=600 parents with B=1000")
    for name in CORE_FILES:
        if digest(Path(__file__).parent / name) != original["source_sha256"][name]:
            raise ValueError(f"parent training/evaluation implementation changed: {name}")
    for name, version in original["versions"].items():
        if metadata.version(name) != version:
            raise ValueError(f"parent library changed: {name}")
    plan = {"kind": "technical smoke" if smoke else "paired fixed-observation-statistics development study",
        "source": str(source), "parent_generation": parent, "final_generation": final,
        "seeds": original["seeds"], "batch_steps": original["batch_steps"],
        "minibatch_size": original["minibatch_size"], "epochs": original["epochs"],
        "evaluation_generations": [parent, final] if smoke else [600, 750, 1000],
        "intervention": "Freeze each parent's own observation mean/variance/count at G=600; continue PPO normally",
        "control": "Reuse unchanged_reward rolling-statistics continuations from the same G=600 parents",
        "additional_training_steps": len(original["seeds"]) * (final - parent) * original["batch_steps"],
        "prefix": "Three independent initializations, paired treatment/control branches; shared history through G=600",
        "evaluation": original["evaluation"], "selection_gate": original["selection_gate"],
        "versions": original["versions"],
        "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    if smoke:
        plan["intervention"] = "Technical-only freeze at G=2, continue to G=4"
        plan["prefix"] = "One technical seed, paired saved control; not research results"
    audit = []
    for seed in plan["seeds"]:
        directory = source / CONTROL_SOURCE / f"seed_{seed}"
        for required in plan["evaluation_generations"]:
            generation, rejected = verified_checkpoint(directory / "models/member_0000",
                generation_settings(plan, seed).pilot(seed), required)
            if generation != required or rejected:
                raise ValueError("parent or matched control checkpoint is incomplete or altered")
        config = ExperimentConfig.load(directory / "experiment.json")
        if config.reward.position_scale != 1. or config.reward.form != "bounded":
            raise ValueError("original bounded training reward must remain unchanged")
        if config != ExperimentConfig.load(directory / "evaluation_experiment.json"):
            raise ValueError("matched control must use the same training and evaluation reward")
        filenames = ["evaluations.json", "experiment.json", "evaluation_experiment.json", *[
            f"models/member_0000/generation_{g:06d}{suffix}"
            for g in plan["evaluation_generations"] for suffix in (".zip", ".normalization.pkl", ".resume.pkl")]]
        audit.append({"seed": seed, "source": str(directory),
                      "sha256": {name: digest(directory / name) for name in filenames}})
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    write_json(output / "plan.json", plan)
    write_json(output / "parent_and_control_audit.json", audit)
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    for seed in plan["seeds"]:
        directory = source / CONTROL_SOURCE / f"seed_{seed}"
        retained = output / "rolling_statistics" / f"seed_{seed}"
        retained.mkdir(parents=True)
        for filename in ("evaluations.json", "experiment.json", "evaluation_experiment.json"):
            shutil.copy2(directory / filename, retained / filename)
        write_json(retained / "retained_source.json", {"source": str(directory), "trained_again": False,
            "checkpoint_files": "Original source retains policy, normalization and full recovery state",
            "evaluation_sha256": digest(directory / "evaluations.json")})
    status = {"status": "running", "process_id": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "completed_tasks": 0, "total_tasks": len(plan["seeds"])}
    write_json(output / "status.json", status)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(run_trial, seed, str(source), plan, str(output)) for seed in plan["seeds"]]
            for future in as_completed(futures):
                results.append(future.result())
                results.sort(key=lambda row: row["seed"])
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
    output = args.output or Path("runs") / f"fixed_normalization_{'smoke' if args.smoke else 'development'}_{stamp}"
    run_study(args.source, output, smoke=args.smoke)
    print(f"Fixed-normalization study finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
