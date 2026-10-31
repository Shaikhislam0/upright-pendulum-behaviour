"""Bounded, fresh-seed development comparison; never extends the failed cohort."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .cohort import EvaluationSettings
from .experiments import ExperimentConfig
from .rewards import RewardConfig
from .train_generations import GenerationSettings, TrainingMember, evaluate_members
from .train_pilot import write_json


CONDITIONS = {
    "historical_baseline": ("quadratic", "global"),
    "bounded_reward_only": ("bounded", "global"),
    "independent_clipping_only": ("quadratic", "independent"),
    "bounded_and_independent": ("bounded", "independent"),
}


def latest_metrics(path):
    import csv
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {key: float(value) for key, value in rows[-1].items() if value and key.startswith("train/")}


def evaluate_development(member, config, settings, generation, output):
    trajectory = output / "trajectories" / f"generation_{generation:06d}.csv"
    record, hashes = evaluate_members([member], config, settings, EvaluationSettings(), generation, output, trajectory)
    import csv
    with trajectory.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    min_angle = 180.
    near_origin_upright = False
    for row in rows:
        theta = float(row["theta_rad_unwrapped"])
        angle = abs(float(np.degrees(np.arctan2(np.sin(theta), np.cos(theta)))))
        min_angle = min(min_angle, angle)
        near_origin_upright |= angle <= 30 and abs(float(row["x_m"])) <= 2
    return {"generation": generation, "event": record["members"][0],
        "max_abs_x_m": max(abs(float(row["x_m"])) for row in rows),
        "min_abs_angle_deg": min_angle, "ever_upright_30deg_within_2m": near_origin_upright,
        "episode_return": sum(float(row["reward"]) for row in rows if row["reward"]),
        "metrics": {} if generation == 0 else latest_metrics(output / "models" / member.member_id / "ppo_log/progress.csv")}


def run_task(condition, seed, generations, evaluation_grid, root):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    form, clipping = CONDITIONS[condition]
    config = ExperimentConfig(name=condition, reward=RewardConfig(form=form))
    settings = GenerationSettings(1, generations, seed_base=seed)
    output = Path(root) / condition / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    config.save(output / "experiment.json")
    member = None
    evaluations = []
    start = time.perf_counter()
    progress = {"status": "running", "condition": condition, "seed": seed,
                "last_training_generation": 0, "last_evaluation_generation": None}
    try:
        member = TrainingMember("member_0000", config, settings.pilot(seed), output / "models/member_0000",
                                gradient_clipping=clipping)
        for generation in range(generations + 1):
            if generation:
                member.update(generation)
            if generation in evaluation_grid:
                result = evaluate_development(member, config, settings, generation, output)
                evaluations.append(result)
                write_json(output / "evaluations.json", evaluations)
                progress["last_evaluation_generation"] = generation
                print(f"{condition}, seed={seed}, generation={generation}/{generations}: "
                      f"success={result['event']['success_time']}, max |x|={result['max_abs_x_m']:.2f} m, "
                      f"closest angle={result['min_abs_angle_deg']:.1f} deg", flush=True)
            else:
                member.save(generation)
            progress["last_training_generation"] = generation
            progress["updated_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "progress.json", progress)
        progress["status"] = "complete"
        return {"condition": condition, "seed": seed, "evaluations": evaluations,
                "elapsed_wall_s": time.perf_counter() - start}
    except BaseException as error:
        progress["status"] = "failed"
        progress["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "progress.json", progress)
        if member is not None:
            member.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.generations < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds) or args.workers not in (1, 2):
        parser.error("use positive generations, distinct seeds, and one or two workers")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"development_ablation_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))
    grid = sorted({0, *[g for g in (10, 25, 50, 100) if g <= args.generations], args.generations})
    plan = {"kind": "development comparison, not main-study evidence", "conditions": CONDITIONS,
        "seeds": args.seeds, "generations": args.generations, "experience_steps_per_generation": 1000,
        "training_steps_total": len(CONDITIONS) * len(args.seeds) * args.generations * 1000,
        "evaluation_generations": grid, "evaluation_horizon_s": 20,
        "selection_gate": "At least 2/3 development seeds succeed at the final predeclared generation with no post-success exit, finite training, and unchanged physical/event checks; otherwise no large follow-on run.",
        "fixed_settings": asdict(GenerationSettings(1, args.generations).pilot(args.seeds[0])),
        "versions": {name: metadata.version(name) for name in ("numpy", "gymnasium", "torch", "stable-baselines3")},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob("*.py")}}
    write_json(output / "plan.json", plan)
    snapshot = output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    status = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(), "completed_tasks": 0,
              "total_tasks": len(CONDITIONS) * len(args.seeds), "process_id": os.getpid()}
    write_json(output / "status.json", status)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(run_task, condition, seed, args.generations, grid, str(output.resolve()))
                       for condition in CONDITIONS for seed in args.seeds]
            for future in as_completed(futures):
                results.append(future.result())
                results.sort(key=lambda row: (row["condition"], row["seed"]))
                write_json(output / "results.json", results)
                status["completed_tasks"] = len(results)
                write_json(output / "status.json", status)
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)
    print(f"Completed development comparison: {output.resolve()}")


if __name__ == "__main__":
    main()
