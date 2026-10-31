"""Continue an interrupted development comparison without replacing its source."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import shutil
import time

from .calibrate import CONDITIONS, evaluate_development, run_task
from .checkpoint_state import restore_member
from .experiments import ExperimentConfig
from .train_generations import GenerationSettings
from .train_pilot import write_json


def verified_checkpoint(directory, settings, maximum):
    """Choose the newest committed checkpoint, excluding incomplete later writes."""
    directory = Path(directory)
    rejected = []
    for sidecar in sorted(directory.glob("generation_*.resume.pkl"), reverse=True):
        generation = int(sidecar.name.split("_")[1].split(".")[0])
        if generation > maximum:
            continue
        try:
            with sidecar.open("rb") as file:
                state = pickle.load(file)  # Trusted local research output only.
            if (state["schema"] != 1 or state["generation"] != generation or
                    state["member_id"] != "member_0000" or state["settings"] != asdict(settings)):
                raise ValueError("recovery identity/settings disagree")
            stem = directory / f"generation_{generation:06d}"
            for field, suffix in (("checkpoint", ".zip"), ("statistics", ".normalization.pkl")):
                if hashlib.sha256(Path(str(stem) + suffix).read_bytes()).hexdigest() != state["sha256"][field]:
                    raise ValueError(f"{field} differs from committed hash")
            return generation, rejected
        except (OSError, EOFError, ValueError, KeyError, pickle.UnpicklingError) as error:
            rejected.append({"generation": generation, "reason": str(error)})
    raise ValueError(f"no verified committed checkpoint in {directory}; rejected={rejected}")


def validate_source(source, plan):
    current = Path(__file__).parent
    for name, expected in plan["source_sha256"].items():
        if hashlib.sha256((current / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"original implementation changed: {name}")
    for name, expected in plan["versions"].items():
        if metadata.version(name) != expected:
            raise ValueError(f"original library version changed: {name}")
    if {name: list(value) for name, value in CONDITIONS.items()} != plan["conditions"]:
        raise ValueError("development conditions changed")
    if json.loads((source / "status.json").read_text())["status"] not in ("interrupted", "failed"):
        raise ValueError("source must be marked interrupted or failed")


def seed_log_from_boundary(member, source, generation):
    """Keep the last committed update's diagnostics available for boundary evaluation."""
    log = source / "models/member_0000/ppo_log/progress.csv"
    if not log.is_file():
        return
    with log.open(newline="") as file:
        rows = [row for row in csv.DictReader(file) if float(row["time/generation"]) == generation]
    if rows:
        for key, value in rows[-1].items():
            if value:
                member.model.logger.record(key, float(value))
        member.model.logger.dump(step=generation * member.settings.rollout_steps)


def resume_task(condition, seed, generations, grid, root, source, generation):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source = Path(source)
    output = Path(root) / condition / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    config = ExperimentConfig.load(source / "experiment.json")
    config.save(output / "experiment.json")
    settings = GenerationSettings(1, generations, seed_base=seed)
    history_path = source / "evaluations.json"
    history = json.loads(history_path.read_text()) if history_path.is_file() else []
    evaluations = [row for row in history if row["generation"] <= generation]
    for row in evaluations:
        name = f"generation_{row['generation']:06d}.csv"
        shutil.copy2(source / "trajectories" / name, output / "trajectories" / name)
    write_json(output / "evaluations.json", evaluations)
    progress = {"status": "running", "condition": condition, "seed": seed,
                "last_training_generation": generation,
                "last_evaluation_generation": evaluations[-1]["generation"] if evaluations else None,
                "source": str(source), "restored_generation": generation}
    member = None
    start = time.perf_counter()
    try:
        member = restore_member("member_0000", config, settings.pilot(seed),
                                output / "models/member_0000", source / "models/member_0000", generation)
        if member.model.pendulum_gradient_clipping != CONDITIONS[condition][1]:
            raise ValueError("restored optimizer variant differs from declared condition")
        seed_log_from_boundary(member, source, generation)
        member.save(generation)
        for g in range(generation, generations + 1):
            if g > generation:
                member.update(g)
            if g in grid and not any(row["generation"] == g for row in evaluations):
                result = evaluate_development(member, config, settings, g, output)
                evaluations.append(result)
                write_json(output / "evaluations.json", evaluations)
                progress["last_evaluation_generation"] = g
                print(f"{condition}, seed={seed}, generation={g}/{generations}: "
                      f"success={result['event']['success_time']}, max |x|={result['max_abs_x_m']:.2f} m, "
                      f"closest angle={result['min_abs_angle_deg']:.1f} deg", flush=True)
            elif g > generation:
                member.save(g)
            progress.update(last_training_generation=g, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "progress.json", progress)
        progress["status"] = "complete"
        return {"condition": condition, "seed": seed, "evaluations": evaluations,
                "resumed_from_generation": generation, "elapsed_continuation_wall_s": time.perf_counter() - start}
    except BaseException as error:
        progress.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(output / "progress.json", progress)
        if member is not None:
            member.close()


def continue_comparison(source, output, workers=2):
    source, output = Path(source).resolve(), Path(output).resolve()
    plan = json.loads((source / "plan.json").read_text())
    validate_source(source, plan)
    tasks, completed, audit = [], [], []
    for condition in plan["conditions"]:
        for seed in plan["seeds"]:
            directory = source / condition / f"seed_{seed}"
            if not directory.is_dir():
                tasks.append((condition, seed, None, None))
                audit.append({"condition": condition, "seed": seed, "action": "fresh"})
                continue
            settings = GenerationSettings(1, plan["generations"], seed_base=seed).pilot(seed)
            generation, rejected = verified_checkpoint(directory / "models/member_0000", settings, plan["generations"])
            history_path = directory / "evaluations.json"
            history = json.loads(history_path.read_text()) if history_path.is_file() else []
            finished = generation == plan["generations"] and {r["generation"] for r in history} == set(plan["evaluation_generations"])
            audit.append({"condition": condition, "seed": seed, "action": "retain" if finished else "resume",
                          "generation": generation, "rejected_commits": rejected})
            if finished:
                completed.append({"condition": condition, "seed": seed, "evaluations": history,
                                  "retained_source": str(directory)})
            else:
                tasks.append((condition, seed, str(directory), generation))
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    write_json(output / "plan.json", {**plan, "continuation_source": str(source),
        "continuation_sources_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}})
    write_json(output / "recovery_audit.json", audit)
    snapshot = output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    for result in completed:
        src = Path(result["retained_source"])
        dest = output / result["condition"] / f"seed_{result['seed']}"
        dest.mkdir(parents=True)
        for name in ("experiment.json", "evaluations.json", "progress.json"):
            shutil.copy2(src / name, dest / name)
        write_json(dest / "retained_source.json", {"source": str(src), "reason": "completed model retained without further training"})
    status = {"status": "running", "process_id": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "completed_tasks": len(completed), "total_tasks": len(CONDITIONS) * len(plan["seeds"])}
    write_json(output / "status.json", status)
    write_json(output / "results.json", completed)
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
            futures = []
            for condition, seed, directory, generation in tasks:
                args = (condition, seed, plan["generations"], plan["evaluation_generations"], str(output))
                futures.append(executor.submit(run_task, *args) if directory is None else
                               executor.submit(resume_task, *args, directory, generation))
            for future in as_completed(futures):
                completed.append(future.result())
                completed.sort(key=lambda row: (row["condition"], row["seed"]))
                write_json(output / "results.json", completed)
                status["completed_tasks"] = len(completed)
                write_json(output / "status.json", status)
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"development_resumed_{stamp}"
    continue_comparison(args.source, output)
    print(f"Completed comparison: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
