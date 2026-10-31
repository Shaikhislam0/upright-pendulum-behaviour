"""Paired development test of a declared training failure boundary versus longer training."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import multiprocessing as mp
import os
from pathlib import Path
import shutil

import numpy as np

from .calibrate import evaluate_development
from .experiments import ExperimentConfig
from .rewards import RewardConfig
from .train_generations import GenerationSettings, TrainingMember
from .train_pilot import write_json


CONDITIONS = {"unbounded_training": None, "training_failure_at_3m": 3.0}


def run_trial(condition, seed, generations, grid, root, *, smoke=False):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root = Path(root)
    output = root / condition / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    config = ExperimentConfig(name="bounded_independent_boundary_study", reward=RewardConfig(form="bounded"))
    config.save(output / "experiment.json")
    settings = GenerationSettings(1, generations, seed_base=seed,
        batch_steps=32 if smoke else 1000, minibatch_size=32 if smoke else 250, epochs=2 if smoke else 10)
    progress = {"status": "running", "condition": condition, "seed": seed,
                "last_training_generation": None, "last_evaluation_generation": None}
    member, evaluations = None, []
    try:
        if (root / "stop.request").exists():
            progress["status"] = "interrupted"
            return {**progress, "evaluations": []}
        member = TrainingMember("member_0000", config, settings.pilot(seed), output / "models/member_0000",
                                gradient_clipping="independent", training_boundary_m=CONDITIONS[condition],
                                boundary_failure_cost=10.0)
        for generation in range(generations + 1):
            if generation:
                member.update(generation)
            if generation in grid:
                result = evaluate_development(member, config, settings, generation, output)
                if not all(np.isfinite(value) for value in result["metrics"].values()):
                    raise FloatingPointError("nonfinite training diagnostics")
                evaluations.append(result)
                write_json(output / "evaluations.json", evaluations)
                progress["last_evaluation_generation"] = generation
                print(f"{condition}, seed={seed}, generation={generation}/{generations}: "
                      f"success={result['event']['success_time']}, max |x|={result['max_abs_x_m']:.2f} m, "
                      f"closest angle={result['min_abs_angle_deg']:.1f} deg", flush=True)
            else:
                member.save(generation)
            progress.update(last_training_generation=generation, updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "progress.json", progress)
            if (root / "stop.request").exists() and generation < generations:
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


def run_study(output, *, generations=500, seeds=(5000, 5001, 5002), smoke=False):
    output = Path(output).resolve()
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("use distinct, nonempty seeds")
    settings = GenerationSettings(1, generations, seed_base=seeds[0],
        batch_steps=32 if smoke else 1000, minibatch_size=32 if smoke else 250, epochs=2 if smoke else 10)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    grid = sorted({0, *[g for g in (50, 100, 200, 300, 500) if g <= generations], generations})
    plan = {"kind": "technical smoke" if smoke else "paired development study; not the main 100-model experiment",
            "conditions_training_boundary_m": CONDITIONS, "seeds": list(seeds), "generations": generations,
            "evaluation_generations": grid, "experience_steps_per_generation": settings.batch_steps,
            "training_steps_total": len(CONDITIONS) * len(seeds) * generations * settings.batch_steps,
            "ppo_settings": asdict(settings.pilot(seeds[0])), "gradient_clipping": "independent",
            "reward_form": "bounded", "boundary_failure_cost": 10.0,
            "boundary_semantics": "terminal failure in training MDP at control endpoint |x|>=3m; no state clipping; no angle termination",
            "evaluation": "Both conditions reload frozen policies on original unbounded rail, exact downward start, 20s, original four tolerances and 2s hold at 2ms monitoring",
            "selection_gate": "At least 2/3 seeds achieve the original event and have no later exit at final predeclared G=500, all diagnostics finite and physical checks passed. No best-checkpoint selection or automatic main-study launch.",
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")},
            "versions": {name: metadata.version(name) for name in ("numpy", "gymnasium", "torch", "stable-baselines3")}}
    write_json(output / "plan.json", plan)
    snapshot = output / "source"
    snapshot.mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    status = {"status": "running", "process_id": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "completed_tasks": 0, "total_tasks": len(CONDITIONS) * len(seeds)}
    write_json(output / "status.json", status)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(run_trial, condition, seed, generations, grid, str(output), smoke=smoke)
                       for seed in seeds for condition in CONDITIONS]
            for future in as_completed(futures):
                results.append(future.result())
                results.sort(key=lambda row: (row["condition"], row["seed"]))
                write_json(output / "results.json", results)
                status["completed_tasks"] = sum(row["status"] == "complete" for row in results)
                write_json(output / "status.json", status)
        status["status"] = "complete" if all(row["status"] == "complete" for row in results) else "interrupted"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "status.json", status)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"boundary_{'smoke' if args.smoke else 'development'}_{stamp}"
    run_study(output, generations=2 if args.smoke else 500,
              seeds=(5999,) if args.smoke else (5000, 5001, 5002), smoke=args.smoke)
    print(f"Study finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
