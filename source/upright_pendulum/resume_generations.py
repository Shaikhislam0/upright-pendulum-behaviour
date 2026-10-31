"""Recover a saved complete cohort into a new directory, preserving the source."""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import time

from .cohort import EvaluationSettings
from .cohort_results import surface
from .experiments import ExperimentConfig
from .train_generations import GenerationSettings
from .workers import WorkerPool


def _core_ast(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RandomStream":
            result[node.name] = ast.dump(node, include_attributes=False)
        if isinstance(node, ast.ClassDef) and node.name == "TrainingMember":
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in ("__init__", "update"):
                    result[method.name] = ast.dump(method, include_attributes=False)
    return result


def validate_legacy_implementation(source, manifest):
    """Reject reconstruction if learning code, core physics, or libraries changed."""
    current = Path(__file__).parent
    snapshot = Path(str(source) + ".sources")
    if not snapshot.is_dir():
        raise ValueError("legacy source snapshot is required")
    if _core_ast(snapshot / "train_generations.py") != _core_ast(current / "train_generations.py"):
        raise ValueError("legacy model initialization/update/RNG implementation changed")
    for name in ("physics.py", "env.py", "stabilization.py", "observations.py", "rewards.py",
                 "actions.py", "experiments.py", "normalization.py", "train_pilot.py"):
        if hashlib.sha256((current / name).read_bytes()).hexdigest() != manifest["source_sha256"][name]:
            raise ValueError(f"legacy recovery core implementation changed: {name}")
    for name in ("numpy", "torch", "stable-baselines3", "gymnasium"):
        if metadata.version(name) != manifest["versions"][name]:
            raise ValueError(f"legacy recovery requires original {name} version")


def validate_full_implementation(source, manifest):
    """Full state does not authorize changing the equations or learning code."""
    current = Path(__file__).parent
    snapshot = Path(source) / "source"
    expected = manifest.get("continuation_source_sha256", manifest["source_sha256"])
    if not snapshot.is_dir():
        raise ValueError("full recovery requires its source snapshot")
    if _core_ast(snapshot / "train_generations.py") != _core_ast(current / "train_generations.py"):
        raise ValueError("saved model initialization/update/RNG implementation changed")
    for name in ("physics.py", "env.py", "stabilization.py", "observations.py", "rewards.py", "actions.py",
                 "experiments.py", "normalization.py", "train_pilot.py", "ppo_variants.py", "training_recipe.py",
                 "checkpoint_state.py", "cohort_execution.py", "cohort.py"):
        if name in expected:
            saved = snapshot / name
            if (hashlib.sha256(saved.read_bytes()).hexdigest() != expected[name] or
                    hashlib.sha256((current / name).read_bytes()).hexdigest() != expected[name]):
                raise ValueError(f"full recovery core implementation changed: {name}")
    for name in ("numpy", "torch", "stable-baselines3", "gymnasium", "scipy"):
        if name in manifest["versions"] and metadata.version(name) != manifest["versions"][name]:
            raise ValueError(f"full recovery requires original {name} version")


def run_continuation(source, output, *, generation, workers=2):
    source, output = Path(source).resolve(), Path(output).resolve()
    original = json.loads((source / "manifest.json").read_text())
    settings = GenerationSettings(**original["settings"])
    evaluation = EvaluationSettings(**original["evaluation"])
    config = ExperimentConfig.load(source / "experiment.json")
    from .training_recipe import TrainingRecipe
    recipe = TrainingRecipe(**original.get("training_recipe", {}))
    from .cohort_execution import atomic_json, collect_generations, validate_reserve
    confidence_level = original.get("inference", {}).get("confidence_level")
    min_free_gib = original.get("resource_policy", {}).get("min_free_gib", 2.)
    validate_reserve(min_free_gib)
    if original["status"] not in ("interrupted", "failed"):
        raise ValueError("source must be explicitly marked interrupted or failed")
    if type(generation) is not int or not 0 <= generation <= settings.generations:
        raise ValueError("invalid recovery generation")
    records = json.loads((source / "generations.json").read_text())
    # Ignore only uncommitted rows in a new output; preserve the source intact.
    records = [row for row in records if row["generation"] in original["completed_generations"]]
    if records:
        surface(records)
    last_evaluated = records[-1]["generation"] if records else 0
    if settings.evaluation_every == 1:
        maximum = last_evaluated + 1
    else:
        maximum = original.get("last_fully_saved_policy_generation", last_evaluated)
    if generation < last_evaluated or generation > maximum:
        raise ValueError("recover a committed cohort at or after its last completed evaluation")
    if type(workers) is not int or not 2 <= workers <= settings.models:
        raise ValueError("continuation requires 2..n persistent workers")
    sidecars = []
    for i in range(settings.models):
        stem = source / "models" / f"member_{i:04d}" / f"generation_{generation:06d}"
        for suffix in (".zip", ".normalization.pkl"):
            if not Path(str(stem) + suffix).is_file():
                raise ValueError(f"incomplete saved cohort: {stem}{suffix}")
        sidecars.append(Path(str(stem) + ".resume.pkl").is_file())
    if any(sidecars) and not all(sidecars):
        raise ValueError("mixed or partially committed recovery states; use an earlier complete cohort")
    legacy = not any(sidecars)
    if legacy:
        if settings.batch_steps != 1000:
            raise ValueError("legacy reconstruction requires complete 1,000-step episodes")
        validate_legacy_implementation(source, original)
    else:
        validate_full_implementation(source, original)
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    config.save(output / "experiment.json")
    snapshot = output / "source"
    snapshot.mkdir()
    for file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(file, snapshot / file.name)
    # Preserve completed source records; replace only the re-evaluated boundary
    # in this new continuation directory, never in the source run.
    source_records_hash = hashlib.sha256((source / "generations.json").read_bytes()).hexdigest()
    records = [record for record in records if record["generation"] < generation]
    manifest = copy.deepcopy(original)
    manifest.update(status="running", kind="recovered_generation_cohort", finished_utc=None,
        error=None, started_utc=datetime.now(timezone.utc).isoformat(),
        completed_generations=[record["generation"] for record in records], checkpoint_sha256={},
        source_checkpoint_sha256=original["checkpoint_sha256"], timings=[],
        execution={"workers": workers, "torch_threads_per_worker": 1,
                   "generation_barriers": "all updates before evaluation; all evaluations before next update"},
        process_id=os.getpid(),
        recovery={"source_run": str(source), "generation": generation,
                  "reason": original.get("interruption_cause", "interrupted checkpoint recovery"), "source_event_records_sha256": source_records_hash,
                  "training_state_method": "validated legacy sampling-schedule reconstruction" if legacy else "committed full recovery state",
                  "source_completed_generations": original["completed_generations"],
                  "note": "Earlier measurements refer to source run; evaluate only declared checkpoints after restoration, without adding non-grid measurements."},
        continuation_source_sha256={file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in snapshot.glob("*.py")})
    manifest["training_recipe"] = original.get("training_recipe", {})
    manifest["evaluation_generations"] = settings.evaluation_generations()
    for field in ("last_complete_evaluation_generation", "last_fully_saved_policy_generation",
                  "interruption_detected_utc", "environment_steps_per_model_completed", "cohort_training_steps_completed"):
        manifest.pop(field, None)
    atomic_json(output / "manifest.json", manifest)
    pool = None
    start = time.perf_counter()
    try:
        pool = WorkerPool(workers, config, settings, evaluation, output,
            recovery={"source": str(source), "generation": generation, "legacy": legacy}, recipe=recipe)
        manifest["initialization_s"] = time.perf_counter() - start
        return collect_generations([], pool, config, settings, evaluation, output, manifest, records,
            start_generation=generation, restored=True, confidence_level=confidence_level, min_free_gib=min_free_gib)
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["elapsed_wall_s"] = time.perf_counter() - start
        last = manifest.get("last_training_generation", generation)
        manifest["environment_steps_per_model_completed"] = last * settings.batch_steps
        manifest["cohort_training_steps_completed"] = last * settings.batch_steps * settings.models
        atomic_json(output / "manifest.json", manifest)
        if pool is not None:
            pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"{args.source.name}_resumed_{timestamp}"
    print(f"Recovering cohort at generation {args.generation}; source data remains in {args.source}", flush=True)
    run_continuation(args.source, output, generation=args.generation, workers=args.workers)
    status = json.loads((output / "manifest.json").read_text())["status"]
    print(f"Run {status}. Results: {output.resolve()}")


if __name__ == "__main__":
    main()
