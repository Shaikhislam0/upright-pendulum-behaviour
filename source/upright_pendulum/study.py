"""Check a fixed study configuration; --start explicitly executes that study."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import shutil

from .cohort import EvaluationSettings
from .cohort_execution import atomic_json, validate_reserve
from .cohort_results import validate_confidence
from .experiments import ExperimentConfig
from .train_generations import GenerationSettings
from .training_recipe import TrainingRecipe


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declared_seeds(data):
    result = set(data.get("seeds", []))
    result.update(data.get("member_seeds", {}).values())
    if type(data.get("seed")) is int:
        result.add(data["seed"])
    settings = data.get("settings", {})
    if "seed_base" in settings and "models" in settings:
        result.update(range(settings["seed_base"], settings["seed_base"] + settings["models"]))
    return result


def prepare_study(path, workspace=None):
    workspace = Path(workspace or Path.cwd()).resolve()
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if data["schema_version"] != 1:
        raise ValueError("unsupported study schema")
    settings = GenerationSettings(**data["settings"])
    recipe = TrainingRecipe(**data["training_recipe"])
    evaluation = EvaluationSettings(**data["evaluation"])
    experiment_path = workspace / data["experiment_file"]
    config = ExperimentConfig.load(experiment_path)
    validate_confidence(data["confidence_level"])
    validate_reserve(data["min_free_gib"])
    if type(data["workers"]) is not int or not 1 <= data["workers"] <= settings.models:
        raise ValueError("invalid worker count")
    if data["workers"] > 1 and evaluation.parallel:
        raise ValueError("nested evaluation workers are not supported")
    safety = data["storage_safety_factor"]
    if isinstance(safety, bool) or not isinstance(safety, (int, float)) or not math.isfinite(safety) or safety < 1:
        raise ValueError("storage safety factor must be finite and at least one")
    seeds = set(range(settings.seed_base, settings.seed_base + settings.models))
    metadata_paths = []
    for directory in (workspace / "runs", workspace / "analysis"):
        if directory.is_dir():
            metadata_paths.extend(directory.rglob("manifest.json"))
            metadata_paths.extend(directory.rglob("plan.json"))
    collisions = []
    for prior in metadata_paths:
        values = declared_seeds(json.loads(prior.read_text()))
        overlap = sorted(seeds & values)
        if overlap:
            collisions.append({"source": str(prior), "seeds": overlap})
    if collisions:
        raise ValueError(f"study seed range overlaps registered prior work: {collisions}")
    sample_paths = [workspace / name for name in data["representative_artifacts"]]
    sample_bytes = sum(item.stat().st_size for item in sample_paths)
    estimated_bytes = sample_bytes * settings.models * len(settings.evaluation_generations())
    reserve_bytes = data["min_free_gib"] * 1024**3
    required_bytes = math.ceil(safety * estimated_bytes + reserve_bytes)
    free = shutil.disk_usage(workspace).free
    if free < required_bytes:
        raise ValueError("available disk space does not cover the study estimate, safety multiplier and reserve")
    report = {"status": "preflight passed; study not launched by preparation",
        "checked_utc": datetime.now(timezone.utc).isoformat(), "study_config": str(path),
        "settings": asdict(settings), "training_recipe": asdict(recipe), "experiment": config.to_dict(),
        "evaluation": asdict(evaluation), "workers": data["workers"],
        "confidence_level": data["confidence_level"], "inference": "Pointwise Clopper-Pearson exact binomial; not simultaneous bands",
        "member_seeds": sorted(seeds), "seed_registry_files_checked": len(metadata_paths),
        "training_interactions": settings.models * settings.generations * settings.batch_steps,
        "evaluation_generations": settings.evaluation_generations(),
        "evaluation_episodes": settings.models * len(settings.evaluation_generations()),
        "evaluation_interactions": settings.models * len(settings.evaluation_generations()) * evaluation.steps,
        "storage": {"representative_bytes_per_checkpoint_and_trajectory": sample_bytes,
                    "estimated_artifact_bytes": estimated_bytes, "safety_factor": safety,
                    "min_free_gib": data["min_free_gib"], "required_free_bytes": required_bytes,
                    "currently_free_bytes": free,
                    "limitation": "Empirical projection, not a guaranteed bound; runtime reserve checks can pause the run"},
        "versions": {name: metadata.version(name) for name in ("numpy", "gymnasium", "torch", "stable-baselines3", "scipy", "matplotlib")},
        "source_sha256": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "input_sha256": {str(p): sha256(p) for p in [path, experiment_path]}}
    return report, data, config, settings, recipe, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/study_100x1000.json"))
    parser.add_argument("--report", type=Path, default=Path("NEXT_STUDY_READINESS.json"))
    parser.add_argument("--start", action="store_true", help="Start the fixed study after successful preflight")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, data, config, settings, recipe, evaluation = prepare_study(args.config)
    atomic_json(args.report, report)
    print(f"Preflight passed: {settings.models} models, {settings.generations} generations, "
          f"{len(settings.evaluation_generations())} evaluations per model.", flush=True)
    if not args.start:
        print(f"Preparation only. No models started. Report: {args.report.resolve()}", flush=True)
        return
    from .train_generations import run_generations
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"learning_n{settings.models}_g{settings.generations}_{stamp}"
    if output.exists():
        raise FileExistsError("new studies require a fresh output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(Path(str(output) + ".preflight.json"), report)
    run_generations(config, settings, output, evaluation=evaluation, recipe=recipe, workers=data["workers"],
        confidence_level=data["confidence_level"], min_free_gib=data["min_free_gib"])
    print(f"Run {json.loads((output / 'manifest.json').read_text())['status']}: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
