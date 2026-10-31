"""Prepare or start the declared paired follow-up; never relax fresh-study checks."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil

from .cohort import EvaluationSettings
from .cohort_execution import atomic_json
from .experiments import ExperimentConfig
from .study import declared_seeds, sha256
from .train_generations import GenerationSettings
from .training_recipe import TrainingRecipe


def checked_file(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"sealed reference differs: {path}")


def reference_chain(source):
    paths = set()
    while source is not None:
        source = Path(source).resolve()
        if source in paths:
            raise ValueError("reference recovery chain contains a cycle")
        paths.add(source)
        manifest = json.loads((source / "manifest.json").read_text())
        source = manifest.get("recovery", {}).get("source_run")
    return paths


def quadratic_counterpart(reference):
    """Keep every scale/weight/input; change the complete reward form only."""
    if reference.reward.form != "bounded":
        raise ValueError("reference arm must use the declared bounded reward")
    return replace(reference, name="quadratic_boundary_rolling_followup",
                   reward=replace(reference.reward, form="quadratic"))


def prepare_comparison(path, workspace=None):
    workspace = Path(workspace or Path.cwd()).resolve()
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    if plan["schema_version"] != 1 or plan["scope"] != "paired_followup_reuse_completed_bounded_arm":
        raise ValueError("unsupported reward-comparison scope")
    completion_path = workspace / plan["reference_completion_file"]
    checked_file(completion_path, plan["reference_completion_sha256"])
    completion = json.loads(completion_path.read_text())
    analysis = Path(completion["analysis_directory"])
    checked_file(analysis / "audit.json", completion["audit_sha256"])
    checked_file(analysis / "artifact_hashes.json", completion["analysis_artifact_hashes_sha256"])
    artifact_hashes = json.loads((analysis / "artifact_hashes.json").read_text())
    for name, expected in artifact_hashes.items():
        checked_file(analysis / name, expected)
    if not completion["status"].startswith("complete:"):
        raise ValueError("reference must have a completed verified analysis")
    original = json.loads((analysis / "declared_study_config.json").read_text())
    initial = json.loads((analysis / "initial_preflight.json").read_text())
    settings = GenerationSettings(**plan["settings"])
    recipe = TrainingRecipe(**plan["training_recipe"])
    evaluation = EvaluationSettings(**plan["evaluation"])
    for key, value in (("settings", asdict(settings)), ("training_recipe", asdict(recipe)),
                       ("evaluation", asdict(evaluation)), ("workers", plan["workers"]),
                       ("confidence_level", plan["confidence_level"])):
        if value != original[key]:
            raise ValueError(f"paired comparison must preserve reference {key}")
    bounded = ExperimentConfig.load(analysis / "declared_reward_input_config.json")
    quadratic = ExperimentConfig.load(workspace / plan["experiment_file"])
    if quadratic != quadratic_counterpart(bounded):
        raise ValueError("only the declared reward-form change is allowed")
    current = Path(__file__).parent
    for name, expected in initial["source_sha256"].items():
        checked_file(current / name, expected)
    for name, version in initial["versions"].items():
        if metadata.version(name) != version:
            raise ValueError(f"reference library version changed: {name}")
    source = Path(completion["final_run"])
    manifest = json.loads((source / "manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["last_complete_evaluation_generation"] != settings.generations
            or manifest["versions"]["python"] != platform.python_version()):
        raise ValueError("reference completion or Python version differs")
    allowed = reference_chain(source)
    seeds = set(range(settings.seed_base, settings.seed_base + settings.models))
    collisions = []
    registry_count = 0
    for root in (workspace / "runs", workspace / "analysis"):
        for pattern in ("manifest.json", "plan.json"):
            for prior in root.rglob(pattern):
                registry_count += 1
                if prior.parent.resolve() in allowed:
                    continue
                overlap = seeds & declared_seeds(json.loads(prior.read_text()))
                if overlap:
                    collisions.append(str(prior))
    if collisions:
        raise ValueError(f"paired seeds already used outside the sealed reference arm: {collisions}")
    # Use actual final-cohort sizes, including the raw trajectory's per-member share.
    stem = source / "models/member_0000" / f"generation_{settings.generations:06d}"
    bundle_bytes = sum(Path(str(stem) + suffix).stat().st_size
                       for suffix in (".zip", ".normalization.pkl", ".resume.pkl"))
    trace_bytes = (source / "trajectories" / f"generation_{settings.generations:06d}.csv").stat().st_size
    estimate = (bundle_bytes * settings.models + trace_bytes) * len(settings.evaluation_generations())
    safety, reserve = plan["storage_safety_factor"], plan["min_free_gib"]
    if (isinstance(safety, bool) or not math.isfinite(safety) or safety < 1
            or isinstance(reserve, bool) or not math.isfinite(reserve) or reserve < 2):
        raise ValueError("invalid storage policy")
    required = math.ceil(estimate * safety + reserve * 1024**3)
    free = shutil.disk_usage(workspace).free
    report = {
        "status": "ready" if free >= required else "blocked: insufficient disk space",
        "checked_utc": datetime.now(timezone.utc).isoformat(),
        "comparison_config": str(path), "comparison_config_sha256": sha256(path),
        "experiment_sha256": sha256(workspace / plan["experiment_file"]),
        "reference_completion_sha256": sha256(completion_path),
        "reference_analysis_artifacts_verified": len(artifact_hashes),
        "reference_source_runs": sorted(map(str, allowed)),
        "unchanged_original_source_files": len(initial["source_sha256"]),
        "source_sha256": {p.name: sha256(p) for p in current.glob("*.py")},
        "versions": {"python": platform.python_version(), **initial["versions"]},
        "settings": asdict(settings), "training_recipe": asdict(recipe),
        "evaluation": asdict(evaluation), "member_seeds": sorted(seeds),
        "paired_unit": "training seed; not independent arms or time/generation cells",
        "new_training_interactions": settings.models * settings.generations * settings.batch_steps,
        "new_evaluation_episodes": settings.models * len(settings.evaluation_generations()),
        "new_evaluation_interactions": settings.models * len(settings.evaluation_generations()) * evaluation.steps,
        "storage": {"estimated_artifact_bytes": estimate, "safety_factor": safety,
                    "min_free_gib": reserve, "required_free_bytes": required, "currently_free_bytes": free},
        "seed_registry_files_checked": registry_count,
        "reference_final_run": str(source), "reference_analysis": str(analysis),
        "limitation": "Follow-up designed after seeing bounded-arm outcomes; not an independent confirmatory replication",
    }
    return report, plan, quadratic, settings, recipe, evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/reward_comparison_followup.json"))
    parser.add_argument("--report", type=Path, default=Path("REWARD_COMPARISON_READINESS.json"))
    parser.add_argument("--initialization-check", type=Path)
    parser.add_argument("--start", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, plan, config, settings, recipe, evaluation = prepare_comparison(args.config)
    atomic_json(args.report, report)
    print(f"Comparison preflight: {report['status']}; {settings.models} paired seeds, G={settings.generations}", flush=True)
    if not args.start:
        print("Preparation only; no training started.", flush=True)
        return
    if report["status"] != "ready":
        raise ValueError(report["status"])
    if args.initialization_check is None:
        raise ValueError("all-seed paired initialization verification is required before starting")
    checks = json.loads(args.initialization_check.read_text())
    if (checks["status"] != "passed" or checks["comparison_config_sha256"] != report["comparison_config_sha256"]
            or checks["member_seeds"] != report["member_seeds"] or checks["source_sha256"] != report["source_sha256"]):
        raise ValueError("initialization check is absent, stale or incomplete")
    from .train_generations import run_generations
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"quadratic_followup_n100_g1000_{stamp}"
    if output.exists():
        raise FileExistsError("follow-up requires a fresh output directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(Path(str(output) + ".preflight.json"), report)
    atomic_json(Path(str(output) + ".comparison.json"), {
        "plan": plan, "reference_completion_sha256": report["reference_completion_sha256"],
        "initialization_check_sha256": sha256(args.initialization_check),
        "initialization_check": str(args.initialization_check.resolve())})
    run_generations(config, settings, output, evaluation=evaluation, recipe=recipe,
        workers=plan["workers"], confidence_level=plan["confidence_level"], min_free_gib=plan["min_free_gib"])


if __name__ == "__main__":
    main()
