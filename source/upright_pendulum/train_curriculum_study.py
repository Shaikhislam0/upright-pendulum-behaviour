"""Queue a fixed training-start curriculum against retained matched controls."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
import zipfile

import numpy as np

from .calibrate import evaluate_development
from .curriculum import CurriculumConfig, attach_curriculum
from .experiments import ExperimentConfig
from .resume_calibration import verified_checkpoint
from .train_generations import GenerationSettings, TrainingMember
from .train_pilot import write_json


CONTROL = "training_failure_at_3m"
CORE_FILES = ("physics.py", "env.py", "stabilization.py", "observations.py", "rewards.py",
              "actions.py", "experiments.py", "normalization.py", "train_pilot.py",
              "train_generations.py", "ppo_variants.py", "cohort.py", "calibrate.py", "training_boundary.py")


def validate_control_plan(source, smoke):
    plan = json.loads((source / "plan.json").read_text())
    if plan["conditions_training_boundary_m"].get(CONTROL) != 3.0 or plan["boundary_failure_cost"] != 10.0:
        raise ValueError("matched controls must have the declared 3m boundary and cost 10")
    if plan["gradient_clipping"] != "independent" or plan["reward_form"] != "bounded":
        raise ValueError("matched control reward/optimizer differs")
    expected_g, expected_b = (2, 32) if smoke else (500, 1000)
    if plan["generations"] != expected_g or plan["experience_steps_per_generation"] != expected_b:
        raise ValueError("control budget differs from this declared comparison")
    for name in CORE_FILES:
        if hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() != plan["source_sha256"][name]:
            raise ValueError(f"control training/evaluation implementation changed: {name}")
    for name, version in plan["versions"].items():
        if metadata.version(name) != version:
            raise ValueError(f"control library version changed: {name}")
    return plan


def settings_for(plan, seed):
    settings = GenerationSettings(1, plan["generations"], seed_base=seed,
        batch_steps=plan["experience_steps_per_generation"], minibatch_size=plan["ppo_settings"]["batch_size"],
        epochs=plan["ppo_settings"]["epochs"])
    if asdict(settings.pilot(seed)) != {**plan["ppo_settings"], "seed": seed}:
        raise ValueError("PPO scalar settings do not match retained controls")
    return settings


def check_initial_policy(member, source):
    import torch
    path = source / "models/member_0000/generation_000000.zip"
    with zipfile.ZipFile(path) as archive:
        expected = torch.load(io.BytesIO(archive.read("policy.pth")), weights_only=True, map_location="cpu")
    actual = member.model.policy.state_dict()
    if actual.keys() != expected.keys() or any(not torch.equal(actual[key], expected[key]) for key in actual):
        raise ValueError("curriculum/control initial policy parameters differ")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_trial(seed, plan, source, root):
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    root, source = Path(root), Path(source)
    output = root / "curriculum_starts" / f"seed_{seed}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "trajectories").mkdir()
    settings = settings_for(plan, seed)
    config = ExperimentConfig.load(source / "experiment.json")
    config.save(output / "experiment.json")
    progress = {"status": "running", "seed": seed, "last_training_generation": None,
                "last_evaluation_generation": None}
    member, evaluations, history = None, [], []
    try:
        if (root / "stop.request").exists():
            progress["status"] = "interrupted"
            return {**progress, "evaluations": []}
        member = TrainingMember("member_0000", config, settings.pilot(seed), output / "models/member_0000",
                                gradient_clipping="independent", training_boundary_m=3.0, boundary_failure_cost=10.)
        initial_hash = check_initial_policy(member, source)
        layer = attach_curriculum(member, CurriculumConfig(total_generations=settings.generations))
        write_json(output / "trial.json", {"control_source": str(source), "control_initial_checkpoint_sha256": initial_hash,
                                           "identical_initial_policy": True, "curriculum": asdict(layer.config)})
        for generation in range(settings.generations + 1):
            layer.set_generation(generation)
            if generation:
                member.update(generation)
            if generation in plan["evaluation_generations"]:
                result = evaluate_development(member, config, settings, generation, output)
                if not all(np.isfinite(v) for v in result["metrics"].values()):
                    raise FloatingPointError("nonfinite training diagnostics")
                evaluations.append(result)
                write_json(output / "evaluations.json", evaluations)
                progress["last_evaluation_generation"] = generation
                print(f"curriculum, seed={seed}, generation={generation}/{settings.generations}: "
                      f"success={result['event']['success_time']}, max |x|={result['max_abs_x_m']:.2f} m", flush=True)
            else:
                member.save(generation)
            history.append({"generation": generation, "phase_for_next_reset": layer.phase + 1,
                            "cumulative_start_counts": dict(layer.start_counts), "last_start": layer.last_start})
            write_json(output / "curriculum_history.json", history)
            progress.update(last_training_generation=generation, phase_for_next_reset=layer.phase + 1,
                            updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(output / "progress.json", progress)
            if (root / "stop.request").exists() and generation < settings.generations:
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


def run_study(source, output, *, wait_for_control=False, smoke=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    control_plan = validate_control_plan(source, smoke)
    for seed in control_plan["seeds"]:
        settings_for(control_plan, seed)
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    plan = {"kind": "technical smoke" if smoke else "paired training-start curriculum development comparison",
            "control_source": str(source), "seeds": control_plan["seeds"], "generations": control_plan["generations"],
            "evaluation_generations": control_plan["evaluation_generations"],
            "curriculum": asdict(CurriculumConfig(total_generations=control_plan["generations"])),
            "additional_training_steps": len(control_plan["seeds"]) * control_plan["generations"] * control_plan["experience_steps_per_generation"],
            "retained_condition": CONTROL, "ppo_settings": control_plan["ppo_settings"],
            "fixed_training": "3m terminal failure, cost 10, bounded reward, independent clipping, identical initial weights per paired seed",
            "evaluation": control_plan["evaluation"],
            "selection_gate": "At least 2/3 curriculum seeds succeed without later exit at final predeclared G=500, all diagnostics finite and original physics/event checks passed. Earlier points describe learning speed; no best-checkpoint selection or automatic 100-model launch.",
            "source_sha256": hashes, "versions": control_plan["versions"]}
    write_json(output / "plan.json", plan)
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    status = {"status": "waiting_for_control", "process_id": os.getpid(),
              "queued_utc": datetime.now(timezone.utc).isoformat(), "completed_tasks": 0,
              "total_tasks": len(control_plan["seeds"]), "control_source": str(source)}
    write_json(output / "status.json", status)
    results = []
    try:
        while True:
            if (output / "stop.request").exists():
                status["status"] = "interrupted"
                return []
            try:
                control_status = json.loads((source / "status.json").read_text())["status"]
            except (OSError, json.JSONDecodeError):
                time.sleep(1)
                continue
            if control_status == "complete":
                break
            if control_status != "running" or not wait_for_control:
                raise RuntimeError(f"control study is {control_status}; recover/complete it before starting curriculum")
            time.sleep(10)
        if hashes != {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}:
            raise RuntimeError("source changed while queued; review the declared snapshot before starting")
        validate_control_plan(source, smoke)
        retained = []
        for seed in control_plan["seeds"]:
            directory = source / CONTROL / f"seed_{seed}"
            generation, rejected = verified_checkpoint(directory / "models/member_0000",
                settings_for(control_plan, seed).pilot(seed), control_plan["generations"])
            if generation != control_plan["generations"] or rejected:
                raise ValueError("control final checkpoint is not committed and intact")
            evaluations = json.loads((directory / "evaluations.json").read_text())
            if [r["generation"] for r in evaluations] != control_plan["evaluation_generations"]:
                raise ValueError("control evaluation grid is incomplete")
            dest = output / "fixed_downward_start" / f"seed_{seed}"
            dest.mkdir(parents=True)
            for name in ("evaluations.json", "experiment.json", "progress.json"):
                shutil.copy2(directory / name, dest / name)
            evidence_hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                               for name in ("evaluations.json", "experiment.json",
                                            f"models/member_0000/generation_{generation:06d}.zip",
                                            f"models/member_0000/generation_{generation:06d}.normalization.pkl",
                                            f"models/member_0000/generation_{generation:06d}.resume.pkl")}
            write_json(dest / "retained_source.json", {"source": str(directory), "sha256": evidence_hashes,
                       "reason": "matched completed control retained without retraining"})
            retained.append({"seed": seed, "source": str(directory), "verified_final_generation": generation,
                             "sha256": evidence_hashes})
        write_json(output / "retained_controls.json", retained)
        status.update(status="running", started_utc=datetime.now(timezone.utc).isoformat(), retained_control_models=len(retained))
        write_json(output / "status.json", status)
        print("Controls verified; starting fresh curriculum policies.", flush=True)
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
            futures = [executor.submit(run_trial, seed, control_plan, str(source / CONTROL / f"seed_{seed}"), str(output))
                       for seed in control_plan["seeds"]]
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
    parser.add_argument("--control-study", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-for-control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"curriculum_{'smoke' if args.smoke else 'development'}_{stamp}"
    run_study(args.control_study, output, wait_for_control=args.wait_for_control, smoke=args.smoke)
    print(f"Curriculum runner finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
