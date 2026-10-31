"""Resume an intentionally stopped fixed-normalization study without replaying experience."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib import metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import zipfile

from .resume_calibration import verified_checkpoint
from .train_centering_study import CORE_FILES
from .train_fixed_normalization import CONDITION, CONTROL_SOURCE, digest, generation_settings, run_trial
from .train_pilot import write_json


def recovery_plan(source):
    source = Path(source).resolve()
    status = json.loads((source / "status.json").read_text())
    if status["status"] not in ("interrupted", "failed"):
        raise ValueError("resume requires a stopped study")
    plan = json.loads((source / "plan.json").read_text())
    for name in (*CORE_FILES, "checkpoint_state.py", "fixed_normalization.py", "resume_calibration.py"):
        if digest(Path(__file__).parent / name) != plan["source_sha256"][name]:
            raise ValueError(f"training, evaluation or recovery implementation changed: {name}")
    for name, version in plan["versions"].items():
        if metadata.version(name) != version:
            raise ValueError(f"training library version changed: {name}")
    # Follow immutable ancestor links if a later continuation was stopped before
    # its queued task could write another checkpoint.
    records = []
    for seed in plan["seeds"]:
        ancestor = source
        seen = set()
        while True:
            if ancestor in seen:
                raise ValueError("cyclic continuation ancestry")
            seen.add(ancestor)
            trial = ancestor / CONDITION / f"seed_{seed}"
            directory = trial / "models/member_0000"
            if directory.is_dir() and any(directory.glob("generation_*.resume.pkl")):
                generation, rejected = verified_checkpoint(directory,
                    generation_settings(plan, seed).pilot(seed), plan["final_generation"])
                if generation < plan["parent_generation"]:
                    raise ValueError("fixed checkpoint precedes its freeze anchor")
                with zipfile.ZipFile(directory / f"generation_{generation:06d}.zip") as archive:
                    data = json.loads(archive.read("data"))
                if (data.get("pendulum_fixed_observation_statistics") is not True or
                        data.get("pendulum_observation_freeze_generation") != plan["parent_generation"]):
                    raise ValueError("saved fixed-statistics marker or anchor differs")
                records.append({"seed": seed, "generation": generation, "source_trial": str(trial),
                    "already_frozen": True, "rejected_incomplete_checkpoints": rejected,
                    "checkpoint_sha256": digest(directory / f"generation_{generation:06d}.zip"),
                    "statistics_sha256": digest(directory / f"generation_{generation:06d}.normalization.pkl")})
                break
            ancestor_plan = json.loads((ancestor / "plan.json").read_text())
            previous = ancestor_plan.get("continuation_of")
            if previous:
                ancestor = Path(previous).resolve()
                continue
            parent_trial = Path(plan["source"]) / CONTROL_SOURCE / f"seed_{seed}"
            generation, rejected = verified_checkpoint(parent_trial / "models/member_0000",
                generation_settings(plan, seed).pilot(seed), plan["parent_generation"])
            if generation != plan["parent_generation"] or rejected:
                raise ValueError("queued member's original parent is incomplete")
            records.append({"seed": seed, "generation": generation, "source_trial": str(parent_trial),
                            "already_frozen": False, "rejected_incomplete_checkpoints": []})
            break
    return plan, records


def run_continuation(source, output, *, workers=2):
    if workers not in (1, 2):
        raise ValueError("use one or two workers")
    source, output = Path(source).resolve(), Path(output).resolve()
    old_plan, records = recovery_plan(source)
    # Verify retained controls before copying them into the new output.
    audit = json.loads((source / "parent_and_control_audit.json").read_text())
    for item in audit:
        for name, expected in item["sha256"].items():
            if digest(Path(item["source"]) / name) != expected:
                raise ValueError("retained parent/control source was modified")
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    plan = {**old_plan, "continuation_of": str(source), "workers": workers,
            "additional_training_steps": sum((old_plan["final_generation"] - row["generation"]) *
                old_plan["batch_steps"] for row in records),
            "source_sha256": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}}
    write_json(output / "plan.json", plan)
    write_json(output / "recovery_plan.json", records)
    write_json(output / "parent_and_control_audit.json", audit)
    shutil.copytree(source / "rolling_statistics", output / "rolling_statistics")
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    status = {"status": "running", "process_id": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
              "completed_tasks": 0, "total_tasks": len(records)}
    write_json(output / "status.json", status)
    results = []

    def accept(result):
        results.append(result)
        results.sort(key=lambda row: row["seed"])
        write_json(output / "results.json", results)
        status["completed_tasks"] = sum(row["status"] == "complete" for row in results)
        write_json(output / "status.json", status)

    try:
        tasks = [(row["seed"], plan["source"], plan, str(output), row if row["already_frozen"] else None)
                 for row in records]
        if workers == 1:
            for args in tasks:
                accept(run_trial(*args))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as executor:
                futures = [executor.submit(run_trial, *args) for args in tasks]
                for future in as_completed(futures):
                    accept(future.result())
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
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output or Path("runs") / f"fixed_normalization_resumed_{stamp}"
    run_continuation(args.source, output, workers=args.workers)
    print(f"Fixed-normalization continuation finished: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
