"""Shared generation schedule, durable cohort boundaries and resource stops."""

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate_reserve(min_free_gib):
    if (isinstance(min_free_gib, bool) or not isinstance(min_free_gib, (int, float)) or
            not math.isfinite(min_free_gib) or min_free_gib < 0):
        raise ValueError("min_free_gib must be finite and nonnegative")


def save_members(members, generation, output):
    output = Path(output)
    paths = []
    for member in members:
        checkpoint, statistics = member.save(generation)
        paths.extend((checkpoint, statistics, checkpoint.with_suffix(".resume.pkl")))
    return {path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def collect_generations(members, pool, config, settings, evaluation, output, manifest, records,
                        *, start_generation=0, restored=False, confidence_level=None, min_free_gib=2.):
    from .cohort_results import export_surface
    from .train_generations import evaluate_members, merge_trajectories
    output = Path(output)
    grid = set(settings.evaluation_generations())
    manifest["last_training_generation"] = start_generation
    atomic_json(output / "generations.json", records)

    def stop_reason():
        if (output / "stop.request").exists():
            return "user stop.request"
        if shutil.disk_usage(output).free < min_free_gib * 1024**3:
            return "available disk space is below the declared reserve"
        return None

    def pause(generation, reason):
        if manifest.get("last_fully_saved_policy_generation") != generation:
            hashes = pool.save(generation) if pool is not None else save_members(members, generation, output)
            manifest["checkpoint_sha256"].update(hashes)
            manifest["last_fully_saved_policy_generation"] = generation
        manifest.update(status="interrupted", interruption_cause=reason,
                        last_progress_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(output / "manifest.json", manifest)

    for generation in range(start_generation, settings.generations + 1):
        phase = time.perf_counter()
        if generation > start_generation:
            reason = stop_reason()
            if reason:
                pause(generation - 1, reason)
                break
            if pool is not None:
                pool.update(generation)
            else:
                for member in members:
                    member.update(generation)
        training_s = time.perf_counter() - phase
        manifest["last_training_generation"] = generation
        reason = stop_reason()
        if reason:
            pause(generation, reason)
            break
        phase = time.perf_counter()
        if generation in grid:
            trajectory = output / "trajectories" / f"generation_{generation:06d}.csv"
            if pool is not None:
                record, hashes, parts = pool.evaluate(generation)
                merge_trajectories(parts, trajectory)
            else:
                record, hashes = evaluate_members(members, config, settings, evaluation, generation, output, trajectory)
            for index in range(settings.models):
                path = output / "models" / f"member_{index:04d}" / f"generation_{generation:06d}.resume.pkl"
                hashes[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest["checkpoint_sha256"].update(hashes)
            manifest["last_fully_saved_policy_generation"] = generation
            record.update(generation=generation, environment_steps_per_model=generation * settings.batch_steps)
            for index, row in enumerate(record["members"]):
                row["training_seed"] = settings.seed_base + index
            records.append(record)
            export_surface(records, output, technical=manifest.get("kind") == "technical_smoke",
                           confidence_level=confidence_level)
            manifest["completed_generations"].append(generation)
            manifest["last_complete_evaluation_generation"] = generation
            final = record["samples"][-1]
            print(f"Generation {generation}/{settings.generations}: by {evaluation.horizon_s:g} s, "
                  f"{final['achieved_balance']}/{settings.models} achieved balance ({final['percentage']:.1f}%)", flush=True)
        elif restored and generation == start_generation:
            # A resumed stop checkpoint need not be a measurement checkpoint.
            hashes = pool.save(generation) if pool is not None else save_members(members, generation, output)
            manifest["checkpoint_sha256"].update(hashes)
            manifest["last_fully_saved_policy_generation"] = generation
        manifest["timings"].append({"generation": generation, "training_s": training_s,
                                   "evaluation_export_s": time.perf_counter() - phase})
        manifest["last_progress_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(output / "manifest.json", manifest)
    else:
        manifest["status"] = "complete"
    return records
