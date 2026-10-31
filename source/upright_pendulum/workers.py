"""Persistent CPU workers owning disjoint groups of independent PPO models.

Only generation-level commands cross process boundaries. Training barriers
precede evaluation; physics, optimization, and each model's RNG stay local.
"""

from dataclasses import asdict
import multiprocessing as mp
from pathlib import Path
import traceback

from .cohort import EvaluationSettings, cumulative_counts
from .experiments import ExperimentConfig


def _worker(connection, config_data, settings_data, evaluation_data, output, indices, worker_index, recovery, recipe_data):
    members = []
    try:
        import torch
        from .train_generations import GenerationSettings, TrainingMember, evaluate_members
        from .training_recipe import TrainingRecipe, check_member_recipe, make_member
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        config = ExperimentConfig.from_dict(config_data)
        settings = GenerationSettings(**settings_data)
        evaluation = EvaluationSettings(**evaluation_data)
        recipe = TrainingRecipe(**recipe_data)
        output = Path(output)
        trajectory_dir = output / "trajectories" / f"worker_{worker_index:03d}"
        trajectory_dir.mkdir(parents=True, exist_ok=False)
        for index in indices:
            member_id = f"member_{index:04d}"
            if recovery is None:
                member = make_member(member_id, config, settings.pilot(settings.seed_base + index),
                                     output / "models" / member_id, recipe)
            else:
                from .checkpoint_state import restore_member
                member = restore_member(member_id, config, settings.pilot(settings.seed_base + index),
                    output / "models" / member_id, Path(recovery["source"]) / "models" / member_id,
                    recovery["generation"], legacy=recovery["legacy"])
                try:
                    check_member_recipe(member, recipe)
                except BaseException:
                    member.close()
                    raise
            members.append(member)
        connection.send({"ok": True, "members": [member.member_id for member in members]})
        while True:
            command, generation = connection.recv()
            if command == "close":
                break
            if command == "update":
                for member in members:
                    member.update(generation)
                connection.send({"ok": True})
            elif command == "evaluate":
                trajectory = trajectory_dir / f"generation_{generation:06d}.csv"
                record, hashes = evaluate_members(members, config, settings, evaluation,
                    generation, output, trajectory)
                connection.send({"ok": True, "record": record, "hashes": hashes,
                                 "trajectory": str(trajectory)})
            elif command == "save":
                from .cohort_execution import save_members
                connection.send({"ok": True, "hashes": save_members(members, generation, output)})
            else:
                raise ValueError(f"unknown worker command: {command}")
    except BaseException as error:
        try:
            connection.send({"ok": False, "error": f"{type(error).__name__}: {error}",
                             "traceback": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for member in members:
            member.close()
        connection.close()


def merge_cohorts(shards, expected_ids):
    """Reconstruct the fixed-n count from first events, never average percentages."""
    if not shards:
        raise ValueError("no worker cohorts received")
    members = [member for shard in shards for member in shard["members"]]
    ids = [member["member_id"] for member in members]
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):
        raise RuntimeError("worker results do not contain the complete, unique cohort")
    horizon = shards[0]["horizon_s"]
    times = [row["evaluation_time_s"] for row in shards[0]["samples"]]
    for shard in shards:
        if (shard["horizon_s"] != horizon or
                [row["evaluation_time_s"] for row in shard["samples"]] != times or
                shard["n_simulations"] != len(shard["members"])):
            raise RuntimeError("worker evaluation grids or cohort sizes disagree")
    if any(abs(member["followup_time"] - horizon) > 1e-10 for member in members):
        raise RuntimeError("a worker returned an incomplete evaluation")
    members.sort(key=lambda member: expected_ids.index(member["member_id"]))
    n = len(members)
    counts = cumulative_counts([member["success_time"] for member in members], times)
    return {"n_simulations": n, "horizon_s": horizon, "members": members,
            "samples": [{"evaluation_time_s": t, "achieved_balance": int(count),
                         "n_simulations": n, "percentage": 100.0 * count / n}
                        for t, count in zip(times, counts)]}


class WorkerPool:
    def __init__(self, count, config, settings, evaluation, output, *, recovery=None, recipe=None):
        from .training_recipe import TrainingRecipe
        recipe = recipe if recipe is not None else TrainingRecipe()
        if not isinstance(recipe, TrainingRecipe):
            raise ValueError("recipe must be a TrainingRecipe")
        if type(count) is not int or not 1 < count <= settings.models:
            raise ValueError("worker count must be between 2 and the number of models")
        if evaluation.parallel:
            raise ValueError("worker pools already parallelize evaluation; do not nest physics workers")
        self.connections = []
        self.processes = []
        self.ids = [f"member_{i:04d}" for i in range(settings.models)]
        context = mp.get_context("spawn")
        try:
            for index in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, config.to_dict(),
                    asdict(settings), asdict(evaluation), str(Path(output).resolve()),
                    list(range(index, settings.models, count)), index, recovery, asdict(recipe)), name=f"ppo-worker-{index}")
                process.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(process)
            ready = self.receive()
            if sorted(member for result in ready for member in result["members"]) != self.ids:
                raise RuntimeError("worker initialization lost cohort members")
        except BaseException:
            self.close()
            raise

    def receive(self):
        results = []
        for connection, process in zip(self.connections, self.processes):
            while not connection.poll(.2):
                if not process.is_alive():
                    raise RuntimeError(f"worker {process.name} exited with code {process.exitcode}")
            try:
                result = connection.recv()
            except EOFError as error:
                raise RuntimeError(f"worker {process.name} closed its connection") from error
            if not result["ok"]:
                raise RuntimeError(f"worker {process.name}: {result['error']}\n{result['traceback']}")
            results.append(result)
        return results

    def update(self, generation):
        for connection in self.connections:
            connection.send(("update", generation))
        self.receive()  # Barrier: every model finishes training before evaluation.

    def evaluate(self, generation):
        for connection in self.connections:
            connection.send(("evaluate", generation))
        results = self.receive()  # Barrier: no model trains during evaluation.
        return (merge_cohorts([result["record"] for result in results], self.ids),
                {key: value for result in results for key, value in result["hashes"].items()},
                [Path(result["trajectory"]) for result in results])

    def save(self, generation):
        for connection in self.connections:
            connection.send(("save", generation))
        return {key: value for result in self.receive() for key, value in result["hashes"].items()}

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try:
                    connection.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        for connection, process in zip(self.connections, self.processes):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            connection.close()
