"""Export the empirical generation/time surface without smoothing or selection."""

import csv
import math
import os
from pathlib import Path

import numpy as np

from .cohort import cumulative_counts
from .train_pilot import write_json


def validate_confidence(confidence_level):
    if confidence_level is None:
        return
    if (isinstance(confidence_level, bool) or not isinstance(confidence_level, (int, float)) or
            not math.isfinite(confidence_level) or not 0 < confidence_level < 1):
        raise ValueError("confidence_level must be between 0 and 1")
    from scipy.stats import beta  # Optional; required only when intervals are requested.


def exact_binomial_intervals(counts, n, confidence_level=.95):
    """Two-sided Clopper-Pearson pointwise bounds, in probability units."""
    validate_confidence(confidence_level)
    if confidence_level is None:
        raise ValueError("a confidence level is required")
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    values = np.asarray(counts)
    if (values.dtype.kind not in "iu" or np.any(values < 0) or np.any(values > n)):
        raise ValueError("counts must be integers between zero and n")
    from scipy.stats import beta
    tail = (1 - confidence_level) / 2
    lower = np.zeros(values.shape, dtype=float)
    upper = np.ones(values.shape, dtype=float)
    interior_lower = values > 0
    interior_upper = values < n
    lower[interior_lower] = beta.ppf(tail, values[interior_lower], n - values[interior_lower] + 1)
    upper[interior_upper] = beta.ppf(1 - tail, values[interior_upper] + 1, n - values[interior_upper])
    return lower, upper


def surface(records):
    """Validate complete cohorts and return a [time, generation] percentage grid."""
    if not records:
        raise ValueError("at least one generation is required")
    generations = [record["generation"] for record in records]
    if any(type(g) is not int or g < 0 for g in generations) or any(b <= a for a, b in zip(generations, generations[1:])):
        raise ValueError("generations must be nonnegative and strictly increasing")
    reference_ids = [member["member_id"] for member in records[0]["members"]]
    n = len(reference_ids)
    if n == 0 or len(set(reference_ids)) != n:
        raise ValueError("cohort must have unique members")
    times = np.array([row["evaluation_time_s"] for row in records[0]["samples"]], dtype=float)
    columns = []
    for record in records:
        ids = [member["member_id"] for member in record["members"]]
        if record["n_simulations"] != n or len(ids) != n or set(ids) != set(reference_ids):
            raise ValueError("the same complete cohort must be present at every generation")
        record_times = np.array([row["evaluation_time_s"] for row in record["samples"]], dtype=float)
        if not np.array_equal(record_times, times) or times[0] != 0 or times[-1] != record["horizon_s"]:
            raise ValueError("every generation must cover the same time grid from zero to the horizon")
        events = []
        for member in record["members"]:
            event = member["success_time"]
            if not np.isclose(member["followup_time"], record["horizon_s"], rtol=0, atol=1e-10):
                raise ValueError("an incomplete evaluation cannot enter the surface")
            if bool(member["event_observed"]) != (event is not None) or bool(member["right_censored"]) != (event is None):
                raise ValueError("event and censoring flags disagree")
            if event is not None and event > record["horizon_s"] + 1e-12:
                raise ValueError("event exceeds the evaluation horizon")
            events.append(event)
        counts = cumulative_counts(events, times)
        percentages = 100.0 * counts / n
        for row, count, percentage in zip(record["samples"], counts, percentages):
            if row["n_simulations"] != n or row["achieved_balance"] != count or not np.isclose(row["percentage"], percentage):
                raise ValueError("reported fraction disagrees with first-event times")
        columns.append(percentages)
    return np.asarray(generations), times, np.column_stack(columns), n


def export_surface(records, output, *, technical=False, confidence_level=None):
    """Retain event records and exact percentages alongside a scientific figure."""
    generations, times, percentages, n = surface(records)
    validate_confidence(confidence_level)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((output / ".mplconfig").resolve()))
    from .cohort_execution import atomic_json
    atomic_json(output / "generations.json", records)
    with (output / "fractions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["generation", "evaluation_time_s", "achieved_balance", "n_simulations", "percentage"])
        writer.writeheader()
        for record in records:
            for row in record["samples"]:
                writer.writerow({"generation": record["generation"], **row})
    extras = {}
    if confidence_level is not None:
        counts = np.rint(percentages * n / 100).astype(np.int64)
        lower, upper = exact_binomial_intervals(counts, n, confidence_level)
        extras = {"lower_percentage": 100 * lower, "upper_percentage": 100 * upper,
                  "confidence_level": confidence_level}
        with (output / "confidence.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["generation", "evaluation_time_s", "achieved_balance", "n_simulations", "percentage",
                             "lower_percentage", "upper_percentage", "confidence_level", "method"])
            for j, generation in enumerate(generations):
                for i, time in enumerate(times):
                    writer.writerow([generation, time, counts[i,j], n, percentages[i,j],
                                     100 * lower[i,j], 100 * upper[i,j], confidence_level, "Clopper-Pearson pointwise"])
    np.savez(output / "surface.npz", generations=generations, evaluation_times_s=times,
             percentages=percentages, n_simulations=n, **extras)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    def edges(values):
        if len(values) == 1:
            return np.array([values[0] - .5, values[0] + .5])
        midpoints = (values[:-1] + values[1:]) / 2
        return np.r_[values[0] - (midpoints[0] - values[0]), midpoints,
                     values[-1] + (values[-1] - midpoints[-1])]

    image = ax.pcolormesh(edges(generations), edges(times), percentages, shading="flat", cmap="cividis", vmin=0, vmax=100)
    ax.set(xlabel="Generation (one PPO rollout/update cycle)", ylabel="Elapsed evaluation time (s)",
           ylim=(0, times[-1]), title=f"{'Technical smoke — ' if technical else ''}Completed stabilization, n = {n}")
    if len(generations) <= 12:
        ax.set_xticks(generations)
    if len(generations) > 1 and np.any(np.diff(generations) > 1):
        ax.set_xlabel("Evaluated generation (intermediate policies not evaluated)")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Controllers that achieved balance by time t (%)")
    fig.savefig(output / "heatmap.png", dpi=180)
    fig.savefig(output / "heatmap.pdf")
    plt.close(fig)

