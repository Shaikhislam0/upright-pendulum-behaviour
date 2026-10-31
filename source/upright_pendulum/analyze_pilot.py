"""Offline diagnostics for an existing pilot; never trains or alters a model.

Run: python -m upright_pendulum.analyze_pilot runs/<pilot-directory>
Derived JSON and a trajectory figure are saved in that run's diagnostics/.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def numeric_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return [{key: float(value) if value not in (None, "") and value not in ("True", "False")
                 else value for key, value in row.items()} for row in csv.DictReader(file)]


def analyze(run):
    run = Path(run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_task", {}).get("name") == "local_balance":
        raise ValueError("For a local-balancing diagnostic, read local_balance_summary.json and evaluations.json; "
                         "this analyzer plots single-episode swing-up pilots.")
    evaluations = json.loads((run / "evaluations.json").read_text(encoding="utf-8"))
    if not evaluations:
        raise ValueError("the run has no completed checkpoint evaluations")
    diagnostics = run / "diagnostics"
    diagnostics.mkdir(exist_ok=True)
    summaries, trajectories = [], []
    cost_names = ("position_cost", "cart_velocity_cost", "angular_velocity_cost", "force_cost")
    for evaluation in evaluations:
        rows = numeric_rows(run / "trajectories" / evaluation["trajectory_file"])
        active = rows[1:]
        times = np.array([r["time_s"] for r in rows])
        x = np.array([r["x_m"] for r in rows])
        theta = np.array([r["theta_rad_unwrapped"] for r in rows])
        angle_error = np.abs(np.arctan2(np.sin(theta), np.cos(theta)))
        forces = np.array([r["force_N"] for r in active])
        costs = {name: math.fsum(r[name] for r in active) for name in cost_names}
        upright = math.fsum(r["upright"] for r in active)
        reconstructed_return = upright - math.fsum(costs.values())
        if not math.isclose(reconstructed_return, evaluation["episode_return"], rel_tol=1e-10, abs_tol=1e-8):
            raise ValueError("trajectory reward components do not reconstruct the recorded return")
        if len(active) != evaluation["evaluation_steps"] or times[-1] != evaluation["followup_time"]:
            raise ValueError("trajectory length/duration differs from the recorded evaluation")
        effort = float(np.sum(forces**2 * np.diff(times)))
        if not math.isclose(effort, evaluation["force_squared_integral_N2_s"], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError("trajectory force effort differs from the recorded evaluation")
        total_cost = math.fsum(costs.values())
        summaries.append({
            "environment_steps": evaluation["environment_steps"],
            "event_observed": evaluation["event_observed"],
            "episode_return": evaluation["episode_return"],
            "final_cart_position_m": float(x[-1]),
            "max_abs_cart_position_m": float(np.max(np.abs(x))),
            "max_abs_cart_velocity_m_s": max(abs(r["v_m_s"]) for r in rows),
            "minimum_absolute_angle_error_deg": float(np.rad2deg(np.min(angle_error))),
            "mean_force_N": float(np.mean(forces)),
            "rms_force_N": float(np.sqrt(np.mean(forces**2))),
            "positive_force_fraction": float(np.mean(forces > 0)),
            "negative_force_fraction": float(np.mean(forces < 0)),
            "saturated_action_fraction": evaluation["saturated_action_fraction"],
            "total_upright_benefit": upright, "summed_penalty_components": costs,
            "position_fraction_of_total_penalties": costs["position_cost"] / total_cost if total_cost else None,
            "maximum_single_step_position_penalty": max(r["position_cost"] for r in active),
        })
        trajectories.append((evaluation["environment_steps"], times, x, np.rad2deg(angle_error)))

    # DictReader retains slashes in SB3's column names.
    with (run / "progress.csv").open(newline="", encoding="utf-8") as file:
        logs = list(csv.DictReader(file))
    statistics = {}
    for key in ("train/value_loss", "train/explained_variance", "train/approx_kl", "train/std", "train/clip_fraction"):
        values = np.array([float(row[key]) for row in logs if row.get(key, "") != ""])
        values = values[np.isfinite(values)]
        if values.size:
            statistics[key] = {"min": float(values.min()), "median": float(np.median(values)),
                               "max": float(values.max()), "logged_rows": int(values.size)}
    package = Path(__file__).parent
    hashes_match = all((package / name).exists()
                       and hashlib.sha256((package / name).read_bytes()).hexdigest() == digest
                       for name, digest in manifest["source_sha256"].items())
    report = {
        "purpose": "Describe one existing pilot; causal effects and success probabilities are not estimated",
        "run": run.name, "training_seed": manifest["pilot_settings"]["seed"],
        "training_reward_scale": manifest["pilot_settings"].get("training_reward_scale", 1.0),
        "normalize_observations": manifest["pilot_settings"].get("normalize_observations", False),
        "original_source_hashes_match_current_files": hashes_match,
        "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_summaries": summaries, "ppo_logged_statistics": statistics,
        "arithmetic_checks_passed": True,
        "limitations": [
            "One training seed and one deterministic evaluation per checkpoint.",
            "Reward scale, input scale, exploration, and optimization are confounded in this pilot.",
            "Near-zero explained variance is evidence of poor value prediction, not proof of its cause.",
            "Return magnitudes are only comparable under the same reward definition and scale.",
            "Value loss is in squared scaled-return units; its magnitude alone cannot establish improvement across training scales.",
        ],
    }
    (diagnostics / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_trajectories(diagnostics, trajectories, manifest["stabilization_criteria"]["angle"])
    for row in summaries:
        print(f"{row['environment_steps']:>6} steps | max |x|={row['max_abs_cart_position_m']:8.2f} m"
              f" | closest to upright={row['minimum_absolute_angle_error_deg']:6.2f} deg"
              f" | mean F={row['mean_force_N']:6.3f} N")
    print(f"Diagnostics: {diagnostics}")
    return report


def plot_trajectories(output, trajectories, angle_bound):
    import os
    cache = output / ".mplconfig"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    colors = ["#303030", "#dca76b", "#c18749", "#9d622d", "#75451f", "#492b17"]
    for index, (step, time, x, angle) in enumerate(trajectories):
        label = "0 (initial)" if step == 0 else f"{step / 1000:g}k"
        style = "--" if step == 0 else "-"
        color = colors[index % len(colors)]
        axes[0].plot(time, x, style, color=color, linewidth=1.4, label=label)
        axes[1].plot(time, angle, style, color=color, linewidth=1.4)
    bound = math.degrees(angle_bound)
    axes[1].axhline(bound, color="#ad3b31", linestyle=":", linewidth=1.2)
    axes[1].text(19.7, bound + 3, f"{bound:g} degree angle bound", ha="right", color="#ad3b31", fontsize=9)
    axes[0].set(title="A  Cart motion", xlabel="Evaluation time (s)", ylabel="Cart position (m)")
    axes[1].set(title="B  Distance from upright", xlabel="Evaluation time (s)",
                ylabel="Absolute angular error (degrees)", ylim=(0, 185))
    for ax in axes:
        ax.set_xlim(0, 20)
        ax.grid(alpha=0.16)
    fig.suptitle("PPO pilot · deterministic evaluation trajectories", x=0.01, ha="left", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Training environment steps · one training seed",
               loc="outside lower center", ncol=min(len(labels), 6), frameon=False)
    fig.savefig(output / "trajectories.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    analyze(parser.parse_args().run)
