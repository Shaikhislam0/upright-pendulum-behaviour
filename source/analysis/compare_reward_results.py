"""Declared paired analysis of sealed reward arms; no training or policy selection."""

import argparse
import csv
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from analysis.final_study_analysis import digest, write_csv, write_json
from upright_pendulum.cohort_results import exact_binomial_intervals
from upright_pendulum.paired_reward_results import paired_endpoint


def load_events(path, seeds, generations):
    """Align by the declared training seed, never CSV order or successful subsets."""
    seed_index = {s: i for i, s in enumerate(seeds)}
    generation_index = {g: i for i, g in enumerate(generations)}
    times = np.full((len(seeds), len(generations)), np.inf)
    exits = np.full(times.shape, np.inf)
    seen = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            seed, g = int(row["training_seed"]), int(row["generation"])
            if seed not in seed_index or g not in generation_index or (seed, g) in seen:
                raise ValueError("extra or duplicate seed/generation pair")
            seen.add((seed, g))
            ts = float(row["success_time"]) if row["success_time"] else np.inf
            exit_time = float(row["first_post_success_exit_time"]) if row["first_post_success_exit_time"] else np.inf
            event = np.isfinite(ts)
            if not (ts == np.inf or 2 <= ts <= 20) or math.isnan(exit_time):
                raise ValueError("invalid first-hold or exit time")
            if exit_time != np.inf and not (event and ts <= exit_time <= 20):
                raise ValueError("later exit without a preceding first event")
            flags = (row["event_observed"], row["right_censored"], row["has_succeeded"])
            if flags != (("True", "False", "True") if event else ("False", "True", "False")):
                raise ValueError("censoring flags disagree with the event time")
            if float(row["followup_time"]) != 20 or float(row["observed_time"]) != (ts if event else 20):
                raise ValueError("incomplete or inconsistent follow-up")
            i, j = seed_index[seed], generation_index[g]
            times[i, j], exits[i, j] = ts, exit_time
    if len(seen) != len(seeds) * len(generations):
        raise ValueError("missing seed/generation pair; failures must stay in the cohort")
    return times, exits


def counts_from_events(events, times):
    return (events[:, :, None] <= times[None, None, :] + 1e-12).sum(axis=0).T


def arm_summary(events, exits, counts, generations):
    final = events[:, -1]
    observed = final[np.isfinite(final)]
    success_count = len(observed)
    lo, hi = exact_binomial_intervals(np.array([success_count]), len(final), .95)
    endpoint = counts[-1]
    return {
        "final_successes_by_20s": success_count,
        "pointwise_95_interval_percentage": [float(100 * lo[0]), float(100 * hi[0])],
        "final_censored_at_20s": len(final) - success_count,
        "first_evaluated_generation_with_success": next((int(g) for g, c in zip(generations, endpoint) if c), None),
        "scanned_maximum_count_descriptive_only": int(endpoint.max()),
        "generations_at_scanned_maximum": generations[endpoint == endpoint.max()].tolist(),
        "final_mean_capped_first_event_time_s": float(np.minimum(final, 20).mean()),
        "final_success_conditioned_median_s": float(np.median(observed)) if success_count else None,
        "final_time_to_at_least_50_events_s": float(np.sort(observed)[49]) if success_count >= 50 else None,
        "final_successes_with_later_exit": int(np.isfinite(exits[:, -1]).sum()),
        "final_successes_with_no_observed_later_exit": int(success_count - np.isfinite(exits[:, -1]).sum()),
        "final_counts_at_times": {str(t): int(np.sum(final <= t + 1e-12)) for t in (2, 5, 10, 13, 15, 20)},
    }


def figures(output, generations, times, bounded, quadratic, events_b, events_q, primary):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp/mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    colors = {"bounded": "#176b66", "quadratic": "#a15428"}

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=200, facecolor="white")
        fig.savefig(output / f"{name}.pdf", facecolor="white")
        plt.close(fig)

    delta = bounded - quadratic
    limit = max(10, math.ceil(float(np.abs(delta).max()) / 10) * 10)
    yedges = np.r_[times[0] - .05, (times[:-1] + times[1:]) / 2, times[-1] + .05]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True, layout="constrained")
    for index, (ax, values, title) in enumerate(zip(axes, (bounded, quadratic, delta),
            ("Bounded reward", "Original quadratic reward", "Bounded − quadratic"))):
        mesh = ax.pcolormesh(np.arange(102) - .5, yedges, values, shading="flat", rasterized=True,
                             cmap="cividis" if index < 2 else "BrBG",
                             vmin=0 if index < 2 else -limit, vmax=100 if index < 2 else limit)
        ax.set(title=title, xlim=(-.5, 100.5), ylim=(0, 20), xlabel="Evaluated PPO generation, g")
        ax.set_xticks(np.arange(0, 101, 20), generations[::20])
        ax.set_yticks(np.arange(0, 21, 5))
        cbar = fig.colorbar(mesh, ax=ax, pad=.025, shrink=.92)
        cbar.set_label("First hold completed by t (%)" if index < 2 else "Difference (percentage points)")
        if index < 2:
            cbar.set_ticks(np.arange(0, 101, 20))
    axes[0].set_ylabel("Elapsed physical evaluation time, t (s)")
    fig.suptitle("First sustained stabilization across learning — 100 paired training seeds", fontsize=14)
    fig.get_layout_engine().set(rect=(0, .075, 1, .925))
    fig.text(.035, .025, "Measured g = 0, 10, …, 1000; no generation interpolation. First event completes a 2 s hold. Difference panel is descriptive, not a significance map.", fontsize=9)
    save(fig, "comparison_heatmaps")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), layout="constrained")
    ax = axes[0]
    for name, values in (("bounded", bounded[-1]), ("quadratic", quadratic[-1])):
        lo, hi = exact_binomial_intervals(values.astype(int), 100, .95)
        ax.errorbar(generations, values, yerr=np.vstack((values - lo * 100, hi * 100 - values)),
                    fmt="o", markersize=2.8, elinewidth=.65, capsize=0, color=colors[name], alpha=.75,
                    label=name.capitalize())
        ax.annotate(f"{values[-1]:.0f}%", (1000, values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=colors[name], fontweight="bold")
    ax.set(xlim=(-15, 1100), ylim=(0, 100), title="Success by 20 s", xlabel="Evaluated generation, g",
           ylabel="Controllers completing the hold (%)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
    ax = axes[1]
    values = delta[-1]
    ax.axhline(0, color="#777777", linewidth=.8)
    ax.plot(generations, values, "o", color=colors["bounded"], markersize=3)
    low, high = primary["difference_interval_percentage_points"]
    ax.errorbar([1000], [values[-1]], yerr=[[values[-1] - low], [high - values[-1]]],
                fmt="o", color="#242424", capsize=4, markersize=4, linewidth=1.2,
                label="Final endpoint: conservative 95% paired interval")
    ax.set(xlim=(-15, 1100), ylim=(min(-15, float(values.min()) - 5, low - 5), max(45, high + 5)),
           title="Paired reward difference", xlabel="Evaluated generation, g",
           ylabel="Bounded − quadratic (percentage points)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
    fig.get_layout_engine().set(rect=(0, .085, 1, .915))
    fig.text(.045, .025, "Left: pointwise 95% exact arm intervals. Right: measured descriptive differences; only G=1000 has the declared paired inferential interval.", fontsize=8.5)
    save(fig, "endpoint_comparison")

    fig, ax = plt.subplots(figsize=(8.8, 5.1), layout="constrained")
    for name, events in (("bounded", events_b[:, -1]), ("quadratic", events_q[:, -1])):
        observed = np.sort(events[np.isfinite(events)])
        x = np.unique(np.r_[0., observed, 20.])
        values = np.searchsorted(observed, x + 1e-12, side="right")
        lo, hi = exact_binomial_intervals(values.astype(int), 100, .95)
        ax.step(x, values, where="post", color=colors[name], linewidth=1.9,
                label=f"{name.capitalize()}: {len(observed)}/100 by 20 s")
        ax.fill_between(x, lo * 100, hi * 100, step="post", color=colors[name], alpha=.12)
    ax.set(xlim=(0, 20), ylim=(0, 100), title="Final generation: when the first hold is completed",
           xlabel="Elapsed physical evaluation time, t (s)", ylabel="Controllers completing the hold by t (%)")
    ax.set_xticks(np.arange(0, 21, 2))
    ax.set_yticks(np.arange(0, 101, 20))
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="#e5e5e5", linewidth=.6)
    fig.get_layout_engine().set(rect=(0, .07, 1, .93))
    fig.text(.07, .025, "Exact empirical step curves, G=1000. Shading: pointwise 95% arm intervals, not simultaneous bands or intervals for the paired difference.", fontsize=8)
    save(fig, "final_cumulative")


def report(output, summary):
    b, q, p = (summary[k] for k in ("bounded", "quadratic", "primary_paired_endpoint"))
    ci = p["difference_interval_percentage_points"]
    selected = "\n".join(f"| {t} | {b['final_counts_at_times'][str(t)]}/100 | {q['final_counts_at_times'][str(t)]}/100 |"
                         for t in (5, 10, 13, 15, 20))
    curve_rows = "\n".join(f"| {r['generation']} | {r['bounded_successes']} | {r['quadratic_successes']} | {r['difference_pp']:+.0f} |"
                           for r in summary["selected_checkpoints"])
    text = f'''# Reward comparison: completed paired follow-up

All 100 quadratic controllers completed the fixed G=1000 budget. At the declared final endpoint, the bounded reward had **68/100** first holds completed by 20 s; the original quadratic reward had **52/100**. The observed difference is **+16 percentage points** for the bounded reward.

The prespecified conservative 95% paired difference interval is **{ci[0]:.1f} to {ci[1]:.1f} percentage points**, which includes zero. The two-sided exact McNemar test gives **p={p['exact_mcnemar_two_sided_p_value']:.5f}**. Report both: the deliberately conservative interval and exact conditional test are different procedures and need not give matching zero-exclusion decisions. The sample favors the bounded form, while the effect magnitude remains uncertain. This follow-up was designed after observing the bounded arm; it is not an independent confirmatory replication or evidence of an optimal reward.

## The original research object

H(g,t) is the fraction of independently trained controllers whose FIRST qualifying sustained hold is completed by physical time t after training generation g. Every valid failed controller remains in the denominator. Each event completes a 2 s residence interval; later exits do not undo it. Horizontal axis: measured generation. Vertical axis: evaluation time. Color: cumulative percentage.

![Measured reward heatmaps](<{(output / 'comparison_heatmaps.png').as_posix()}>)

Both arm heatmaps share 0–100% scales. The difference panel uses percentage points; positive means more bounded-reward controllers completed by that cell. Only g=0,10,...,1000 are measured. No generation interpolation, smoothing or significance filtering is used. Events were monitored every 2 ms; reporting rows are 0.1 s. The difference surface is descriptive, not a simultaneous significance map.

## Declared primary paired endpoint: G=1000, t=20 s

| Paired outcome | Controllers |
| --- | ---: |
| Both rewards succeed | {p['both_succeed']} |
| Bounded only succeeds | {p['bounded_only']} |
| Quadratic only succeeds | {p['quadratic_only']} |
| Neither succeeds | {p['neither_succeeds']} |
| Total paired seeds | 100 |

The sampling unit is a training seed, not a heatmap cell. Seed 20000 in one arm is paired with seed 20000 in the other. There are 100 paired seeds, not 200 independent replications. Starting weights, optimizer state, private random streams and normalization were verified identical for all pairs; reward-dependent later trajectories and learned statistics may differ.

Marginal pointwise 95% Clopper–Pearson arm intervals are {b['pointwise_95_interval_percentage'][0]:.1f}–{b['pointwise_95_interval_percentage'][1]:.1f}% for bounded and {q['pointwise_95_interval_percentage'][0]:.1f}–{q['pointwise_95_interval_percentage'][1]:.1f}% for quadratic. Overlap of these marginal intervals is not the paired comparison test.

For the fixed paired interval, Delta=p10-p01. Two 97.5% Clopper–Pearson intervals jointly cover the two discordance probabilities with at least 95% confidence by the union bound. Opposite endpoints are subtracted. This method is conservative and retains uncertainty even if no discordances are seen. The exact McNemar calculation conditions on the {p['bounded_only'] + p['quadratic_only']} discordant pairs and tests equal directions, using a two-sided binomial test. There is one declared primary endpoint at alpha=0.05; no heatmap-wide p-value search or later interval replacement was performed.

## Learning across measured generations

| Generation | Bounded successes by 20 s | Quadratic successes by 20 s | Difference (pp) |
| --- | ---: | ---: | ---: |
{curve_rows}

![Learning and paired endpoint difference](<{(output / 'endpoint_comparison.png').as_posix()}>)

The first evaluated nonzero checkpoint was G={b['first_evaluated_generation_with_success']} for bounded and G={q['first_evaluated_generation_with_success']} for quadratic. Policies between measured generations are unknown; these are not exact learning-onset times. Both cohorts may improve or regress with generation. All observed columns and failures are retained; the endpoint stays G=1000 rather than a scanned maximum. The bounded scanned maximum was {b['scanned_maximum_count_descriptive_only']}/100 and the quadratic scanned maximum was {q['scanned_maximum_count_descriptive_only']}/100; these are descriptive only.

The direction depends on training progress: at G=600 the quadratic arm had 68/100 successes against 56/100 for bounded; at the final G=1000 the ordering reversed to 52/100 against 68/100. Thus the bounded formulation did not dominate the complete learning history. This observed rise and regression in the quadratic arm is retained as a finding, not erased by choosing its best checkpoint. These intermediate comparisons are descriptive.

## Final-generation evaluation timing and continued holding

| Physical time | Bounded completed first hold | Quadratic completed first hold |
| --- | ---: | ---: |
{selected}

![Final cumulative timing](<{(output / 'final_cumulative.png').as_posix()}>)

The direction also depends on physical evaluation time: at G=1000, quadratic had 13/100 completions by 5 s against 4/100 for bounded, while bounded had 55/100 against 48/100 by 10 s and 68/100 against 52/100 by 20 s. The empirical final cumulative curves cross; neither arm is uniformly above the other at every evaluation time. This is why the full H(g,t) surface, rather than only a final percentage or a success-conditioned median, is central to the research question.

Mean capped first-event time, mean[min(T_s,20)], was **{b['final_mean_capped_first_event_time_s']:.3f} s** for bounded and **{q['final_mean_capped_first_event_time_s']:.3f} s** for quadratic (bounded minus quadratic **{summary['descriptive_capped_time_difference_s']:+.3f} s**; negative favors earlier/more frequent bounded success). This includes failures at the cap and is a descriptive restricted-to-20-s quantity. It is not an uncensored mean completion time. Success-conditioned medians were {b['final_success_conditioned_median_s']:.3f} s and {q['final_success_conditioned_median_s']:.3f} s; those omit failures and must not be used alone to rank the methods.

There were {b['final_censored_at_20s']} bounded and {q['final_censored_at_20s']} quadratic no-event observations censored at 20 s. Among successes, {b['final_successes_with_later_exit']} bounded and {q['final_successes_with_later_exit']} quadratic controllers later exited the tolerance region; {b['final_successes_with_no_observed_later_exit']} and {q['final_successes_with_no_observed_later_exit']} respectively had no observed later exit before 20 s. A late first success has less remaining follow-up, and no observed exit is not proof of permanent stability. These are secondary descriptive outcomes.

## What changed, and what this comparison can establish

The intervention is the complete reward form in REWARD_COMPARISON_PROTOCOL.md. Both use the same numerical scales and penalty weights. The bounded form gates upright benefit by cart position and speeds and saturates penalties; the quadratic form uses the ungated upright term and quadratic penalties. This comparison cannot attribute the result solely to boundedness, one gate or one coefficient. Neither formula is claimed to be universally standard or optimal. Returns from the two formulas are not a common performance score.

All other scientific settings were fixed: 100 fresh private controllers per arm, seeds 20000..20099; G=1000; 1000 new interactions per generation; ten PPO epochs and minibatches of 250; separate actor/critic gradient clipping; sin/cos input; private rolling observation normalization frozen during evaluation. Training alone has a terminal 3 m cart boundary and cost 10. Evaluation uses the original unbounded rail, deterministic matching policy/preprocessing and exact downward start [0,0,pi,0], always for the full 20 s. No curriculum, teacher, reward rescaling, best-checkpoint selection or seed replacement was introduced.

The nonlinear physical model remains the 1 kg cart and uniform 0.1 kg, 1 m rod, +/-10 N force, zero damping, 20 ms held control and float64 RK4 at 2 ms. A first event completes a 2 s hold simultaneously within |x|<=0.1 m, |v|<=0.1 m/s, wrapped upright angle<=5 degrees and |omega|<=0.2 rad/s. The raw 20 ms traces are consistent with events but cannot independently establish continuous residence between 2 ms monitoring samples.

Worker count and per-process Windows CPU scheduling changed execution speed only. Exact policy/optimizer/normalization/trajectory equivalence was verified on the bounded technical benchmark, and all 100 G=1 production recoveries matched private states exactly. The original physical and learning source files were unchanged.

## Completion, wall-clock time and audit

Quadratic final evaluation completed **{summary['completed_local_time']}**. The accelerated continuation elapsed {summary['continuation_elapsed_wall_hours']:.2f} h. Its G=992 update alone recorded {summary['largest_elapsed_generation_hours']:.2f} h; all other recorded continuation phases total {summary['other_recorded_phases_hours']:.2f} h. The saved logs do not establish the cause of that wall-clock gap, so it must not be called verified CPU time or a confirmed laptop sleep. The earlier roughly ten-hour estimate was a pace projection, not the actual elapsed duration. Step-count-based physical evaluation time and the fixed scientific budget are unaffected by a wall-clock pause.

The quadratic arm retained 100M training interactions and 10,100 complete checkpoint evaluations. Its new audit verified {summary['quadratic_audit']['source_artifact_hashes_verified']:,} committed artifact hashes, 10,100 canonical policy/normalization/recovery bundles, all 10,110,100 raw trajectory rows, source/runtime/declarations, actual final recipe metadata and saved fractions/confidence exports. It also verified the 21 sealed files of the previously completed bounded analysis. The original bounded audit and all seven reference sources remain preserved; this analysis reuses that sealed audit rather than claiming to have re-read all original raw traces again.

Reproducible exports: [summary.json](<{(output / 'summary.json').as_posix()}>), [paired_seed_results.csv](<{(output / 'paired_seed_results.csv').as_posix()}>), [paired_surface.npz](<{(output / 'paired_surface.npz').as_posix()}>), [endpoint_comparison.csv](<{(output / 'endpoint_comparison.csv').as_posix()}>), [quadratic audit](<{(output / 'quadratic_audit/audit.json').as_posix()}>). PNG and PDF figures use the same data. The bounded reference completion/report remain unchanged. No training or new experiment was launched during analysis.

## Interpretation and next decision

The primary H(g,t) measurement and the planned reward-form follow-up are now reported, including failures, timing and regressions. The observed final advantage is 16 percentage points for this bounded formulation and recipe; the conservative interval is wide and the study is a follow-up using an already observed arm. It does not establish general superiority, optimal penalties or robustness. Robustness requires a separately fixed perturbation design and seed-level analysis; no further training, coefficient search or robustness campaign starts automatically.
'''
    (output / "RESULTS.md").write_text(text, encoding="utf-8")
    (ROOT / "REWARD_COMPARISON_RESULTS.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / "analysis")
    audit_dir = output / "quadratic_audit"
    audit = json.loads((audit_dir / "audit.json").read_text())
    checks = json.loads((audit_dir / "followup_checks.json").read_text())
    assert audit["status"] == checks["status"] == "passed"
    # Recheck the quadratic audit artifacts before treating them as verified input.
    for name, expected in json.loads((audit_dir / "artifact_hashes.json").read_text()).items():
        assert digest(audit_dir / name) == expected
    plan = json.loads((ROOT / "configs/research/reward_comparison_followup.json").read_text())
    completion_path = ROOT / plan["reference_completion_file"]
    assert digest(completion_path) == plan["reference_completion_sha256"]
    completion = json.loads(completion_path.read_text())
    reference = Path(completion["analysis_directory"])
    assert digest(reference / "artifact_hashes.json") == completion["analysis_artifact_hashes_sha256"]
    for name, expected in json.loads((reference / "artifact_hashes.json").read_text()).items():
        assert digest(reference / name) == expected
    seeds, generations, times = list(range(20000, 20100)), np.arange(0, 1001, 10), np.arange(201) / 10
    b, exit_b = load_events(reference / "individual_events.csv", seeds, generations)
    q, exit_q = load_events(audit_dir / "individual_events.csv", seeds, generations)
    bc, qc = counts_from_events(b, times), counts_from_events(q, times)
    for directory, counts in ((reference, bc), (audit_dir, qc)):
        with np.load(directory / "surface.npz") as values:
            assert np.array_equal(values["generations"], generations)
            assert np.array_equal(values["counts"], counts)
    primary = paired_endpoint(np.isfinite(b[:, -1]), np.isfinite(q[:, -1]))
    sb, sq = arm_summary(b, exit_b, bc, generations), arm_summary(q, exit_q, qc, generations)
    final_manifest = json.loads((Path(audit["final_run"]) / "manifest.json").read_text())
    timings = final_manifest["timings"]
    large = max(timings, key=lambda row: row["training_s"])
    selected = [{"generation": int(g), "bounded_successes": int(bc[-1, j]),
                 "quadratic_successes": int(qc[-1, j]), "difference_pp": float(bc[-1, j] - qc[-1, j])}
                for j, g in enumerate(generations) if g in (0, 100, 300, 500, 600, 750, 900, 1000)]
    summary = {"status": "paired analysis completed", "computed_utc": datetime.now(timezone.utc).isoformat(),
               "models_per_arm": 100, "paired_seeds": 100, "measured_generations": 101,
               "evaluation_time_points": 201, "bounded": sb, "quadratic": sq,
               "primary_paired_endpoint": primary,
               "descriptive_capped_time_difference_s": float((np.minimum(b[:, -1], 20) - np.minimum(q[:, -1], 20)).mean()),
               "selected_checkpoints": selected, "quadratic_audit": audit,
               "reference_completion_sha256": digest(completion_path),
               "completed_local_time": datetime.fromisoformat(final_manifest["finished_utc"]).astimezone(timezone(timedelta(hours=5))).isoformat(),
               "continuation_elapsed_wall_hours": final_manifest["elapsed_wall_s"] / 3600,
               "largest_elapsed_generation": large["generation"],
               "largest_elapsed_generation_hours": large["training_s"] / 3600,
               "other_recorded_phases_hours": sum(r["training_s"] + r["evaluation_export_s"]
                   for r in timings if r is not large) / 3600,
               "followup_limitation": "Designed after reference outcomes were known; not an independent confirmatory replication",
               "no_new_training_started": True}
    write_json(output / "summary.json", summary)
    write_csv(output / "paired_seed_results.csv", [
        {"training_seed": seed, "bounded_success_time_s": "" if not np.isfinite(b[i, -1]) else b[i, -1],
         "quadratic_success_time_s": "" if not np.isfinite(q[i, -1]) else q[i, -1],
         "bounded_success": bool(np.isfinite(b[i, -1])), "quadratic_success": bool(np.isfinite(q[i, -1])),
         "bounded_capped_time_s": min(b[i, -1], 20), "quadratic_capped_time_s": min(q[i, -1], 20),
         "paired_capped_time_difference_s": min(b[i, -1], 20) - min(q[i, -1], 20),
         "bounded_later_exit_s": "" if not np.isfinite(exit_b[i, -1]) else exit_b[i, -1],
         "quadratic_later_exit_s": "" if not np.isfinite(exit_q[i, -1]) else exit_q[i, -1]}
        for i, seed in enumerate(seeds)])
    blo, bhi = exact_binomial_intervals(bc, 100, .95)
    qlo, qhi = exact_binomial_intervals(qc, 100, .95)
    write_csv(output / "endpoint_comparison.csv", [
        {"generation": int(g), "bounded_successes": int(bc[-1, j]), "quadratic_successes": int(qc[-1, j]),
         "difference_percentage_points": int(bc[-1, j] - qc[-1, j]),
         "bounded_pointwise_95_lower_pct": float(100 * blo[-1, j]),
         "bounded_pointwise_95_upper_pct": float(100 * bhi[-1, j]),
         "quadratic_pointwise_95_lower_pct": float(100 * qlo[-1, j]),
         "quadratic_pointwise_95_upper_pct": float(100 * qhi[-1, j])}
        for j, g in enumerate(generations)])
    np.savez_compressed(output / "paired_surface.npz", training_seeds=seeds, generations=generations,
                        evaluation_times_s=times, bounded_counts=bc, quadratic_counts=qc,
                        difference_percentage_points=bc - qc, bounded_event_times_s=b, quadratic_event_times_s=q,
                        event_censoring_note="inf means no first hold observed by 20 s", n_paired_seeds=100)
    figures(output, generations, times, bc, qc, b, q, primary)
    report(output, summary)
    shutil.copy2(Path(__file__), output / "comparison_script_snapshot.py")
    shutil.copy2(ROOT / "upright_pendulum/paired_reward_results.py", output / "paired_statistics_snapshot.py")
    shutil.copy2(ROOT / "REWARD_COMPARISON_PROTOCOL.md", output / "declared_protocol_snapshot.md")
    shutil.copy2(ROOT / "configs/research/reward_comparison_followup.json", output / "declared_comparison.json")
    write_json(output / "artifact_hashes.json", {p.relative_to(output).as_posix(): digest(p)
        for p in output.rglob("*") if p.is_file() and p != output / "artifact_hashes.json"})
    print(json.dumps({"primary": primary, "bounded": sb, "quadratic": sq, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
