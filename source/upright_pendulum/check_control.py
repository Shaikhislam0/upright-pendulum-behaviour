"""Reference local-balancing check, with no RL training.

Run: python -m upright_pendulum.check_control
Uses the nonlinear experiment environment, actuator and success measurement.
The LQR sees physical state directly, rather than encoded policy observations.
"""

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import numpy as np

from .actions import NormalizedForceWrapper
from .balance_control import balance_force, design_balance_lqr
from .experiments import ExperimentConfig, make_experiment_env
from .physics import Parameters
from .stabilization import StabilizationCriteria


def evaluate_case(initial, gain, *, feedback):
    env = NormalizedForceWrapper(make_experiment_env())
    rows = []
    try:
        env.reset(seed=2026, options={"state": initial})
        physical = env.unwrapped
        rows.append([0.0, *physical.state, 0.0])
        max_force = 0.0
        saturated = 0
        total_return = 0.0
        for _ in range(physical.max_episode_steps):
            force = balance_force(physical.state, gain, force_limit=physical.force_limit) if feedback else 0.0
            command = np.array([force / physical.force_limit], dtype=np.float32)
            _, reward, terminated, truncated, info = env.step(command)
            applied = info["applied_force"]
            max_force = max(max_force, abs(applied))
            saturated += abs(applied) >= physical.force_limit - 1e-7
            total_return += reward
            rows.append([info["elapsed_time"], *physical.state, applied])
            if terminated or truncated:
                break
        measurement = info["episode_measurement"]
        final = physical.state
        error = final.copy()
        error[2] = math.atan2(math.sin(error[2]), math.cos(error[2]))
        return {
            "initial_state": list(initial), "controller": "local_LQR" if feedback else "zero_force",
            **measurement, "final_state": final.tolist(), "final_error_state": error.tolist(),
            "final_in_region": StabilizationCriteria().contains(final),
            "max_abs_force_N": max_force, "saturated_action_fraction": saturated / (len(rows) - 1),
            "max_abs_cart_position_m": max(abs(row[1]) for row in rows),
            "raw_episode_return": total_return,
        }, rows
    finally:
        env.close()


def run_checks(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    design = design_balance_lqr()
    gain = design["K"]
    ad, bd = design["Ad"], design["Bd"]
    poles = np.linalg.eigvals(ad - bd @ gain)
    controllability = np.column_stack([np.linalg.matrix_power(ad, i) @ bd for i in range(4)])
    cases = {}
    for name, initial, feedback in (
        ("displaced_tilted", [0.2, 0.0, 0.1, 0.0], True),
        ("mirrored", [-0.2, 0.0, -0.1, 0.0], True),
        ("moving", [0.0, 0.25, 0.0, 0.3], True),
        ("equivalent_after_rotation", [0.2, 0.0, 2 * math.pi + 0.1, 0.0], True),
        ("same_perturbation_without_control", [0.2, 0.0, 0.1, 0.0], False),
        ("downward_without_control", [0.0, 0.0, math.pi, 0.0], False),
    ):
        result, rows = evaluate_case(initial, gain, feedback=feedback)
        result["trajectory_file"] = f"{name}.csv"
        cases[name] = result
        with (output / result["trajectory_file"]).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "x_m", "v_m_s", "theta_rad_unwrapped", "omega_rad_s", "force_N"])
            writer.writerows(rows)
        label = "success" if result["event_observed"] else "no success"
        suffix = f" at {result['success_time']:.3f} s" if result["event_observed"] else ""
        print(f"{name}: {label}{suffix}")

    positive = [r for r in cases.values() if r["controller"] == "local_LQR"]
    negative = [r for r in cases.values() if r["controller"] == "zero_force"]
    checks = {
        "linear_controllability_rank_four": np.linalg.matrix_rank(controllability) == 4,
        "linear_closed_loop_poles_inside_unit_circle": np.max(np.abs(poles)) < 1,
        "all_local_feedback_cases_succeed_and_stay": all(
            r["event_observed"] and r["final_in_region"] and r["first_post_success_exit_time"] is None
            for r in positive),
        "all_final_local_error_components_below_1e_minus_4": all(
            max(abs(value) for value in r["final_error_state"]) < 1e-4 for r in positive),
        "zero_force_cases_do_not_succeed": all(not r["event_observed"] for r in negative),
        "all_episodes_followed_for_twenty_seconds": all(r["followup_time"] == 20 for r in cases.values()),
        "actuator_limit_respected": all(r["max_abs_force_N"] <= 10 for r in cases.values()),
    }
    package = Path(__file__).parent
    report = {
        "purpose": "Check local feedback, nonlinear integration, actuator and success measurement; no PPO training",
        "limitations": ["Fixed deterministic local cases, not a basin-of-attraction proof.",
                        "Does not demonstrate swing-up from hanging down or PPO learnability.",
                        "LQR design cost is separate from the unchanged RL reward."],
        "versions": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scipy", "gymnasium")}},
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(package.glob("*.py"))},
        "parameters": asdict(Parameters()),
        "experiment": ExperimentConfig().to_dict(),
        "criteria": asdict(StabilizationCriteria()),
        "control_dt_s": 0.02, "integration_dt_s": 0.002, "evaluation_horizon_s": 20.0,
        "force_limit_N": 10.0,
        "design": {name: value.tolist() for name, value in design.items()},
        "closed_loop_poles_real_imag": [[float(p.real), float(p.imag)] for p in poles],
        "cases": cases, "checks": {name: bool(value) for name, value in checks.items()},
        "passed": bool(all(checks.values())),
        "reference": "https://underactuated.mit.edu/lqr.html",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {output / 'report.json'}")
    print(json.dumps({"passed": report["passed"]}))
    if not report["passed"]:
        raise AssertionError("reference control check failed; inspect report.json")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("control_checks"))
    run_checks(parser.parse_args().output)
