"""Reproducible finite-time physics validation, independent of any controller.

Run: python -m upright_pendulum.validate --output validation
"""

import argparse
import csv
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
from scipy.integrate import simpson, solve_ivp

from .physics import Parameters, energy, horizontal_momentum, integrate


def reference_rhs(time, state, force, p):
    """Matrix-form ODE for the independent adaptive integration reference."""
    _, v, theta, omega = state
    b = p.pole_mass * p.com_distance * np.cos(theta)
    mass_matrix = [[p.cart_mass + p.pole_mass, b], [b, p.inertia_pivot]]
    rhs = [force - p.cart_damping * v + p.pole_mass * p.com_distance * np.sin(theta) * omega**2,
           p.pole_mass * p.gravity * p.com_distance * np.sin(theta) - p.pivot_damping * omega]
    a, alpha = np.linalg.solve(mass_matrix, rhs)
    return [v, a, omega, alpha]


def run_validation(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    p = Parameters()
    dt, h, duration = 0.02, 0.002, 60.0
    times = np.arange(round(duration / dt) + 1) * dt
    cases = {
        "oscillation": [0, 0, 2.4, 0],
        "rotation": [0, 0, np.pi, 10],
        "coupled_motion": [0.2, 1.5, 0.2, -2],
        "near_upright": [0, 0, 0.01, 0],
    }
    tolerances = {"max_energy_drift_J": 1e-6, "max_momentum_drift_kg_m_s": 1e-7,
                  "reference_max_component_error": 1e-7, "power_balance_residual_J": 1e-6}
    report = {
        "stage": 1, "parameters": asdict(p), "inertia_pivot_kg_m2": p.inertia_pivot,
        "control_dt_s": dt, "integration_dt_s": h, "unforced_duration_s": duration,
        "arithmetic": "float64", "integrator": "classical RK4, zero-order-held force",
        "tolerances": tolerances,
        "versions": {"python": platform.python_version(),
                     **{name: version(name) for name in ("numpy", "gymnasium", "scipy", "matplotlib")}},
        "unforced_cases": {},
    }
    trajectories = {}
    for name, initial in cases.items():
        states = [np.asarray(initial, dtype=np.float64)]
        for _ in times[1:]:
            states.append(integrate(states[-1], 0, dt, p, max_step=h))
        states = np.array(states)
        energies = np.array([energy(s, p) for s in states])
        momenta = np.array([horizontal_momentum(s, p) for s in states])
        de, dp = energies - energies[0], momenta - momenta[0]
        trajectories[name] = (states, de, dp)
        report["unforced_cases"][name] = {
            "initial_state": initial, "initial_energy_J": energies[0],
            "max_abs_energy_drift_J": float(np.max(np.abs(de))),
            "scaled_energy_drift": float(np.max(np.abs(de)) / max(abs(energies[0]), 2*p.pole_mass*p.gravity*p.com_distance)),
            "max_abs_momentum_drift_kg_m_s": float(np.max(np.abs(dp))),
        }
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "x_m", "v_m_s", "theta_rad_unwrapped", "omega_rad_s",
                             "energy_J", "energy_drift_J", "momentum_kg_m_s", "momentum_drift_kg_m_s"])
            writer.writerows(zip(times, *states.T, energies, de, momenta, dp))

    initial, force, reference_duration = [0, 0, 2.4, 1.3], 1.7, 2.0
    ref_times = np.linspace(0, reference_duration, 101)
    ref = solve_ivp(reference_rhs, [0, reference_duration], initial, args=(force, p),
                    method="DOP853", rtol=1e-12, atol=1e-14, t_eval=ref_times)
    if not ref.success:
        raise RuntimeError(ref.message)
    steps = [0.02, 0.01, 0.005, 0.002]
    errors = [float(np.linalg.norm(integrate(initial, force, reference_duration, p, max_step=s)
                                 - ref.y[:, -1])) for s in steps]
    state = np.asarray(initial, dtype=float)
    reference_errors = [np.zeros(4)]
    for index in range(1, len(ref_times)):
        state = integrate(state, force, dt, p, max_step=h)
        reference_errors.append(state - ref.y[:, index])
    component_errors = np.max(np.abs(reference_errors), axis=0)
    report["reference_and_convergence"] = {
        "solver": "SciPy DOP853, independent matrix-form RHS",
        "rtol": 1e-12, "atol": 1e-14, "duration_s": reference_duration,
        "initial_state": initial, "force_N": force,
        "max_component_errors_x_v_theta_omega": component_errors.tolist(),
        "step_sizes_s": steps,
        "final_state_error_norms": errors,
        "halving_error_ratios": [errors[0]/errors[1], errors[1]/errors[2]],
        "norm_note": "Euclidean norm of raw SI components, used only as a convergence diagnostic",
    }

    # A constant force does exact work F * (x_final - x_initial).
    final = integrate(initial, force, 5, p, max_step=h)
    work_residual = energy(final, p) - energy(initial, p) - force * (final[0] - initial[0])
    damped = Parameters(cart_damping=0.1, pivot_damping=0.01)
    damped_initial = np.array([0, 0.4, 2.2, 3.0])
    states = [damped_initial]
    damped_times = np.arange(2501) * h
    for _ in damped_times[1:]:
        states.append(integrate(states[-1], 0, h, damped, max_step=h))
    states = np.array(states)
    dissipation = simpson(damped.cart_damping * states[:, 1]**2
                          + damped.pivot_damping * states[:, 3]**2, x=damped_times)
    damping_residual = energy(states[-1], damped) - energy(damped_initial, damped) + dissipation
    report["power_balance"] = {
        "duration_s": 5.0, "constant_force_work_residual_J": float(work_residual),
        "damped_unforced_balance_residual_J": float(damping_residual),
        "damped_parameters": asdict(damped), "damped_initial_state": damped_initial.tolist(),
        "dissipation_quadrature": "composite Simpson at 0.002 s",
    }
    checks = [
        all(c["max_abs_energy_drift_J"] < tolerances["max_energy_drift_J"]
            and c["max_abs_momentum_drift_kg_m_s"] < tolerances["max_momentum_drift_kg_m_s"]
            for c in report["unforced_cases"].values()),
        bool(np.max(component_errors) < tolerances["reference_max_component_error"]),
        all(10 < ratio < 22 for ratio in report["reference_and_convergence"]["halving_error_ratios"]),
        abs(work_residual) < tolerances["power_balance_residual_J"],
        abs(damping_residual) < tolerances["power_balance_residual_J"],
    ]
    report["passed"] = bool(all(checks))
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_diagnostics(output, times, trajectories, steps, errors)
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise AssertionError("Physics validation failed; inspect report.json")


def plot_diagnostics(output, times, trajectories, steps, errors):
    import os
    cache = output.resolve() / ".mplconfig"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlelocation": "left", "axes.labelcolor": "#252525"})
    colors = ["#252525", "#bc4b2f", "#26756b", "#89722b"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for (name, (_, de, dp)), color in zip(trajectories.items(), colors):
        axes[0].plot(times, de * 1e9, color=color, label=name.replace("_", " "), linewidth=1)
        axes[1].plot(times, dp * 1e9, color=color, linewidth=1)
    axes[0].set(title="A  Mechanical energy", xlabel="Elapsed time (s)", ylabel="E(t) − E(0) (nJ)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(title="B  Horizontal momentum", xlabel="Elapsed time (s)", ylabel="p(t) − p(0) (10⁻⁹ kg m/s)")
    axes[2].loglog(steps, errors, "o-", color="#252525", label="RK4 vs DOP853")
    guideline = errors[0] * (np.array(steps) / steps[0])**4
    axes[2].loglog(steps, guideline, "--", color="#bc4b2f", label="Fourth-order slope")
    axes[2].set(title="C  Timestep convergence", xlabel="Integration step (s)", ylabel="Final state error norm (raw SI)")
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    axes[2].xaxis.set_major_locator(FixedLocator([0.002, 0.005, 0.01, 0.02]))
    axes[2].xaxis.set_major_formatter(FixedFormatter(["0.002", "0.005", "0.010", "0.020"]))
    axes[2].xaxis.set_minor_locator(NullLocator())
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.15)
    fig.suptitle("Cart–rigid-pendulum · Stage 1 physics validation", x=0.01, ha="left", fontsize=15)
    fig.savefig(output / "physics_validation.png", dpi=180)
    fig.savefig(output / "physics_validation.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("validation"))
    run_validation(parser.parse_args().output)
