"""Validate full swing-up feasibility with the unchanged plant/success rule."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from .experiments import ExperimentConfig
from .swingup_control import EnergySwingupController
from .train_pilot import make_pilot_env, write_json


def evaluate_reference(initial=None, *, direction=1):
    initial = [0, 0, math.pi, 0] if initial is None else initial
    controller = EnergySwingupController(kick_direction=direction)
    env = make_pilot_env(ExperimentConfig())
    rows = []
    try:
        env.reset(options={"state": initial})
        rows.append([0., *env.unwrapped.state, 0.])
        for _ in range(1000):
            force = controller.force(env.unwrapped.state)
            _, _, terminated, truncated, info = env.step(np.array([force / 10], dtype=np.float32))
            rows.append([info["elapsed_time"], *env.unwrapped.state, info["applied_force"]])
        assert not terminated and truncated
        result = {**info["episode_measurement"], "initial_state": list(initial),
                  "max_abs_cart_position_m": max(abs(row[1]) for row in rows),
                  "max_abs_force_N": max(abs(row[-1]) for row in rows),
                  "final_state": env.unwrapped.state.tolist()}
        return result, rows
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("swingup_checks"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, state, direction in (
        ("exact_downward", [0, 0, math.pi, 0], 1),
        ("mirrored_kick", [0, 0, -math.pi, 0], -1),
        ("perturbed_downward", [.1, 0, math.pi + .02, 0], 1),
        ("periodic_equivalent", [0, 0, 3 * math.pi, 0], 1),
    ):
        result, rows = evaluate_reference(state, direction=direction)
        cases[name] = result
        with (args.output / f"{name}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["time_s", "x_m", "v_m_s", "theta_rad_unwrapped", "omega_rad_s", "force_N"])
            writer.writerows(rows)
        print(f"{name}: success={result['success_time']}, max |x|={result['max_abs_cart_position_m']:.3f} m", flush=True)
    passed = all(result["event_observed"] and result["first_post_success_exit_time"] is None
                 and result["followup_time"] == 20 and result["max_abs_force_N"] <= 10 for result in cases.values())
    write_json(args.output / "report.json", {"purpose": "physical full-task feasibility, not PPO performance", "passed": passed,
        "controller": {"energy_gain": 5, "cart_kp": .5, "cart_kd": 1, "kick_N": 2, "kick_duration_s": .2,
                       "capture_angle_rad": .25, "capture_angular_speed_rad_s": 2.5}, "cases": cases})
    if not passed:
        raise RuntimeError("full-task reference did not meet the unchanged success definition")


if __name__ == "__main__":
    main()
