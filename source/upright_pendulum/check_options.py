"""Check reward/input options without creating or training a policy.

Run from the project directory: python -m upright_pendulum.check_options
Optionally select one file with --config configs/sin_cos_small_penalty.json.
"""

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform

import numpy as np

from .env import CartPendulumEnv
from .experiments import ExperimentConfig, make_experiment_env
from .observations import encode_observation
from .rewards import reward_components
from .stabilization import StabilizationCriteria, StabilizationWrapper


def check_config(config):
    cases = {
        "upright_stationary": ([0, 0, 0, 0], 0),
        "downward_stationary": ([0, 0, math.pi, 0], 0),
        "cart_displaced_1m": ([1, 0, 0, 0], 0),
        "cart_moving_1m_s": ([0, 1, 0, 0], 0),
        "pole_rotating_5rad_s": ([0, 0, 0, 5], 0),
        "upright_with_10N_force": ([0, 0, 0, 0], 10),
    }
    values = {
        name: {"physical_state": state, "applied_force_N": force,
               "reward_components": reward_components(state, force, config.reward),
               "policy_observation": encode_observation(state, config.observation).tolist()}
        for name, (state, force) in cases.items()
    }
    checks = {
        "upright_reward_is_one": values["upright_stationary"]["reward_components"]["total"] == 1,
        "downward_reward_is_zero": values["downward_stationary"]["reward_components"]["total"] == 0,
    }

    # Same prescribed forces for both environments: no feedback controller.
    reference = StabilizationWrapper(CartPendulumEnv(max_episode_steps=100))
    candidate = make_experiment_env(config, max_episode_steps=100)
    reference.reset(seed=2026)
    observation, _ = candidate.reset(seed=2026)
    checks["observations_in_declared_space"] = candidate.observation_space.contains(observation)
    checks["physics_identical"] = True
    checks["success_measurement_identical"] = True
    checks["reward_breakdown_consistent"] = True
    for index in range(100):
        action = [2 * math.sin(index / 10)]
        _, _, terminated, truncated, info = reference.step(action)
        observation, reward, actual_terminated, actual_truncated, actual_info = candidate.step(action)
        checks["physics_identical"] &= bool(np.array_equal(reference.unwrapped.state, candidate.unwrapped.state))
        checks["success_measurement_identical"] &= (
            actual_info["stabilization"] == info["stabilization"]
            and (actual_terminated, actual_truncated) == (terminated, truncated))
        checks["observations_in_declared_space"] &= candidate.observation_space.contains(observation)
        components = actual_info["reward_components"]
        reconstructed = components["upright"] - sum(
            components[name] for name in ("position_cost", "cart_velocity_cost", "angular_velocity_cost", "force_cost"))
        checks["reward_breakdown_consistent"] &= math.isclose(reward, reconstructed, abs_tol=1e-12)
    checks["horizon_measurement_identical"] = actual_info["episode_measurement"] == info["episode_measurement"]
    reference.close()
    candidate.close()

    mode = config.observation.angle_representation
    equivalent_delta = float(np.linalg.norm(
        encode_observation([0, 0, 0.3, 0], config.observation)
        - encode_observation([0, 0, 0.3 + 2 * math.pi, 0], config.observation)))
    branch_delta = float(np.linalg.norm(
        encode_observation([0, 0, math.pi - 1e-5, 0], config.observation)
        - encode_observation([0, 0, math.pi + 1e-5, 0], config.observation)))
    checks["expected_rotation_encoding"] = equivalent_delta < 1e-6 if mode != "unwrapped" else equivalent_delta > 1.99
    checks["expected_downward_branch_behavior"] = branch_delta > 1.99 if mode == "wrapped" else branch_delta < 1e-4
    return {
        "configuration": config.to_dict(), "known_states": values,
        "observation_diagnostics": {
            "equivalent_orientations_input_distance": equivalent_delta,
            "input_distance_across_downward_branch": branch_delta,
            "note": "Encoding properties only; no evidence of improved learning yet",
        },
        "diagnostic_rollout_duration_s": 2.0,
        "checks": {key: bool(value) for key, value in checks.items()},
        "passed": bool(all(checks.values())),
    }


def run_checks(paths, output):
    if not paths:
        raise ValueError("no configurations found; run from the project directory or use --config")
    configs = [ExperimentConfig.load(path) for path in paths]
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("configuration names must be unique")
    physical = CartPendulumEnv()
    report = {
        "purpose": "Validate configurable rewards and policy inputs; no training or policy comparison",
        "versions": {"python": platform.python_version(), "numpy": version("numpy"),
                     "gymnasium": version("gymnasium")},
        "physical_baseline": {
            "parameters": asdict(physical.parameters), "force_limit_N": physical.force_limit,
            "control_dt_s": physical.control_dt, "integration_dt_s": physical.integration_dt,
            "default_evaluation_horizon_s": physical.max_episode_steps * physical.control_dt,
        },
        "stabilization_criteria": asdict(StabilizationCriteria()),
        "configurations": [check_config(config) for config in configs],
    }
    physical.close()
    report["passed"] = all(item["passed"] for item in report["configurations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for item in report["configurations"]:
        label = "PASS" if item["passed"] else "FAIL"
        config = item["configuration"]
        print(f"{label}: {config['name']} | angular-velocity penalty={config['reward']['angular_velocity_penalty']:g}")
    print(f"Checked {len(configs)} configurations. Report: {output}")
    print(json.dumps({"passed": report["passed"]}))
    if not report["passed"]:
        raise AssertionError("reward/input checks failed; inspect the report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="check one JSON file; default checks configs/*.json")
    parser.add_argument("--output", type=Path, default=Path("option_checks/report.json"))
    args = parser.parse_args()
    paths = [args.config] if args.config else sorted(Path("configs").glob("*.json"))
    run_checks(paths, args.output)
