"""Measurement sanity checks; no trained controller is involved.

Run: python -m upright_pendulum.check_stabilization
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .env import CartPendulumEnv
from .stabilization import StabilizationCriteria, StabilizationMonitor, StabilizationWrapper


def run_checks(output: Path):
    criteria = StabilizationCriteria()
    cases = {}
    for name, initial in (
        ("physical_downward_zero_force", [0, 0, math.pi, 0]),
        ("physical_exact_upright_zero_force", [0, 0, 0, 0]),
    ):
        env = StabilizationWrapper(CartPendulumEnv(), criteria)
        env.reset(seed=2026, options={"state": initial})
        for _ in range(1000):
            _, _, terminated, truncated, info = env.step([0])
        assert not terminated and truncated
        cases[name] = env.monitor.summary()
        env.close()

    monitor = StabilizationMonitor(criteria)
    for index in range(3151):
        monitor.observe(index * 0.002, [0, 0, 0 if index >= 2150 else math.pi, 0])
    cases["synthetic_entry_4_30_completion_6_30"] = monitor.summary()

    monitor = StabilizationMonitor(criteria)
    for index in range(2001):
        monitor.observe(index * 0.002, [0, 0, math.pi if index == 503 else 0, 0])
    cases["synthetic_2ms_violation_at_1_006"] = monitor.summary()

    checks = [
        cases["physical_downward_zero_force"]["right_censored"],
        cases["physical_downward_zero_force"]["followup_time"] == 20.0,
        cases["physical_exact_upright_zero_force"]["success_time"] == 2.0,
        math.isclose(cases["synthetic_entry_4_30_completion_6_30"]["success_time"], 6.3, abs_tol=1e-12),
        math.isclose(cases["synthetic_2ms_violation_at_1_006"]["success_time"], 3.008, abs_tol=1e-12),
    ]
    report = {
        "purpose": "Validate the measurement definition, not controller performance",
        "criteria": asdict(criteria), "evaluation_horizon_s": 20.0,
        "control_dt_s": 0.02, "monitoring_max_gap_s": 0.002,
        "cases": cases, "passed": bool(all(checks)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["passed"]:
        raise AssertionError("stabilization measurement check failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation_checks/report.json"))
    run_checks(parser.parse_args().output)
