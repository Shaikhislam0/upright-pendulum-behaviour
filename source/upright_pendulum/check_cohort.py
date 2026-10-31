"""Check the intended percentage measurement; this command does not train PPO."""

import argparse
from pathlib import Path

import numpy as np

from .cohort import EvaluationSettings, cumulative_counts, evaluate_cohort
from .train_pilot import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("cohort_checks"))
    parser.add_argument("--parallel", action="store_true", help="Use a worker process for each physical simulation")
    args = parser.parse_args()
    # One exact upright equilibrium succeeds at 2 s; one downward never does.
    result = evaluate_cohort([lambda obs: [0.0], lambda obs: [0.0]],
        settings=EvaluationSettings(horizon_s=2.2, parallel=args.parallel),
        initial_states=[[0, 0, 0, 0], [0, 0, np.pi, 0]])
    expected = [0.0 if row["evaluation_time_s"] < 2 - 1e-12 else 50.0 for row in result["samples"]]
    assert [row["percentage"] for row in result["samples"]] == expected
    # Exact completion times and non-events, with all four members retained.
    assert cumulative_counts([2.0, 6.3, None, None], [0, 1.9, 2, 6.29, 6.3, 20]).tolist() == [0, 0, 1, 1, 2, 2]
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "report.json", {"purpose": "measurement check, not learned performance", "passed": True, **result})
    print("PASS: 0% before the 2 s completed hold; 50% thereafter; both members retained.")
    print(f"Report: {args.output / 'report.json'}")


if __name__ == "__main__":
    main()
