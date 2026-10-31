import csv

import numpy as np
import pytest

from analysis.compare_reward_results import arm_summary, counts_from_events, load_events


def rows():
    output = []
    for generation in (0, 10):
        for seed in (10, 11):
            ts = 3. if generation == 10 and seed == 10 else None
            output.append({"generation": generation, "training_seed": seed,
                           "success_time": "" if ts is None else ts,
                           "first_post_success_exit_time": "", "event_observed": ts is not None,
                           "right_censored": ts is None, "has_succeeded": ts is not None,
                           "observed_time": ts if ts is not None else 20., "followup_time": 20.})
    return output


def save(path, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def test_pair_alignment_is_by_seed_and_generation_not_row_order(tmp_path):
    path = tmp_path / "events.csv"
    save(path, rows()[::-1])
    events, exits = load_events(path, [10, 11], [0, 10])
    np.testing.assert_array_equal(events, [[np.inf, 3.], [np.inf, np.inf]])
    assert np.isinf(exits).all()
    counts = counts_from_events(events, np.array([0., 2., 3., 20.]))
    np.testing.assert_array_equal(counts, [[0, 0], [0, 0], [0, 1], [0, 1]])
    summary = arm_summary(events, exits, counts, np.array([0, 10]))
    assert summary["final_successes_by_20s"] == 1
    assert summary["final_censored_at_20s"] == 1
    assert summary["final_mean_capped_first_event_time_s"] == 11.5
    assert summary["final_success_conditioned_median_s"] == 3.
    assert summary["first_evaluated_generation_with_success"] == 10


@pytest.mark.parametrize("mutation,match", [
    (lambda values: values[:-1], "missing seed"),
    (lambda values: values + values[:1], "duplicate seed"),
    (lambda values: [{**values[0], "followup_time": 19.}, *values[1:]], "follow-up"),
    (lambda values: [{**values[0], "event_observed": True}, *values[1:]], "censoring flags"),
])
def test_incomplete_duplicate_or_miscensored_pairs_are_rejected(tmp_path, mutation, match):
    path = tmp_path / "events.csv"
    save(path, mutation(rows()))
    with pytest.raises(ValueError, match=match):
        load_events(path, [10, 11], [0, 10])
