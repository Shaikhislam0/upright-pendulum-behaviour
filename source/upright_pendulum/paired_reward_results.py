"""Declared paired endpoint statistics; no independent-arm approximation."""

import numpy as np

from .cohort_results import exact_binomial_intervals


def paired_endpoint(bounded, quadratic, confidence_level=.95):
    """Positive difference favors bounded reward; one row is one paired seed.

    For D=B-Q, E[D]=p10-p01. Each discordance count has a binomial marginal.
    Two (1-alpha/2) Clopper-Pearson intervals jointly cover p10,p01 with
    probability at least 1-alpha by the union bound, regardless of dependence.
    Subtract opposite endpoints to obtain a conservative paired difference CI.
    This remains nonzero-width when no discordances were observed.
    """
    bounded, quadratic = np.asarray(bounded), np.asarray(quadratic)
    if (bounded.dtype.kind != "b" or quadratic.dtype.kind != "b"
            or bounded.ndim != 1 or bounded.shape != quadratic.shape or not bounded.size):
        raise ValueError("supply equal nonempty boolean vectors aligned by training seed")
    n = len(bounded)
    both = int(np.sum(bounded & quadratic))
    bounded_only = int(np.sum(bounded & ~quadratic))
    quadratic_only = int(np.sum(~bounded & quadratic))
    neither = n - both - bounded_only - quadratic_only
    # The helper validates the requested confidence before Bonferroni adjustment.
    from .cohort_results import validate_confidence
    validate_confidence(confidence_level)
    if confidence_level is None:
        raise ValueError("a confidence level is required")
    lower, upper = exact_binomial_intervals(
        np.array([bounded_only, quadratic_only]), n, 1 - (1 - confidence_level) / 2)
    from scipy.stats import binomtest
    discordant = bounded_only + quadratic_only
    p_value = float(binomtest(bounded_only, discordant, .5).pvalue) if discordant else 1.
    return {
        "paired_seeds": n, "both_succeed": both, "bounded_only": bounded_only,
        "quadratic_only": quadratic_only, "neither_succeeds": neither,
        "bounded_successes": both + bounded_only, "quadratic_successes": both + quadratic_only,
        "difference_percentage_points": 100 * (bounded_only - quadratic_only) / n,
        "difference_interval_percentage_points": [100 * float(lower[0] - upper[1]),
                                                   100 * float(upper[0] - lower[1])],
        "confidence_level": confidence_level,
        "interval_method": "Conservative paired difference: Bonferroni joint Clopper-Pearson discordance bounds",
        "exact_mcnemar_two_sided_p_value": p_value,
        "inference_scope": "One declared final endpoint; follow-up design after reference outcomes were known",
    }
