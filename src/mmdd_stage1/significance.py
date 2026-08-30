"""Paired uncertainty estimates for retrieval comparisons."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def paired_bootstrap_delta(
    per_query_a: Sequence[float],
    per_query_b: Sequence[float],
    *,
    iterations: int = 10_000,
    seed: int = 13,
) -> dict[str, float | int]:
    """Bootstrap the paired per-query delta ``a - b``.

    Queries are resampled as pairs, preserving the correlation between the two
    systems on each query.  The returned interval is a percentile 95% CI.
    """

    if len(per_query_a) != len(per_query_b):
        raise ValueError("Paired bootstrap inputs must have the same length")
    if not per_query_a:
        raise ValueError("Paired bootstrap requires at least one query")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    deltas = np.asarray(per_query_a, dtype=np.float64) - np.asarray(
        per_query_b, dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(deltas), size=(iterations, len(deltas)))
    bootstrap_means = deltas[sample_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    result = {
        "mean": float(deltas.mean()),
        "delta_mean": float(deltas.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_delta_lt_0": float(np.mean(bootstrap_means < 0.0)),
        "iterations": int(iterations),
        "queries": int(len(deltas)),
        "seed": int(seed),
    }
    return result
