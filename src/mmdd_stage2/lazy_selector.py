"""Exact top-branch selection using the table prior as an upper bound.

For every column, log P(T,c) = log P(T) + log P(c|T) <= log P(T).
The denominator always contains the original candidate pool. We stop only on
a strict inequality, so ties retain the exhaustive scheduler's ordering.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def log_softmax(values: Sequence[float]) -> list[float]:
    """Use the same arithmetic as the R5b exhaustive scheduler."""
    if not values or not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Expected nonempty finite logits")
    maximum = max(values)
    denominator = maximum + math.log(sum(math.exp(v - maximum) for v in values))
    return [float(v - denominator) for v in values]


def select_lazy(
    base: Sequence[str],
    path_logits: Mapping[str, float],
    eligible: Mapping[str, Sequence[int]],
    score_table: Callable[[str], dict[str, Any]],
    *,
    branches: int = 10,
    per_table: int = 3,
) -> dict[str, Any]:
    """Evaluate only tables which can still enter the exact capped top-k.

    ``score_table`` must return the same independent per-table logits as the
    exhaustive path. The result contains *only* evaluated logits; skipped
    tables receive explicit upper bounds, never invented scores.
    """
    if len(base) != len(set(base)) or not base or min(branches, per_table) <= 0:
        raise ValueError("Invalid candidate pool or branch budget")
    prior = dict(zip(base, log_softmax([path_logits[t] for t in base])))
    rank = {t: i + 1 for i, t in enumerate(base)}
    order = sorted(base, key=lambda t: (-prior[t], rank[t]))
    results: dict[str, Any] = {}
    pairs: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for target in order:
        if len(selected) == branches and prior[target] < selected[-1]["pair_score"]:
            break
        result = score_table(target)
        columns = result["column_logits"]
        ids = [int(c["column_id"]) for c in columns]
        if (len(ids) != len(set(ids)) or set(ids) != set(eligible[target])
                or set(result["eligible_column_ids"]) != set(ids)):
            raise ValueError(f"Selector eligibility changed: {target}")
        results[target] = result
        normalized = log_softmax([float(c["logit"]) for c in columns]) if columns else []
        for column, value in zip(columns, normalized):
            pairs.append({"target_id": target, "column_id": int(column["column_id"]),
                          "stage1_rank": rank[target], "pair_score": prior[target] + value})
        counts: Counter[str] = Counter()
        selected = []
        for pair in sorted(pairs, key=lambda p: (-p["pair_score"], p["stage1_rank"], p["column_id"])):
            if counts[pair["target_id"]] == per_table:
                continue
            counts[pair["target_id"]] += 1
            selected.append(pair)
            if len(selected) == branches:
                break
    pruned = {t: prior[t] for t in order if t not in results}
    return {
        "selector": results,
        "selected_pairs": selected,
        "certificate": {
            "rule": "log_P_column_given_table_le_zero_strict_bound_v1",
            "normalization_candidates": list(base),
            "evaluated_targets": list(results),
            "pruned_upper_bounds": pruned,
            "threshold": selected[-1]["pair_score"] if len(selected) == branches else None,
            "available_pairs_after_cap": sum(min(per_table, len(eligible[t])) for t in base),
        },
    }


def build_lazy_plan(
    query_id: str,
    base: Sequence[str],
    path_logits: Mapping[str, float],
    retained_paths: dict[str, Any],
    tables: dict[str, Any],
    result: dict[str, Any],
    build_plan: Callable[..., Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse R5b view construction with all C50 priors and sparse evaluated pairs.

    Empty entries are internal scheduler placeholders, not persisted selector
    outputs. The certificate records why those tables require no forward.
    """
    scores = {t: result["selector"].get(t, {"column_logits": [], "eligible_column_ids": []}) for t in base}
    plan, pairs = build_plan(query_id, list(base), path_logits, scores, retained_paths, tables)
    plan["available_pairs_after_cap"] = result["certificate"]["available_pairs_after_cap"]
    return plan, pairs
