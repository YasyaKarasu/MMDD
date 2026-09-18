"""S2-R4 Phase J: log-space table x column joint selection over a real candidate pool.

The exact contract, with no fusion weight and no temperature:

    log_p_table = log_softmax(all native table logits of the query's candidate pool)
    log_rho_T   = log_softmax(all GENUINE column logits of table T)
    log_joint   = log_p_table[T] + log_rho_T[c]
    J(T,c)      = exp(log_joint)

Three outputs are kept apart and must never be substituted for one another:

* ``PAIR_RANK``          - every (T, c) globally ordered by log_joint.
* ``TABLE_RANK_MAX``     - each T once, ordered by max_c log_joint. This is *not* the
                           table-only ranking; it prefers tables with concentrated columns.
* ``TABLE_MARGINAL``     - sum_c J(T,c) == p_T. A unit test that table-only order is
                           reproduced by the marginal; it is not a new reranker.

Truncating to Top-k columns never renormalises: the kept probabilities still sum to
less than one, so cross-table comparisons stay in the original scale.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch

SENTINEL_MARKERS = ('ranking_only_sentinels', 'sentinel', 'pvr_shortlist_outside')


def assert_no_sentinel(payload: dict[str, Any]) -> None:
    """PVR shortlist-outside scores are ranking-only sentinels, not probabilities."""
    marker = str(payload.get('outside_scores', '')).lower()
    if marker in SENTINEL_MARKERS:
        raise ValueError(
            'PVR shortlist-outside scores are ranking-only sentinels and cannot be '
            'softmaxed into cross-table probabilities'
        )


def log_joint(table_logits: torch.Tensor, column_logits: torch.Tensor,
              column_mask: torch.Tensor, *, target_ids: Sequence[str] | None = None,
              column_ids: Sequence[Sequence[int]] | None = None,
              original_ranks: Sequence[int] | None = None) -> dict[str, Any]:
    """Exact log-space product. ``column_mask`` marks genuine columns only."""
    if table_logits.ndim != 1:
        raise ValueError('table_logits must be 1-D over the candidate pool')
    targets, columns = column_logits.shape
    if column_logits.ndim != 2 or column_mask.shape != column_logits.shape:
        raise ValueError('column tensors must be [targets, columns]')
    if targets != table_logits.shape[0]:
        raise ValueError('one table logit per candidate target is required')
    table_logits = table_logits.to(torch.float64)
    column_logits = column_logits.to(torch.float64)
    column_mask = column_mask.to(torch.bool)
    if not bool(torch.isfinite(table_logits).all()):
        raise ValueError('non-finite native table logit; the J arm is BLOCKED for this query')

    log_p_table = torch.log_softmax(table_logits, dim=-1)

    # A table with no genuine column keeps its table probability but contributes no pair.
    row_has_column = column_mask.any(dim=-1)
    safe_logits = torch.where(column_mask, column_logits, torch.full_like(column_logits, -torch.inf))
    log_rho = torch.log_softmax(safe_logits, dim=-1)
    log_rho = torch.where(column_mask, log_rho, torch.full_like(log_rho, -torch.inf))

    pair_log = log_p_table.unsqueeze(-1) + log_rho
    pair_log = torch.where(column_mask, pair_log, torch.full_like(pair_log, -torch.inf))
    pair_probability = torch.where(column_mask, torch.exp(pair_log), torch.zeros_like(pair_log))

    marginal = pair_probability.sum(dim=-1)
    table_probability = torch.exp(log_p_table)

    order: list[tuple[str, int]] = []
    if target_ids is not None and column_ids is not None:
        ranks = list(original_ranks) if original_ranks is not None else list(range(targets))
        candidates = []
        for t in range(targets):
            if not bool(row_has_column[t]):
                continue
            for c in range(columns):
                if not bool(column_mask[t, c]):
                    continue
                candidates.append((
                    -float(pair_log[t, c]),
                    int(ranks[t]),
                    str(target_ids[t]),
                    int(column_ids[t][c]),
                ))
        candidates.sort()
        order = [(target, column) for _, _, target, column in candidates]

    best = torch.where(row_has_column, pair_log.max(dim=-1).values,
                       torch.full_like(pair_log[:, 0], -torch.inf))
    return {
        'table_log_probability': log_p_table,
        'table_probability': table_probability,
        'column_log_probability': log_rho,
        'column_probability': torch.where(column_mask, torch.exp(log_rho), torch.zeros_like(log_rho)),
        'pair_log_probability': pair_log,
        'pair_probability': pair_probability,
        'marginal': marginal,
        'best_pair_log_probability': best,
        'tables_with_columns': int(row_has_column.sum()),
        'tables_without_columns': int((~row_has_column).sum()),
        'pair_order': order,
    }


def table_rank_max(joint: dict[str, Any], target_ids: Sequence[str]) -> list[str]:
    """Deduplicated table order: first occurrence of each T in the global pair order.

    Equivalent to ordering tables by max_c log_joint, with ties broken by the pair
    ordering contract (original rank, target id, canonical column id).
    """
    seen: list[str] = []
    known = set(target_ids)
    for target, _ in joint['pair_order']:
        if target in known and target not in seen:
            seen.append(target)
    return seen


def table_rank_only(joint: dict[str, Any], target_ids: Sequence[str],
                    original_ranks: Sequence[int]) -> list[str]:
    """J0: the table-only ranking. Column scores are diagnostic and must not move it."""
    order = sorted(range(len(target_ids)),
                   key=lambda i: (-float(joint['table_probability'][i]), int(original_ranks[i]),
                                  str(target_ids[i])))
    return [str(target_ids[i]) for i in order]


def top_k_pairs(joint: dict[str, Any], k: int, *, max_columns_per_table: int | None = None) -> list[tuple[str, int]]:
    """First k pairs in global order, optionally capped per table. No renormalisation."""
    chosen: list[tuple[str, int]] = []
    counts: dict[str, int] = {}
    for target, column in joint['pair_order']:
        if max_columns_per_table is not None and counts.get(target, 0) >= max_columns_per_table:
            continue
        chosen.append((target, column))
        counts[target] = counts.get(target, 0) + 1
        if len(chosen) >= k:
            break
    return chosen


def joint_table_column_recall(joint: dict[str, Any], gold: dict[str, set[int]], *, top_tables: int,
                              top_columns: int, target_ids: Sequence[str]) -> float | None:
    """Correct table inside TABLE_RANK_MAX@K and a correct column inside its own Top-k."""
    order = table_rank_max(joint, target_ids)
    position = {target: i for i, target in enumerate(order)}
    hits = []
    for target, columns in gold.items():
        if not columns or target not in position:
            continue
        if position[target] >= top_tables:
            hits.append(0.0)
            continue
        row = joint['column_probability'][list(target_ids).index(target)]
        ranked = sorted(range(len(row)), key=lambda c: (-float(row[c]), c))
        hits.append(1.0 if set(ranked[:top_columns]) & columns else 0.0)
    if not hits:
        return None
    return sum(hits) / len(hits)
