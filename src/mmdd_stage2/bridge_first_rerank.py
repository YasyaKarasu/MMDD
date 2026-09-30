"""Bridge-first reranking over a frozen candidate scope.

Renamed and merged form of the LEXICO_IDF ablation (``work/LEXICO_IDF_ABLATION_V1``). The ordering
has three fixed tiers:

1. candidates with recovered-bridge evidence, ordered by the row-count bridge score;
2. the remaining candidates, ordered by an inverse-document-frequency weighted visible score;
3. everything else, left in the frozen Stage-1 order.

The IDF weight is ``log((K+1)/(df+1)) / log(K+1)`` for a scope of ``K`` candidates, where ``df``
counts distinct candidates whose compatible column holds a matching value. The visible score keeps
a fixed denominator of five and is never renormalized by the weight sum, so a generic column cannot
buy back a full score by matching many low-information cells. Tier 2 breaks ties on the unweighted
row score and then on the Stage-1 rank, so a selection with no bridge evidence follows Stage 1.

The visible tier deliberately ignores column headers: physical (query column, target column) value
pairs define the match, exactly as in the Direct scorer. Column headers still key the bridge tier.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .matched_row_score import DENOMINATOR_ROWS, UnitCosineMatcher, header_key, value_info

RRF_CONSTANT = 60
RRF_ROUND_DIGITS = 12
BRIDGE_ROUND_DIGITS = 8
IDF_ROUND_DIGITS = 12
ENTITY_URL_HEADER = "entity_url"
VISIBLE_COLUMN_MIN_DISTINCT_KEYS = 2


def visible_query_columns(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Eligible query columns: at least two distinct typed values across the five rows.

    ``entity_url`` is excluded because it identifies the entity rather than describing it, and
    URL cells are never comparable values. The distinct-key floor is what keeps an identifier or
    constant column from acting as a free join key.
    """
    if len(rows) != DENOMINATOR_ROWS:
        raise ValueError(f"query must hold exactly {DENOMINATOR_ROWS} rows")
    columns: dict[int, dict[str, Any]] = {}
    for row in rows:
        seen: set[int] = set()
        for cell in row["cells"]:
            index = int(cell["column_id"])
            if index in seen:
                raise ValueError("duplicate query cell")
            seen.add(index)
            attribute = header_key(cell["column_name"])
            if attribute == ENTITY_URL_HEADER:
                continue
            column = columns.setdefault(index, {"column_id": index, "column_name": cell["column_name"],
                                                "attribute": attribute, "rows": [None] * DENOMINATOR_ROWS})
            if column["attribute"] != attribute:
                raise ValueError("inconsistent query column header")
            info = value_info(cell["text"])
            if info["key"] is not None and info["kind"] != "URL":
                column["rows"][int(row["query_row_id"])] = info
    eligible = []
    for index in sorted(columns):
        column = columns[index]
        column["distinct_keys"] = sorted({value["key"] for value in column["rows"] if value is not None})
        column["nonempty_rows"] = sum(value is not None for value in column["rows"])
        if len(column["distinct_keys"]) >= VISIBLE_COLUMN_MIN_DISTINCT_KEYS:
            eligible.append(column)
    return eligible


def visible_target_values(column: dict[str, Any]) -> list[dict[str, Any]]:
    """Comparable values of a target column: typed, non-empty and not a URL."""
    if header_key(column["column_name"]) == ENTITY_URL_HEADER:
        return []
    return [value for value in column["values"] if value.get("key") is not None and value["kind"] not in {"URL", "EMPTY"}]


def idf_weight(document_frequency: int, scope_size: int) -> float:
    """Fixed IDF weight; no smoothing constant beyond the ``+1`` in numerator and denominator."""
    if scope_size < 1:
        raise ValueError("candidate scope must be non-empty")
    return math.log((scope_size + 1) / (document_frequency + 1)) / math.log(scope_size + 1)


def visible_row_scores(
    rows: Sequence[dict[str, Any]],
    base: Sequence[str],
    tables: Mapping[str, dict[str, Any]],
    matcher: UnitCosineMatcher,
) -> tuple[dict[str, float], dict[tuple[int, str], int], dict[tuple[int, int], set[str]]]:
    """Unweighted visible score: max over (query column, target column) of matched rows / 5.

    Also returns the per-row best match count and, for the IDF step, which candidates each visible
    query cell matched (counted once per candidate).
    """
    columns = visible_query_columns(rows)
    scores = {target: 0.0 for target in base}
    best_rows: dict[tuple[int, str], int] = {}
    cell_hits: dict[tuple[int, int], set[str]] = {}
    for column_index, query_column in enumerate(columns):
        for row_id in range(DENOMINATOR_ROWS):
            cell = query_column["rows"][row_id]
            if cell is None:
                continue
            for target in base:
                for target_column in tables[target]["columns"]:
                    values = visible_target_values(target_column)
                    if values and matcher.best(cell, values)["matched"]:
                        cell_hits.setdefault((column_index, row_id), set()).add(target)
        for target in base:
            best = 0
            for target_column in tables[target]["columns"]:
                values = visible_target_values(target_column)
                if not values:
                    continue
                matched = sum(1 for row_id in range(DENOMINATOR_ROWS)
                              if query_column["rows"][row_id] is not None
                              and matcher.best(query_column["rows"][row_id], values)["matched"])
                best = max(best, matched)
            best_rows[(column_index, target)] = best
            scores[target] = max(scores[target], best / DENOMINATOR_ROWS)
    return scores, best_rows, cell_hits


def visible_idf_scores(
    rows: Sequence[dict[str, Any]],
    base: Sequence[str],
    tables: Mapping[str, dict[str, Any]],
    matcher: UnitCosineMatcher,
    cell_hits: Mapping[tuple[int, int], set[str]],
) -> dict[str, float]:
    """IDF-weighted visible score ``1/5 * sum_i match(x_i, column) * w(x_i)``.

    The denominator is the fixed row count, never the weight sum: a column that matches many
    generic values still has to win row-by-row.
    """
    columns = visible_query_columns(rows)
    scope_size = len(base)
    scores = {target: 0.0 for target in base}
    for column_index, query_column in enumerate(columns):
        weights = {row_id: idf_weight(len(cell_hits.get((column_index, row_id), ())), scope_size)
                   for row_id in range(DENOMINATOR_ROWS)}
        for target in base:
            for target_column in tables[target]["columns"]:
                values = visible_target_values(target_column)
                if not values:
                    continue
                total = 0.0
                for row_id in range(DENOMINATOR_ROWS):
                    cell = query_column["rows"][row_id]
                    if cell is None:
                        continue
                    if matcher.best(cell, values)["matched"]:
                        total += weights[row_id]
                scores[target] = max(scores[target], total / DENOMINATOR_ROWS)
    return scores


def rrf_fuse(base: Sequence[str], pure: Sequence[str]) -> list[str]:
    """Reciprocal-rank fusion of the Stage-1 order and a reranked order."""
    stage1 = {target: rank for rank, target in enumerate(base, 1)}
    reranked = {target: rank for rank, target in enumerate(pure, 1)}
    fused = {target: 1 / (RRF_CONSTANT + stage1[target]) + 1 / (RRF_CONSTANT + reranked[target]) for target in base}
    return sorted(base, key=lambda target: (-round(fused[target], RRF_ROUND_DIGITS), stage1[target], target))


def bridge_first_orders(
    base: Sequence[str],
    bridge: Mapping[str, float],
    visible_row: Mapping[str, float],
    visible_idf: Mapping[str, float],
) -> dict[str, list[str]]:
    """Pure and RRF-fused orders for the three-tier bridge-first reranking."""
    stage1 = {target: rank for rank, target in enumerate(base, 1)}
    with_bridge = sorted((target for target in base if bridge[target] > 0),
                         key=lambda target: (-round(bridge[target], BRIDGE_ROUND_DIGITS), stage1[target], target))
    rest = [target for target in base if bridge[target] == 0]
    scored = sorted((target for target in rest if visible_idf[target] > 0),
                    key=lambda target: (-round(visible_idf[target], IDF_ROUND_DIGITS),
                                        -round(visible_row[target], BRIDGE_ROUND_DIGITS), stage1[target], target))
    unscored = sorted((target for target in rest if visible_idf[target] <= 0), key=lambda target: stage1[target])
    pure = with_bridge + scored + unscored
    return {"PURE": pure, "RRF60": rrf_fuse(base, pure)}


def rerank(
    base: Sequence[str],
    bridges: Sequence[dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    tables: Mapping[str, dict[str, Any]],
    matcher: UnitCosineMatcher,
    bridge_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Full bridge-first reranking pass over one query's frozen candidate scope."""
    if len(set(base)) != len(base):
        raise ValueError("candidate scope must be unique")
    visible_row, best_rows, cell_hits = visible_row_scores(rows, base, tables, matcher)
    visible_idf = visible_idf_scores(rows, base, tables, matcher, cell_hits)
    orders = bridge_first_orders(base, bridge_scores, visible_row, visible_idf)
    return {"orders": orders, "visible_row_scores": visible_row, "visible_idf_scores": visible_idf,
            "bridge_scores": dict(bridge_scores), "best_matched_rows": best_rows,
            "scope_size": len(base)}
