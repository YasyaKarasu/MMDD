"""Visible-column IDF scores and the B+IDF ranking of the candidate scope.

Visible score (no recovery, no headers): a query column is eligible when its five rows hold at
least two distinct typed non-URL values (``entity_url`` excluded). For an eligible query column
and a target column, ``match(x_i)`` is the D-0.98 match of row value ``x_i`` against that column's
non-URL values.

    vis_row(T) = max over (query col, target col) of  (1/5) * sum_i match(x_i)
    df(x_i)    = number of candidates in the scope with any column matching x_i
    w(x_i)     = log((K + 1) / (df + 1)) / log(K + 1),   K = scope size
    vis_idf(T) = max over (query col, target col) of  (1/5) * sum_i match(x_i) * w(x_i)

The denominator stays five; weights are never renormalized, so a column of values that every
candidate contains (df = K -> w = 0) cannot buy back a full score.

B+IDF order of the scope: tier 1 = targets with a recovered-bridge score > 0, by bridge score;
tier 2 = the rest with vis_idf > 0, by (vis_idf, vis_row); tier 3 = Stage-1 order. Ties fall back
to Stage-1 rank. The tiered order is fused with Stage 1 by RRF.
"""
from __future__ import annotations

import math
from typing import Any

from .matching import Matcher, comparable_texts, rrf_fuse
from .values import ROWS, norm, value_info


def visible_query_columns(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns: dict[int, dict[str, Any]] = {}
    for row in rows:
        for cell in row["cells"]:
            if norm(cell["column_name"]) == "entity_url":
                continue
            column = columns.setdefault(int(cell["column_id"]), {"column_id": int(cell["column_id"]), "rows": [None] * ROWS})
            info = value_info(cell["text"])
            if info["key"] is not None and info["kind"] != "URL":
                column["rows"][row["query_row_id"]] = info
    return [columns[i] for i in sorted(columns)
            if len({v["key"] for v in columns[i]["rows"] if v is not None}) >= 2]


def visible_target_values(column: dict[str, Any]) -> list[dict[str, Any]]:
    if column["attribute"] == "entity_url":
        return []
    return [v for v in column["values"] if v["kind"] not in ("URL", "EMPTY")]


def visible_texts(rows: list[dict[str, Any]], tables: dict[str, dict[str, Any]]) -> set[str]:
    cells = [v for column in visible_query_columns(rows) for v in column["rows"] if v is not None]
    return {text for table in tables.values() for column in table["columns"]
            for text in comparable_texts(cells, visible_target_values(column))}


def idf_weight(df: int, scope: int) -> float:
    return math.log((scope + 1) / (df + 1)) / math.log(scope + 1)


def visible_scores(candidates: list[str], rows: list[dict[str, Any]], tables: dict[str, dict[str, Any]],
                   matcher: Matcher) -> tuple[dict[str, float], dict[str, float]]:
    """``(vis_row, vis_idf)`` for every candidate; df is counted within ``candidates``."""
    query_columns = visible_query_columns(rows)
    target_columns = {t: [(c["column_id"], vals) for c in tables[t]["columns"] if (vals := visible_target_values(c))]
                      for t in candidates}
    # hits[qi][rid][t] = target column ids of t matching query cell (qi, rid)
    hits = [[{t: [cid for cid, vals in target_columns[t] if matcher.match(x, ("visible", t, cid), vals)]
              for t in candidates} if x is not None else None for x in column["rows"]] for column in query_columns]
    vis_row = {t: 0.0 for t in candidates}
    vis_idf = {t: 0.0 for t in candidates}
    for qi, column in enumerate(query_columns):
        weights = {rid: idf_weight(sum(bool(cols) for cols in by_target.values()), len(candidates))
                   for rid, by_target in enumerate(hits[qi]) if by_target is not None}
        for t in candidates:
            for cid, _ in target_columns[t]:
                matched = [rid for rid in weights if cid in hits[qi][rid][t]]
                vis_row[t] = max(vis_row[t], len(matched) / ROWS)
                vis_idf[t] = max(vis_idf[t], sum(weights[rid] for rid in matched) / ROWS)
    return vis_row, vis_idf


def bidf_order(candidates: list[str], bridge: dict[str, float], vis_row: dict[str, float],
               vis_idf: dict[str, float]) -> list[str]:
    """Tiered PURE order of the scope (bridge, then visible IDF, then Stage 1)."""
    rank = {t: i for i, t in enumerate(candidates)}
    tier1 = sorted((t for t in candidates if bridge[t] > 0), key=lambda t: (-round(bridge[t], 8), rank[t], t))
    rest = [t for t in candidates if bridge[t] == 0]
    tier2 = sorted((t for t in rest if vis_idf[t] > 0),
                   key=lambda t: (-round(vis_idf[t], 12), -round(vis_row[t], 8), rank[t], t))
    return tier1 + tier2 + [t for t in rest if vis_idf[t] <= 0]


def bidf_ranking(candidates: list[str], bridge: dict[str, float], vis_row: dict[str, float],
                 vis_idf: dict[str, float], rrf_constant: int) -> list[str]:
    return rrf_fuse(candidates, bidf_order(candidates, bridge, vis_row, vis_idf), rrf_constant)
