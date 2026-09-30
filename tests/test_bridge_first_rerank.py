from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mmdd_stage2.bridge_first_rerank import (
    bridge_first_orders,
    idf_weight,
    rerank,
    rrf_fuse,
    visible_query_columns,
    visible_target_values,
)
from mmdd_stage2.matched_row_score import DictVectorStore, UnitCosineMatcher, value_info


def rows(*values_per_column: tuple[str, list[str | None]]) -> list[dict]:
    """Five query rows from (column name, per-row values)."""
    columns = [(name, values) for name, values in values_per_column]
    return [{"query_row_id": row_id,
             "cells": [{"column_id": index, "column_name": name, "text": values[row_id] or ""}
                       for index, (name, values) in enumerate(columns)]}
            for row_id in range(5)]


def table(header: str, values: list[str]) -> dict:
    return {"columns": [{"column_id": 0, "column_name": header, "values": [value_info(v) for v in values]}]}


def store(**vectors: list[float]) -> DictVectorStore:
    return DictVectorStore({name.replace("_", " "): np.asarray(vector, dtype=np.float32)
                            for name, vector in vectors.items()})


def name_space_store() -> DictVectorStore:
    """One-hot vectors so only equal names reach the cosine threshold."""
    return DictVectorStore({name: np.asarray([1.0 if position == index else 0.0 for position in range(5)],
                                             dtype=np.float32)
                            for index, name in enumerate(["acme", "beta", "gamma", "delta", "epsilon"])})


def test_visible_columns_drop_entity_url_and_low_information_columns():
    query = rows(("Entity_URL", ["https://a", "https://b", "https://c", "https://d", "https://e"]),
                 ("City", ["Paris", "Paris", "Paris", "Paris", "Paris"]),
                 ("Name", ["Acme", "Beta", "Gamma", "Delta", "Epsilon"]),
                 ("Note", ["", "", "", "", ""]))
    columns = visible_query_columns(query)
    assert [column["column_name"] for column in columns] == ["Name"]


def test_visible_columns_require_exactly_five_rows():
    four = [{"query_row_id": row_id, "cells": [{"column_id": 0, "column_name": "Name", "text": str(row_id)}]}
            for row_id in range(4)]
    with pytest.raises(ValueError, match="exactly 5 rows"):
        visible_query_columns(four)


def test_visible_target_values_skip_urls_and_empty_cells():
    column = {"column_name": "Link", "values": [value_info("https://x/y")]}
    assert visible_target_values(column) == []
    column = {"column_name": "Name", "values": [value_info("Acme"), value_info("--"), value_info("42")]}
    assert [value["key"] for value in visible_target_values(column)] == ["TEXT:acme", "NUMBER:42"]


def test_idf_weight_penalizes_values_present_in_every_candidate():
    scope = 50
    assert idf_weight(1, scope) > idf_weight(10, scope) > idf_weight(49, scope) > 0
    assert idf_weight(scope, scope) == pytest.approx(0.0)
    assert idf_weight(0, scope) == pytest.approx(1.0, abs=1e-9)


def test_bridge_tier_precedes_visible_tier_and_rrf_keeps_stage1_tail():
    base = ["t1", "t2", "t3", "t4"]
    bridge = {"t1": 0.0, "t2": 0.2, "t3": 0.0, "t4": 0.0}
    visible_row = {"t1": 1.0, "t2": 0.0, "t3": 0.4, "t4": 0.0}
    visible_idf = {"t1": 0.9, "t2": 0.0, "t3": 0.1, "t4": 0.0}
    orders = bridge_first_orders(base, bridge, visible_row, visible_idf)
    assert orders["PURE"] == ["t2", "t1", "t3", "t4"]
    assert sorted(orders["RRF60"]) == sorted(base)
    # Under k=60 a single-rank promotion only ties Stage-1 rank 1, so Stage-1 order wins the tie.
    assert orders["RRF60"][:2] == ["t1", "t2"]


def test_tier_two_breaks_ties_on_the_unweighted_row_score():
    base = ["t1", "t2", "t3"]
    bridge = {target: 0.0 for target in base}
    visible_idf = {"t1": 0.5, "t2": 0.5, "t3": 0.0}
    visible_row = {"t1": 0.2, "t2": 0.8, "t3": 0.0}
    assert bridge_first_orders(base, bridge, visible_row, visible_idf)["PURE"] == ["t2", "t1", "t3"]


def test_rrf_fusion_is_stable_for_a_pure_order_equal_to_stage1():
    base = ["a", "b", "c"]
    assert rrf_fuse(base, base) == base


def test_rerank_promotes_a_bridge_target_over_a_visible_only_candidate():
    matcher = UnitCosineMatcher(name_space_store())
    query = rows(("Name", ["Acme", "Beta", "Gamma", "Delta", "Epsilon"]))
    tables = {"t_visible": table("Name", ["Beta", "Gamma", "Delta", "Epsilon"]),
              "t_bridge": table("Ref", ["Acme"])}
    bridges = [{"attribute": "ref",
                "slots": [{"row_id": row_id, "status": "VALUE", "value_key": "TEXT:acme"} for row_id in range(5)],
                "domain": [value_info("Acme")]}]
    base = ["t_visible", "t_bridge"]
    bridge_scores = {"t_visible": 0.0, "t_bridge": 1.0}
    result = rerank(base, bridges, query, tables, matcher, bridge_scores)
    assert result["orders"]["PURE"][0] == "t_bridge"
    assert result["scope_size"] == 2
    assert result["visible_row_scores"]["t_visible"] > 0  # the visible tier still scores candidates


def test_scope_size_drives_the_idf_denominator():
    matcher = UnitCosineMatcher(name_space_store())
    query = rows(("Name", ["Acme", "Beta", "Gamma", "Delta", "Epsilon"]))
    tables = {"t1": table("Name", ["Acme"]), "t2": table("Name", ["Acme"])}
    bridges = [{"attribute": "ref", "slots": [{"row_id": r, "status": "VALUE", "value_key": "TEXT:acme"}
                                              for r in range(5)], "domain": [value_info("Acme")]}]
    result = rerank(["t1", "t2"], bridges, query, tables, matcher, {"t1": 0.0, "t2": 0.0})
    # the value matches every candidate in scope, so its IDF weight is zero
    assert result["visible_idf_scores"]["t1"] == pytest.approx(0.0)
    assert math.isclose(result["visible_row_scores"]["t1"], 0.2)
